from __future__ import annotations

import argparse
import ast
import json
import math
import pickle
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from pathlib import Path


BASE_DIR = Path(__file__).parent

DEFAULT_FEATURE_CSV = BASE_DIR / "assets" / "climate_soil_data.csv"
DEFAULT_MAPPING_NOTEBOOK = BASE_DIR / "assets" / "crop_logic.ipynb"
DEFAULT_CATALOG_DOCX = BASE_DIR / "assets" / "project_crop_recommendation.docx"
DEFAULT_OUTPUT_DIR = Path("outputs")
RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "rainfall",
    "temperature",
    "humidity",
    "ph",
    "organic_carbon",
    "sand",
    "clay",
    "silt",
    "ph_match_score",
    "soil_match_score",
    "temp_match_score",
    "rain_match_score",
    "suitability_score",
]
CATEGORICAL_FEATURES = ["district", "Season", "soil_texture", "crop"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

DISTRICT_ALIASES = {
    "Chittaurgarh": "Chittorgarh",
    "Chittorgarh": "Chittorgarh",
    "Dholpur": "Dhaulpur",
    "Dhaulpur": "Dhaulpur",
    "Ganganagar": "Sri Ganganagar",
    "Jalor": "Jalore",
    "Jalore": "Jalore",
    "Jhunjhunun": "Jhunjhunu",
    "Jhunjhunu": "Jhunjhunu",
    "Sri Ganganagar": "Sri Ganganagar",
}

CROP_ALIASES = {
    "Indian Mustard": "Mustard",
    "Paddy (Basmati)": "Rice",
    "Paddy Basmati": "Rice",
    "Basmati": "Rice",
    "Cluster Bean": "Guar",
    "Cluster Bean (Guar)": "Guar",
    "Guar Bean": "Guar",
    "Cumin (Jeera)": "Cumin",
    "Jeera": "Cumin",
    "Isabgol (Psyllium)": "Isabgol",
    "Psyllium": "Isabgol",
    "Garden Pea": "Pea",
    "Green Pea": "Pea",
    "Malt Barley": "Barley",
    "Red Chilli": "Chilli",
    "Nigella": "Kalonji",
    "Henna": "Mehandi",
    "Henna (Mehandi)": "Mehandi",
    "Urdbean": "Blackgram",
    "Urd Bean": "Blackgram",
    "Black Gram": "Blackgram",
    "Sesamum": "Sesame",
}


@dataclass(frozen=True)
class RangeValue:
    low: float | None
    high: float | None

    def contains(self, value: float | None) -> bool | None:
        if value is None or math.isnan(value):
            return None
        if self.low is not None and value < self.low:
            return False
        if self.high is not None and value > self.high:
            return False
        return True


def normalize_spaces(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip())


def normalize_district(value: Any) -> str:
    text = normalize_spaces(value)
    return DISTRICT_ALIASES.get(text, text)


def normalize_crop(value: Any) -> str:
    text = normalize_spaces(value).replace("_", " ")
    text = re.sub(r"\s*\([^)]*\)\s*", " ", text)
    text = normalize_spaces(text)
    return CROP_ALIASES.get(text, text)


def normalize_season(value: Any) -> str:
    text = normalize_spaces(value).lower()
    if "perennial" in text or "bahar" in text or "year" in text:
        return "Perennial"
    if "rabi" in text or "winter" in text:
        return "Rabi"
    if "kharif" in text or "monsoon" in text:
        return "Kharif"
    if "zaid" in text or "summer" in text or "spring" in text:
        return "Zaid"
    return normalize_spaces(value)


def parse_numeric_range(text: Any, kind: str) -> RangeValue:
    raw = normalize_spaces(text)
    if not raw:
        return RangeValue(None, None)
    lowered = raw.lower()
    if any(token in lowered for token in ["not found", "notfound", "rain-free", "moist", "irrig"]):
        return RangeValue(None, None)
    if kind == "rain" and not any(ch.isdigit() for ch in raw):
        return RangeValue(None, None)

    cleaned = (
        raw.replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
        .replace("~", "")
        .replace(",", "")
    )
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", cleaned)]
    if not nums:
        return RangeValue(None, None)

    if "<" in cleaned:
        return RangeValue(None, nums[0])
    if ">" in cleaned or "≥" in raw:
        return RangeValue(nums[0], None)

    if len(nums) >= 2:
        low, high = nums[0], nums[1]
        if low > high:
            low, high = high, low
        return RangeValue(low, high)

    value = nums[0]
    if kind == "ph":
        return RangeValue(max(0.0, value - 0.5), min(14.0, value + 0.5))
    return RangeValue(value, value)


def range_to_columns(prefix: str, range_value: RangeValue) -> dict[str, float | None]:
    return {f"{prefix}_min": range_value.low, f"{prefix}_max": range_value.high}


def canonical_soil_text(text: Any) -> str:
    lowered = normalize_spaces(text).lower()
    lowered = lowered.replace("-", " ")
    lowered = lowered.replace("/", " ")
    lowered = lowered.replace("loamy", "loam")
    lowered = lowered.replace("med ", "medium ")
    lowered = re.sub(r"[^a-z\s]", " ", lowered)
    stop = {
        "well",
        "drained",
        "fertile",
        "heavy",
        "light",
        "medium",
        "deep",
        "textured",
        "soil",
        "black",
        "brown",
        "red",
        "alluvial",
        "stony",
        "coarse",
        "shallow",
    }
    tokens = [token for token in lowered.split() if token and token not in stop]
    return " ".join(tokens)


def soil_matches(observed: Any, requirement: Any) -> bool | None:
    observed_text = canonical_soil_text(observed)
    requirement_text = canonical_soil_text(requirement)
    if not observed_text or not requirement_text:
        return None
    if observed_text in requirement_text or requirement_text in observed_text:
        return True
    observed_tokens = set(observed_text.split())
    requirement_tokens = set(requirement_text.split())
    if not observed_tokens or not requirement_tokens:
        return None
    important = {"sand", "sandy", "clay", "loam", "silt"}
    overlap = (observed_tokens & requirement_tokens) & important
    return bool(overlap)


def score_or_unknown(match: bool | None) -> float:
    if match is None:
        return 0.5
    return 1.0 if match else 0.0


def load_final_crop_mappings(notebook_path: Path) -> dict[str, dict[str, list[str]]]:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    for cell in notebook.get("cells", []):
        source = "".join(cell.get("source", []))
        if "final_crop_mappings" not in source or "final_crop_mappings =" not in source:
            continue
        module = ast.parse(source)
        for node in module.body:
            if not isinstance(node, ast.Assign):
                continue
            if any(isinstance(target, ast.Name) and target.id == "final_crop_mappings" for target in node.targets):
                raw = ast.literal_eval(node.value)
                return {
                    normalize_district(district): {
                        normalize_season(season): [normalize_crop(crop) for crop in crops]
                        for season, crops in season_map.items()
                    }
                    for district, season_map in raw.items()
                }
    raise ValueError(f"Could not find final_crop_mappings in {notebook_path}")


def build_district_crop_evidence(mapping: dict[str, dict[str, list[str]]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for district, season_map in mapping.items():
        for season, crops in season_map.items():
            for rank, crop in enumerate(crops, start=1):
                rows.append(
                    {
                        "district": normalize_district(district),
                        "Season": normalize_season(season),
                        "crop": normalize_crop(crop),
                        "source": "institutional_research_mapping",
                        "source_rank": rank,
                    }
                )
    evidence = pd.DataFrame(rows).drop_duplicates(["district", "Season", "crop"])
    return evidence.sort_values(["district", "Season", "source_rank", "crop"]).reset_index(drop=True)


def extract_docx_tables(docx_path: Path) -> list[list[list[str]]]:
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    with zipfile.ZipFile(docx_path) as archive:
        xml = archive.read("word/document.xml")
    root = ET.fromstring(xml)
    tables: list[list[list[str]]] = []
    for table in root.findall(".//w:tbl", ns):
        rows: list[list[str]] = []
        for tr in table.findall("./w:tr", ns):
            cells: list[str] = []
            for tc in tr.findall("./w:tc", ns):
                paragraphs: list[str] = []
                for para in tc.findall("./w:p", ns):
                    text = "".join(t.text or "" for t in para.findall(".//w:t", ns)).strip()
                    if text:
                        paragraphs.append(text)
                cells.append(normalize_spaces(" ".join(paragraphs)))
            if any(cells):
                rows.append(cells)
        if rows:
            tables.append(rows)
    return tables


def header_index(header: list[str], *needles: str) -> int | None:
    lowered = [cell.lower() for cell in header]
    for needle in needles:
        for idx, cell in enumerate(lowered):
            if needle in cell:
                return idx
    return None


def table_rows_to_catalog(tables: list[list[list[str]]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    previous_header: list[str] | None = None
    for table_idx, table in enumerate(tables, start=1):
        first = [cell.lower() for cell in table[0]]
        has_header = any("crop" in cell for cell in first) and any("soil" in cell for cell in first)
        if has_header:
            header = table[0]
            data_rows = table[1:]
            previous_header = header
        elif previous_header and len(table[0]) >= 7:
            header = previous_header
            data_rows = table
        elif len(table[0]) >= 7:
            header = [
                "Crop",
                "Soil Texture",
                "Soil pH",
                "Organic Carbon",
                "Temperature",
                "Rainfall",
                "Humidity",
                "Season",
            ]
            data_rows = table
            previous_header = header
        else:
            continue

        crop_idx = header_index(header, "crop")
        soil_idx = header_index(header, "soil texture", "soil")
        ph_idx = header_index(header, "ph")
        oc_idx = header_index(header, "organic")
        temp_idx = header_index(header, "temp")
        rain_idx = header_index(header, "rain", "water")
        humidity_idx = header_index(header, "humid")
        season_idx = header_index(header, "season")
        if crop_idx is None:
            continue

        for row in data_rows:
            if crop_idx >= len(row):
                continue
            crop = normalize_crop(row[crop_idx])
            if not crop or crop.lower() in {"crop", "crop name", "high-value crop"}:
                continue
            soil_text = row[soil_idx] if soil_idx is not None and soil_idx < len(row) else ""
            season_text = row[season_idx] if season_idx is not None and season_idx < len(row) else ""
            ph_text = row[ph_idx] if ph_idx is not None and ph_idx < len(row) else ""
            temp_text = row[temp_idx] if temp_idx is not None and temp_idx < len(row) else ""
            rain_text = row[rain_idx] if rain_idx is not None and rain_idx < len(row) else ""
            oc_text = row[oc_idx] if oc_idx is not None and oc_idx < len(row) else ""
            humidity_text = row[humidity_idx] if humidity_idx is not None and humidity_idx < len(row) else ""
            ph_range = parse_numeric_range(ph_text, "ph")
            temp_range = parse_numeric_range(temp_text, "temp")
            rain_range = parse_numeric_range(rain_text, "rain")
            rows.append(
                {
                    "crop": crop,
                    "valid_season": normalize_season(season_text),
                    "soil_requirement": soil_text,
                    "ph_text": ph_text,
                    "temp_text": temp_text,
                    "rain_text": rain_text,
                    "organic_carbon_text": oc_text,
                    "humidity_text": humidity_text,
                    "source": f"Project Crop Recommendation.docx table {table_idx}",
                    **range_to_columns("ph", ph_range),
                    **range_to_columns("temp", temp_range),
                    **range_to_columns("rain", rain_range),
                }
            )
    if not rows:
        return pd.DataFrame()
    catalog = pd.DataFrame(rows)
    catalog = catalog.drop_duplicates(
        [
            "crop",
            "valid_season",
            "soil_requirement",
            "ph_text",
            "temp_text",
            "rain_text",
        ]
    )
    return catalog.sort_values(["crop", "valid_season", "source"]).reset_index(drop=True)


def ensure_catalog_covers_evidence(catalog: pd.DataFrame, evidence: pd.DataFrame) -> pd.DataFrame:
    existing = set(catalog["crop"]) if not catalog.empty else set()
    fallback_rows: list[dict[str, Any]] = []
    for crop, seasons in evidence.groupby("crop")["Season"]:
        if crop in existing:
            continue
        for season in sorted(set(seasons)):
            fallback_rows.append(
                {
                    "crop": crop,
                    "valid_season": season,
                    "soil_requirement": "",
                    "ph_text": "",
                    "temp_text": "",
                    "rain_text": "",
                    "organic_carbon_text": "",
                    "humidity_text": "",
                    "source": "evidence_mapping_fallback",
                    "ph_min": None,
                    "ph_max": None,
                    "temp_min": None,
                    "temp_max": None,
                    "rain_min": None,
                    "rain_max": None,
                }
            )
    if fallback_rows:
        catalog = pd.concat([catalog, pd.DataFrame(fallback_rows)], ignore_index=True)
    return catalog.sort_values(["crop", "valid_season", "source"]).reset_index(drop=True)


def load_and_validate_feature_source(feature_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(feature_csv)
    required = {
        "district",
        "Year",
        "Month",
        "Season",
        "rainfall",
        "temperature",
        "humidity",
        "ph",
        "organic_carbon",
        "sand",
        "clay",
        "silt",
        "soil_texture",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Feature CSV is missing required columns: {missing}")
    df = df.copy()
    df["district"] = df["district"].map(normalize_district)
    df["Season"] = df["Season"].map(normalize_season)
    df["soil_texture"] = df["soil_texture"].map(normalize_spaces)

    ph_min, ph_max = df["ph"].min(), df["ph"].max()
    if ph_min < 2 or ph_max > 11:
        raise ValueError(
            "Invalid pH scale detected. Use the aggregate source with pH around 5-9; "
            f"found range {ph_min:.3f}-{ph_max:.3f}."
        )
    texture_sum = df[["sand", "clay", "silt"]].sum(axis=1)
    if not texture_sum.between(98, 102).all():
        bad = df.loc[~texture_sum.between(98, 102), ["district", "Year", "Month", "sand", "clay", "silt"]].head()
        raise ValueError(f"Sand+clay+silt should be about 100. Example bad rows:\n{bad}")
    return df


def aggregate_district_season_year(df: pd.DataFrame) -> pd.DataFrame:
    aggregated = (
        df.groupby(["district", "Year", "Season"], as_index=False)
        .agg(
            rainfall=("rainfall", "sum"),
            temperature=("temperature", "mean"),
            humidity=("humidity", "mean"),
            ph=("ph", "mean"),
            organic_carbon=("organic_carbon", "mean"),
            sand=("sand", "mean"),
            clay=("clay", "mean"),
            silt=("silt", "mean"),
            soil_texture=("soil_texture", lambda x: x.mode().iat[0] if not x.mode().empty else x.iloc[0]),
        )
        .sort_values(["district", "Year", "Season"])
        .reset_index(drop=True)
    )
    duplicate_count = aggregated.duplicated(["district", "Year", "Season"]).sum()
    if duplicate_count:
        raise ValueError(f"Expected one row per district-year-season, found {duplicate_count} duplicates.")
    return aggregated


def build_prediction_lookup(district_season_features: pd.DataFrame) -> pd.DataFrame:
    lookup = (
        district_season_features.groupby(["district", "Season"], as_index=False)
        .agg(
            rainfall=("rainfall", "mean"),
            temperature=("temperature", "mean"),
            humidity=("humidity", "mean"),
            ph=("ph", "mean"),
            organic_carbon=("organic_carbon", "mean"),
            sand=("sand", "mean"),
            clay=("clay", "mean"),
            silt=("silt", "mean"),
            soil_texture=("soil_texture", lambda x: x.mode().iat[0] if not x.mode().empty else x.iloc[0]),
        )
        .sort_values(["district", "Season"])
        .reset_index(drop=True)
    )
    return lookup


def best_catalog_row(catalog: pd.DataFrame, crop: str, season: str) -> pd.Series | None:
    crop_rows = catalog[catalog["crop"] == crop]
    if crop_rows.empty:
        return None
    season_rows = crop_rows[crop_rows["valid_season"].isin([season, ""])]
    if not season_rows.empty:
        crop_rows = season_rows

    def available_count(row: pd.Series) -> int:
        count = 0
        for prefix in ["ph", "temp", "rain"]:
            if pd.notna(row.get(f"{prefix}_min")) or pd.notna(row.get(f"{prefix}_max")):
                count += 1
        if normalize_spaces(row.get("soil_requirement", "")):
            count += 1
        return count

    ranked = crop_rows.assign(_available=crop_rows.apply(available_count, axis=1))
    return ranked.sort_values(["_available", "source"], ascending=[False, True]).iloc[0]


def compute_audit_features(base_row: pd.Series | dict[str, Any], crop: str, catalog: pd.DataFrame) -> dict[str, Any]:
    season = normalize_season(base_row["Season"])
    catalog_row = best_catalog_row(catalog, crop, season)
    if catalog_row is None:
        ph_match = soil_match = temp_match = rain_match = None
        source = "missing_catalog"
    else:
        ph_match = RangeValue(
            catalog_row.get("ph_min") if pd.notna(catalog_row.get("ph_min")) else None,
            catalog_row.get("ph_max") if pd.notna(catalog_row.get("ph_max")) else None,
        ).contains(float(base_row["ph"]))
        temp_match = RangeValue(
            catalog_row.get("temp_min") if pd.notna(catalog_row.get("temp_min")) else None,
            catalog_row.get("temp_max") if pd.notna(catalog_row.get("temp_max")) else None,
        ).contains(float(base_row["temperature"]))
        rain_match = RangeValue(
            catalog_row.get("rain_min") if pd.notna(catalog_row.get("rain_min")) else None,
            catalog_row.get("rain_max") if pd.notna(catalog_row.get("rain_max")) else None,
        ).contains(float(base_row["rainfall"]))
        soil_match = soil_matches(base_row["soil_texture"], catalog_row.get("soil_requirement", ""))
        source = catalog_row.get("source", "unknown")

    scores = [score_or_unknown(x) for x in [ph_match, soil_match, temp_match, rain_match]]
    return {
        "crop": crop,
        "ph_match": ph_match,
        "soil_match": soil_match,
        "temp_match": temp_match,
        "rain_match": rain_match,
        "ph_match_score": scores[0],
        "soil_match_score": scores[1],
        "temp_match_score": scores[2],
        "rain_match_score": scores[3],
        "suitability_score": float(np.mean(scores)),
        "catalog_source": source,
    }


def deterministic_sample(values: list[str], n: int, seed_key: str) -> list[str]:
    if n >= len(values):
        return list(values)
    seed = abs(hash(seed_key)) % (2**32)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(values), size=n, replace=False)
    return [values[i] for i in sorted(idx)]


def build_ranking_dataset(
    district_season_features: pd.DataFrame,
    evidence: pd.DataFrame,
    catalog: pd.DataFrame,
    negative_ratio: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    evidence_lookup = (
        evidence.groupby(["district", "Season"])["crop"].apply(lambda x: sorted(set(x))).to_dict()
    )
    season_crop_pool = evidence.groupby("Season")["crop"].apply(lambda x: sorted(set(x))).to_dict()
    rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []

    for _, feature_row in district_season_features.iterrows():
        key = (feature_row["district"], feature_row["Season"])
        positives = evidence_lookup.get(key, [])
        if not positives:
            continue
        pool = season_crop_pool.get(feature_row["Season"], [])
        negatives = [crop for crop in pool if crop not in positives]
        n_negatives = min(len(negatives), max(3, len(positives) * negative_ratio))
        sampled_negatives = deterministic_sample(
            negatives,
            n_negatives,
            f"{feature_row['district']}|{feature_row['Year']}|{feature_row['Season']}",
        )
        candidates = [(crop, 1) for crop in positives] + [(crop, 0) for crop in sampled_negatives]
        for crop, label in candidates:
            audit = compute_audit_features(feature_row, crop, catalog)
            output_row = feature_row.to_dict()
            output_row.update(audit)
            output_row["label"] = label
            output_row["group_id"] = f"{feature_row['district']}|{int(feature_row['Year'])}|{feature_row['Season']}"
            rows.append(output_row)
            audit_rows.append(
                {
                    "district": feature_row["district"],
                    "Year": feature_row["Year"],
                    "Season": feature_row["Season"],
                    "crop": crop,
                    "label": label,
                    "ph_match": audit["ph_match"],
                    "soil_match": audit["soil_match"],
                    "temp_match": audit["temp_match"],
                    "rain_match": audit["rain_match"],
                    "suitability_score": audit["suitability_score"],
                    "catalog_source": audit["catalog_source"],
                }
            )
    ranking = pd.DataFrame(rows)
    audit = pd.DataFrame(audit_rows)
    return ranking, audit


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def train_models(ranking_df: pd.DataFrame, output_dir: Path) -> tuple[dict[str, Any], dict[str, Pipeline]]:
    X = ranking_df[MODEL_FEATURES]
    y = ranking_df["label"].astype(int)
    groups = ranking_df["group_id"]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    models: dict[str, Pipeline] = {
        "logistic_regression": Pipeline(
            [
                ("preprocess", make_preprocessor()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        solver="lbfgs",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("preprocess", make_preprocessor()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=400,
                        max_depth=12,
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
    }

    try:
        from xgboost import XGBClassifier

        positives = max(1, int(y_train.sum()))
        negatives = max(1, int((1 - y_train).sum()))
        models["xgboost"] = Pipeline(
            [
                ("preprocess", make_preprocessor()),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=300,
                        max_depth=4,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        eval_metric="logloss",
                        scale_pos_weight=negatives / positives,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
    except Exception as exc:
        print(f"Skipping XGBoost because it is unavailable: {exc}")

    metrics: dict[str, Any] = {
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "positive_rate": float(y.mean()),
        "models": {},
    }
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    trained: dict[str, Pipeline] = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="binary", zero_division=0
        )
        metrics["models"][name] = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "classification_report": classification_report(y_test, y_pred, zero_division=0, output_dict=True),
        }
        with (model_dir / f"{name}.pkl").open("wb") as handle:
            pickle.dump(model, handle)
        trained[name] = model
    return metrics, trained


def build_candidate_rows(
    feature_row: pd.Series | dict[str, Any],
    candidate_crops: list[str],
    catalog: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for crop in candidate_crops:
        audit = compute_audit_features(feature_row, crop, catalog)
        row = dict(feature_row)
        row.update(audit)
        rows.append(row)
    return pd.DataFrame(rows)


def recommend_top3(
    model: Pipeline,
    district: str,
    season: str,
    lookup: pd.DataFrame,
    evidence: pd.DataFrame,
    catalog: pd.DataFrame,
    fallback_pool: bool = True,
) -> pd.DataFrame:
    district = normalize_district(district)
    season = normalize_season(season)
    feature_match = lookup[(lookup["district"] == district) & (lookup["Season"] == season)]
    if feature_match.empty:
        raise ValueError(f"No feature lookup row for {district} / {season}")
    feature_row = feature_match.iloc[0].to_dict()

    district_candidates = sorted(
        evidence[(evidence["district"] == district) & (evidence["Season"] == season)]["crop"].unique()
    )
    if fallback_pool and len(district_candidates) < 3:
        season_pool = sorted(evidence[evidence["Season"] == season]["crop"].unique())
        candidate_crops = sorted(set(district_candidates) | set(season_pool))
    else:
        candidate_crops = district_candidates
    if not candidate_crops:
        raise ValueError(f"No candidate crops for {district} / {season}")

    candidates = build_candidate_rows(feature_row, candidate_crops, catalog)
    probabilities = model.predict_proba(candidates[MODEL_FEATURES])[:, 1]
    candidates["suitability_probability"] = probabilities
    candidates["in_district_evidence"] = candidates["crop"].isin(district_candidates)
    candidates = candidates.sort_values(
        ["in_district_evidence", "suitability_probability", "suitability_score"],
        ascending=[False, False, False],
    )
    return candidates.head(3).reset_index(drop=True)


def evaluate_top3(
    model: Pipeline,
    lookup: pd.DataFrame,
    evidence: pd.DataFrame,
    catalog: pd.DataFrame,
) -> dict[str, float]:
    precision_scores: list[float] = []
    hit_scores: list[float] = []
    for _, row in lookup.iterrows():
        true_crops = set(evidence[(evidence["district"] == row["district"]) & (evidence["Season"] == row["Season"])]["crop"])
        if not true_crops:
            continue
        try:
            recs = recommend_top3(model, row["district"], row["Season"], lookup, evidence, catalog)
        except ValueError:
            continue
        predicted = set(recs["crop"])
        overlap = len(predicted & true_crops)
        precision_scores.append(overlap / max(1, len(predicted)))
        hit_scores.append(1.0 if overlap > 0 else 0.0)
    return {
        "top3_precision_against_evidence": float(np.mean(precision_scores)) if precision_scores else 0.0,
        "top3_hit_rate_against_evidence": float(np.mean(hit_scores)) if hit_scores else 0.0,
        "evaluated_district_seasons": int(len(precision_scores)),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_pipeline(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_features = load_and_validate_feature_source(Path(args.feature_csv))
    district_season_features = aggregate_district_season_year(raw_features)
    lookup = build_prediction_lookup(district_season_features)

    mapping = load_final_crop_mappings(Path(args.mapping_notebook))
    evidence = build_district_crop_evidence(mapping)

    catalog = table_rows_to_catalog(extract_docx_tables(Path(args.catalog_docx)))
    catalog = ensure_catalog_covers_evidence(catalog, evidence)

    ranking_df, mapping_audit = build_ranking_dataset(
        district_season_features[district_season_features["Season"].isin(["Kharif", "Rabi"])],
        evidence[evidence["Season"].isin(["Kharif", "Rabi"])],
        catalog,
        negative_ratio=args.negative_ratio,
    )
    if ranking_df.empty:
        raise ValueError("Ranking dataset is empty. Check district/season names in features and evidence.")

    district_season_features.to_csv(output_dir / "district_season_features.csv", index=False)
    lookup.to_csv(output_dir / "district_season_lookup.csv", index=False)
    evidence.to_csv(output_dir / "district_crop_evidence.csv", index=False)
    catalog.to_csv(output_dir / "crop_catalog.csv", index=False)
    ranking_df.to_csv(output_dir / "training_rank_dataset.csv", index=False)
    mapping_audit.to_csv(output_dir / "mapping_audit.csv", index=False)

    validation = {
        "feature_source": str(Path(args.feature_csv)),
        "districts_in_features": int(raw_features["district"].nunique()),
        "districts_in_evidence": int(evidence["district"].nunique()),
        "district_season_year_rows": int(len(district_season_features)),
        "lookup_rows": int(len(lookup)),
        "evidence_rows": int(len(evidence)),
        "catalog_rows": int(len(catalog)),
        "ranking_rows": int(len(ranking_df)),
        "ranking_positive_rows": int(ranking_df["label"].sum()),
        "ranking_negative_rows": int((1 - ranking_df["label"]).sum()),
        "missing_evidence_districts": sorted(set(raw_features["district"]) - set(evidence["district"])),
        "missing_feature_districts": sorted(set(evidence["district"]) - set(raw_features["district"])),
        "seasons_without_evidence": sorted(set(raw_features["Season"]) - set(evidence["Season"])),
    }
    write_json(output_dir / "validation_summary.json", validation)

    metrics, trained_models = train_models(ranking_df, output_dir)
    model_for_examples = trained_models.get("random_forest") or next(iter(trained_models.values()))
    top3_metrics = evaluate_top3(
        model_for_examples,
        lookup[lookup["Season"].isin(["Kharif", "Rabi"])],
        evidence,
        catalog,
    )
    metrics["top3_random_forest_or_default"] = top3_metrics
    write_json(output_dir / "model_metrics.json", metrics)

    examples: list[pd.DataFrame] = []
    for district in ["Kota", "Udaipur", "Jaisalmer", "Jaipur", "Barmer"]:
        for season in ["Rabi", "Kharif"]:
            try:
                recs = recommend_top3(model_for_examples, district, season, lookup, evidence, catalog)
                recs.insert(0, "requested_district", district)
                recs.insert(1, "requested_season", season)
                examples.append(recs)
            except ValueError as exc:
                examples.append(
                    pd.DataFrame(
                        [
                            {
                                "requested_district": district,
                                "requested_season": season,
                                "crop": "",
                                "warning": str(exc),
                            }
                        ]
                    )
                )
    pd.concat(examples, ignore_index=True).to_csv(output_dir / "top3_examples.csv", index=False)

    print("Pipeline complete.")
    print(f"Outputs written to: {output_dir.resolve()}")
    print(json.dumps(validation, indent=2))
    print(json.dumps({k: v for k, v in metrics.items() if k != "models"}, indent=2))
    for model_name, model_metrics in metrics["models"].items():
        print(
            f"{model_name}: accuracy={model_metrics['accuracy']:.3f}, "
            f"precision={model_metrics['precision']:.3f}, recall={model_metrics['recall']:.3f}, "
            f"f1={model_metrics['f1']:.3f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build crop suitability ranking dataset and train top-3 models.")
    parser.add_argument("--feature-csv", default=str(DEFAULT_FEATURE_CSV))
    parser.add_argument("--mapping-notebook", default=str(DEFAULT_MAPPING_NOTEBOOK))
    parser.add_argument("--catalog-docx", default=str(DEFAULT_CATALOG_DOCX))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--negative-ratio", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    run_pipeline(parse_args())
