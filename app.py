# Step 1:
from __future__ import annotations
import joblib
from joblib import  load
from pathlib import Path

import pandas as pd
import streamlit as st

from crop_recommendation_pipeline import recommend_top3

# ✅ Robust path (works locally + Streamlit Cloud)
BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_PATH = OUTPUT_DIR / "models" / "random_forest.joblib"
IMAGE_PATH = BASE_DIR / "images" / "CTAE_logo.jpg"

st.set_page_config(page_title="Crop Recommendation", layout="wide")


# Step 2
col1, col2 = st.columns([1, 8], vertical_alignment="center")

with col1:
    if IMAGE_PATH.exists():
        st.image(str(IMAGE_PATH), width=70)
    else:
        st.warning("Logo not found")

with col2:
    st.markdown("""
    <h3 style='margin-bottom:0;'>COLLEGE OF TECHNOLOGY AND ENGINEERING, UDAIPUR</h3>
    <p style='margin-top:0;'>Department of Agricultural Engineering</p>
    """, unsafe_allow_html=True)

# Step 3: Load data & Model
@st.cache_data
def load_tables():
    try:
        lookup = pd.read_csv(OUTPUT_DIR / "district_season_lookup.csv")
        evidence = pd.read_csv(OUTPUT_DIR / "district_crop_evidence.csv")
        catalog = pd.read_csv(OUTPUT_DIR / "crop_catalog.csv")
        return lookup, evidence, catalog
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()


@st.cache_resource
def load_model():
    try:
         return load(MODEL_PATH)
    
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        st.stop()


lookup_df, evidence_df, catalog_df = load_tables()
model = load_model()

# Step 4: Styling (UI Design)
st.markdown("""
<style>
/* Correct selector */
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}

.card {
    background: rgba(255,255,255,0.06);
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 20px;
}

.crop-card {
    background: rgba(255,255,255,0.08);
    padding: 15px;
    border-radius: 10px;
    margin-bottom: 12px;
}

.progress-bar {
    height: 8px;
    background: #333;
    border-radius: 5px;
    margin-top: 5px;
}

.progress-fill {
    height: 8px;
    background: #4CAF50;
    border-radius: 5px;
}

/* Button styling */
div.stButton > button {
    width: 100%;
    background-color: #2e7d32;
    color: white;
    border-radius: 8px;
    height: 45px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Step 5: App Title
st.markdown("""
<h1 style='text-align:center; margin-bottom:5px;'>
🌱 High Value Crop Recommendation System
</h1>
<hr>
""", unsafe_allow_html=True)

#Step 6: Input Panel
districts = sorted(lookup_df["district"].unique())
seasons = ["Kharif", "Rabi"]

left, right = st.columns([1, 2])

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown("### Input Parameters")

    district = st.selectbox("Select District", districts)
    season = st.selectbox("Select Season", seasons)

    st.markdown("<br>", unsafe_allow_html=True)

    run = st.button("🔍 Get Recommendation", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# Step 7: Crop Card Function
def crop_card(rank, crop, score):
    percent = int(score * 100)

    st.markdown(f"""
    <div class="crop-card">
        <h4>#{rank} {crop}</h4>
        <p>Suitability: {percent}%</p>
        <div class="progress-bar">
            <div class="progress-fill" style="width:{percent}%"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Step 8: Output Panel
with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown("### 🌾 Recommended Crops")

    if run:
        with st.spinner("Generating recommendations..."):
            try:
                recommendations = recommend_top3(
                    model, district, season,
                    lookup_df, evidence_df, catalog_df
                )

                recommendations = recommendations.sort_values(
                    by="suitability_probability", ascending=False
                ).reset_index(drop=True)

                for i in range(len(recommendations)):
                    row = recommendations.iloc[i]
                    crop_card(i + 1, row["crop"], row["suitability_probability"])

                st.markdown("### 🌿 Why these crops?")
                st.markdown("""
                - Soil and pH are suitable  
                - Climate conditions match  
                - Based on district-level data  
                """)

                st.markdown("### 📊 Conditions Used")

                feature_data = lookup_df[
                    (lookup_df["district"] == district) &
                    (lookup_df["Season"] == season)
                ]

                if not feature_data.empty:
                    row = feature_data.iloc[0]

                    st.write(f"🌧 Rainfall: {row['rainfall']:.2f} mm")
                    st.write(f"🌡 Temperature: {row['temperature']:.2f} °C")
                    st.write(f"💧 Humidity: {row['humidity']:.2f}%")
                    st.write(f"🧪 pH: {row['ph']:.2f}")
                else:
                    st.warning("No data available for selected input")

            except Exception as e:
                st.error(f"Error: {e}")

    else:
        st.info("👈 Select inputs and click the button")

    st.markdown('</div>', unsafe_allow_html=True)
    

# st.caption("Final Year Project | Crop Recommendation System | Rajasthan")
st.markdown("""
<div style="
    background: linear-gradient(90deg, #d4edda, #c3e6cb);
    padding:14px;
    border-radius:12px;
    text-align:center;
    margin-top:35px;
    color:#155724;
    font-weight:600;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
">
    🌱 Final Year Project | Crop Recommendation System | Rajasthan
</div>
""", unsafe_allow_html=True)
