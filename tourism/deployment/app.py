"""app.py – Streamlit front-end for the Wellness Tourism Package predictor.
Deployed on Hugging Face Spaces.
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from huggingface_hub import hf_hub_download

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
HF_USERNAME   = os.getenv("HF_USERNAME", "vivekkumar-hf")
MODEL_REPO    = f"{HF_USERNAME}/tourism-model"
DATASET_REPO  = f"{HF_USERNAME}/tourism-data"
MODEL_FNAME   = "best_tourism_model_v1.joblib"
PREP_FNAME    = "preprocessor.joblib"

# ── LOAD MODEL & PREPROCESSOR (cached) ───────────────────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FNAME, repo_type="model")
    return joblib.load(path)

@st.cache_resource(show_spinner="Loading preprocessor...")
def load_preprocessor():
    path = hf_hub_download(repo_id=DATASET_REPO, filename=PREP_FNAME, repo_type="dataset")
    return joblib.load(path)

model        = load_model()
preprocessor = load_preprocessor()

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Wellness Tourism Package Predictor",
    page_icon="✈️",
    layout="wide",
)

st.title("✈️ Wellness Tourism Package Predictor")
st.markdown(
    "Predict whether a customer is likely to purchase the **Wellness Tourism Package** "
    "based on their profile and interaction data."
)
st.divider()

# ── INPUT FORM ────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Customer Profile")
    age                     = st.slider("Age", 18, 80, 35)
    gender                  = st.selectbox("Gender", ["Male", "Female"])
    marital_status          = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    occupation              = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
    designation             = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    monthly_income          = st.number_input("Monthly Income (₹)", 5000, 100000, 20000, step=1000)

with col2:
    st.subheader("Travel Preferences")
    city_tier               = st.selectbox("City Tier", [1, 2, 3])
    own_car                 = st.selectbox("Owns a Car?", [0, 1], format_func=lambda x: "Yes" if x else "No")
    passport                = st.selectbox("Has Passport?", [0, 1], format_func=lambda x: "Yes" if x else "No")
    preferred_star          = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
    num_trips               = st.slider("Number of Trips per Year", 1, 22, 3)
    num_person_visiting     = st.slider("No. of Persons Visiting", 1, 5, 2)
    num_children_visiting   = st.slider("No. of Children Visiting (<5 yrs)", 0, 3, 0)

with col3:
    st.subheader("Sales Interaction")
    type_of_contact         = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    product_pitched         = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    pitch_satisfaction      = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    num_followups           = st.slider("Number of Follow-ups", 1.0, 6.0, 3.0, step=1.0)
    duration_of_pitch       = st.slider("Duration of Pitch (minutes)", 5, 60, 15)

st.divider()
predict_btn = st.button("🔮 Predict Purchase Likelihood", type="primary", use_container_width=True)

# ── PREDICTION ────────────────────────────────────────────────────────────────
if predict_btn:
    input_data = pd.DataFrame([{
        "Age": age,
        "CityTier": city_tier,
        "DurationOfPitch": duration_of_pitch,
        "NumberOfPersonVisiting": num_person_visiting,
        "NumberOfFollowups": num_followups,
        "PreferredPropertyStar": preferred_star,
        "NumberOfTrips": num_trips,
        "Passport": passport,
        "PitchSatisfactionScore": pitch_satisfaction,
        "OwnCar": own_car,
        "NumberOfChildrenVisiting": num_children_visiting,
        "MonthlyIncome": monthly_income,
        "TypeofContact": type_of_contact,
        "Occupation": occupation,
        "Gender": gender,
        "ProductPitched": product_pitched,
        "MaritalStatus": marital_status,
        "Designation": designation,
    }])

    processed   = preprocessor.transform(input_data)
    prediction  = model.predict(processed)[0]
    probability = model.predict_proba(processed)[0][1]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.success(f"✅ **LIKELY TO PURCHASE** – Confidence: {probability:.1%}")
        st.balloons()
    else:
        st.warning(f"❌ **UNLIKELY TO PURCHASE** – Confidence of NOT purchasing: {1-probability:.1%}")

    # Show probability bar
    st.progress(float(probability), text=f"Purchase probability: {probability:.1%}")

    with st.expander("View input summary"):
        st.dataframe(input_data.T.rename(columns={0: "Value"}))

st.caption("Model: Visit with Us – Wellness Tourism Package | MLOps Pipeline powered by GitHub Actions & Hugging Face")
