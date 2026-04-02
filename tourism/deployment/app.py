"""app.py – Streamlit front-end for the Tourism Package predictor.
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
# Function to load the trained model from Hugging Face Hub
def load_model():
    # Download model file from the specified repository
    path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FNAME)
    # Load and return the model
    return joblib.load(path)

# Function to load the preprocessor from Hugging Face Hub
def load_preprocessor():
    # Download preprocessor file from the specified repository
    path = hf_hub_download(repo_id=DATASET_REPO, filename=PREP_FNAME, repo_type="dataset")
    # Load and return the preprocessor
    return joblib.load(path)

# Load the model and preprocessor for predictions
model        = load_model()
preprocessor = load_preprocessor()

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
# Configure the Streamlit page with title, icon and layout
st.set_page_config(
    page_title="Tourism Package Predictor",
    page_icon="",
    layout="wide",
)

# Set the main title and description for the application
st.title("Tourism Package Predictor")
st.markdown(
    "Predict whether a customer is likely to purchase the **Tourism Package** "
    "based on their profile and interaction data."
)
st.divider()

# ── INPUT FORM ────────────────────────────────────────────────────────────────
# Create a 3-column layout for organizing input fields
col1, col2, col3 = st.columns(3)

# First column: Customer demographic information
with col1:
    st.subheader("Customer Profile")
    age                     = st.slider("Age", 18, 100, 20)  # Age slider with default 35
    gender                  = st.selectbox("Gender", ["Male", "Female"])
    marital_status          = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    occupation              = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
    designation             = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    monthly_income          = st.number_input("Monthly Income (₹)", 5000, 200000, 30000, step=5000)  # Income in rupees

# Second column: Travel preferences and history
with col2:
    st.subheader("Travel Preferences")
    city_tier               = st.selectbox("City Tier", [1, 2, 3])  # City classification by development level
    own_car                 = st.selectbox("Owns a Car?", [0, 1], format_func=lambda x: "Yes" if x else "No")  # Convert 0/1 to No/Yes
    passport                = st.selectbox("Has Passport?", [0, 1], format_func=lambda x: "Yes" if x else "No")  # Convert 0/1 to No/Yes
    preferred_star          = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])  # Hotel star preference
    num_trips               = st.slider("Number of Trips per Year", 1, 25, 5)  # Annual travel frequency
    num_person_visiting     = st.slider("No. of Persons Visiting", 1, 5, 2)  # Group size
    num_children_visiting   = st.slider("No. of Children Visiting (<5 yrs)", 0, 3, 1)  # Number of young children

with col3:
    st.subheader("Sales Data")
    type_of_contact         = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    product_pitched         = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    pitch_satisfaction      = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    num_followups           = st.slider("Number of Follow-ups", 1.0, 6.0, 3.0, step=1.0)
    duration_of_pitch       = st.slider("Duration of Pitch (minutes)", 5, 60, 10)

# Visual separator
st.divider()
# Primary button to trigger prediction
predict_btn = st.button("Purchase Prediction", type="primary", use_container_width=True)

# ── PREDICTION SECTION ────────────────────────────────────────────────────────────────
if predict_btn:
    # Create a DataFrame with all user inputs for model prediction
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

    # Apply preprocessing transformation to input data
    processed   = preprocessor.transform(input_data)
    # Get binary prediction (0 or 1)
    prediction  = model.predict(processed)[0]
    # Get probability of positive class (likelihood of purchase)
    probability = model.predict_proba(processed)[0][1]

    # Display prediction results
    st.subheader("Result")
    if prediction == 1:
        # Show success message with purchase probability if likely to purchase
        st.success(f"✅ **LIKELY TO PURCHASE** – Confidence: {probability:.1%}")
        st.balloons()  # Celebratory animation
    else:
        # Show warning message with non-purchase probability if unlikely to purchase
        st.warning(f"❌ **UNLIKELY TO PURCHASE** – Confidence of NOT purchasing: {1-probability:.1%}")

    # Visual representation of purchase probability
    st.progress(float(probability), text=f"Purchase probability: {probability:.1%}")

    # Expandable section to view all input data
    with st.expander("View input summary"):
        st.dataframe(input_data.T.rename(columns={0: "Value"}))

# Footer with model information
st.caption("Model: Visit with Us – Tourism Package | MLOps Pipeline powered by GitHub Actions & Hugging Face")
