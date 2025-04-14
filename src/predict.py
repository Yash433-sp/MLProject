import pandas as pd
import numpy as np
import streamlit as st
import joblib
import os

# --- Load all pre-trained components safely ---
try:
    best_model = joblib.load("models/best_loan_model.pkl")
    # label_enc_edu = joblib.load("models/label_encoder_education.pkl")
    # label_enc_emp = joblib.load("models/label_encoder_self_employed.pkl")
    # scaler = joblib.load("models/scaler.pkl")
except Exception as e:
    st.error(f"Failed to load model or encoders: {e}")
    st.stop()

# --- Streamlit UI ---
st.title("Loan Approval Prediction")
st.write("Enter the details below to predict loan approval status.")

# Input form
no_of_dependents = st.number_input("Number of Dependents", min_value=0, step=1)
education = st.selectbox("Education", label_enc_edu.classes_)
self_employed = st.selectbox("Self Employed", label_enc_emp.classes_)
income_annum = st.number_input("Annual Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_term = st.number_input("Loan Term (in months)", min_value=0)
cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900)
residential_assets_value = st.number_input("Residential Assets Value", min_value=0)
commercial_assets_value = st.number_input("Commercial Assets Value", min_value=0)
luxury_assets_value = st.number_input("Luxury Assets Value", min_value=0)
bank_asset_value = st.number_input("Bank Asset Value", min_value=0)

# Encode categorical inputs
try:
    education_encoded = label_enc_edu.transform([education])[0]
    self_employed_encoded = label_enc_emp.transform([self_employed])[0]
except Exception as e:
    st.error(f"Encoding error: {e}")
    st.stop()

# Prepare input
user_data = pd.DataFrame([[
    no_of_dependents, education_encoded, self_employed_encoded, income_annum,
    loan_amount, loan_term, cibil_score, residential_assets_value,
    commercial_assets_value, luxury_assets_value, bank_asset_value
]], columns=[
    "no_of_dependents", "education", "self_employed", "income_annum",
    "loan_amount", "loan_term", "cibil_score", "residential_assets_value",
    "commercial_assets_value", "luxury_assets_value", "bank_asset_value"
])

# Scale
try:
    user_scaled = scaler.transform(user_data)
except Exception as e:
    st.error(f"Scaling error: {e}")
    st.stop()

# Prediction
if st.button("Predict"):
    try:
        prediction = best_model.predict(user_scaled)
        result = "Approved" if prediction[0] in [1, "1", "approved"] else "Rejected"
        st.success(f"### Loan Status: {result}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
