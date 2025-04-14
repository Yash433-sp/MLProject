import pandas as pd
import numpy as np
import streamlit as st
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder


best_model = joblib.load("models/best_loan_model.pkl")


data = pd.read_csv("data/loan_approval_dataset.csv", delimiter=";")
data.columns = data.columns.str.strip()
label_enc_edu = LabelEncoder()
data['education'] = label_enc_edu.fit_transform(data['education'])
label_enc_emp = LabelEncoder()
data['self_employed'] = label_enc_emp.fit_transform(data['self_employed'])


scaler = StandardScaler()
X = data.drop(columns=['loan_status', 'loan_id'])
scaler.fit(X)


st.title("Loan Approval Prediction")
st.write("Enter the details below to predict loan approval status.")


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


education_encoded = label_enc_edu.transform([education])[0]
self_employed_encoded = label_enc_emp.transform([self_employed])[0]


user_data = pd.DataFrame([[no_of_dependents, education_encoded, self_employed_encoded, income_annum, loan_amount,
                           loan_term, cibil_score, residential_assets_value, commercial_assets_value,
                           luxury_assets_value, bank_asset_value]],
                         columns=["no_of_dependents", "education", "self_employed", "income_annum", "loan_amount",
                                  "loan_term", "cibil_score", "residential_assets_value", "commercial_assets_value",
                                  "luxury_assets_value", "bank_asset_value"])


user_scaled = scaler.transform(user_data)


if st.button("Predict"):
    prediction = best_model.predict(user_scaled)
    result = "Approved" if prediction[0] == 1 else "Rejected"
    st.write(f"### Loan Status: {result}")
