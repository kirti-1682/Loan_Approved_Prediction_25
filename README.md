import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

# --------------------------
# Load model
# --------------------------
# Make sure you have trained your model and saved it as 'loan_model.pkl'
# Example: pickle.dump(model, open("loan_model.pkl", "wb"))
model = pickle.load(open("loan_model.pkl", "rb"))

# --------------------------
# Title
# --------------------------
st.set_page_config(page_title="Loan Approval Prediction", page_icon="💳", layout="centered")
st.title("💳 Loan Approval Prediction App")
st.write("Fill in the details below to check if your loan will be approved.")

# --------------------------
# Input Fields
# --------------------------
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Co-applicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0)
loan_term = st.selectbox("Loan Amount Term", [12, 36, 60, 84, 120, 180, 240, 300, 360, 480])
credit_history = st.selectbox("Credit History", [1.0, 0.0])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# --------------------------
# Preprocess Input
# --------------------------
def preprocess_input():
    input_data = pd.DataFrame({
        "Gender": [gender],
        "Married": [married],
        "Dependents": [dependents],
        "Education": [education],
        "Self_Employed": [self_employed],
        "ApplicantIncome": [applicant_income],
        "CoapplicantIncome": [coapplicant_income],
        "LoanAmount": [loan_amount],
        "Loan_Amount_Term": [loan_term],
        "Credit_History": [credit_history],
        "Property_Area": [property_area]
    })

    # Encode categorical variables
    for col in ["Gender", "Married", "Education", "Self_Employed", "Property_Area", "Dependents"]:
        input_data[col] = LabelEncoder().fit_transform(input_data[col])

    return input_data

# --------------------------
# Prediction
# --------------------------
if st.button("Predict Loan Approval"):
    processed_data = preprocess_input()
    prediction = model.predict(processed_data)[0]

    if prediction == 1:
        st.success("✅ Congratulations! Your loan is likely to be **Approved**.")
    else:
        st.error("❌ Sorry, your loan is likely to be **Not Approved**.")



Loan approval is a crucial process for financial institutions. This project aims to build a machine learning model that predicts whether a loan application will be approved based on applicant details such as income, education, credit history, and other features.

The project demonstrates data preprocessing, exploratory data analysis (EDA), feature engineering, model building, evaluation, and deployment (optional with Streamlit).
