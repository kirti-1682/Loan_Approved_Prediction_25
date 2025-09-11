import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained model
model = pickle.load(open("loan_model.pkl", "rb"))

# Streamlit app settings
st.set_page_config(page_title="Loan Approval Prediction", page_icon="🏦", layout="centered")

st.title("🏦 Loan Approval Prediction Dashboard")
st.write("Fill in applicant details to check loan approval status.")

# --- Input fields ---
st.title("Loan Approval Prediction")

# Collect input from user
applicant_income = st.number_input("Applicant Income", min_value=0, step=100)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0, step=100)
loan_amount = st.number_input("Loan Amount", min_value=0, step=10)
loan_amount_term = st.number_input("Loan Amount Term (in days)", min_value=0, step=10)

# Convert input to DataFrame
input_data = {
    "ApplicantIncome": [applicant_income],
    "CoapplicantIncome": [coapplicant_income],
    "LoanAmount": [loan_amount],
    "Loan_Amount_Term": [loan_amount_term]
}
features = pd.DataFrame(input_data)
features = np.array(features)


# --- Prediction button ---

if st.button("Predict Loan Approval"):
    
    

    prediction = model.predict(features)
    

    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Not Approved")
# Display prediction result
    st.write(f"Prediction: {'Approved' if prediction[0]==1 else 'Not Approved'}")
    st.write("Note: This is a Machine Learning model prediction and may not be 100% accurate.")
    
        
