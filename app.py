import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Load trained model
model = pickle.load(open('Model.sav', 'rb'))

# Streamlit UI
st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

st.title("📊 Customer Churn Prediction System")

st.markdown("Enter customer details below to predict if they are likely to churn or continue.")

# User Inputs
Dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, value=12)
OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
Contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
MonthlyCharges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=500.0, value=50.0)
TotalCharges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=500.0)

# Prediction Button
if st.button("🔍 Predict Churn"):
    # Prepare input data
    data = [[Dependents, tenure, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, Contract, PaperlessBilling, MonthlyCharges, TotalCharges]]
    df = pd.DataFrame(data, columns=['Dependents', 'tenure', 'OnlineSecurity',
                                     'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                                     'Contract', 'PaperlessBilling', 'MonthlyCharges', 'TotalCharges'])

    # Encode categorical features
    categorical_feature = {feature for feature in df.columns if df[feature].dtypes == 'O'}
    encoder = LabelEncoder()
    for feature in categorical_feature:
        df[feature] = encoder.fit_transform(df[feature])

    # Model Prediction
    single = model.predict(df)
    probability = model.predict_proba(df)[:, 1] * 100

    # Display Results
    if single == 1:
        st.error(f"🚨 **This Customer is likely to churn!**\n💡 Confidence Level: **{np.round(probability[0], 2)}%**")
    else:
        st.success(f"✅ **This Customer is likely to continue!**\n💡 Confidence Level: **{np.round(probability[0], 2)}%**")
