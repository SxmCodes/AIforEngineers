import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load model
model = joblib.load('models/churn_model.pkl')

# App title
st.title("Customer Churn Prediction App")

# User inputs for ALL features (matching training dataset)
st.header("Enter Customer Details")

# Binary/Yes-No features use selectbox for 'Yes'/'No'
gender = st.selectbox("Gender", ["Female", "Male"])
senior_citizen = st.selectbox("Senior Citizen", [0, 1])  # Already numeric
partner = st.selectbox("Partner", ["No", "Yes"])
dependents = st.selectbox("Dependents", ["No", "Yes"])
tenure = st.slider("Tenure (months)", 0, 72, 1)
phone_service = st.selectbox("Phone Service", ["No", "Yes"])
multiple_lines = st.selectbox("Multiple Lines", ["No", "No phone service", "Yes"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", ["No", "No internet service", "Yes"])
online_backup = st.selectbox("Online Backup", ["No", "No internet service", "Yes"])
device_protection = st.selectbox("Device Protection", ["No", "No internet service", "Yes"])
tech_support = st.selectbox("Tech Support", ["No", "No internet service", "Yes"])
streaming_tv = st.selectbox("Streaming TV", ["No", "No internet service", "Yes"])
streaming_movies = st.selectbox("Streaming Movies", ["No", "No internet service", "Yes"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
payment_method = st.selectbox("Payment Method", ["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"])
monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 50.0)
total_charges = st.number_input("Total Charges", 0.0, 10000.0, 100.0)

# Pre-process inputs: Create DataFrame and encode categorically to match training (LabelEncoder alphabetical order)
input_data = pd.DataFrame({
    'gender': [0 if gender == "Female" else 1],  # Female=0, Male=1
    'SeniorCitizen': [senior_citizen],
    'Partner': [0 if partner == "No" else 1],  # No=0, Yes=1
    'Dependents': [0 if dependents == "No" else 1],
    'tenure': [tenure],
    'PhoneService': [0 if phone_service == "No" else 1],
    'MultipleLines': [0 if multiple_lines == "No" else 1 if multiple_lines == "No phone service" else 2],  # No=0, No phone service=1, Yes=2
    'InternetService': [0 if internet_service == "DSL" else 1 if internet_service == "Fiber optic" else 2],  # DSL=0, Fiber optic=1, No=2
    'OnlineSecurity': [0 if online_security == "No" else 1 if online_security == "No internet service" else 2],
    'OnlineBackup': [0 if online_backup == "No" else 1 if online_backup == "No internet service" else 2],
    'DeviceProtection': [0 if device_protection == "No" else 1 if device_protection == "No internet service" else 2],
    'TechSupport': [0 if tech_support == "No" else 1 if tech_support == "No internet service" else 2],
    'StreamingTV': [0 if streaming_tv == "No" else 1 if streaming_tv == "No internet service" else 2],
    'StreamingMovies': [0 if streaming_movies == "No" else 1 if streaming_movies == "No internet service" else 2],
    'Contract': [0 if contract == "Month-to-month" else 1 if contract == "One year" else 2],  # Month-to-month=0, One year=1, Two year=2
    'PaperlessBilling': [0 if paperless_billing == "No" else 1],
    'PaymentMethod': [0 if payment_method == "Bank transfer (automatic)" else 1 if payment_method == "Credit card (automatic)" else 2 if payment_method == "Electronic check" else 3],  # Bank=0, Credit=1, Electronic=2, Mailed=3
    'MonthlyCharges': [monthly_charges],
    'TotalCharges': [total_charges]
})

if st.button("Predict Churn"):
    try:
        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]  # Probability of churn (class 1)
        st.write(f"Prediction: {'Likely to Churn' if prediction == 1 else 'Not Likely to Churn'}")
        st.write(f"Churn Probability: {prob:.2%}")

        # Optional visualization
        fig, ax = plt.subplots()
        sns.barplot(x=['No Churn', 'Churn'], y=[1 - prob, prob], ax=ax)
        st.pyplot(fig)
    except Exception as e:
        st.error("Prediction failed. Please check your inputs and ensure all fields are filled correctly. If the issue persists, verify the model training.")
        # Optionally log the error internally: print(e)  # But don't show to user