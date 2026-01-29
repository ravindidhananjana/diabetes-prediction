import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("rf_model.pkl")

# Feature names 
FEATURE_NAMES = [
    'Pregnancies',
    'Glucose',
    'BloodPressure',
    'SkinThickness',
    'Insulin',
    'BMI',
    'DiabetesPedigreeFunction',
    'Age'
]

# Page title
st.title("🩺 Diabetes Prediction App")
st.write("Enter the following details to predict whether the person is diabetic or not:")

# Input fields
Pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
Glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=120)
BloodPressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
SkinThickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
Insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
BMI = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
DiabetesPedigreeFunction = st.number_input(
    "Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5
)
Age = st.number_input("Age", min_value=1, max_value=120, value=30)

# Predict button
if st.button("Predict"):

    input_data = pd.DataFrame([[
        Pregnancies, Glucose, BloodPressure, SkinThickness,
        Insulin, BMI, DiabetesPedigreeFunction, Age
    ]], columns=FEATURE_NAMES)

    # Probability prediction
    prob = model.predict_proba(input_data)[0][1]

    # Decision
    if prob >= 0.5:
        st.error(f"🩸 Diabetic (Probability: {prob*100:.0f}%)")
    else:
        st.success(f"✅ Non-diabetic (Probability: {prob*100:.0f}%)")
