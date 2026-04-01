import streamlit as st
import pandas as pd
import joblib

st.title("Customer Purchase Prediction")

# Load model and columns
rf_model = joblib.load("rf_random_forest.pkl")
model_columns = joblib.load("model_columns.pkl")

# Create inputs for all features
input_data = {}
for col in model_columns:
    value = st.text_input(f"Enter {col}", "0")
    input_data[col] = float(value)  # convert all to float; adjust if categorical

# Predict button
if st.button("Predict"):
    input_df = pd.DataFrame([input_data])
    prediction = rf_model.predict(input_df)[0]
    st.success(f"Prediction: {'Will Purchase' if prediction == 1 else 'Will Not Purchase'}")