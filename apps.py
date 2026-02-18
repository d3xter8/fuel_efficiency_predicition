import streamlit as st
import joblib
import pandas as pd
import datetime

# Load model
model = joblib.load("models/model_v1.pkl")

st.title("🚗 MLOps Demo: Car Fuel Efficiency Predictor")

st.write("Enter car details:")

# User inputs
cylinders = st.slider("Number of Cylinders", 3, 8, 4)
horsepower = st.slider("Horsepower", 50, 250, 100)
weight = st.slider("Weight (lbs)", 1500, 5000, 3000)
acceleration = st.slider("Acceleration", 8.0, 25.0, 15.0)

if st.button("Predict Fuel Efficiency"):
    input_data = pd.DataFrame(
        [[cylinders, horsepower, weight, acceleration]],
        columns=["cylinders", "horsepower", "weight", "acceleration"],
    )

    prediction = model.predict(input_data)[0]

    result = "High Efficiency 🚗💨" if prediction == 1 else "Low Efficiency ⛽"

    st.success(f"Prediction: {result}")

    # Log prediction
    with open("logs.txt", "a", encoding="utf-8") as f:
        f.write(
            f"{datetime.datetime.now()} - Input: {input_data.values.tolist()} - Prediction: {result}\n"
        )
    st.info("Prediction logged in logs.txt")        
    