import streamlit as st
import pickle
import numpy as np

# Load trained model
model = pickle.load(open("injury_model.pkl", "rb"))

# UI Layout
st.title("🏃 Running Injury Predictor")
st.write("Enter your details to check your injury risk.")

# Input Fields
hours = st.number_input("Weekly Training Hours", min_value=1, max_value=20)
surface = st.selectbox("Running Surface", ["Road", "Grass", "Treadmill"])
foot_strike = st.selectbox("Foot Strike Pattern", ["Heel", "Midfoot", "Forefoot"])
previous_injury = st.radio("Previous Injuries?", ["Yes", "No"])

# Convert Inputs to Model Format
input_data = np.array([[hours, ["Road", "Grass", "Treadmill"].index(surface),
                        ["Heel", "Midfoot", "Forefoot"].index(foot_strike),
                        ["No", "Yes"].index(previous_injury)]])

# Prediction Button
if st.button("Predict Injury Risk"):
    result = model.predict(input_data)
    if result[0] == 1:
        st.error("⚠️ High Risk of Injury! Take Precautions.")
    else:
        st.success("✅ Low Injury Risk! Keep Running Safely.")
