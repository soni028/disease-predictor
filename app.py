import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the saved model
with open('stroke_model.pkl', 'rb') as f:
    model = pickle.pickle.load(f)

# (Optional) Load scaler if you used one
# with open('scaler.pkl', 'rb') as f:
#     scaler = pickle.load(f)

st.title("Brain Stroke Prediction & Wellness Advisor")
st.write("Input patient details below to predict stroke risk and receive lifestyle recommendations.")

# --- USER INPUTS ---
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    age = st.number_input("Age", min_value=1, max_value=120, value=30)
    hypertension = st.selectbox("Hypertension (High BP)?", ["No", "Yes"])
    heart_disease = st.selectbox("Heart Disease?", ["No", "Yes"])
    ever_married = st.selectbox("Ever Married?", ["Yes", "No"])

with col2:
    work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
    residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
    avg_glucose_level = st.number_input("Average Glucose Level", min_value=50.0, max_value=300.0, value=90.0)
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=22.0)
    smoking_status = st.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"])

# --- PREPROCESSING USER INPUT ---
# Map the inputs exactly how you did in your training (0 and 1, or One-Hot Encoding)
# This is a basic example assuming label encoding:
gender_encoded = 1 if gender == "Male" else (0 if gender == "Female" else 2)
hypertension_encoded = 1 if hypertension == "Yes" else 0
heart_disease_encoded = 1 if heart_disease == "Yes" else 0
married_encoded = 1 if ever_married == "Yes" else 0
residence_encoded = 1 if residence_type == "Urban" else 0

# (Note: You will need to map work_type, smoking_status based on your specific Colab encoding)
# For placeholder, let's assume simple mappings or numeric dummy values matching your feature order:
work_mapping = {"Private": 0, "Self-employed": 1, "Govt_job": 2, "children": 3, "Never_worked": 4}
smoke_mapping = {"formerly smoked": 0, "never smoked": 1, "smokes": 2, "Unknown": 3}

features = [
    gender_encoded, age, hypertension_encoded, heart_disease_encoded,
    married_encoded, work_mapping[work_type], residence_encoded,
    avg_glucose_level, bmi, smoke_mapping[smoking_status]
]

# Convert to dataframe or array matching your model's expected input shape
input_data = np.array([features])

# If you scaled your data in Colab, uncomment below:
# input_data[:, [1, 7, 8]] = scaler.transform(input_data[:, [1, 7, 8]]) 

# --- PREDICTION AND RECOMMENDATIONS ---
if st.button("Predict & Get Recommendations"):
    prediction = model.predict(input_data)
    
    if prediction[0] == 1:
        st.error("⚠️ High Risk of Stroke Detected")
        
        st.subheader("📋 Recommended Actions & Diet:")
        st.write("- **Diet:** Follow a DASH or Mediterranean diet. Focus on low-sodium foods, leafy greens, whole grains, and omega-3 fatty acids. Strictly avoid processed meats and trans fats.")
        st.write("- **Exercise:** Light to moderate aerobic exercise (e.g., brisk walking 30 mins/day, 5 days a week). Avoid sudden, high-intensity heavy lifting without medical clearance.")
        st.write("- **Medical:** Please consult a healthcare professional immediately to monitor blood pressure and cholesterol levels.")
    else:
        st.success("✅ Low Risk of Stroke Detected")
        
        st.subheader("🌱 Maintenance Tips:")
        st.write("- **Diet:** Maintain a balanced diet rich in fiber, fruits, and lean proteins to keep blood glucose and BMI within healthy ranges.")
        st.write("- **Exercise:** Mix strength training with cardiovascular exercises (cycling, swimming, running) for 150 minutes weekly.")