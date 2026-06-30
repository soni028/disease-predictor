import streamlit as st
import pickle
import pandas as pd
import numpy as np

# --- 1. LOAD THE TRAINED MODEL ---
# Make sure this filename matches the exact name of the file uploaded to GitHub
MODEL_PATH = 'random_forest_stroke_model.pkl'

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error(f"❌ Error: '{MODEL_PATH}' not found in the repository. Please ensure you have uploaded your model file to GitHub.")
    st.stop()

# --- 2. APP TITLE & DESCRIPTION ---
st.set_page_config(page_title="Stroke Risk Predictor", layout="centered")
st.title("🧠 Brain Stroke Prediction & Wellness Advisor")
st.write("Input patient demographic and clinical details below to predict stroke risk and receive customized recommendations.")
st.write("---")

# --- 3. USER INPUT INTERFACE ---
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    age = st.number_input("Age", min_value=1, max_value=120, value=30, step=1)
    hypertension = st.selectbox("Hypertension (High Blood Pressure)?", ["No", "Yes"])
    heart_disease = st.selectbox("Heart Disease?", ["No", "Yes"])
    ever_married = st.selectbox("Ever Married?", ["Yes", "No"])

with col2:
    work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
    residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
    avg_glucose_level = st.number_input("Average Glucose Level (mg/dL)", min_value=30.0, max_value=350.0, value=90.0, step=0.1)
    bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=70.0, value=22.0, step=0.1)
    smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])

st.write("---")

# --- 4. DATA PREPROCESSING (MAPPING) ---
# Transforming user inputs into the numerical formats expected by your Random Forest model
gender_mapping = {"Male": 1, "Female": 0, "Other": 2}
work_mapping = {"Private": 0, "Self-employed": 1, "Govt_job": 2, "children": 3, "Never_worked": 4}
residence_mapping = {"Urban": 1, "Rural": 0}
smoke_mapping = {"never smoked": 0, "formerly smoked": 1, "smokes": 2, "Unknown": 3}

# Convert binary variables
hypertension_encoded = 1 if hypertension == "Yes" else 0
heart_disease_encoded = 1 if heart_disease == "Yes" else 0
married_encoded = 1 if ever_married == "Yes" else 0

# Construct feature array in the exact order your model was trained on
# Standard dataset order: gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status
features = [
    gender_mapping[gender],
    age,
    hypertension_encoded,
    heart_disease_encoded,
    married_encoded,
    work_mapping[work_type],
    residence_mapping[residence_type],
    avg_glucose_level,
    bmi,
    smoke_mapping[smoking_status]
]

# Convert to 2D array for prediction
input_data = np.array([features])

# --- 5. PREDICTION AND RECOMMENDATIONS ---
if st.button("Predict Risk & Generate Guidelines", type="primary"):
    
    # Run prediction
    prediction = model.predict(input_data)
    
    # If your model supports probability outputs, get the risk percentage
    try:
        prediction_proba = model.predict_proba(input_data)[0][1] * 100
        st.write(f"**Calculated Statistical Risk:** {prediction_proba:.1f}%")
    except:
        pass  # Fallback if predict_proba is not available

    # Display results based on prediction output (0 = No Stroke Risk, 1 = Stroke Risk)
    if prediction[0] == 1:
        st.error("⚠️ High Risk of Stroke Detected")
        
        st.subheader("📋 Recommended Lifestyle Adjustments")
        st.markdown("""
        * **Desired Diet:** * Strictly adopt a **DASH** or **Mediterranean diet** structure.
          * Focus heavily on high-potassium, low-sodium foods (leafy greens, bananas, avocados).
          * Incorporate whole grains, lean proteins (fish, skinless poultry), and healthy fats (olive oil, walnuts).
          * Eliminate processed foods, red meats, sugary beverages, and trans fats entirely.
        * **Targeted Exercise:** * Engage in **moderate aerobic activity** (brisk walking, stationary cycling, water aerobics) for at least 30 minutes a day, 5 days a week.
          * Avoid extreme, sudden heavy weightlifting workouts without primary physician clearance.
        * **Next Medical Steps:** * Routinely monitor your blood pressure and blood sugar levels. 
          * Consult a healthcare professional to review these risks and discuss preventative strategies.
        """)
    else:
        st.success("✅ Low Risk of Stroke Detected")
        
        st.subheader("🌱 Preventative Maintenance Guidelines")
        st.markdown("""
        * **Desired Diet:** * Maintain a balanced caloric intake rich in natural dietary fiber, fresh fruits, and varied vegetables.
          * Keep refined sugar and salt intake within healthy daily limits to preserve optimal blood glucose ranges.
        * **Targeted Exercise:** * Keep active with a mix of cardiovascular routines and moderate functional strength training.
          * Aim for roughly **150 to 300 minutes of moderate activity** weekly to maintain a healthy BMI.
        """)