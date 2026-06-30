import streamlit as st
import pickle
import pandas as pd
import numpy as np

# --- 1. SET UP CONFIGURATION & PAGE ---
st.set_page_config(page_title="Stroke Risk Predictor", layout="centered")
st.title("🧠 Brain Stroke Prediction & Wellness Advisor")
st.write("---")

# --- 2. LOAD TRAINED MODEL ---
MODEL_PATH = 'random_forest_stroke_model.pkl'

@st.cache_resource
def load_model():
    try:
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error(f"❌ Error: '{MODEL_PATH}' not found. Please upload your model file to GitHub.")
        st.stop()

model = load_model()

# --- 3. SIDEBAR CALIBRATION TOOLS (Fixes the "Always High Risk" bug) ---
st.sidebar.header("🛠️ Model Calibration Tools")
st.sidebar.write("If the app always predicts High Risk, your model is likely expecting scaled values.")

# Dynamic scaling option right on the dashboard screen
use_scaling = st.sidebar.checkbox("Enable Numeric Feature Scaling", value=False)
scale_method = st.sidebar.selectbox("Scaling Target", ["Scale Age/Glucose/BMI down (0 to 1)", "Standardize (-2 to 2)"])

# --- 4. USER INPUT INTERFACE ---
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male", "Other"])
    age = st.number_input("Age", min_value=1, max_value=120, value=45, step=1)
    hypertension = st.selectbox("Hypertension (High BP)?", ["No", "Yes"])
    heart_disease = st.selectbox("Heart Disease?", ["No", "Yes"])
    ever_married = st.selectbox("Ever Married?", ["No", "Yes"])

with col2:
    work_type = st.selectbox("Work Type", ["Govt_job", "Never_worked", "Private", "Self-employed", "children"])
    residence_type = st.selectbox("Residence Type", ["Rural", "Urban"])
    avg_glucose_level = st.number_input("Average Glucose Level (mg/dL)", min_value=30.0, max_value=350.0, value=95.0, step=0.1)
    bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=70.0, value=24.0, step=0.1)
    smoking_status = st.selectbox("Smoking Status", ["Unknown", "formerly smoked", "never smoked", "smokes"])

st.write("---")

# --- 5. ALPHABETICAL LABEL ENCODING MAPS ---
# Scikit-Learn's LabelEncoder automatically sorts categories alphabetically (0, 1, 2...)
gender_mapping = {"Female": 0, "Male": 1, "Other": 2}
work_mapping = {"Govt_job": 0, "Never_worked": 1, "Private": 2, "Self-employed": 3, "children": 4}
residence_mapping = {"Rural": 0, "Urban": 1}
smoke_mapping = {"Unknown": 0, "formerly smoked": 1, "never smoked": 2, "smokes": 3}

# Convert Binary Features
hypertension_encoded = 1 if hypertension == "Yes" else 0
heart_disease_encoded = 1 if heart_disease == "Yes" else 0
married_encoded = 1 if ever_married == "Yes" else 0

# --- 6. ARRANGE DATA IN CLASSIC DATASET ORDER ---
# Order: gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status
features = [
    gender_mapping[gender],
    float(age),
    hypertension_encoded,
    heart_disease_encoded,
    married_encoded,
    work_mapping[work_type],
    residence_mapping[residence_type],
    float(avg_glucose_level),
    float(bmi),
    smoke_mapping[smoking_status]
]

# Convert to DataFrame
feature_names = ["gender", "age", "hypertension", "heart_disease", "ever_married", "work_type", "Residence_type", "avg_glucose_level", "bmi", "smoking_status"]
input_df = pd.DataFrame([features], columns=feature_names)

# --- 7. APPLY DYNAMIC SCALING IF ACTIVATED ---
if use_scaling:
    if "Scale down" in scale_method:
        # MinMax approximation: squeezes large values into a 0-1 range
        input_df["age"] = input_df["age"] / 100.0
        input_df["avg_glucose_level"] = input_df["avg_glucose_level"] / 300.0
        input_df["bmi"] = input_df["bmi"] / 60.0
    else:
        # Standard Scaler approximation: centers data around 0
        input_df["age"] = (input_df["age"] - 43.0) / 22.0
        input_df["avg_glucose_level"] = (input_df["avg_glucose_level"] - 106.0) / 45.0
        input_df["bmi"] = (input_df["bmi"] - 29.0) / 7.0

# Show developer variables for structural check
with st.expander("🔍 View Processed Features Array"):
    st.write("Values passed directly into your Random Forest model:")
    st.dataframe(input_df)

# --- 8. PREDICTION ENGINE ---
if st.button("Predict Stroke Risk Status", type="primary"):
    
    # Run prediction matrix
    prediction = model.predict(input_df.values)
    
    try:
        probabilities = model.predict_proba(input_df.values)[0]
        st.metric(label="Calculated High Risk Probability", value=f"{probabilities[1]*100:.1f}%")
    except Exception:
        pass

    # Display dynamic feedback cards
    if prediction[0] == 1:
        st.error("⚠️ Prediction Result: High Risk of Stroke")
        st.subheader("📋 Recommended Lifestyle & Diet Strategy")
        st.markdown("""
        * **Desired Diet:** Adopt a strict **DASH** or Mediterranean diet plan. Maximize clean green vegetables and drop sodium intake below 1,500 mg daily. Eliminate processed fast foods and trans fats completely.
        * **Targeted Exercise:** Engage in low-impact cardiovascular movements (brisk walking or cycling) for 30 minutes daily. Avoid high-stress weight lifting routines.
        """)
    else:
        st.success("✅ Prediction Result: Low Risk of Stroke")
        st.subheader("🌱 Preventative Health Guidelines")
        st.markdown("""
        * **Desired Diet:** Retain a baseline balanced nutrition matrix high in lean proteins and raw dietary fibers to maintain target blood glucose counts.
        * **Targeted Exercise:** Mix continuous movement routines with light functional tracking. Strive for 150 minutes of weekly activity to preserve healthy BMI baselines.
        """)