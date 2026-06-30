import streamlit as st
import pickle
import pandas as pd
import numpy as np

# --- 1. LOAD THE TRAINED MODEL & OPTIONAL SCALER ---
MODEL_PATH = 'random_forest_stroke_model.pkl'
SCALER_PATH = 'scaler.pkl'  # Only used if you uploaded a scaler.pkl to GitHub

@st.cache_resource
def load_artifacts():
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError:
        st.error(f"❌ Error: '{MODEL_PATH}' not found in your GitHub repository.")
        st.stop()
        
    scaler = None
    try:
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
    except FileNotFoundError:
        pass # It's fine if you didn't use a scaler
        
    return model, scaler

model, scaler = load_artifacts()

# --- 2. APP UI CONFIGURATION ---
st.set_page_config(page_title="Stroke Risk Predictor", layout="centered")
st.title("🧠 Brain Stroke Prediction & Wellness Advisor")
st.write("Input patient demographic and clinical details below to analyze stroke risk.")
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
# Check your Colab notebook to ensure these numbers match your LabelEncoder/Mapping!
gender_mapping = {"Male": 1, "Female": 0, "Other": 2}
work_mapping = {"Private": 2, "Self-employed": 3, "Govt_job": 0, "children": 4, "Never_worked": 1}
residence_mapping = {"Urban": 1, "Rural": 0}
smoke_mapping = {"never smoked": 2, "formerly smoked": 1, "smokes": 3, "Unknown": 0}

# Convert binary selections
hypertension_encoded = 1 if hypertension == "Yes" else 0
heart_disease_encoded = 1 if heart_disease == "Yes" else 0
married_encoded = 1 if ever_married == "Yes" else 0

# 🚨 CRITICAL: Build the feature list in the EXACT column order of your Colab X_train dataframe
# Current order: gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status
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

# Convert to 2D DataFrame (Using columns helps keep track of features better than a raw numpy array)
feature_names = [
    "gender", "age", "hypertension", "heart_disease", "ever_married", 
    "work_type", "Residence_type", "avg_glucose_level", "bmi", "smoking_status"
]
input_df = pd.DataFrame([features], columns=feature_names)

# --- 5. SYSTEM DEBUGGER SECTION (Helps you spot mismatches instantly) ---
with st.expander("🛠️ Developer Debugging Tools (Click to expand)"):
    st.write("This is the exact data structure being sent to your Random Forest model:")
    st.dataframe(input_df)
    if scaler is not None:
        st.write("✅ `scaler.pkl` detected and running successfully.")
    else:
        st.write("⚠️ No `scaler.pkl` detected. If your model always predicts High Risk, check if your numeric inputs (Age, Glucose, BMI) need scaling.")

# --- 6. APPLY SCALING (IF SCALER EXISTS) ---
if scaler is not None:
    try:
        # Fits if you scaled your entire data matrix or specific columns
        input_data = scaler.transform(input_df)
    except Exception as e:
        # Fallback if your scaler expects an array structure instead of a dataframe
        input_data = scaler.transform(input_df.values)
else:
    input_data = input_df.values

# --- 7. PREDICTION AND RECOMMENDATIONS ---
if st.button("Predict Risk & Generate Guidelines", type="primary"):
    
    # Run Model Prediction
    prediction = model.predict(input_data)
    
    # Attempt to fetch risk probability for extra insight
    try:
        prediction_proba = model.predict_proba(input_data)[0][1] * 100
        st.info(f"📊 **Calculated Statistical Probability:** {prediction_proba:.1f}%")
    except:
        pass

    # Display Results
    if prediction[0] == 1:
        st.error("⚠️ High Risk of Stroke Detected")
        
        st.subheader("📋 Recommended Lifestyle Adjustments")
        st.markdown("""
        * **Desired Diet:** Strictly adopt a **DASH** or **Mediterranean diet**. Focus on high-potassium, low-sodium options like leafy greens, whole grains, and lean proteins (fish). Completely avoid processed snacks, red meat, and trans fats.
        * **Targeted Exercise:** Engage in **moderate aerobic activity** (brisk walking, swimming, cycling) for 30 minutes a day, 5 days a week. Avoid intense weight lifting without medical clearance.
        * **Next Medical Steps:** Closely monitor your blood pressure and consult a medical professional to review proactive cardiovascular strategies.
        """)
    else:
        st.success("✅ Low Risk of Stroke Detected")
        
        st.subheader("🌱 Preventative Maintenance Guidelines")
        st.markdown("""
        * **Desired Diet:** Maintain a balanced diet rich in natural dietary fiber, fresh fruits, and varied vegetables to keep blood glucose stable.
        * **Targeted Exercise:** Mix cardiovascular exercises with strength training routines. Aim for **150 to 300 minutes of moderate activity** weekly to maintain a healthy BMI.
        """)