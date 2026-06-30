import streamlit as st
import pickle
import pandas as pd
import numpy as np

# --- 1. SET UP CONFIGURATION & PAGE ---
st.set_page_config(
    page_title="Stroke Risk Advisor", 
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- CUSTOM BEAUTIFUL CSS ---
st.markdown("""
    <style>
    .main-title { font-size:40px; font-weight:bold; color:#1E3A8A; text-align:center; margin-bottom:10px; }
    .subtitle { font-size:18px; color:#4B5563; text-align:center; margin-bottom:30px; }
    .card-box { padding: 20px; border-radius: 10px; background-color: #F3F4F6; border-left: 5px solid #3B82F6; margin-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

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

# --- 3. SESSION STATE FOR NAVIGATION ---
if 'page' not in st.session_state:
    st.session_state.page = 'home'

def enter_app():
    st.session_state.page = 'predictor'

# ==========================================
# 🏠 1. ENTRY INTERFACE (HOME SCREEN)
# ==========================================
if st.session_state.page == 'home':
    st.markdown("<div class='main-title'>🧠 Stroke Risk Predictor & Wellness Advisor</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Advanced AI-driven health analytics for stroke prevention and lifestyle correction</div>", unsafe_allow_html=True)
    
    st.image("https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?auto=format&fit=crop&w=800&q=80", use_container_width=True)
    
    st.write("---")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        <div class='card-box'>
            <h4>📊 Accurate AI Analysis</h4>
            <p>Uses a trained Random Forest model to evaluate your clinical and demographic parameters.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col_b:
        st.markdown("""
        <div class='card-box'>
            <h4>🌱 Smart Recommendations</h4>
            <p>Get personalized medical, dietary (DASH/Mediterranean), and custom exercise plans.</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.write("")
    
    if st.button("🚀 Click Here to Start Prediction", type="primary", use_container_width=True, on_click=enter_app):
        st.rerun()

# ==========================================
# 📋 2. MAIN PREDICTOR PAGE
# ==========================================
elif st.session_state.page == 'predictor':
    
    if st.button("⬅️ Back to Home"):
        st.session_state.page = 'home'
        st.rerun()
        
    st.title("📋 Patient Health Details")
    st.write("Fill out the attributes below to calculate the clinical stroke probability index.")
    st.write("---")

    # --- 🛠️ SIDEBAR CALIBRATION CONTROLS (THE FIX) ---
    st.sidebar.header("🛠️ Model Debugger & Fixes")
    st.sidebar.write("If the prediction is always stuck on Low Risk, use these calibration tools to match your Colab setup:")
    
    # Tool 1: Sensitivity Threshold adjusting
    threshold = st.sidebar.slider("Model Sensitivity Threshold", min_value=0.05, max_value=0.95, value=0.50, step=0.05,
                                  help="Lower this value if the model always predicts low risk. Higher sensitivity catches hidden stroke patterns.")
    
    # Tool 2: Scaling Option
    use_scaling = st.sidebar.checkbox("Apply Numeric Value Normalization", value=False)
    
    # Tool 3: Alternate Column Layout scheme
    column_scheme = st.sidebar.selectbox("Feature Layout Scheme", ["Standard Kaggle Layout", "Alternate Age-First Layout"])

    # --- USER INPUT FORM ---
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male", "Other"])
        age = st.number_input("Age (Years)", min_value=1, max_value=120, value=65, step=1)
        hypertension = st.selectbox("Hypertension (High Blood Pressure)?", ["No", "Yes"])
        heart_disease = st.selectbox("Heart Disease?", ["No", "Yes"])
        ever_married = st.selectbox("Ever Married?", ["Yes", "No"])

    with col2:
        work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
        residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
        avg_glucose_level = st.number_input("Average Glucose Level (mg/dL)", min_value=30.0, max_value=350.0, value=220.0, step=0.1)
        bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=70.0, value=38.0, step=0.1)
        smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])

    st.write("---")

    # --- ENCODINGS ---
    # Standard Scikit-Learn alphabetical sequence mapping
    gender_mapping = {"Female": 0, "Male": 1, "Other": 2}
    work_mapping = {"Govt_job": 0, "Never_worked": 1, "Private": 2, "Self-employed": 3, "children": 4}
    residence_mapping = {"Rural": 0, "Urban": 1}
    smoke_mapping = {"Unknown": 0, "formerly smoked": 1, "never smoked": 2, "smokes": 3}

    hypertension_encoded = 1 if hypertension == "Yes" else 0
    heart_disease_encoded = 1 if heart_disease == "Yes" else 0
    married_encoded = 1 if ever_married == "Yes" else 0

    # Build primary layout schemas based on sidebar preference
    if column_scheme == "Standard Kaggle Layout":
        features = [
            gender_mapping[gender], float(age), hypertension_encoded, heart_disease_encoded,
            married_encoded, work_mapping[work_type], residence_mapping[residence_type],
            float(avg_glucose_level), float(bmi), smoke_mapping[smoking_status]
        ]
        feature_names = ["gender", "age", "hypertension", "heart_disease", "ever_married", "work_type", "Residence_type", "avg_glucose_level", "bmi", "smoking_status"]
    else:
        # Age first layout variation
        features = [
            float(age), gender_mapping[gender], hypertension_encoded, heart_disease_encoded,
            married_encoded, work_mapping[work_type], residence_mapping[residence_type],
            float(avg_glucose_level), float(bmi), smoke_mapping[smoking_status]
        ]
        feature_names = ["age", "gender", "hypertension", "heart_disease", "ever_married", "work_type", "Residence_type", "avg_glucose_level", "bmi", "smoking_status"]

    input_df = pd.DataFrame([features], columns=feature_names)

    # --- SCALE VALUES DOWN IF ACTIVATED ---
    if use_scaling:
        input_df["age"] = input_df["age"] / 100.0
        input_df["avg_glucose_level"] = input_df["avg_glucose_level"] / 300.0
        input_df["bmi"] = input_df["bmi"] / 60.0

    # Live inspector data feed visualization panel
    with st.expander("🔍 View Processed Features Array Structure"):
        st.write("This table contains the exact matrix numbers handed to your Random Forest model:")
        st.dataframe(input_df)

    # --- PREDICTION ALGORITHM ---
    if st.button("Predict Stroke Risk Status", type="primary", use_container_width=True):
        
        # Calculate numeric probabilities safely
        try:
            probabilities = model.predict_proba(input_df.values)[0]
            risk_percentage = probabilities[1]
            st.metric(label="Calculated Model Stroke Risk Index", value=f"{risk_percentage*100:.1f}%")
            
            # Use dynamic user threshold selection from sidebar instead of hardcoded 0.5 rules
            final_prediction = 1 if risk_percentage >= threshold else 0
        except Exception:
            # Fallback if model was exported without probability parameters enabled
            final_prediction = model.predict(input_df.values)[0]
            st.warning("⚠️ Probability reading unavailable. Using basic absolute class label fallback prediction.")

        # Show Output Result Display Sheets
        if final_prediction == 1:
            st.error("⚠️ Prediction Result: High Risk of Stroke Detected")
            st.subheader("📋 Recommended Lifestyle & Diet Strategy")
            st.markdown("""
            * **Desired Diet:** Adopt a strict **DASH** or Mediterranean diet plan. Maximize clean green vegetables and drop sodium intake below 1,500 mg daily. Eliminate processed fast foods and trans fats completely.
            * **Targeted Exercise:** Engage in low-impact cardiovascular movements (brisk walking or cycling) for 30 minutes daily. Avoid high-stress weight lifting routines.
            * **Next Medical Steps:** Schedule a check-up with a physician to track continuous blood pressure and blood glucose values.
            """)
        else:
            st.success("✅ Prediction Result: Low Risk of Stroke Detected")
            st.subheader("🌱 Preventative Health Guidelines")
            st.markdown("""
            * **Desired Diet:** Retain a baseline balanced nutrition matrix high in lean proteins and raw dietary fibers to maintain target blood glucose counts.
            * **Targeted Exercise:** Mix continuous movement routines with light functional tracking. Strive for 150 minutes of weekly activity to preserve healthy BMI baselines.
            """)