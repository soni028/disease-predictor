import streamlit as st
import pickle
import pandas as pd
import numpy as np

# --- 1. SET UP CONFIGURATION & PAGE ---
st.set_page_config(
    page_title="Stroke Risk Advisor", 
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM CSS FOR BETTER LOOK ---
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

# --- FUNCTION TO CHANGE PAGES ---
def enter_app():
    st.session_state.page = 'predictor'

# ==========================================
# 🏠 1. ENTRY / WELCOME INTERFACE (HOME)
# ==========================================
if st.session_state.page == 'home':
    st.markdown("<div class='main-title'>🧠 Stroke Risk Predictor & Wellness Advisor</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Advanced AI-driven health analytics for stroke prevention and lifestyle correction</div>", unsafe_allow_html=True)
    
    # Attractive Banner Image / Placeholder
    st.image("https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?auto=format&fit=crop&w=800&q=80", use_container_width=True)
    
    st.write("---")
    
    # Small Information Blocks
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
    
    # Large Action Button to Enter App
    if st.button("🚀 Click Here to Start Prediction", type="primary", use_container_width=True, on_click=enter_app):
        st.rerun()

# ==========================================
# 📋 2. MAIN PREDICTOR PAGE
# ==========================================
elif st.session_state.page == 'predictor':
    
    # Back to Home Button
    if st.button("⬅️ Back to Home"):
        st.session_state.page = 'home'
        st.rerun()
        
    st.title("📋 Patient Health Details")
    st.write("Please fill out the form below carefully to estimate the risk.")
    st.write("---")

    # --- SIDEBAR CALIBRATION TOOLS ---
    st.sidebar.header("🛠️ Model Calibration Tools")
    use_scaling = st.sidebar.checkbox("Enable Numeric Feature Scaling", value=False)
    scale_method = st.sidebar.selectbox("Scaling Target", ["Scale Age/Glucose/BMI down (0 to 1)", "Standardize (-2 to 2)"])

    # --- USER INPUT INTERFACE ---
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

    # --- MAPPINGS ---
    gender_mapping = {"Female": 0, "Male": 1, "Other": 2}
    work_mapping = {"Govt_job": 0, "Never_worked": 1, "Private": 2, "Self-employed": 3, "children": 4}
    residence_mapping = {"Rural": 0, "Urban": 1}
    smoke_mapping = {"Unknown": 0, "formerly smoked": 1, "never smoked": 2, "smokes": 3}

    hypertension_encoded = 1 if hypertension == "Yes" else 0
    heart_disease_encoded = 1 if heart_disease == "Yes" else 0
    married_encoded = 1 if ever_married == "Yes" else 0

    features = [
        gender_mapping[gender], float(age), hypertension_encoded, heart_disease_encoded,
        married_encoded, work_mapping[work_type], residence_mapping[residence_type],
        float(avg_glucose_level), float(bmi), smoke_mapping[smoking_status]
    ]

    input_df = pd.DataFrame([features], columns=["gender", "age", "hypertension", "heart_disease", "ever_married", "work_type", "Residence_type", "avg_glucose_level", "bmi", "smoking_status"])

    # --- DYNAMIC SCALING ---
    if use_scaling:
        if "Scale down" in scale_method:
            input_df["age"] = input_df["age"] / 100.0
            input_df["avg_glucose_level"] = input_df["avg_glucose_level"] / 300.0
            input_df["bmi"] = input_df["bmi"] / 60.0
        else:
            input_df["age"] = (input_df["age"] - 43.0) / 22.0
            input_df["avg_glucose_level"] = (input_df["avg_glucose_level"] - 106.0) / 45.0
            input_df["bmi"] = (input_df["bmi"] - 29.0) / 7.0

    # Developer Debug view
    with st.expander("🔍 View Processed Features Array"):
        st.dataframe(input_df)

    # --- PREDICTION ENGINE ---
    if st.button("Predict Stroke Risk Status", type="primary", use_container_width=True):
        prediction = model.predict(input_df.values)
        
        try:
            probabilities = model.predict_proba(input_df.values)[0]
            st.metric(label="Calculated High Risk Probability", value=f"{probabilities[1]*100:.1f}%")
        except:
            pass

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