import streamlit as st
import joblib
import pandas as pd

# Page Configuration
st.set_page_config(
    page_title="Diabetes Readmission Predictor",
    page_icon="🏥",
    layout="centered"
)

@st.cache_resource
def load_model():
    model = joblib.load("random_forest_model.pkl")
    return model, list(model.feature_names_in_)

try:
    model, expected_features = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Failed to load model: {e}")

st.title("🏥 Patient Diabetes Readmission Predictor")
st.markdown("Enter patient details below to predict the likelihood of hospital readmission.")

if model_loaded:
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=0, max_value=120, value=65)
            time_in_hospital = st.number_input("Days in Hospital", min_value=1, max_value=30, value=4)
            num_lab_procedures = st.number_input("Number of Lab Procedures", min_value=0, max_value=200, value=45)
            num_procedures = st.number_input("Number of Non-Lab Procedures", min_value=0, max_value=10, value=1)
            
        with col2:
            num_medications = st.number_input("Number of Distinct Medications", min_value=0, max_value=100, value=12)
            number_outpatient = st.number_input("Outpatient Visits (Past Year)", min_value=0, max_value=50, value=0)
            number_emergency = st.number_input("Emergency Visits (Past Year)", min_value=0, max_value=50, value=0)
            number_inpatient = st.number_input("Inpatient Visits (Past Year)", min_value=0, max_value=50, value=1)
            
        submitted = st.form_submit_button("Run Prediction")
        
        if submitted:
            # Create feature-aligned DataFrame
            input_df = pd.DataFrame(0, index=[0], columns=expected_features)
            
            # Map user inputs
            user_inputs = {
                "age": age,
                "time_in_hospital": time_in_hospital,
                "num_lab_procedures": num_lab_procedures,
                "num_procedures": num_procedures,
                "num_medications": num_medications,
                "number_outpatient": number_outpatient,
                "number_emergency": number_emergency,
                "number_inpatient": number_inpatient
            }
            
            for key, val in user_inputs.items():
                if key in input_df.columns:
                    input_df.loc[0, key] = val
            
            # Predict
            pred = model.predict(input_df)[0]
            prob = float(model.predict_proba(input_df)[0][1]) if hasattr(model, "predict_proba") else 0.0
            
            st.divider()
            st.subheader("Prediction Results")
            if pred == 1:
                st.warning(f"⚠️ **High Risk:** Patient is predicted to be **Readmitted**.")
            else:
                st.success(f"✅ **Low Risk:** Patient is predicted **Not Readmitted**.")
                
            st.metric(label="Readmission Probability", value=f"{prob:.2%}")
