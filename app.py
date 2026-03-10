import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Hospital Readmission Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: bold;
        color: #1f4e79;
        text-align: center;
    }
    .risk-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        color: white;
        margin: 10px 0;
    }
    .risk-high { background-color: #e74c3c; }
    .risk-moderate { background-color: #f39c12; }
    .risk-low { background-color: #27ae60; }
    .info-text { font-size: 0.9rem; color: #555; }
    .metric-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #1f4e79;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load trained model and preprocessor."""
    try:
        model = joblib.load('models/xgboost_best.pkl')
        preprocessor = joblib.load('models/preprocessor.pkl')
        threshold = joblib.load('models/best_threshold.pkl')
        return model, preprocessor, threshold
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, 0.3


def create_risk_gauge(probability):
    """Create risk gauge using Plotly."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Readmission Risk", 'font': {'size': 24}},
        number={'suffix': "%", 'font': {'size': 40}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "#1f4e79"},
            'bgcolor': "white",
            'steps': [
                {'range': [0, 30], 'color': '#d5f5e3'},
                {'range': [30, 70], 'color': '#fef9e7'},
                {'range': [70, 100], 'color': '#fadbd8'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 3},
                'thickness': 0.75,
                'value': probability * 100
            }
        }
    ))
    fig.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def get_patient_inputs():
    """Collect all patient inputs from sidebar."""
    inputs = {}
    
    with st.sidebar:
        st.header("Patient Information")
        
        # Demographics
        st.subheader("Demographics")
        inputs['age'] = st.selectbox(
            "Age Group",
            ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
             "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"]
        )
        inputs['gender'] = st.selectbox("Gender", ["Male", "Female"])
        inputs['race'] = st.selectbox(
            "Race", 
            ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other"]
        )
        
        # Admission details
        st.subheader("Admission Details")
        inputs['admission_type_id'] = st.selectbox(
            "Admission Type",
            [1, 2, 3, 4],
            format_func=lambda x: {
                1: "Emergency", 2: "Urgent", 3: "Elective", 4: "Newborn"
            }.get(x, "Other")
        )
        inputs['discharge_disposition_id'] = st.selectbox(
            "Discharge Disposition",
            [1, 2, 3, 6],
            format_func=lambda x: {
                1: "Discharged to home",
                2: "Transferred to another hospital",
                3: "Transferred to SNF",
                6: "Home with home health service"
            }.get(x, "Other")
        )
        inputs['admission_source_id'] = st.selectbox(
            "Admission Source",
            [1, 2, 3, 4, 5, 6, 7, 8],
            format_func=lambda x: {
                1: "Physician Referral", 2: "Clinic Referral", 
                3: "HMO Referral", 7: "Emergency Room"
            }.get(x, "Other")
        )
        inputs['time_in_hospital'] = st.slider("Length of Stay (days)", 1, 14, 3)
        
        # Clinical history
        st.subheader("Clinical History")
        inputs['num_lab_procedures'] = st.slider("Lab Procedures", 1, 100, 40)
        inputs['num_procedures'] = st.slider("Procedures", 0, 6, 1)
        inputs['num_medications'] = st.slider("Medications", 1, 50, 10)
        inputs['number_outpatient'] = st.slider("Outpatient Visits (1yr)", 0, 15, 0)
        inputs['number_emergency'] = st.slider("Emergency Visits (1yr)", 0, 15, 0)
        inputs['number_inpatient'] = st.slider("Prior Inpatient Visits (1yr)", 0, 15, 0)
        inputs['number_diagnoses'] = st.slider("Number of Diagnoses", 1, 16, 5)
        
        # Labs and medications
        st.subheader("Laboratory Results")
        inputs['max_glu_serum'] = st.selectbox(
            "Max Glucose", ["None", "Norm", ">200", ">300"]
        )
        inputs['A1Cresult'] = st.selectbox(
            "HbA1c Result", ["None", "Norm", ">7", ">8"]
        )
        inputs['change'] = st.selectbox("Medication Change", ["No", "Ch"])
        inputs['diabetesMed'] = st.selectbox("Diabetes Medication", ["Yes", "No"])
        
        # Add default values for medication features (all No)
        med_cols = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
                   'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
                   'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
                   'miglitol', 'troglitazone', 'tolazamide', 'examide',
                   'citoglipton', 'insulin', 'glyburide-metformin',
                   'glipizide-metformin', 'glimepiride-pioglitazone',
                   'metformin-rosiglitazone', 'metformin-pioglitazone']
        for med in med_cols:
            inputs[med] = 'No'
        
        predict_btn = st.button("Calculate Risk", type="primary", use_container_width=True)
    
    return inputs, predict_btn


def create_input_dataframe(inputs):
    """Convert inputs to DataFrame for model."""
    # Order must match training data
    feature_order = [
        'race', 'gender', 'age', 'admission_type_id', 'discharge_disposition_id',
        'admission_source_id', 'time_in_hospital', 'num_lab_procedures',
        'num_procedures', 'num_medications', 'number_outpatient',
        'number_emergency', 'number_inpatient', 'diag_1', 'diag_2', 'diag_3',
        'number_diagnoses', 'max_glu_serum', 'A1Cresult', 'metformin',
        'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
        'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',
        'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
        'tolazamide', 'examide', 'citoglipton', 'insulin', 'glyburide-metformin',
        'glipizide-metformin', 'glimepiride-pioglitazone',
        'metformin-rosiglitazone', 'metformin-pioglitazone', 'change', 'diabetesMed'
    ]
    
    # Add default values for diagnosis codes
    inputs['diag_1'] = '250'
    inputs['diag_2'] = '250'
    inputs['diag_3'] = '250'
    
    # Create DataFrame with correct column order
    data = {col: inputs.get(col, 'No') for col in feature_order}
    return pd.DataFrame([data])


def get_recommendations(risk_level, probability):
    """Get clinical recommendations based on risk."""
    if risk_level == "High":
        return [
            "**Immediate Actions (0-48 hours):**",
            "• Schedule follow-up within 3-5 days",
            "• Medication reconciliation with pharmacist",
            "• Arrange home health evaluation",
            "• Provide low-literacy discharge instructions"
        ], "risk-high"
    elif risk_level == "Moderate":
        return [
            "**Standard Actions (3-7 days):**",
            "• Schedule primary care follow-up within 7-10 days",
            "• Provide disease-specific education materials",
            "• Phone follow-up within 48 hours"
        ], "risk-moderate"
    else:
        return [
            "**Routine Care:**",
            "• Schedule routine follow-up within 30 days",
            "• Standard discharge instructions"
        ], "risk-low"


def main():
    # Header
    st.markdown('<p class="main-header">Hospital Readmission Risk Predictor</p>', 
                unsafe_allow_html=True)
    st.markdown("**Clinical Decision Support Tool** | COM572 Machine Learning Coursework")
    
    # Load model
    model, preprocessor, threshold = load_model()
    
    if model is None:
        st.error("Model not loaded. Please check installation.")
        st.stop()
    
    # Get inputs
    inputs, predict_btn = get_patient_inputs()
    
    # Main panel
    if predict_btn:
        try:
            # Create input DataFrame
            input_df = create_input_dataframe(inputs)
            
            # Preprocess and predict
            X_processed = preprocessor.transform(input_df)
            probability = model.predict_proba(X_processed)[0][1]
            
            # Determine risk level
            if probability >= 0.7:
                risk_level = "High"
            elif probability >= 0.3:
                risk_level = "Moderate"
            else:
                risk_level = "Low"
            
            # Display results
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.plotly_chart(create_risk_gauge(probability), use_container_width=True)
                
                # Risk banner
                recs, css_class = get_recommendations(risk_level, probability)
                st.markdown(f'<div class="risk-box {css_class}">'
                           f'{risk_level} Risk: {probability:.1%}</div>', 
                           unsafe_allow_html=True)
                
                # Metrics
                st.markdown("### Prediction Details")
                m1, m2, m3 = st.columns(3)
                m1.metric("Model Confidence", f"{max(probability, 1-probability):.1%}")
                m2.metric("Decision Threshold", f"{threshold:.0%}")
                m3.metric("Model Version", "XGBoost v1.0")
            
            with col2:
                st.markdown("### Clinical Recommendations")
                for rec in recs:
                    st.markdown(rec)
                
                st.info("**Note:** This tool supports clinical judgment but does not replace it.")
            
            # Explanation
            st.markdown("---")
            st.markdown("### Prediction Explanation")
            st.write("""
            This prediction is based on machine learning analysis of 100,000+ patient records.
            Key factors in this prediction include:
            - Number of prior hospital visits
            - Diabetes control (HbA1c levels)
            - Number of active diagnoses
            - Length of current admission
            """)
            
        except Exception as e:
            st.error(f"Error generating prediction: {e}")
            st.info("Please ensure all fields are completed.")
    
    else:
        # Initial state
        st.info("👈 Enter patient details in the sidebar and click 'Calculate Risk'")
        
        with st.expander("How to Use This Tool"):
            st.write("""
            1. Enter patient demographics and admission details in the sidebar
            2. Add clinical history including prior visits and diagnoses
            3. Input laboratory results (HbA1c, glucose)
            4. Click 'Calculate Risk' to generate prediction
            5. Review risk score and clinical recommendations
            
            **Interpretation:**
            - **High Risk (≥70%)**: Immediate intervention required
            - **Moderate Risk (30-70%)**: Enhanced discharge planning
            - **Low Risk (<30%)**: Standard care appropriate
            """)
        
        # Model info
        st.markdown("### Model Information")
        st.markdown("""
        | Metric | Value |
        |--------|-------|
        | Algorithm | XGBoost Classifier |
        | Training Data | 101,766 encounters (1999-2008) |
        | PR-AUC | 0.38 |
        | Sensitivity | 82% |
        | Specificity | 64% |
        """)
    
    # Footer disclaimer
    st.markdown("---")
    st.markdown("""
    <div style="font-size: 0.8rem; color: #666;">
    <b>Disclaimer:</b> This clinical decision support tool is for educational purposes only. 
    Predictions are based on historical patterns and may not generalise to all populations. 
    Always use professional clinical judgment. Model trained on UCI Diabetes dataset.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
