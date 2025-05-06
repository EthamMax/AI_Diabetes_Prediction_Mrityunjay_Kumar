import streamlit as st
import pandas as pd
import numpy as np
import pickle
import logging
import os
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app_logs.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load the model and feature names
@st.cache_resource
def load_model():
    try:
        with open('diabetes_model.pkl', 'rb') as file:
            model = pickle.load(file)
        with open('feature_names.pkl', 'rb') as file:
            feature_names = pickle.load(file)
        return model, feature_names
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        st.error("Failed to load the prediction model. Please contact support.")
        return None, None

# Define physiological bounds for each feature
PHYSIOLOGICAL_BOUNDS = {
    'Pregnancies': (0, 25),  # Maximum pregnancies unlikely to exceed 25
    'Glucose': (40, 500),    # Normal range is 70-140, but allowing wider range for extreme cases
    'BloodPressure': (40, 250),  # Normal range is 90/60 to 120/80, but allowing wider range
    'SkinThickness': (0, 100),  # In mm, unlikely to exceed 100mm
    'Insulin': (0, 1000),    # Fasting insulin rarely exceeds 300, but allowing wider range
    'BMI': (10, 80),         # Normal range is 18.5-24.9, but allowing wider range
    'DiabetesPedigreeFunction': (0, 3),  # Based on dataset distribution
    'Age': (0, 120)          # Human age range
}

# Function to validate input within physiological bounds
def validate_input(feature, value):
    if feature in PHYSIOLOGICAL_BOUNDS:
        min_val, max_val = PHYSIOLOGICAL_BOUNDS[feature]
        if value < min_val or value > max_val:
            return False, f"{feature} should be between {min_val} and {max_val}."
    return True, ""

# Function to engineer features from input
def engineer_features(input_data):
    try:
        # Create BMI categories
        bmi = input_data['BMI']
        if bmi < 18.5:
            input_data['BMI_Category_Normal'] = 0
            input_data['BMI_Category_Obese'] = 0
            input_data['BMI_Category_Overweight'] = 0
        elif bmi < 25:
            input_data['BMI_Category_Normal'] = 1
            input_data['BMI_Category_Obese'] = 0
            input_data['BMI_Category_Overweight'] = 0
        elif bmi < 30:
            input_data['BMI_Category_Normal'] = 0
            input_data['BMI_Category_Obese'] = 0
            input_data['BMI_Category_Overweight'] = 1
        else:
            input_data['BMI_Category_Normal'] = 0
            input_data['BMI_Category_Obese'] = 1
            input_data['BMI_Category_Overweight'] = 0
        
        # Create glucose categories
        glucose = input_data['Glucose']
        if glucose < 70:
            input_data['Glucose_Category_Normal'] = 0
            input_data['Glucose_Category_Prediabetes'] = 0
            input_data['Glucose_Category_Diabetes'] = 0
        elif glucose < 100:
            input_data['Glucose_Category_Normal'] = 1
            input_data['Glucose_Category_Prediabetes'] = 0
            input_data['Glucose_Category_Diabetes'] = 0
        elif glucose < 126:
            input_data['Glucose_Category_Normal'] = 0
            input_data['Glucose_Category_Prediabetes'] = 1
            input_data['Glucose_Category_Diabetes'] = 0
        else:
            input_data['Glucose_Category_Normal'] = 0
            input_data['Glucose_Category_Prediabetes'] = 0
            input_data['Glucose_Category_Diabetes'] = 1
        
        # Create interaction features
        input_data['Glucose_BMI'] = input_data['Glucose'] * input_data['BMI']
        input_data['Age_BMI'] = input_data['Age'] * input_data['BMI']
        
        return input_data
    except Exception as e:
        logger.error(f"Error in feature engineering: {e}")
        raise

# Function to make predictions
def predict_diabetes(input_data, model, feature_names):
    try:
        # Engineer features
        engineered_data = engineer_features(input_data)
        
        # Create a DataFrame with the correct feature order
        input_df = pd.DataFrame([engineered_data])
        
        # Ensure all required features are present
        for feature in feature_names:
            if feature not in input_df.columns:
                input_df[feature] = 0
        
        # Select only the features needed by the model
        input_df = input_df[feature_names]
        
        # Make prediction
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)[0][1]
        
        return prediction[0], probability
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        st.error(f"An error occurred during prediction: {e}")
        return None, None

# Function to log user inputs and predictions
def log_prediction(input_data, prediction, probability):
    try:
        log_dir = "prediction_logs"
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, f"predictions_{datetime.now().strftime('%Y%m%d')}.csv")
        
        # Prepare log data
        log_data = input_data.copy()
        log_data['prediction'] = prediction
        log_data['probability'] = probability
        log_data['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Convert to DataFrame
        log_df = pd.DataFrame([log_data])
        
        # Append to log file
        if os.path.exists(log_file):
            log_df.to_csv(log_file, mode='a', header=False, index=False)
        else:
            log_df.to_csv(log_file, index=False)
            
        logger.info(f"Prediction logged: {prediction}, probability: {probability:.4f}")
    except Exception as e:
        logger.error(f"Error logging prediction: {e}")

# Main Streamlit app
def main():
    st.set_page_config(
        page_title="Diabetes Risk Prediction System",
        page_icon="🩺",
        layout="wide"
    )
    
    st.title("Diabetes Risk Prediction System")
    st.write("""
    ### Supply Chain Approach to Healthcare
    This system uses machine learning to predict diabetes risk based on clinical and demographic features.
    Please enter your information below for an assessment.
    """)
    
    # Load the model
    model, feature_names = load_model()
    if model is None:
        st.stop()
    
    # Create a sidebar for inputs
    st.sidebar.header("Patient Information")
    
    # Initialize error message container
    error_container = st.empty()
    
    try:
        # Input form
        with st.sidebar.form("patient_data_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=25, value=0)
                glucose = st.number_input("Glucose Level (mg/dL)", min_value=40, max_value=500, value=120)
                blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=40, max_value=250, value=80)
                skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)
            
            with col2:
                insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0, max_value=1000, value=79)
                bmi = st.number_input("BMI", min_value=10.0, max_value=80.0, value=25.0, format="%.1f")
                dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, format="%.3f")
                age = st.number_input("Age", min_value=0, max_value=120, value=33)
            
            submit_button = st.form_submit_button("Predict Diabetes Risk")
        
        # When form is submitted
        if submit_button:
            # Collect all inputs
            input_data = {
                'Pregnancies': pregnancies,
                'Glucose': glucose,
                'BloodPressure': blood_pressure,
                'SkinThickness': skin_thickness,
                'Insulin': insulin,
                'BMI': bmi,
                'DiabetesPedigreeFunction': dpf,
                'Age': age
            }
            
            # Validate all inputs
            all_valid = True
            error_messages = []
            
            for feature, value in input_data.items():
                valid, message = validate_input(feature, value)
                if not valid:
                    all_valid = False
                    error_messages.append(message)
            
            if not all_valid:
                error_container.error("\n".join(error_messages))
            else:
                error_container.empty()
                
                # Make prediction
                prediction, probability = predict_diabetes(input_data, model, feature_names)
                
                if prediction is not None:
                    # Log the prediction
                    log_prediction(input_data, prediction, probability)
                    
                    # Display results
                    st.subheader("Prediction Results")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if prediction == 1:
                            st.error("⚠️ High Risk of Diabetes")
                        else:
                            st.success("✅ Low Risk of Diabetes")
                    
                    with col2:
                        st.metric("Probability", f"{probability:.1%}")
                    
                    # Risk interpretation
                    st.subheader("Risk Interpretation")
                    
                    if probability < 0.3:
                        st.write("**Low Risk:** The model predicts a low probability of diabetes.")
                        st.write("Recommendation: Maintain a healthy lifestyle with regular exercise and balanced diet.")
                    elif probability < 0.7:
                        st.write("**Moderate Risk:** There are some indicators that suggest an elevated risk of diabetes.")
                        st.write("Recommendation: Consider consulting with a healthcare provider for further evaluation.")
                    else:
                        st.write("**High Risk:** The model predicts a high probability of diabetes.")
                        st.write("Recommendation: Please consult with a healthcare provider as soon as possible for proper diagnosis and care.")
                    
                    # Feature importance visualization
                    st.subheader("Key Risk Factors")
                    
                    # Simplified risk factor display
                    risk_factors = []
                    
                    if glucose > 125:
                        risk_factors.append(("High Glucose Level", f"{glucose} mg/dL", "Elevated blood glucose is a primary indicator of diabetes."))
                    
                    if bmi > 30:
                        risk_factors.append(("Obesity", f"BMI: {bmi:.1f}", "Obesity is strongly associated with type 2 diabetes."))
                    
                    if dpf > 0.8:
                        risk_factors.append(("Family History", f"DPF: {dpf:.3f}", "A high diabetes pedigree function indicates genetic predisposition."))
                    
                    if age > 45:
                        risk_factors.append(("Age", f"{age} years", "Risk of type 2 diabetes increases with age."))
                    
                    if risk_factors:
                        for factor, value, description in risk_factors:
                            st.write(f"**{factor}:** {value}")
                            st.write(description)
                    else:
                        st.write("No specific high-risk factors identified.")
                    
                    # Disclaimer
                    st.info("""
                    **Disclaimer:** This prediction is based on a machine learning model and should not be considered as medical advice. 
                    Please consult with a healthcare professional for proper diagnosis and treatment.
                    """)
        
        # Display information about the system
        with st.expander("About this System"):
            st.write("""
            This diabetes prediction system uses a machine learning model trained on the Pima Indians Diabetes Dataset.
            The model considers several factors that are associated with diabetes risk:
            
            - **Pregnancies:** Number of times pregnant
            - **Glucose:** Plasma glucose concentration (mg/dL)
            - **Blood Pressure:** Diastolic blood pressure (mm Hg)
            - **Skin Thickness:** Triceps skin fold thickness (mm)
            - **Insulin:** 2-Hour serum insulin (mu U/ml)
            - **BMI:** Body mass index (weight in kg/(height in m)²)
            - **Diabetes Pedigree Function:** A function that represents the genetic influence
            - **Age:** Age in years
            
            The system implements a supply chain approach to healthcare data processing, ensuring data quality,
            efficient processing, and reliable predictions.
            """)
        
        with st.expander("Understanding Your Results"):
            st.write("""
            ### How to Interpret Your Results
            
            The prediction is based on statistical patterns found in historical data and provides an estimate of diabetes risk.
            
            **Probability Score:**
            - Below 30%: Generally considered low risk
            - 30% to 70%: Moderate risk that warrants attention
            - Above 70%: High risk that should be discussed with a healthcare provider
            
            ### Next Steps
            
            Regardless of your risk score, the following steps are recommended for diabetes prevention:
            
            1. **Maintain a healthy diet** rich in fruits, vegetables, and whole grains
            2. **Regular physical activity** (at least 150 minutes of moderate exercise per week)
            3. **Maintain a healthy weight**
            4. **Regular health check-ups** with your healthcare provider
            5. **Monitor your blood glucose** if you have risk factors
            
            Remember that this tool is meant for educational purposes and does not replace professional medical advice.
            """)
    
    except Exception as e:
        logger.error(f"Error in Streamlit app: {e}")
        st.error(f"An unexpected error occurred: {e}")
        st.write("Please try again later or contact support if the problem persists.")

if __name__ == "__main__":
    main()
