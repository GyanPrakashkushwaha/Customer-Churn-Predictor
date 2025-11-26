import streamlit as st
import pandas as pd
from churnPredictor.pipeline.prediction import PredictionPipeline

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊", layout="wide")

st.title('📊 Customer Churn Prediction')
st.markdown("Enter customer details below to predict if they are likely to churn.")

# --- Input Form ---
with st.form("churn_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input(label='Age', min_value=18, max_value=100, value=30)
        gender = st.selectbox(label='Gender', options=['Male', 'Female'])
        location = st.selectbox(label='Location', options=['Houston', 'Los Angeles', 'Miami', 'New York'])
    
    with col2:
        subscription_length = st.number_input(label='Subscription Length (Months)', min_value=1, max_value=24, value=12)
        monthly_bill = st.number_input(label='Monthly Bill ($)', min_value=1.0, value=50.0)
        total_usage = st.number_input(label='Total Usage (GB)', min_value=1.0, value=100.0)

    submit_button = st.form_submit_button(label='Predict Churn')

# --- Prediction Logic ---
if submit_button:
    # Prepare the data dictionary (Match the column names from your training data exactly)
    data_dict = {
        'Age': [age],
        'Gender': [gender],
        'Location': [location],
        'Subscription_Length_Months': [subscription_length],
        'Monthly_Bill': [monthly_bill],
        'Total_Usage_GB': [total_usage]
    }
    
    # Convert to DataFrame
    df = pd.DataFrame(data_dict)
    
    try:
        # Call our pipeline
        pipeline = PredictionPipeline()
        prediction = pipeline.predict(df)
        
        result = prediction[0]
        
        # Display Result
        st.divider()
        if result == 1:
            st.error("⚠️ Prediction: This customer is likely to **CHURN**.")
        else:
            st.success("✅ Prediction: This customer is likely to **STAY**.")
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")