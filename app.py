# app.py
import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from sklearn.preprocessing import LabelEncoder

# Set page config
st.set_page_config(
    page_title="Financial Inclusion Predictor",
    page_icon="💰",
    layout="wide"
)

# Title and description
st.title("Financial Inclusion Prediction App")
st.markdown("""
This app predicts whether an individual has a bank account based on various demographic and socioeconomic factors.
Please fill in all the fields below to get a prediction.
""")

@st.cache_resource
def load_model():
    # Load the trained model using joblib
    return load('streamlit_check2/FINANCIAL.pkl')

def preprocess_input(data):
    # Create a copy of the input data
    df_processed = data.copy()
    
    # Initialize label encoders for categorical variables
    categorical_mappings = {
        'country': {'Kenya': 0, 'Rwanda': 1, 'Tanzania': 2, 'Uganda': 3},
        'location_type': {'Rural': 0, 'Urban': 1},
        'cellphone_access': {'No': 0, 'Yes': 1},
        'gender_of_respondent': {'Female': 0, 'Male': 1},
        'relationship_with_head': {'Head of Household': 0, 'Other': 1, 'Spouse': 2},
        'marital_status': {'Divorced/Separated': 0, 'Married/Living together': 1, 
                          'Single/Never Married': 2, 'Widowed': 3},
        'education_level': {'No formal education': 0, 'Primary education': 1, 
                          'Secondary education': 2, 'Tertiary education': 3, 
                          'Vocational/Specialised training': 4},
        'job_type': {'Dont Know/Refuse to answer': 0, 'Formally employed Government': 1, 
                    'Formally employed Private': 2, 'Government Dependent': 3, 
                    'Informally employed': 4, 'No Income': 5, 'Other Income': 6, 
                    'Self employed': 7}
    }
    
    # Convert categorical inputs using mappings
    for col, mapping in categorical_mappings.items():
        df_processed[col] = df_processed[col].map(mapping)
    
    return df_processed

# Create the input form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        country = st.selectbox(
            'Country',
            ['Kenya', 'Rwanda', 'Tanzania', 'Uganda']
        )
        
        year = st.number_input(
            'Year',
            min_value=2016,
            max_value=2024,
            value=2018
        )
        
        location_type = st.selectbox(
            'Location Type',
            ['Rural', 'Urban']
        )
        
        cellphone_access = st.selectbox(
            'Cellphone Access',
            ['No', 'Yes']
        )
        
        household_size = st.number_input(
            'Household Size',
            min_value=1,
            max_value=50,
            value=5
        )
        
        age = st.number_input(
            'Age',
            min_value=16,
            max_value=100,
            value=30
        )
    
    with col2:
        gender = st.selectbox(
            'Gender',
            ['Female', 'Male']
        )
        
        relationship_with_head = st.selectbox(
            'Relationship with Head',
            ['Head of Household', 'Spouse', 'Other']
        )
        
        marital_status = st.selectbox(
            'Marital Status',
            ['Married/Living together', 'Single/Never Married', 
             'Divorced/Separated', 'Widowed']
        )
        
        education_level = st.selectbox(
            'Education Level',
            ['No formal education', 'Primary education', 'Secondary education', 
             'Tertiary education', 'Vocational/Specialised training']
        )
        
        job_type = st.selectbox(
            'Job Type',
            ['Self employed', 'Government Dependent', 'Formally employed Private',
             'Formally employed Government', 'Informally employed', 'No Income',
             'Other Income', 'Dont Know/Refuse to answer']
        )

    submit_button = st.form_submit_button(label='Predict')

if submit_button:
    # Create a dictionary with the input values
    input_data = {
        'country': [country],
        'year': [year],
        'location_type': [location_type],
        'cellphone_access': [cellphone_access],
        'household_size': [household_size],
        'age_of_respondent': [age],
        'gender_of_respondent': [gender],
        'relationship_with_head': [relationship_with_head],
        'marital_status': [marital_status],
        'education_level': [education_level],
        'job_type': [job_type]
    }
    
    # Convert to DataFrame
    input_df = pd.DataFrame(input_data)
    
    # Preprocess the input
    processed_input = preprocess_input(input_df)
    
    # Load model and make prediction
    try:
        model = load_model()
        prediction = model.predict(processed_input)
        probability = model.predict_proba(processed_input)
        
        # Display prediction with a more detailed explanation
        st.subheader('Prediction Results')
        
        # Create columns for prediction and probability
        pred_col, prob_col = st.columns(2)
        
        with pred_col:
            if prediction[0] == 1:
                st.success('✅ Has Bank Account')
            else:
                st.error('❌ No Bank Account')
                
        with prob_col:
            st.metric(
                label="Confidence",
                value=f"{probability[0][prediction[0]]:.1%}"
            )
        
        # Add a progress bar for the probability
        st.progress(float(probability[0][prediction[0]]))
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Please make sure all fields are filled correctly and try again.")

# Add additional information about the project
st.markdown("""
---
### About this Project
This application uses a Random Forest Classifier trained on financial inclusion data from various African countries. 
The model predicts whether an individual is likely to have a bank account based on various demographic and socioeconomic factors.

### Features Used
- Country of residence
- Location type (Rural/Urban)
- Access to cell phone
- Household size
- Age
- Gender
- Relationship with household head
- Marital status
- Education level
- Job type

### Model Performance
The model was trained on historical data and achieved good performance metrics. However, predictions should be used as guidance rather than absolute determinations.
""")