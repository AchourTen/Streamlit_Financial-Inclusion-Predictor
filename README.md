# Streamlit_Financial-Inclusion-Predictor

## Description
This project implements a machine learning model to predict whether an individual has a bank account based on various demographic and socioeconomic factors. The application includes both a model training pipeline and a Streamlit web interface for making predictions.

## Features
- Data preprocessing and exploratory data analysis
- Machine learning model training with Random Forest Classifier
- Handling class imbalance using SMOTE
- Feature importance analysis
- Model evaluation with multiple metrics
- Interactive web interface built with Streamlit
- Support for multiple African countries (Kenya, Rwanda, Tanzania, Uganda)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/AchourTen/Streamlit_Financial-Inclusion-Predictor.git
cd Streamlit_Financial-Inclusion-Predictor
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model
To train the model, run:
```bash
python model_training.py
```

### Running the Web Application
To start the Streamlit web interface:
```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

## File Structure
```
financial-inclusion-predictor/
├── app.py                    # Streamlit web application
├── train_model.py           # Model training script
├── requirements.txt         # Project dependencies
├── README.md               # Project documentation
└── streamlit_check2/       # Model artifacts directory
    └── FINANCIAL.pkl       # Trained model (will appare after the training)

```

## Variable Definitions
Below are the definitions for all variables used in the dataset:

| Variable | Description |
|----------|-------------|
| country | Country interviewee is in |
| year | Year survey was done in |
| uniqueid | Unique identifier for each interviewee |
| location_type | Type of location: Rural, Urban |
| cellphone_access | If interviewee has access to a cellphone: Yes, No |
| household_size | Number of people living in one house |
| age_of_respondent | The age of the interviewee |
| gender_of_respondent | Gender of interviewee: Male, Female |
| relationship_with_head | The interviewee's relationship with the head of the house: Head of Household, Spouse, Child, Parent, Other relative, Other non-relatives, Dont know |
| marital_status | The martial status of the interviewee: Married/Living together, Divorced/Seperated, Widowed, Single/Never Married, Don't know |
| education_level | Highest level of education: No formal education, Primary education, Secondary education, Vocational/Specialised training, Tertiary education, Other/Dont know/RTA |
| job_type | Type of job interviewee has: Farming and Fishing, Self employed, Formally employed Government, Formally employed Private, Informally employed, Remittance Dependent, Government Dependent, Other Income, No Income, Dont Know/Refuse to answer |

```

```

## Model Features
The model uses the following features for prediction:
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

## Technical Details
- Model: Random Forest Classifier with SMOTE for handling class imbalance
- Preprocessing: Standard scaling and label encoding
- Evaluation metrics: ROC-AUC, Precision-Recall, Confusion Matrix
- Cross-validation: 5-fold with stratification

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
