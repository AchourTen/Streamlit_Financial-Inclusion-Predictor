import pandas as pd
import joblib

import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_curve, 
    auc, 
    precision_recall_curve,
    average_precision_score
)

from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
    
    
df  = pd.read_csv('/Users/aymen/Downloads/Financial_inclusion_dataset.csv')


def explore_data(df):
    print("Dataset Shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nData Types:\n", df.dtypes)
    print("\nMissing Values:\n", df.isnull().sum())
    print("\nDuplicate Rows:", df.duplicated().sum())
    
    # Basic statistics for numerical columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    print("\nNumerical Statistics:\n", df[numerical_cols].describe())
    
    # Value counts for categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        print(f"\nValue counts for {col}:\n", df[col].value_counts().head())

def handle_outliers_zscore(df, columns, threshold=3):
    """
    Handle outliers using z-score method
    Parameters:
    - df: DataFrame
    - columns: list of columns to check for outliers
    - threshold: z-score threshold (default=3)
    """
    df_clean = df.copy()
    for column in columns:
        # Calculate z-scores
        z_scores = np.abs((df_clean[column] - df_clean[column].mean()) / df_clean[column].std())
        
        # Print number of outliers found
        outliers_count = len(z_scores[z_scores > threshold])
        print(f"Number of outliers found in {column}: {outliers_count}")
        
        # Replace outliers with the threshold value
        df_clean[column] = df_clean[column].mask(
            z_scores > threshold,
            df_clean[column].mean() + threshold * df_clean[column].std() * np.sign(df_clean[column] - df_clean[column].mean())
        )
    return df_clean

def preprocess_data(df):
    # Create a copy to avoid modifying the original dataframe
    df_processed = df.copy()
    
    # Drop unnecessary columns
    df_processed.drop('uniqueid', axis=1, inplace=True)
    
    # Handle outliers using z-score method for numerical columns
    numerical_cols = ['household_size', 'age_of_respondent']
    print("\nHandling outliers using z-score method:")
    df_processed = handle_outliers_zscore(df_processed, numerical_cols, threshold=3)
    
    # Encode categorical variables
    le = LabelEncoder()
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        df_processed[col] = le.fit_transform(df_processed[col])
    
    return df_processed



def plot_distributions(df_original, df_processed, numerical_cols):
    for col in numerical_cols:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Before outlier handling
        sns.histplot(data=df_original, x=col, ax=ax1)
        ax1.set_title(f'{col} - Before Z-score')
        
        # After outlier handling
        sns.histplot(data=df_processed, x=col, ax=ax2)
        ax2.set_title(f'{col} - After Z-score')
        
        plt.tight_layout()
        plt.show()
        
def train_improved_model(df_processed):
    # Separate features and target
    X = df_processed.drop('bank_account', axis=1)
    y = df_processed['bank_account']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train the improved model
    model = create_balanced_model(X_train, y_train, X_test, y_test)
    
    # Get feature importance from the random forest component
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.named_steps['rf'].feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
    plt.title('Top 10 Most Important Features')
    plt.show()
    
    return model, feature_importance
        

def evaluate_model_performance(model, X_train, X_test, y_train, y_test):
    """
    Comprehensive model evaluation with multiple metrics
    """
    # Get predictions and probabilities
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Basic metrics
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, 
             label=f'Precision-Recall curve (AP = {avg_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.show()
    
    # Cross-validation scores
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    print("\nCross-validation scores:", cv_scores)
    print(f"Average CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    return {
        'confusion_matrix': cm,
        'roc_auc': roc_auc,
        'average_precision': avg_precision,
        'cv_scores': cv_scores
    }
   
def create_balanced_model(X_train, y_train, X_test, y_test):
    """
    Creates and trains a balanced model with optimized hyperparameters
    """
    # Define the parameter grid for RandomizedSearchCV
    param_grid = {
        'rf__n_estimators': [100, 200, 300],
        'rf__max_depth': [10, 20, 30, None],
        'rf__min_samples_split': [2, 5, 10],
        'rf__min_samples_leaf': [1, 2, 4],
        'rf__max_features': ['sqrt', 'log2'],
        'rf__class_weight': ['balanced', 'balanced_subsample']
    }
    
    # Create pipeline with SMOTE and RandomForest
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('rf', RandomForestClassifier(random_state=42))
    ])
    
    # Perform RandomizedSearchCV
    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=20,
        cv=5,
        scoring='f1_weighted',
        random_state=42,
        n_jobs=-1
    )
    
    # Fit the model
    random_search.fit(X_train, y_train)
    
    # Print best parameters
    print("\nBest parameters found:")
    print(random_search.best_params_)
    
    # Get predictions
    y_pred = random_search.predict(X_test)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    # Perform cross-validation to check for overfitting
    cv_scores = cross_val_score(
        random_search.best_estimator_,
        X_train, 
        y_train,
        cv=5,
        scoring='f1_weighted'
    )
    
    print("\nCross-validation scores:")
    print(f"Mean F1: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    return random_search.best_estimator_

# Main execution
# Main execution remains the same until model training
print("Data Exploration of Original Dataset:")
explore_data(df)

# Process the data
df_processed = preprocess_data(df)

# Plot distributions before and after outlier handling
numerical_cols = ['household_size', 'age_of_respondent']
plot_distributions(df, df_processed, numerical_cols)

print("Data Shape after Preprocessing:", df_processed.shape)
print(df_processed.head())

# Train and evaluate the model with enhanced metrics
model, feature_importance = train_improved_model(df_processed)
# Save the model
joblib.dump(model, 'streamlit_check2/FINANCIAL.pkl')

