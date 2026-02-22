# Import necessary libraries here
import pandas as pd
import joblib
import lightgbm as lgb
import numpy as np

def preprocess(df):
    # Create a copy to avoid SettingWithCopyWarning, but do it efficiently
    df_proc = df.copy()

    # Fill basic missing values
    df_proc['Child_Dependents'] = df_proc['Child_Dependents'].fillna(-1)
    df_proc['Region_Code'] = df_proc['Region_Code'].fillna('Unknown')
    df_proc['Deductible_Tier'] = df_proc['Deductible_Tier'].fillna('Unknown')
    df_proc['Acquisition_Channel'] = df_proc['Acquisition_Channel'].fillna('Unknown')
    df_proc['Broker_ID'] = df_proc['Broker_ID'].fillna(-1)
    df_proc['Employer_ID'] = df_proc['Employer_ID'].fillna(-1)

    # Convert object columns to category for LightGBM
    cat_cols = ['Region_Code', 'Broker_Agency_Type', 'Deductible_Tier',
                'Acquisition_Channel', 'Payment_Schedule', 'Employment_Status',
                'Policy_Start_Month']
    for col in cat_cols:
        if col in df_proc.columns:
            df_proc[col] = df_proc[col].astype('category')
            
    # Feature Engineering (Combined Dependents and Risk)
    df_proc['Total_Dependents'] = df_proc['Adult_Dependents'] + df_proc['Child_Dependents'].replace(-1, 0) + df_proc['Infant_Dependents']
    df_proc['Income_per_Dependent'] = df_proc['Estimated_Annual_Income'] / (df_proc['Total_Dependents'] + 1)
    df_proc['Risk_Ratio'] = df_proc['Previous_Claims_Filed'] / (df_proc['Years_Without_Claims'] + 1)

    # CRITICAL: Drop noisy IDs and highly correlated Risk_Score_Proxy
    cols_to_drop = ['Broker_ID', 'Employer_ID', 'Risk_Score_Proxy']
    df_proc = df_proc.drop(columns=[col for col in cols_to_drop if col in df_proc.columns])

    # Downcast numeric types for memory optimization (1GB constraint)
    float_cols = df_proc.select_dtypes(include=['float64']).columns
    df_proc[float_cols] = df_proc[float_cols].astype('float32')
    
    int_cols = df_proc.select_dtypes(include=['int64']).columns
    for col in int_cols:
        if col != 'User_ID' and col != 'Purchased_Coverage_Bundle':
             df_proc[col] = pd.to_numeric(df_proc[col], downcast='integer')

    return df_proc

def load_model():
    return joblib.load('model.pkl')

def predict(df, model):
    X = df.drop(columns=['User_ID'])
    preds = model.predict(X)
    
    predictions = pd.DataFrame({
        'User_ID': df['User_ID'].values,
        'Purchased_Coverage_Bundle': preds.astype(int)
    })
    return predictions

# ----------------------------------------------------------------
# Your code will be called in the following way:
# Note that we will not be using the function defined below.
# ----------------------------------------------------------------

def run(df) -> tuple[float, float, float]:
    from time import time

    # Load the processed data:
    df_processed = preprocess(df)

    # Load the model:
    model = load_model()
    size = get_model_size(model)

    # Get the predictions and time taken:
    start = time.perf_counter()
    predictions = predict(df_processed, model)
    duration = time.perf_counter() - start
    accuracy = get_model_accuracy(predictions)

    return size, accuracy, duration

def get_model_size(model):
    pass

def get_model_accuracy(predictions):
    pass
