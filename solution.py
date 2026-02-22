# ----------------------------------------------------------------
# IMPORTANT: This template will be used to evaluate your solution.
#
# Do NOT change the function signatures.
# And ensure that your code runs within the time limits.
# The time calculation will be computed for the predict function only.
#
# Good luck!
# ----------------------------------------------------------------

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
    # ------------------ PREDICTION LOGIC ------------------

    # Ignore User_ID in features
    X = df.drop(columns=['User_ID'])
    
    # Predict (model is already constrained to n_jobs=1 from training)
    # Flatten the prediction array since CatBoost returns a 2D column vector automatically
    preds = model.predict(X).flatten()
    
    # Ultra-fast DataFrame construction using numpy values
    predictions = pd.DataFrame({
        'User_ID': df['User_ID'].values,
        'Purchased_Coverage_Bundle': preds.astype(int)
    })

    # ------------------ END PREDICTION LOGIC ------------------
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
    predictions = predict(
        df_processed, model
    )  # NOTE: Don't call the `preprocess` function here.

    duration = time.perf_counter() - start
    accuracy = get_model_accuracy(predictions)

    return size, accuracy, duration


# ----------------------------------------------------------------
# Helper functions you should not disturb yourself with.
# ----------------------------------------------------------------


def get_model_size(model):
    pass


def get_model_accuracy(predictions):
    pass
