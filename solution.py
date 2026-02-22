# Import necessary libraries here
import pandas as pd
import joblib
import lightgbm as lgb
import numpy as np

def preprocess(df):
    # 2. DROP FIRST
    cols_to_drop = ['Broker_ID', 'Employer_ID']
    df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)
    
    # 1. IN-PLACE PREPROCESSING / Vectorized engineering
    df['Total_Deps'] = df['Adult_Dependents'] + df['Child_Dependents'].fillna(0) + df['Infant_Dependents']
    df['Income_per_Dep'] = df['Estimated_Annual_Income'] / (df['Total_Deps'] + 1)
    df['Risk_Ratio'] = df['Previous_Claims_Filed'] / (df['Years_Without_Claims'] + 1)

    # 3. CATEGORICAL CASTING
    cat_cols = ['Region_Code', 'Broker_Agency_Type', 'Deductible_Tier', 
                'Acquisition_Channel', 'Payment_Schedule', 'Employment_Status', 
                'Policy_Start_Month']
    
    # Fast casting via dictionary mapping
    cat_dtypes = {col: 'category' for col in cat_cols if col in df.columns}
    if cat_dtypes:
        df = df.astype(cat_dtypes, copy=False)
        
    # 4. DATATYPE COMPRESSION
    float_cols = df.select_dtypes(include=['float64']).columns
    if len(float_cols) > 0:
        df[float_cols] = df[float_cols].astype('float32', copy=False)
        
    int_cols = df.select_dtypes(include=['int64']).columns
    int_cols = [c for c in int_cols if c not in ['User_ID', 'Purchased_Coverage_Bundle']]
    if len(int_cols) > 0:
        df[int_cols] = df[int_cols].astype('int32', copy=False)

    return df

def load_model():
    return joblib.load('model.pkl')

def predict(df, model):
    # Eagerly capture and drop User_ID
    user_ids = df['User_ID'].values
    df.drop(columns=['User_ID'], inplace=True)
    
    # Predict directly, dropping the DataFrame to just raw values handled natively
    preds = model.predict(df, raw_score=False).astype(int)
    
    # Final rapid DF construction
    predictions = pd.DataFrame({
        'User_ID': user_ids,
        'Purchased_Coverage_Bundle': preds
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
