# Import necessary libraries here
import pandas as pd
import joblib
import lightgbm as lgb
import numpy as np

def preprocess(df):
    # Avoid .copy() if possible to save RAM and CPU cycles
    # Work directly on the dataframe or use a very lean selection
    
    # Vectorized engineering (much faster than apply/loops)
    df['Total_Deps'] = df['Adult_Dependents'] + df['Child_Dependents'].fillna(0) + df['Infant_Dependents']
    df['Income_per_Dep'] = df['Estimated_Annual_Income'] / (df['Total_Deps'] + 1)
    df['Risk_Ratio'] = df['Previous_Claims_Filed'] / (df['Years_Without_Claims'] + 1)

    # Fast categorical conversion
    cat_cols = ['Region_Code', 'Deductible_Tier', 'Acquisition_Channel', 'Employment_Status']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
            
    # Final downcast to float32
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')

    # Drop noisy IDs
    cols_to_drop = ['Broker_ID', 'Employer_ID']
    df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)
        
    return df

def load_model():
    return joblib.load('model.pkl')

def predict(df, model):
    X = df.drop(columns=['User_ID'])
    
    # Vectorization & Flattening: Access numpy array via .values, predict, and cast
    # (Note: Passing X natively handles LightGBM categorical types better than X.values)
    preds = model.predict(X).astype(int)
    
    # Final rapid DF construction
    predictions = pd.DataFrame({
        'User_ID': df['User_ID'].values,
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
