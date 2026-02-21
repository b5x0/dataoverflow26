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

def preprocess(df):
    # Create a copy to avoid SettingWithCopyWarning, but do it efficiently
    df_proc = df.copy()

    # Fill basic missing values
    df_proc['Child_Dependents'] = df_proc['Child_Dependents'].fillna(-1)
    df_proc['Region_Code'] = df_proc['Region_Code'].fillna('Unknown')
    df_proc['Deductible_Tier'] = df_proc['Deductible_Tier'].fillna('Unknown')
    df_proc['Acquisition_Channel'] = df_proc['Acquisition_Channel'].fillna('Unknown')
    df_proc['Broker_ID'] = df_proc['Broker_ID'].fillna(-1).astype(str)
    df_proc['Employer_ID'] = df_proc['Employer_ID'].fillna(-1).astype(str)

    # Convert object columns to category for LightGBM
    cat_cols = ['Region_Code', 'Broker_Agency_Type', 'Deductible_Tier',
                'Acquisition_Channel', 'Payment_Schedule', 'Employment_Status',
                'Policy_Start_Month']
    for col in cat_cols:
        if col in df_proc.columns:
            df_proc[col] = df_proc[col].astype('category')
            
    # Feature Engineering (Combined Dependents)
    df_proc['Total_Dependents'] = df_proc['Adult_Dependents'] + df_proc['Child_Dependents'].replace(-1, 0) + df_proc['Infant_Dependents']
    df_proc['Risk_Score_Proxy'] = df_proc['Years_Without_Claims'] - df_proc['Previous_Claims_Filed']

    # Deep features
    df_proc['Income_per_Dependent'] = df_proc['Estimated_Annual_Income'] / (df_proc['Total_Dependents'] + 1)
    df_proc['Risk_Ratio'] = df_proc['Risk_Score_Proxy'] / (df_proc['Years_Without_Claims'] + 1)

    # Downcast numeric types for memory optimization (1GB constraint)
    float_cols = df_proc.select_dtypes(include=['float64']).columns
    df_proc[float_cols] = df_proc[float_cols].astype('float32')
    
    int_cols = df_proc.select_dtypes(include=['int64']).columns
    for col in int_cols:
        # Need to keep User_ID undisturbed, so exclude it from int casting if it sneaks in.
        if col != 'User_ID' and col != 'Purchased_Coverage_Bundle':
             # downcast if possible to save memory
             df_proc[col] = pd.to_numeric(df_proc[col], downcast='integer')

    return df_proc


def load_model():
    model = None
    # ------------------ MODEL LOADING LOGIC ------------------

    # Inside this block, load your trained model.
    # We now load a dictionary containing both the classifier and the encoder
    model = joblib.load('model.pkl')

    # ------------------ END MODEL LOADING LOGIC ------------------
    return model


def predict(df, model):
    # ------------------ PREDICTION LOGIC ------------------

    # Ignore User_ID in features
    X = df.drop(columns=['User_ID'])
    
    # Unpack loaded model components (Combined dict format)
    lgbm_model = model['model']
    encoding_dict = model['encoding_dict']
    
    # Manually map the target encodings natively for judge compatibility
    for col, mapping in encoding_dict.items():
        if col in X.columns:
            # Map values, fill unseen categories with the global mean of the training target
            X[col] = X[col].map(mapping).fillna(mapping.get('__global_mean__', 0))
    
    # Ensure encoded columns are floats and downcast for memory
    enc_float = X.select_dtypes(include=['float64']).columns
    X[enc_float] = X[enc_float].astype('float32')

    # Generate predictions
    preds = lgbm_model.predict(X)
    predictions = pd.DataFrame({
        'User_ID': df['User_ID'],
        'Purchased_Coverage_Bundle': preds.astype(int) # Explicit integer cast
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
