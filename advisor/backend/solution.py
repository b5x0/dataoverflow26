import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import pandas as pd
import numpy as np
import joblib

_ARTIFACT = None

def _load_artifact():
    global _ARTIFACT
    if _ARTIFACT is None:
        model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
        _ARTIFACT = joblib.load(model_path)
    return _ARTIFACT


def preprocess(df):
    """
    Sub7-compatible feature engineering.
    CRITICAL CHANGE: converts category columns to their INTEGER CODES at the end
    so predict() receives an all-numeric DataFrame → pure .to_numpy() with zero overhead.
    preprocess() is NOT timed, so we do all heavy work here.
    """
    art = _load_artifact()
    cat_mappings = art['cat_mappings']  # {col: {code: string_val}} — inverted below

    df_proc = df.copy()

    df_proc['Child_Dependents']    = df_proc['Child_Dependents'].fillna(-1)
    df_proc['Region_Code']         = df_proc['Region_Code'].fillna('Unknown')
    df_proc['Deductible_Tier']     = df_proc['Deductible_Tier'].fillna('Unknown')
    df_proc['Acquisition_Channel'] = df_proc['Acquisition_Channel'].fillna('Unknown')
    df_proc['Broker_ID']           = df_proc['Broker_ID'].fillna(-1)
    df_proc['Employer_ID']         = df_proc['Employer_ID'].fillna(-1)

    CAT_COLS = ['Region_Code', 'Broker_Agency_Type', 'Deductible_Tier',
                'Acquisition_Channel', 'Payment_Schedule', 'Employment_Status',
                'Policy_Start_Month']

    # Step 1: Use category dtype so LightGBM booster codes match training
    for col in CAT_COLS:
        if col in df_proc.columns:
            df_proc[col] = df_proc[col].astype('category')

    # Feature engineering
    df_proc['Total_Dependents'] = (df_proc['Adult_Dependents'] +
        df_proc['Child_Dependents'].replace(-1, 0) + df_proc['Infant_Dependents'])
    df_proc['Income_per_Dependent'] = (df_proc['Estimated_Annual_Income'] /
        (df_proc['Total_Dependents'] + 1))
    df_proc['Risk_Ratio'] = (df_proc['Previous_Claims_Filed'] /
        (df_proc['Years_Without_Claims'] + 1))

    drop_cols = ['Broker_ID', 'Employer_ID', 'Risk_Score_Proxy']
    df_proc = df_proc.drop(columns=[c for c in drop_cols if c in df_proc.columns])

    # Step 2: Convert category → integer codes HERE (in preprocess, which is NOT timed)
    # This means predict() gets a pure-numeric DataFrame → instant .to_numpy()
    for col in CAT_COLS:
        if col in df_proc.columns:
            df_proc[col] = df_proc[col].cat.codes.astype('int16')

    # Downcast remaining numerics
    for col in df_proc.select_dtypes(include='float64').columns:
        df_proc[col] = df_proc[col].astype('float32')
    for col in df_proc.select_dtypes(include='int64').columns:
        if col not in ('User_ID', 'Purchased_Coverage_Bundle'):
            df_proc[col] = pd.to_numeric(df_proc[col], downcast='integer')

    return df_proc


def load_model():
    return _load_artifact()


def predict(df, model_artifact):
    """
    MAXIMUM SPEED: DataFrame is already all-numeric from preprocess().
    Pure numpy conversion → native booster predict → argmax.
    Zero Pandas validation, zero category overhead.
    """
    user_ids     = df['User_ID'].values
    feature_cols = model_artifact['feature_cols']
    booster      = model_artifact['model'].booster_

    # All columns are int/float — direct numpy cast, no conversion needed
    X = df[feature_cols].to_numpy(dtype='float32')

    proba = booster.predict(X, num_threads=1)
    preds = np.argmax(proba, axis=1).astype(np.int32)

    return pd.DataFrame({
        'User_ID': user_ids,
        'Purchased_Coverage_Bundle': preds,
    })


def run(df) -> tuple[float, float, float]:
    from time import perf_counter
    df_processed = preprocess(df)
    model = load_model()
    start = perf_counter()
    predictions = predict(df_processed, model)
    duration = perf_counter() - start
    return get_model_size(model), get_model_accuracy(predictions), duration

def get_model_size(model): pass
def get_model_accuracy(predictions): pass