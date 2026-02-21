import pandas as pd
import numpy as np
import optuna
import joblib
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
import category_encoders as ce
import warnings

warnings.filterwarnings('ignore')

def preprocess_base(df):
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

    # Downcast numerics to honor 1GB memory limits strictly
    float_cols = df_proc.select_dtypes(include=['float64']).columns
    df_proc[float_cols] = df_proc[float_cols].astype('float32')
    
    int_cols = df_proc.select_dtypes(include=['int64']).columns
    for col in int_cols:
        if col not in ['User_ID', 'Purchased_Coverage_Bundle']:
             df_proc[col] = pd.to_numeric(df_proc[col], downcast='integer')

    return df_proc

def objective(trial, X, y):
    params = {
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'class_weight': 'balanced',
        'n_estimators': 100, # Fast trials
        'num_leaves': trial.suggest_int('num_leaves', 10, 80),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'random_state': 42,
        'verbose': -1,
        'n_jobs': 1
    }
    
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in skf.split(X, y):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        te = ce.TargetEncoder(cols=['Region_Code', 'Broker_ID', 'Employer_ID'])
        X_train_enc = te.fit_transform(X_train, y_train)
        X_val_enc = te.transform(X_val)
        
        # Float32 Downcast encodings for memory
        enc_float = X_train_enc.select_dtypes(include=['float64']).columns
        X_train_enc[enc_float] = X_train_enc[enc_float].astype('float32')
        X_val_enc[enc_float] = X_val_enc[enc_float].astype('float32')

        model = LGBMClassifier(**params)
        model.fit(X_train_enc, y_train)
        
        preds = model.predict(X_val_enc)
        f1 = f1_score(y_val, preds, average='macro')
        scores.append(f1)
        
    return np.mean(scores)

def run_optimization():
    print("Loading initial data...")
    train = pd.read_csv('train.csv')
    
    X = preprocess_base(train.drop(columns=['Purchased_Coverage_Bundle']))
    X = X.drop(columns=['User_ID'])
    y = train['Purchased_Coverage_Bundle']

    print("Beginning Optuna Optimization Sprint (50 Trials)...")
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X, y), n_trials=50)
    
    print("Optimization Complete.")
    print("Best F1 Macro Score:", study.best_value)
    print("Best Parameters:", study.best_params)
    
    print("\nTraining Final Model...")
    te_final = ce.TargetEncoder(cols=['Region_Code', 'Broker_ID', 'Employer_ID'])
    X_enc = te_final.fit_transform(X, y)
    
    enc_float = X_enc.select_dtypes(include=['float64']).columns
    X_enc[enc_float] = X_enc[enc_float].astype('float32')
    
    final_params = study.best_params
    final_params.update({
        'objective': 'multiclass',
        'class_weight': 'balanced',
        'n_estimators': 150, 
        'random_state': 42,
        'verbose': -1,
        'n_jobs': 1
    })
    
    final_model = LGBMClassifier(**final_params)
    final_model.fit(X_enc, y)
    
    print("Exporting Artifacts...")
    joblib.dump(final_model, 'model_opt.pkl')
    joblib.dump(te_final, 'target_encoder.pkl')
    
    print("TargetEncoder export complete: 'target_encoder.pkl'. Please integrate logic into solution.py")
    
if __name__ == "__main__":
    run_optimization()
