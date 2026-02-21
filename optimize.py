import pandas as pd
import lightgbm as lgb
import optuna
import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from solution import preprocess
import warnings
warnings.filterwarnings('ignore')

def objective(trial):
    train_df = pd.read_csv("train.csv")
    
    # Apply exactly what will happen in the test environment
    df_proc = preprocess(train_df)
    
    X = df_proc.drop(columns=['User_ID', 'Purchased_Coverage_Bundle'])
    y = df_proc['Purchased_Coverage_Bundle']
    
    # Robust safe constraints defined by Senior Principal ML Engineer
    params = {
        'objective': 'multiclass',
        'num_class': 10,
        'metric': 'multi_error', 
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 7),
        'num_leaves': trial.suggest_int('num_leaves', 10, 40),
        'min_child_samples': trial.suggest_int('min_child_samples', 30, 100),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
        'n_estimators': 150, # Keep estimators low for <10s inference and <200MB size
        'class_weight': 'balanced',
        'random_state': 42,
        'verbose': -1
    }
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = pd.Series(0, index=X.index)
    
    for train_idx, val_idx in skf.split(X, y):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
        )
        
        # Predict class
        val_preds = model.predict(X_val)
        oof_preds.iloc[val_idx] = val_preds
        
    score = f1_score(y, oof_preds, average='macro')
    return score

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)
    
    print("Best trial params:", study.best_trial.params)
    print("Best OOF Macro F1:", study.best_value)
    
    # Retrain final model on 100% of data to maximize knowledge
    train_df = pd.read_csv("train.csv")
    df_proc = preprocess(train_df)
    X = df_proc.drop(columns=['User_ID', 'Purchased_Coverage_Bundle'])
    y = df_proc['Purchased_Coverage_Bundle']
    
    best_params = study.best_trial.params
    best_params['objective'] = 'multiclass'
    best_params['num_class'] = 10
    best_params['boosting_type'] = 'gbdt'
    best_params['class_weight'] = 'balanced'
    best_params['n_estimators'] = 150
    best_params['random_state'] = 42
    best_params['verbose'] = -1
    
    print("Training final robust model on full dataset...")
    final_model = lgb.LGBMClassifier(**best_params)
    final_model.fit(X, y)
    
    joblib.dump(final_model, 'model.pkl')
    import os
    print(f"Model saved to model.pkl. Size: {os.path.getsize('model.pkl') / (1024*1024):.2f} MB")
