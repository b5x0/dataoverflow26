import pandas as pd
import lightgbm as lgb
import optuna
import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from optuna.integration import LightGBMPruningCallback
from solution import preprocess
import warnings
import os
warnings.filterwarnings('ignore')

def objective(trial):
    train_df = pd.read_csv("train.csv")
    df_proc = preprocess(train_df)
    
    X = df_proc.drop(columns=['User_ID', 'Purchased_Coverage_Bundle'])
    y = df_proc['Purchased_Coverage_Bundle']
    
    weights = {0: 1.25, 1: 0.74, 2: 0.57, 3: 0.99, 4: 0.77, 5: 0.67, 6: 1.02, 7: 0.75, 8: 2.35, 9: 0.89}
    
    params = {
        'objective': 'multiclass',
        'num_class': 10,
        'metric': 'multi_error', 
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 0.03, 0.07),
        'max_depth': 10,
        'num_leaves': trial.suggest_int('num_leaves', 40, 80),
        'min_child_samples': trial.suggest_int('min_child_samples', 30, 100),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
        'n_estimators': 250,
        'n_jobs': -1, # Train full-speed using all cores
        'class_weight': weights,
        'random_state': 42,
        'verbose': -1
    }
    
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    oof_preds = pd.Series(0, index=X.index)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=20, verbose=False)
            ]
        )
        
        val_preds = model.predict(X_val)
        oof_preds.iloc[val_idx] = val_preds
        
    score = f1_score(y, oof_preds, average='macro')
    return score

if __name__ == "__main__":
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
    study = optuna.create_study(direction="maximize", pruner=pruner)
    print("Starting 20-Trial Speed Sniper Sprint...")
    study.optimize(objective, n_trials=20)
    
    print("Best trial params:", study.best_trial.params)
    print("Best OOF Macro F1:", study.best_value)
    
    print("Retraining final Speed Sniper on full dataset...")
    train_df = pd.read_csv("train.csv")
    df_proc = preprocess(train_df)
    X = df_proc.drop(columns=['User_ID', 'Purchased_Coverage_Bundle'])
    y = df_proc['Purchased_Coverage_Bundle']
    
    best_params = study.best_trial.params
    best_params['objective'] = 'multiclass'
    best_params['num_class'] = 10
    best_params['boosting_type'] = 'gbdt'
    weights = {0: 1.25, 1: 0.74, 2: 0.57, 3: 0.99, 4: 0.77, 5: 0.67, 6: 1.02, 7: 0.75, 8: 2.35, 9: 0.89}
    best_params['class_weight'] = weights
    
    # HARDCODE values not tuned
    best_params['n_estimators'] = 250
    best_params['max_depth'] = 10
    
    # CRITICAL: SET TO 1 CORE FOR INFERENCE TO PREVENT LATENCY PENALTY!
    best_params['n_jobs'] = 1
    best_params['random_state'] = 42
    best_params['verbose'] = -1
    
    final_model = lgb.LGBMClassifier(**best_params)
    final_model.fit(X, y)
    
    print("Saving speed model to model.pkl...")
    joblib.dump(final_model, 'model.pkl')
    print(f"Model saved. Size: {os.path.getsize('model.pkl') / (1024*1024):.2f} MB")
