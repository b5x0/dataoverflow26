import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report

import warnings
warnings.filterwarnings('ignore')

# We need the original preprocessing before we drop things
def preprocess_for_analysis(df):
    df_proc = df.copy()
    
    # Missing values
    temp_cols = ['Child_Dependents', 'Broker_ID', 'Employer_ID']
    for c in temp_cols:
        if c in df_proc: df_proc[c] = df_proc[c].fillna(-1)
        
    for c in ['Region_Code', 'Deductible_Tier', 'Acquisition_Channel']:
        if c in df_proc: df_proc[c] = df_proc[c].fillna('Unknown')
        
    cat_cols = ['Region_Code', 'Broker_Agency_Type', 'Deductible_Tier',
                'Acquisition_Channel', 'Payment_Schedule', 'Employment_Status',
                'Policy_Start_Month']
    for col in cat_cols:
        if col in df_proc.columns:
            df_proc[col] = df_proc[col].astype('category')
            
    # Feature Engineering
    df_proc['Total_Dependents'] = df_proc['Adult_Dependents'] + df_proc['Child_Dependents'].replace(-1, 0) + df_proc['Infant_Dependents']
    df_proc['Risk_Score_Proxy'] = df_proc['Years_Without_Claims'] - df_proc['Previous_Claims_Filed']
    df_proc['Income_per_Dependent'] = df_proc['Estimated_Annual_Income'] / (df_proc['Total_Dependents'] + 1)
    df_proc['Risk_Ratio'] = df_proc['Previous_Claims_Filed'] / (df_proc['Years_Without_Claims'] + 1)
    
    # Drop IDs explicitly
    cols_to_drop = ['Broker_ID', 'Employer_ID']
    df_proc = df_proc.drop(columns=[col for col in cols_to_drop if col in df_proc.columns])
    
    return df_proc

def run_analysis():
    print("Loading data...")
    train_df = pd.read_csv("train.csv")
    df_proc = preprocess_for_analysis(train_df)
    
    X = df_proc.drop(columns=['User_ID', 'Purchased_Coverage_Bundle'])
    y = df_proc['Purchased_Coverage_Bundle']
    
    # Check correlations for numeric features
    numeric_df = X.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr().abs()
    
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find features with correlation greater than 0.9
    to_drop_corr = [column for column in upper.columns if any(upper[column] > 0.9)]
    print(f"\nHighly correlated features to drop (>0.90): {to_drop_corr}")
    
    # Parameters for analysis
    params = {
        'objective': 'multiclass',
        'num_class': 10,
        'metric': 'multi_error',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'max_depth': 6,
        'num_leaves': 31,
        'n_estimators': 100,
        'n_jobs': 1,
        'class_weight': 'balanced',
        'random_state': 42,
        'verbose': -1
    }
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = pd.Series(0, index=X.index)
    
    feature_importances = np.zeros(X.shape[1])
    
    print("\nTraining models to get OOF predictions...")
    for train_idx, val_idx in skf.split(X, y):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(10, verbose=False)])
        
        oof_preds.iloc[val_idx] = model.predict(X_val)
        feature_importances += model.feature_importances_ / skf.n_splits
        
    print("\n--- Features Importance (Gain) ---")
    fi_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
    fi_df = fi_df.sort_values(by='Importance', ascending=False)
    print(fi_df.to_string(index=False))
    
    to_drop_imp = fi_df[fi_df['Importance'] < 5]['Feature'].tolist()
    print(f"\nWeak features to drop (Importance < 5): {to_drop_imp}")
    
    print("\n--- Classification Report ---")
    report = classification_report(y, oof_preds, output_dict=True)
    print(classification_report(y, oof_preds))
    
    # Calculate custom class weights
    # If a class has low f1, we give it higher weight
    class_weights = {}
    f1_scores = []
    for cls in range(10):
        f1 = report[str(cls)]['f1-score']
        f1_scores.append(f1)
        # Weight inversely proportional to F1, smoothed
        class_weights[cls] = 1.0 / (f1 + 0.1)
        
    # Normalize weights so they average to 1
    avg_weight = sum(class_weights.values()) / len(class_weights)
    for cls in class_weights:
        class_weights[cls] = class_weights[cls] / avg_weight
        
    print(f"\nCalculated Custom Class Weights:\n{class_weights}")

if __name__ == "__main__":
    run_analysis()
