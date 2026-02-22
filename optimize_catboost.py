import pandas as pd
import catboost as cb
from solution import preprocess

def optimize_catboost():
    print("Loading data...")
    train_df = pd.read_csv("train.csv")
    
    print("Preprocessing...")
    df_proc = preprocess(train_df)
    
    X = df_proc.drop(columns=['User_ID', 'Purchased_Coverage_Bundle'])
    y = df_proc['Purchased_Coverage_Bundle']
    
    cat_cols = ['Region_Code', 'Broker_Agency_Type', 'Deductible_Tier',
                'Acquisition_Channel', 'Payment_Schedule', 'Employment_Status',
                'Policy_Start_Month']
                
    # Filter only columns that still exist after dropping
    cat_features = [col for col in cat_cols if col in X.columns]
    
    # CatBoost parameters for fast single-core inference but high F1
    params = {
        'iterations': 400,
        'learning_rate': 0.05,
        'depth': 6,
        'loss_function': 'MultiClass',
        'eval_metric': 'TotalF1:average=Macro',
        'thread_count': 1, # CRITICAL for 1-core Docker
        'random_seed': 42,
        'verbose': 50,
        'auto_class_weights': 'Balanced', # Handle target imbalance
    }
    
    print("Training CatBoost model...")
    model = cb.CatBoostClassifier(**params)
    model.fit(X, y, cat_features=cat_features)
    
    print("Saving model via joblib to model.pkl...")
    import joblib
    joblib.dump(model, 'model.pkl')
    
    import os
    print(f"Model saved. Size: {os.path.getsize('model.pkl') / (1024*1024):.2f} MB")

if __name__ == "__main__":
    optimize_catboost()
