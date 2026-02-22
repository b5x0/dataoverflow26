import pandas as pd
import lightgbm as lgb
import joblib
from solution import preprocess
import os

def train_v2_sniper():
    print("Loading data...")
    train_df = pd.read_csv("train.csv")
    
    print("Preprocessing using final rules...")
    df_proc = preprocess(train_df)
    
    X = df_proc.drop(columns=['User_ID', 'Purchased_Coverage_Bundle'])
    y = df_proc['Purchased_Coverage_Bundle']
    
    # Custom class weights from Error Analysis
    weights = {0: 1.25, 1: 0.74, 2: 0.57, 3: 0.99, 4: 0.77, 5: 0.67, 6: 1.02, 7: 0.75, 8: 2.35, 9: 0.89}
    
    print("Training V2 Sniper (LightGBM)...")
    # Using constraints from Checkpoint 4 + increased min_child_samples for generalization
    model = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=10,
        boosting_type='gbdt',
        learning_rate=0.08,
        max_depth=6,
        num_leaves=31,
        min_child_samples=120, # Increased from ~70 to prevent overfitting
        colsample_bytree=0.7,
        n_estimators=100,      # Capped to 100 for fast latency
        n_jobs=1,              # STRICT single core requirement
        class_weight=weights,  # Apply our custom weights
        random_state=42,
        verbose=-1
    )
    
    model.fit(X, y)
    
    print("Saving model to model.pkl...")
    joblib.dump(model, 'model.pkl')
    print(f"Model saved. Size: {os.path.getsize('model.pkl') / (1024*1024):.2f} MB")

if __name__ == "__main__":
    train_v2_sniper()
