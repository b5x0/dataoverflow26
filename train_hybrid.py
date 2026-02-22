import pandas as pd
import lightgbm as lgb
import joblib
from solution import preprocess
import os

def train_hybrid_sniper():
    print("Loading data...")
    train_df = pd.read_csv("train.csv")
    
    print("Preprocessing using Hybrid Sniper rules...")
    df_proc = preprocess(train_df)
    
    X = df_proc.drop(columns=['User_ID', 'Purchased_Coverage_Bundle'])
    y = df_proc['Purchased_Coverage_Bundle']
    
    # Custom class weights
    weights = {0: 1.25, 1: 0.74, 2: 0.57, 3: 0.99, 4: 0.77, 5: 0.67, 6: 1.02, 7: 0.75, 8: 2.35, 9: 0.89}
    
    print("Training Hybrid Sniper (Medium Weight LightGBM)...")
    
    model = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=10,
        boosting_type='gbdt',
        learning_rate=0.05,
        max_depth=8,
        num_leaves=50,
        min_child_samples=40,
        colsample_bytree=0.75,
        n_estimators=150,
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
    train_hybrid_sniper()
