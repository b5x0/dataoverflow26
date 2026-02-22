import pandas as pd
import lightgbm as lgb
import joblib
from solution import preprocess
import os

def train_wide_sniper():
    print("Loading data...")
    train_df = pd.read_csv("train.csv")
    
    print("Preprocessing using Wide-Leaf Sniper rules...")
    df_proc = preprocess(train_df)
    
    X = df_proc.drop(columns=['User_ID', 'Purchased_Coverage_Bundle'])
    y = df_proc['Purchased_Coverage_Bundle']
    
    # Custom class weights
    weights = {0: 1.25, 1: 0.74, 2: 0.57, 3: 0.99, 4: 0.77, 5: 0.67, 6: 1.02, 7: 0.75, 8: 2.35, 9: 0.89}
    
    print("Training Wide-Leaf Sniper (LightGBM)...")
    
    model = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=10,
        boosting_type='gbdt',
        learning_rate=0.15,
        max_depth=-1, # No explicit limit ensures pure num_leaves constraint
        num_leaves=150,
        min_child_samples=120, # Also known as min_data_in_leaf
        n_estimators=72,
        n_jobs=1,              # STRICT single core requirement
        class_weight=weights,  
        random_state=42,
        verbose=-1
    )
    
    model.fit(X, y)
    
    print("Saving model to model.pkl...")
    joblib.dump(model, 'model.pkl')
    print(f"Model saved. Size: {os.path.getsize('model.pkl') / (1024*1024):.2f} MB")

if __name__ == "__main__":
    train_wide_sniper()
