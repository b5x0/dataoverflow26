import os

# Ultra-strict single thread environment variables BEFORE any data libraries are loaded
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import pandas as pd
import time
import sys
from solution import preprocess, load_model, predict

def mock_inference_test():
    print("Testing 1-Core simulation inference...")
    try:
        # Load the entire test file
        test_df = pd.read_csv("test.csv")
    except FileNotFoundError:
        print("test.csv not found")
        sys.exit(1)

    print(f"Data Loaded. Shape: {test_df.shape}")
    
    print("Preprocessing...")
    proc_df = preprocess(test_df)
    
    print("Loading model...")
    model = load_model()
    
    print("Running predict()")
    start = time.perf_counter()
    preds = predict(proc_df, model)
    duration = time.perf_counter() - start
    
    print(f"Predictions shape: {preds.shape}")
    print(f"Inference duration on {len(test_df)} rows: {duration:.4f} seconds")
    
    if duration < 1.0:
        print("Latency is well under 1 second!")
        sys.exit(0)
    else:
        print("Latency exceeded 1 second.")
        sys.exit(1)

if __name__ == "__main__":
    mock_inference_test()
