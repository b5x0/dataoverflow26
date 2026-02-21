import joblib
import pandas as pd
import os

print("Starting native model consolidation...")
try:
    lgbm_model = joblib.load('model_opt.pkl')
    te = joblib.load('target_encoder.pkl')
    
    encoding_dict = {}
    global_mean = 0.5515
    
    # te.mapping appears to be a dict: {'col_name': Series, ...}
    for col, mapping_series in te.mapping.items():
        # Convert the Series mapping to a dict
        col_mapping = mapping_series.to_dict()
        col_mapping['__global_mean__'] = global_mean
        encoding_dict[col] = col_mapping
        print(f"Mapped {col} with {len(col_mapping)} categories.")

    # Save consolidated artifacts in a native dictionary
    output_dict = {
        'model': lgbm_model,
        'encoding_dict': encoding_dict
    }
    joblib.dump(output_dict, 'model.pkl')
    print(f"Success! model.pkl created ({os.path.getsize('model.pkl')/1024/1024:.2f} MB)")

except Exception as e:
    print(f"Error during consolidation: {str(e)}")
