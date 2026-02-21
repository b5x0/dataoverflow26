import zipfile
import os

def build_submission():
    # We no longer wipe requirements.txt so that API and notebook dependencies are preserved.
    print("Using existing requirements.txt")
    
    # Zip up the files
    files_to_zip = ["solution.py", "model.pkl", "requirements.txt"]
    zip_filename = "submission.zip"
    
    for file in files_to_zip:
        if not os.path.exists(file):
            print(f"Error: Required file {file} is missing.")
            return
            
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in files_to_zip:
            zipf.write(file)
            print(f"Added {file} to {zip_filename}")
            
    print(f"\nSuccess! Built {zip_filename} ({os.path.getsize(zip_filename) / (1024 * 1024):.2f} MB)")

if __name__ == "__main__":
    build_submission()
