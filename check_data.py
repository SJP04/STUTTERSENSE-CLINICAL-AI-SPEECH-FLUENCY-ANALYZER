import os
import pandas as pd

# Check if the file exists in the current folder
file_name = os.path.join("sep28k_mfcc", "sep28k-mfcc.csv")

if os.path.exists(file_name):
    print(f"✅ Success! '{file_name}' is in the correct folder.")
    # Load a tiny piece of it to be sure
    data = pd.read_csv(file_name, nrows=5)
    print("Data Preview:")
    print(data.head())
else:
    print(f"❌ Error: '{file_name}' not found. Current directory is: {os.getcwd()}")