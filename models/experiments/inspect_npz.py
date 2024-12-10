import numpy as np
import pandas as pd

# Load the .npz file with allow_pickle=True
file_path = "src/models/experiments/synthetic_data_finetuned_2024-12-07_00-22-08_By0/predictions/best_predictions.npz"  # Replace with your file's path
data = np.load(file_path, allow_pickle=True)  # Enable allow_pickle

# Display keys (array names) in the file
print("Keys in .npz file:", data.files)

# Create a dictionary to hold DataFrames for each key
dataframes = {}

# Extract each array and convert to DataFrame
for key in data.files:
    array_data = data[key]
    # Convert to DataFrame
    df = pd.DataFrame(array_data)
    dataframes[key] = df
    # Save each DataFrame to a CSV file
    df.to_csv(f"{key}.csv", index=False)
    print(f"Saved {key} to {key}.csv with shape {df.shape}")

print("All data has been converted to CSV files.")