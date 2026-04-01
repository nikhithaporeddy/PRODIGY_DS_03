import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load dataset (IMPORTANT: sep=";")
data_path = os.path.join("data", "bank.csv")
data = pd.read_csv(data_path, sep=";")

print("📊 Original Columns:")
print(data.columns)

# Encode categorical columns
le = LabelEncoder()
for col in data.select_dtypes(include="object").columns:
    data[col] = le.fit_transform(data[col])

# Save processed file
output_path = os.path.join("data", "bank_processed.csv")
data.to_csv(output_path, index=False)

print("✅ Preprocessing done!")
print("Saved at:", output_path)
print("📊 Processed Columns:")
print(data.columns)