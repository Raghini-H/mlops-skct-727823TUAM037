# Name: Raghini H | RollNo: 727823TUAM037

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import os

print("Name : Raghini H | RollNo: 727823TUAM037 | Timestamp:", datetime.now())

# Load dataset
data = pd.read_csv("data/secom.data", sep=" ", header=None)
labels = pd.read_csv("data/secom_labels.data", sep=" ", header=None)

# Merge features + labels
data['label'] = labels[0]

# Replace missing values
imputer = SimpleImputer(strategy='mean')
data_imputed = imputer.fit_transform(data)

# Normalize
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_imputed)

# Save processed data
os.makedirs("outputs", exist_ok=True)
pd.DataFrame(data_scaled).to_csv("outputs/processed_data.csv", index=False)

print("Data preprocessing completed.")