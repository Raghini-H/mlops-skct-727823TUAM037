# Name: Raghini H | RollNo: 727823TUAM037

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import os

print("Name : Raghini H | RollNo: 727823TUAM037 | Timestamp:", datetime.now())

data = pd.read_csv(r"C:\College\Semester 6\MLOPS\Assignment\ML\data\secom.data", sep=r'\s+', header=None)
labels = pd.read_csv(r"C:\College\Semester 6\MLOPS\Assignment\ML\data\secom_labels.data", sep=r'\s+', header=None)

data['label'] = labels[0]
data.columns = data.columns.astype(str)

imputer = SimpleImputer(strategy='mean')
data_imputed = imputer.fit_transform(data)

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_imputed)

os.makedirs("outputs", exist_ok=True)
pd.DataFrame(data_scaled).to_csv("outputs/processed_data.csv", index=False)

print("Data preprocessing completed.")