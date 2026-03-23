# Name: Raghini H | RollNo: 727823TUAM037

import pandas as pd
import numpy as np
from datetime import datetime
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score

print("Name : Raghini H | RollNo: 727823TUAM037 | Timestamp:", datetime.now())

# -----------------------------
# LOAD DATA
# -----------------------------
data = pd.read_csv(
    r"C:\College\Semester 6\MLOPS\Assignment\ML\data\secom.data",
    sep=r'\s+',
    header=None
)

labels = pd.read_csv(
    r"C:\College\Semester 6\MLOPS\Assignment\ML\data\secom_labels.data",
    sep=r'\s+',
    header=None
)

# -----------------------------
# SPLIT FEATURES & TARGET
# -----------------------------
X = data
y = labels[0]

# Convert labels (IMPORTANT: anomaly → binary classification)
y = y.replace({1: 0, -1: 1})

print("\nClass distribution:")
print(y.value_counts())

# Fix column names (IMPORTANT)
X.columns = X.columns.astype(str)

# -----------------------------
# PREPROCESSING (ONLY X)
# -----------------------------
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

scaler = StandardScaler()
X = scaler.fit_transform(X)

# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# MODEL TRAINING
# -----------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(X_train, y_train)

# -----------------------------
# PREDICTION & EVALUATION
# -----------------------------
y_pred = model.predict(X_test)

f1 = f1_score(y_test, y_pred)

print("\nF1 Score:", round(f1, 4))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# -----------------------------
# SAVE OUTPUTS
# -----------------------------
os.makedirs("outputs", exist_ok=True)

# Save model
joblib.dump(model, "outputs/model.pkl")

# Save test data
pd.DataFrame(X_test).to_csv("outputs/X_test.csv", index=False)
pd.DataFrame(y_test).to_csv("outputs/y_test.csv", index=False)

print("\n✅ Model training completed successfully!")