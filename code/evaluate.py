# Name: Raghini H | RollNo: 727823TUAM037

import pandas as pd
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

print("Name : Raghini H | RollNo: 727823TUAM037 | Timestamp:", datetime.now())

# Load model & test data
model = joblib.load("outputs/model.pkl")
X_test = pd.read_csv("outputs/X_test.csv")
y_test = pd.read_csv("outputs/y_test.csv")

# Predictions
y_pred = model.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("Evaluation Results:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)