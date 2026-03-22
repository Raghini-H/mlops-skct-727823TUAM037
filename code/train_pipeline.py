# Name: Raghini H | RollNo: 727823TUAM037

import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

print("Name : Raghini H | RollNo: 727823TUAM037 | Timestamp:", datetime.now())

data = pd.read_csv("outputs/processed_data.csv")

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

os.makedirs("outputs", exist_ok=True)
joblib.dump(model, "outputs/model.pkl")

X_test.to_csv("outputs/X_test.csv", index=False)
y_test.to_csv("outputs/y_test.csv", index=False)

print("Model training completed.")