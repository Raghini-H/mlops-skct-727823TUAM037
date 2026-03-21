import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import time
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, IsolationForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.naive_bayes import GaussianNB

# Metrics
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score

# -------------------------
# USER DETAILS (EDIT THIS)
# -------------------------
STUDENT_NAME = "Raghini H"
ROLL_NO = "727823TUAM037"
DATASET_NAME = "SecomManufacturing"

# -------------------------
# LOAD DATA
# -------------------------
data = pd.read_csv(r"C:\College\Semester 6\MLOPS\Assignment\ML\data\secom.data", sep=r'\s+', header=None)
labels = pd.read_csv(r"C:\College\Semester 6\MLOPS\Assignment\ML\data\secom_labels.data", sep=r'\s+', header=None)

X = data
y = labels[0]

# Convert labels: -1 (fault) → 1 (anomaly)
y = y.replace({1: 0, -1: 1})

print("Class distribution:")
print(y.value_counts())

# -------------------------
# PREPROCESSING
# -------------------------
imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# MLflow Setup
# -------------------------
# mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("SKCT_727823TUAM037_SECOM")
# mlflow.set_tracking_uri()

# -------------------------
# METRIC FUNCTION
# -------------------------
def evaluate(y_true, y_pred):
    return {
        "f1_score": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "auc_pr": average_precision_score(y_true, y_pred)
    }

# -------------------------
# 12 DIFFERENT MODELS
# -------------------------
models = [
    ("LogisticRegression", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ("RandomForest", RandomForestClassifier(n_estimators=100)),
    ("DecisionTree", DecisionTreeClassifier()),
    ("GradientBoosting", GradientBoostingClassifier()),
    ("AdaBoost", AdaBoostClassifier()),
    ("KNN", KNeighborsClassifier()),

    ("ExtraTrees", ExtraTreesClassifier()),
    ("NaiveBayes", GaussianNB()),

    # Anomaly Models
    ("IsolationForest", IsolationForest()),
    ("OneClassSVM", OneClassSVM()),
    ("LocalOutlierFactor", LocalOutlierFactor(novelty=True)),

    # 12th model (variation)
    ("RandomForest_200", RandomForestClassifier(n_estimators=200))
]

# -------------------------
# TRAINING LOOP
# -------------------------
for i, (name, model) in enumerate(models):

    with mlflow.start_run(run_name=name):

        # TAGS (REQUIRED)
        mlflow.set_tag("student_name", STUDENT_NAME)
        mlflow.set_tag("roll_number", ROLL_NO)
        mlflow.set_tag("dataset", DATASET_NAME)

        start_time = time.time()

        # TRAIN
        if name in ["IsolationForest", "OneClassSVM", "LocalOutlierFactor"]:
            model.fit(X_train)
        else:
            model.fit(X_train, y_train)

        end_time = time.time()

        # PREDICT
        y_pred = model.predict(X_test)

        # Fix anomaly model outputs (-1 → 1)
        y_pred = np.where(y_pred == -1, 1, 0)

        # METRICS
        metrics = evaluate(y_test, y_pred)

        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        # OPERATIONAL METRICS
        mlflow.log_metric("training_time_seconds", end_time - start_time)
        mlflow.log_metric("random_seed", 42 + i)
        mlflow.log_metric("n_features", X.shape[1])

        # MODEL SIZE
        temp_path = "temp_model.pkl"
        joblib.dump(model, temp_path)
        size_mb = os.path.getsize(temp_path) / (1024 * 1024)
        mlflow.log_metric("model_size_mb", size_mb)

        # SAVE MODEL
        mlflow.sklearn.log_model(model, "model")

        # -------------------------
        # PRINT OUTPUT (FOR YOU)
        # -------------------------
        print("\n==============================")
        print(f"Model: {name}")
        print(f"Student: {STUDENT_NAME}")
        print(f"Roll No: {ROLL_NO}")
        print(f"Dataset: {DATASET_NAME}")
        print("------------------------------")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
        print("==============================\n")