# Equipment Fault Detection using SECOM Dataset

# 👩‍🎓 Student Details
- **Name:** Raghini H  
- **Roll Number:** 727823TUAM037  

---

# 📊 Dataset
- **Dataset Name:** SECOM Manufacturing Dataset  
- **Description:**  
  The SECOM dataset contains sensor data collected from a semiconductor manufacturing process.  
  It consists of **1567 samples** and **590+ features**, with labels indicating whether a fault has occurred.

- **Class Distribution:**
  - Fault (1): 1463  
  - Normal (0): 104  

- **Challenges:**
  - High dimensionality  
  - Missing values  
  - Severe class imbalance  

# 🚀 Setup Steps

# 1️⃣ Create Virtual Environment
python -m venv mlflow_env


# 2️⃣ Activate Environment
Windows:
mlflow_env\Scripts\activate

Linux/Mac:
source mlflow_env/bin/activate


# 3️⃣ Install Dependencies
pip install -r requirements.txt


# 4️⃣ Place Dataset
Ensure dataset is inside:
data/
 ├── secom.data
 ├── secom_labels.data

 
# 5️⃣ Run Pipeline (Step by Step)
python data_prep.py
python train_pipeline.py
python evaluate.py

# ⚠️ Notes
Dataset is highly imbalanced, which affects model performance
Some models may predict only one class


# 🧠 Models Used
Logistic Regression

Random Forest

Decision Tree

Gradient Boosting

AdaBoost

KNN

Extra Trees

Naive Bayes

Isolation Forest

One-Class SVM

Local Outlier Factor

# ✅ Best Model
One-Class SVM performed best for anomaly detection

# 📦 requirements.txt (Pinned Versions)

pandas==2.2.2

numpy==1.26.4

scikit-learn==1.4.2

matplotlib==3.8.4

joblib==1.4.2

mlflow==2.12.1

scipy==1.13.0

threadpoolctl==3.5.0

