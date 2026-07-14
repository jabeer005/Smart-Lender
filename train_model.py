# ===============================
# SMART LENDER - TRAIN MODEL
# ===============================

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier

# -------------------------------
# Load Dataset
# -------------------------------
data = pd.read_csv("loan_prediction.csv")
print(data.columns)
print(data.head())

print("Dataset Loaded Successfully")
print(data.head())

# -------------------------------
# Remove Loan_ID column
# -------------------------------
data = data.drop("Loan_ID", axis=1)

# -------------------------------
# Encode Text Columns
# -------------------------------
# -------------------------------
# Encode Text Columns
# -------------------------------

from sklearn.preprocessing import LabelEncoder

encoders = {}

columns = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
    "Loan_Status"
]

for col in columns:
    encoder = LabelEncoder()
    data[col] = encoder.fit_transform(data[col])
    encoders[col] = encoder

# -------------------------------
# Input and Output
# -------------------------------
X = data.drop("Loan_Status", axis=1)
y = data["Loan_Status"]

# -------------------------------
# Split Dataset
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# -------------------------------
# Models
# -------------------------------
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "XGBoost": XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss"
    )
}

best_model = None
best_accuracy = 0

print("\nModel Accuracies\n")

for name, model in models.items():

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    accuracy = accuracy_score(y_test, prediction)

    print(name, ":", round(accuracy * 100, 2), "%")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

# -------------------------------
# Save Best Model
# -------------------------------
joblib.dump(best_model, "model.pkl")
joblib.dump(encoders, "encoders.pkl")
print("\nBest Model Saved Successfully")
print("Accuracy :", round(best_accuracy * 100, 2), "%")