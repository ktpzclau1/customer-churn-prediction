import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
import joblib

# Find project folder
project_folder = Path(__file__).resolve().parent.parent

print("CUSTOMER CHURN - MODEL TRAINING")
print("--------------------------------")

# Load training and testing datasets
train = pd.read_csv(
    project_folder / "data" / "processed" / "train.csv"
)

test = pd.read_csv(
    project_folder / "data" / "processed" / "test.csv"
)

print("\nTraining dataset:", train.shape)
print("Testing dataset:", test.shape)

# Separate features and targets
X_train = train.drop(columns=["Churn"])
y_train = train["Churn"].astype(int)

X_test = test.drop(columns=["Churn"])
y_test = test["Churn"].astype(int)

print("\nTraining features:", X_train.shape)
print("Testing features:", X_test.shape)

print("\nTraining churn distribution:")
print(y_train.value_counts())

# Create our machine learning pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

# Train the model using training data
model.fit(X_train, y_train)

print("\nModel training completed successfully!")

# Predict churn for test customers
y_pred = model.predict(X_test)

# Predict probability of churn
y_prob = model.predict_proba(X_test)[:, 1]

print("\nFirst 10 predictions:")
print(y_pred[:10])

print("\nFirst 10 churn probabilities:")
print(y_prob[:10].round(3))

# Calculate model accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nMODEL PERFORMANCE")
print("-----------------")
print(f"Accuracy: {accuracy:.2%}")

# Baseline model: always predicts the majority class
baseline = DummyClassifier(strategy="most_frequent")

baseline.fit(X_train, y_train)

baseline_pred = baseline.predict(X_test)

baseline_accuracy = accuracy_score(
    y_test,
    baseline_pred
)

print(f"\nBaseline accuracy: {baseline_accuracy:.2%}")
print(f"Logistic Regression accuracy: {accuracy:.2%}")
print(f"Improvement: {accuracy - baseline_accuracy:.2%}")

# Create models folder
models_folder = project_folder / "models"
models_folder.mkdir(exist_ok=True)

# Save trained model pipeline
model_path = models_folder / "churn_model.joblib"

joblib.dump(model, model_path)

print("\nModel saved successfully!")
print("Saved to:", model_path)
