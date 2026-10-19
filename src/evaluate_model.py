import pandas as pd
import matplotlib.pyplot as plt
import joblib

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_curve,
    roc_auc_score
)

# Find the project folder
project_folder = Path(__file__).resolve().parent.parent

# Load the test dataset
test = pd.read_csv(
    project_folder / "data" / "processed" / "test.csv"
)

# Separate features and target
X_test = test.drop(columns=["Churn"])
y_test = test["Churn"].astype(int)

# Load the saved trained model
model = joblib.load(
    project_folder / "models" / "churn_model.joblib"
)

# Create output folders
images_folder = project_folder / "images"
results_folder = project_folder / "results"

images_folder.mkdir(exist_ok=True)
results_folder.mkdir(exist_ok=True)

# Portfolio colors
BURGUNDY = "#800020"
PURPLE = "#583D72"

print("CUSTOMER CHURN - MODEL EVALUATION")
print("---------------------------------")
print("Test customers:", len(X_test))

# Predict whether customers leave
y_pred = model.predict(X_test)

# Predict churn probabilities
y_prob = model.predict_proba(X_test)[:, 1]

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("\nMODEL PERFORMANCE")
print("-----------------")

print(f"Accuracy:  {accuracy:.2%}")
print(f"Precision: {precision:.2%}")
print(f"Recall:    {recall:.2%}")
print(f"F1-score:  {f1:.2%}")
print(f"ROC-AUC:   {roc_auc:.3f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Stayed", "Left"]
    )
)

# CHART 1: CONFUSION MATRIX

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Stayed", "Left"]
)

fig, ax = plt.subplots(figsize=(7, 6))

display.plot(
    ax=ax,
    cmap="Purples",
    colorbar=False,
    values_format="d"
)

ax.set_title(
    "Customer Churn - Confusion Matrix",
    fontsize=15,
    fontweight="bold",
    pad=15
)

fig.tight_layout()

fig.savefig(
    images_folder / "05_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print("\nConfusion matrix saved!")

# CHART 2: MODEL PERFORMANCE METRICS

metrics = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1-score": f1
}

fig, ax = plt.subplots(figsize=(9, 5))

bars = ax.bar(
    metrics.keys(),
    metrics.values(),
    color=[PURPLE, BURGUNDY, PURPLE, BURGUNDY],
    width=0.6
)

ax.set_title(
    "Customer Churn - Model Performance",
    fontsize=15,
    fontweight="bold",
    pad=15
)

ax.set_ylabel("Score")
ax.set_ylim(0, 1.1)

ax.bar_label(
    bars,
    fmt="%.2f",
    padding=5
)

fig.tight_layout()

fig.savefig(
    images_folder / "06_model_metrics.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print("Model metrics chart saved!")

# Save evaluation metrics to CSV
metrics_df = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "ROC-AUC"
    ],
    "Score": [
        accuracy,
        precision,
        recall,
        f1,
        roc_auc
    ]
})

metrics_df.to_csv(
    results_folder / "model_metrics.csv",
    index=False
)

# Save the classification report
report = classification_report(
    y_test,
    y_pred,
    target_names=["Stayed", "Left"]
)

with open(
    results_folder / "classification_report.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write(report)

print("\nEvaluation files saved successfully!")
print("Lesson 6 complete!")

# CHART 3: ROC CURVE

from sklearn.metrics import roc_curve, roc_auc_score

# Calculate ROC curve values
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# Calculate ROC-AUC score
roc_auc = roc_auc_score(y_test, y_prob)

# Create the chart
fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(
    fpr,
    tpr,
    color="#800020",
    linewidth=2.5,
    label=f"Logistic Regression (AUC = {roc_auc:.3f})"
)

# Add reference line
ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="gray",
    label="Random classifier"
)

ax.set_title("Customer Churn - ROC Curve")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend(loc="lower right")
ax.grid(alpha=0.2)

fig.tight_layout()

# Save the image
fig.savefig(
    images_folder / "07_roc_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print("ROC curve saved successfully!")
