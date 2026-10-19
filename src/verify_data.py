import pandas as pd
from pathlib import Path

# Locate the project folder
project_folder = Path(__file__).resolve().parent.parent

# Load prepared datasets
train = pd.read_csv(
    project_folder / "data" / "processed" / "train.csv"
)

test = pd.read_csv(
    project_folder / "data" / "processed" / "test.csv"
)

print("DATA VERIFICATION")
print("-----------------")

# Check dataset sizes
print("\nTraining shape:", train.shape)
print("Testing shape:", test.shape)

# Check missing values
print("\nTraining missing values:")
print(train.isnull().sum().sum())

print("\nTesting missing values:")
print(test.isnull().sum().sum())

# Check matching columns
print("\nColumns match:")
print(train.columns.equals(test.columns))

# Check numeric data
print("\nAll columns numeric:")
print(
    all(pd.api.types.is_numeric_dtype(dtype)
        for dtype in train.dtypes)
)

# Check target distribution
print("\nTraining churn rate:")
print(train["Churn"].mean().round(4))

print("\nTesting churn rate:")
print(test["Churn"].mean().round(4))

# Verify essential conditions
assert len(train) + len(test) == 7032
assert train.isnull().sum().sum() == 0
assert test.isnull().sum().sum() == 0
assert train.columns.equals(test.columns)
assert set(train["Churn"].unique()).issubset({0, 1})
assert set(test["Churn"].unique()).issubset({0, 1})

print("\nAll basic verification checks passed!")