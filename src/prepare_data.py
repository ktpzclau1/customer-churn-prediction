import pandas as pd
from pathlib import Path

# Find the project folder
project_folder = Path(__file__).resolve().parent.parent

# Load the original customer dataset
file_path = project_folder / "data" / "telco_churn.csv"
df = pd.read_csv(file_path)

print("CUSTOMER CHURN - DATA PREPARATION")
print("---------------------------------")

print("\nOriginal dataset shape:")
print(df.shape)

print("\nFirst five rows:")
print(df.head())

# Convert TotalCharges into numeric values
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Count missing values
print("\nMissing values before cleaning:")
print(df.isnull().sum())

# Remove rows with missing TotalCharges
df = df.dropna(subset=["TotalCharges"]).copy()

print("\nDataset shape after cleaning:")
print(df.shape)

print("\nRemaining missing values:")
print(df.isnull().sum().sum())

# Remove customer identifier
df = df.drop(columns=["customerID"])

print("\nColumns after removing customerID:")
print(df.columns.tolist())

# Convert Churn to numeric labels
df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

# Check the results
print("\nChurn target values:")
print(df["Churn"].value_counts())

# Confirm that all target values are valid
assert df["Churn"].notna().all()

# Separate features from target
X = df.drop(columns=["Churn"])
y = df["Churn"]

print("\nFeature dataset shape:")
print(X.shape)

print("\nTarget shape:")
print(y.shape)

from sklearn.model_selection import train_test_split

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining features:", X_train.shape)
print("Testing features:", X_test.shape)

print("Training targets:", y_train.shape)
print("Testing targets:", y_test.shape)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Identify categorical columns
categorical_columns = X_train.select_dtypes(
    include=["object", "str", "category"]
).columns.tolist()

# Keep numeric columns unchanged
numeric_columns = [
    column for column in X_train.columns
    if column not in categorical_columns
]

# Create the preprocessing transformer
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categories",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_columns
        ),
        (
            "numbers",
            "passthrough",
            numeric_columns
        )
    ],
    verbose_feature_names_out=False
)

# Learn categories from training data only
X_train_prepared = preprocessor.fit_transform(X_train)

# Apply the same encoding to test data
X_test_prepared = preprocessor.transform(X_test)

# Get the new feature names
feature_names = preprocessor.get_feature_names_out()

print("\nEncoded training shape:")
print(X_train_prepared.shape)

print("\nEncoded testing shape:")
print(X_test_prepared.shape)

# Convert prepared arrays into DataFrames
train_df = pd.DataFrame(
    X_train_prepared,
    columns=feature_names,
    index=X_train.index
)

test_df = pd.DataFrame(
    X_test_prepared,
    columns=feature_names,
    index=X_test.index
)

# Add the target values
train_df["Churn"] = y_train
test_df["Churn"] = y_test

# Create the processed data folder
processed_folder = project_folder / "data" / "processed"
processed_folder.mkdir(parents=True, exist_ok=True)

# Save prepared datasets
train_df.to_csv(
    processed_folder / "train.csv",
    index=False
)

test_df.to_csv(
    processed_folder / "test.csv",
    index=False
)

print("\nPrepared datasets saved successfully!")
print("Training data:", train_df.shape)
print("Testing data:", test_df.shape)
