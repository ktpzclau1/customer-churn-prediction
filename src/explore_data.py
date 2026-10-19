import pandas as pd
from pathlib import Path

# Find the project folder
project_folder = Path(__file__).resolve().parent.parent

# Load our dataset
file_path = project_folder / "data" / "telco_churn.csv"
df = pd.read_csv(file_path)

print("CUSTOMER CHURN DATA EXPLORATION")
print("--------------------------------")

# Display the first 5 customers
print("\nFirst 5 rows:")
print(df.head())

# Display the dataset size
print("\nDataset shape:")
print(df.shape)

# Display column names
print("\nColumn names:")
print(df.columns.tolist())

# Display data types and non-null counts
print("\nDataset information:")
df.info()

# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Check duplicate rows
print("\nDuplicate rows:")
print(df.duplicated().sum())

# Check the TotalCharges column
print("\nTotalCharges data type:")
print(df["TotalCharges"].dtype)

# Find values that cannot be converted to numbers
total_charges = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

print("\nInvalid or missing TotalCharges:")
print(total_charges.isna().sum())

# Count customers who stayed and left
print("\nCustomer churn counts:")
print(df["Churn"].value_counts())

# Calculate percentages
print("\nCustomer churn percentages:")
print(df["Churn"].value_counts(normalize=True) * 100)

# Calculate the overall churn rate
churn_rate = (df["Churn"] == "Yes").mean() * 100

print(f"\nOverall churn rate: {churn_rate:.2f}%")

# Churn rate by contract type
print("\nChurn rate by contract type:")

contract_churn = (
    df.groupby("Contract")["Churn"]
    .apply(lambda x: (x == "Yes").mean() * 100)
)

print(contract_churn.round(2))

# Compare monthly charges
print("\nAverage monthly charges by churn:")

monthly_charges = (
    df.groupby("Churn")["MonthlyCharges"].mean()
)

print(monthly_charges.round(2))

# Compare customer tenure
print("\nAverage tenure by churn:")

average_tenure = (
    df.groupby("Churn")["tenure"].mean()
)

print(average_tenure.round(2))

# Create a folder for analysis results
output_folder = project_folder / "results"
output_folder.mkdir(exist_ok=True)

# Save contract churn rates
contract_churn.to_csv(
    output_folder / "contract_churn_rates.csv",
    header=["ChurnRatePercent"]
)

print("\nAnalysis saved successfully!")