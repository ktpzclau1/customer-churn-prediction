import pandas as pd
from pathlib import Path

# Find the dataset inside our project
project_folder = Path(__file__).resolve().parent.parent
file_path = project_folder / "data" / "telco_churn.csv"

# Load the customer dataset
df = pd.read_csv(file_path)

# Show the first five customers
print("CUSTOMER CHURN PREDICTION")
print("-------------------------")
print(df.head())

# Check the size of the dataset
print("\nNumber of rows and columns:")
print(df.shape)

# Count customers who stayed or left
print("\nCustomer churn counts:")
print(df["Churn"].value_counts())