import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Find the main project folder
project_folder = Path(__file__).resolve().parent.parent

# Load the customer dataset
file_path = project_folder / "data" / "telco_churn.csv"
df = pd.read_csv(file_path)

# Create the images folder if needed
images_folder = project_folder / "images"
images_folder.mkdir(exist_ok=True)

# Colors for our charts
BURGUNDY = "#800020"
PURPLE = "#583D72"
LIGHT_PURPLE = "#B8A2C8"

# Set chart appearance
plt.rcParams["figure.figsize"] = (9, 5)
plt.rcParams["font.size"] = 11
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

print("Customer data loaded successfully!")
print("Total customers:", len(df))

# CHART 1: CUSTOMER CHURN COUNT

churn_counts = df["Churn"].value_counts()
churn_counts = churn_counts.reindex(["No", "Yes"])

fig, ax = plt.subplots()

bars = ax.bar(
    ["Stayed", "Left"],
    churn_counts.values,
    color=[PURPLE, BURGUNDY],
    width=0.55
)

ax.set_title(
    "Customer Churn Distribution",
    fontsize=16,
    fontweight="bold",
    pad=15
)

ax.set_ylabel("Number of Customers")
ax.bar_label(bars, padding=5)

fig.tight_layout()

fig.savefig(
    images_folder / "01_customer_churn.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print("Chart 1 saved!")

# CHART 2: CHURN RATE BY CONTRACT TYPE

contract_churn = (
    df.groupby("Contract")["Churn"]
    .apply(lambda x: (x == "Yes").mean() * 100)
)

contract_order = [
    "Month-to-month",
    "One year",
    "Two year"
]

contract_churn = contract_churn.reindex(contract_order)

fig, ax = plt.subplots()

bars = ax.bar(
    contract_churn.index,
    contract_churn.values,
    color=[BURGUNDY, PURPLE, LIGHT_PURPLE],
    width=0.6
)

ax.set_title(
    "Customer Churn Rate by Contract Type",
    fontsize=16,
    fontweight="bold",
    pad=15
)

ax.set_ylabel("Churn Rate (%)")
ax.set_ylim(0, 55)

ax.bar_label(
    bars,
    fmt="%.1f%%",
    padding=5
)

fig.tight_layout()

fig.savefig(
    images_folder / "02_contract_churn.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print("Chart 2 saved!")

# CHART 3: AVERAGE MONTHLY CHARGES

monthly_charges = (
    df.groupby("Churn")["MonthlyCharges"]
    .mean()
    .reindex(["No", "Yes"])
)

fig, ax = plt.subplots()

bars = ax.bar(
    ["Stayed", "Left"],
    monthly_charges.values,
    color=[PURPLE, BURGUNDY],
    width=0.55
)

ax.set_title(
    "Average Monthly Charges by Churn",
    fontsize=16,
    fontweight="bold",
    pad=15
)

ax.set_ylabel("Average Monthly Charges ($)")
ax.bar_label(
    bars,
    fmt="$%.2f",
    padding=5
)

ax.set_ylim(0, monthly_charges.max() * 1.2)

fig.tight_layout()

fig.savefig(
    images_folder / "03_monthly_charges.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print("Chart 3 saved!")

# CHART 4: AVERAGE CUSTOMER TENURE

average_tenure = (
    df.groupby("Churn")["tenure"]
    .mean()
    .reindex(["No", "Yes"])
)

fig, ax = plt.subplots()

bars = ax.bar(
    ["Stayed", "Left"],
    average_tenure.values,
    color=[PURPLE, BURGUNDY],
    width=0.55
)

ax.set_title(
    "Average Customer Tenure by Churn",
    fontsize=16,
    fontweight="bold",
    pad=15
)

ax.set_ylabel("Average Tenure (Months)")
ax.bar_label(
    bars,
    fmt="%.1f",
    padding=5
)

ax.set_ylim(0, average_tenure.max() * 1.2)

fig.tight_layout()

fig.savefig(
    images_folder / "04_customer_tenure.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print("Chart 4 saved!")
print("All visualizations completed successfully!")
