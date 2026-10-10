"""
Week 1 Project: Data Acquisition, Cleaning, and Exploratory Data Analysis (EDA)
Dataset: Iris measurements (CSV included in this project)

Run:
    pip install pandas numpy matplotlib
    python analysis.py

The script reads iris_dataset.csv from the same folder, checks data quality,
cleans missing/duplicate records if present, prints summary statistics, and
creates plots in the charts/ folder.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "iris_dataset.csv"
CHARTS_DIR = BASE_DIR / "charts"
CHARTS_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)
print("First five rows:")
print(df.head())
print("\nShape before cleaning:", df.shape)
print("\nData types:")
print(df.dtypes)
print("\nMissing values by column:")
print(df.isna().sum())

# Clean missing values: numeric columns are filled with their column median;
# categorical columns are filled with their mode.
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())
for col in categorical_cols:
    if df[col].isna().any():
        df[col] = df[col].fillna(df[col].mode().iloc[0])

# Remove duplicate rows
before_duplicates = len(df)
df = df.drop_duplicates()
removed_duplicates = before_duplicates - len(df)

print("\nDuplicate rows removed:", removed_duplicates)
print("Shape after cleaning:", df.shape)
print("\nDescriptive statistics:")
print(df.describe(include="all"))

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
if "species" in df.columns:
    print("\nSpecies counts:")
    print(df["species"].value_counts())

# Visualizations
if "species" in df.columns:
    df["species"].value_counts().sort_index().plot(kind="bar")
    plt.title("Number of Samples by Species")
    plt.xlabel("Species")
    plt.ylabel("Number of samples")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "species_counts.png", dpi=160)
    plt.close()

df[numeric_cols].hist(figsize=(9, 7), bins=15, edgecolor="black")
plt.suptitle("Distributions of Numeric Features", y=1.02)
plt.tight_layout()
plt.savefig(CHARTS_DIR / "feature_distributions.png", dpi=160, bbox_inches="tight")
plt.close()

if "species" in df.columns and "petal_length_cm" in df.columns:
    df.boxplot(column="petal_length_cm", by="species")
    plt.title("Petal Length by Species")
    plt.suptitle("")
    plt.xlabel("Species")
    plt.ylabel("Petal length (cm)")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "petal_length_by_species.png", dpi=160)
    plt.close()

corr = df[numeric_cols].corr()
plt.figure(figsize=(6.5, 5))
plt.imshow(corr, aspect="auto")
plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=30, ha="right")
plt.yticks(range(len(numeric_cols)), numeric_cols)
plt.colorbar(label="Correlation")
plt.title("Correlation Between Numeric Features")
for i in range(len(numeric_cols)):
    for j in range(len(numeric_cols)):
        plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center")
plt.tight_layout()
plt.savefig(CHARTS_DIR / "correlation_heatmap.png", dpi=160)
plt.close()

# Save cleaned copy and a correlation table for reproducibility
df.to_csv(BASE_DIR / "iris_cleaned.csv", index=False)
corr.round(3).to_csv(BASE_DIR / "correlation_matrix.csv")
print("\nSaved cleaned data and plots to:", BASE_DIR)
