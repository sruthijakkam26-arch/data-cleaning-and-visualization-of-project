# data-cleaning-and-visualization-of-project
Data cleaning and visualization using Python, Pandas, Matplotlib, and Seaborn.
# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load Dataset
df = pd.read_csv("data.csv")

print("Original Dataset:")
print(df.head())

# -------------------------
# Data Cleaning
# -------------------------

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Fill numerical missing values with mean
for col in df.select_dtypes(include=np.number).columns:
    df[col].fillna(df[col].mean(), inplace=True)

# Remove duplicate rows
df.drop_duplicates(inplace=True)

print("\nDataset Shape After Cleaning:")
print(df.shape)

# -------------------------
# Outlier Detection
# -------------------------

numeric_cols = df.select_dtypes(include=np.number).columns

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df = df[(df[col] >= lower) & (df[col] <= upper)]

print("\nDataset Shape After Removing Outliers:")
print(df.shape)

# -------------------------
# Visualization
# -------------------------

# Histogram
df[numeric_cols[0]].hist()
plt.title("Histogram")
plt.xlabel(numeric_cols[0])
plt.ylabel("Frequency")
plt.show()

# Box Plot
plt.boxplot(df[numeric_cols[0]])
plt.title("Box Plot")
plt.show()

# Correlation Heatmap
import seaborn as sns

plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True),
            annot=True,
            cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Save Cleaned Dataset
df.to_csv("cleaned_data.csv", index=False)

print("\nData Cleaning and Visualization Completed Successfully!")