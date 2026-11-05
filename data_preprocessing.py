"""
Task 1: Data Cleaning & Preprocessing
Dataset: Titanic passenger dataset (DataScienceDojo public CSV)
Run: python data_preprocessing.py
"""
from pathlib import Path
import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

def load_data():
    local_file = DATA_DIR / "titanic.csv"
    if not local_file.exists():
        print("Downloading Titanic dataset...")
        urllib.request.urlretrieve(DATA_URL, local_file)
    return pd.read_csv(local_file)

def main():
    df = load_data()
    print("\\n--- Initial dataset ---")
    print("Shape:", df.shape)
    print(df.head())
    print("\\n--- Data types ---")
    print(df.dtypes)
    print("\\n--- Missing values ---")
    print(df.isnull().sum())

    # Keep useful columns and remove identifiers / mostly-empty Cabin.
    df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], errors="ignore")

    # Missing-value treatment: median for numeric columns, mode for categorical.
    for col in ["Age", "Fare"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    for col in ["Embarked"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    # Outlier visualisation before treatment.
    numeric_for_outliers = [c for c in ["Age", "Fare"] if c in df.columns]
    for col in numeric_for_outliers:
        plt.figure(figsize=(6, 3))
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot of {col} (before outlier removal)")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{col.lower()}_boxplot_before.png")
        plt.close()

    # IQR filtering: remove rows outside 1.5*IQR for Age and Fare.
    before = len(df)
    for col in numeric_for_outliers:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        df = df[df[col].between(lower, upper)]
    print(f"\\nRows removed as outliers: {before - len(df)}")

    # Encode categorical features using one-hot encoding.
    categorical = [c for c in ["Sex", "Embarked"] if c in df.columns]
    df = pd.get_dummies(df, columns=categorical, drop_first=True, dtype=int)

    # Standardize continuous numerical features (excluding target Survived and Pclass).
    scale_cols = [c for c in ["Age", "Fare", "SibSp", "Parch"] if c in df.columns]
    scaler = StandardScaler()
    if scale_cols:
        df[scale_cols] = scaler.fit_transform(df[scale_cols])

    # Save cleaned data and final diagnostics.
    df.to_csv(OUTPUT_DIR / "titanic_cleaned.csv", index=False)
    print("\\n--- Final dataset ---")
    print("Shape:", df.shape)
    print(df.head())
    print("\\nMissing values after preprocessing:")
    print(df.isnull().sum())
    print("\\nCleaned dataset saved to outputs/titanic_cleaned.csv")

if __name__ == "__main__":
    main()
