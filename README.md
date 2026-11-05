# Task 1 — Data Cleaning & Preprocessing

## Objective
Clean and prepare the Titanic passenger dataset for machine-learning use with Python, Pandas, NumPy, Matplotlib, Seaborn and scikit-learn.

## Dataset
The project downloads the public Titanic CSV from:
https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv

On first run, the CSV is saved to `data/titanic.csv`. An internet connection is required for this first download. You can also place the CSV there manually.

## Steps performed
1. Load the dataset and inspect its shape, first rows, data types and missing values.
2. Remove identifier/text columns (`PassengerId`, `Name`, `Ticket`) and `Cabin`, which has substantial missing data.
3. Fill missing `Age` and `Fare` values with their medians; fill missing `Embarked` values with its mode.
4. Plot boxplots for `Age` and `Fare` to inspect outliers.
5. Apply the 1.5×IQR rule to remove rows with outlier values in `Age` and `Fare`.
6. Convert `Sex` and `Embarked` categorical columns into numerical one-hot encoded columns.
7. Standardize continuous numerical features using `StandardScaler`.
8. Save the processed dataset and boxplots in the `outputs/` folder.

## Run the project
```bash
pip install -r requirements.txt
python data_preprocessing.py
```

## Output
- `outputs/titanic_cleaned.csv` — cleaned and preprocessed dataset.
- `outputs/age_boxplot_before.png` and `outputs/fare_boxplot_before.png` — outlier visualizations.

## Notes
- The target column `Survived` is retained and is not scaled.
- `Pclass` is retained as an ordinal categorical feature and is not standardized.
- This is a learning exercise. In a production ML workflow, fit imputers, encoders and scalers on the training split only, then apply the fitted transformations to validation/test data to avoid data leakage.
