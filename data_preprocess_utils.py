
import pandas as pd
import numpy as np
from scipy.stats import zscore

def preprocess_employee_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # === 1. Basic Cleaning & Type Casting ===
    df['employee_id'] = df['employee_id'].astype(str)
    df['salary'] = pd.to_numeric(df['salary'], errors='coerce')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['tenure_years'] = pd.to_numeric(df['tenure_years'], errors='coerce')
    df['has_dependents'] = df['has_dependents'].astype(bool).astype(int)
    df['enrolled'] = df['enrolled'].astype(int)
    df['employment_type'] = df['employment_type'].astype(str)
    df['region'] = df['region'].astype(str)

    # === 2. Age Bucketing ===
    age_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, np.inf]
    age_labels = list(range(1, len(age_bins)))
    df['age_bucketed'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False)
    df['age_bucketed'] = df['age_bucketed'].astype('category')

    # === 3. Binary Columns ===
    df['is_fulltime'] = (df['employment_type'].str.lower() == 'full-time').astype(int)

    # === 4. Engineered Features ===
    df['salary_per_year_of_age'] = df['salary'] / df['age'].replace(0, np.nan)
    df['salary_per_year_of_age']=df['salary_per_year_of_age'].fillna(0)
    df['salary_per_tenure'] = df['salary'] / df['tenure_years'].replace(0, np.nan)
    df['salary_per_tenure']=df['salary_per_tenure'].fillna(0)
    df['age_tenure_ratio'] = df['age'] / df['tenure_years'].replace(0, np.nan)
    df['age_tenure_ratio']=df['age_tenure_ratio'].fillna(0)
    df['salary_dependents_ratio'] = df['salary'] / (1 + df['has_dependents'])

    df['is_senior_fulltime'] = ((df['age'] > 45) & (df['is_fulltime'] == 1)).astype(int)
    df['is_young_with_dependents'] = ((df['age'] < 30) & (df['has_dependents'] == 1)).astype(int)
    df['has_high_salary_and_long_tenure'] = (
        (df['salary'] > df['salary'].median()) &
        (df['tenure_years'] > df['tenure_years'].median())
    ).astype(int)

    df['log_salary'] = np.log1p(df['salary'])
    df['age_squared'] = df['age'] ** 2

    # === 5. Group-Based Features ===
    df['salary_zscore_by_region'] = df.groupby('region')['salary'].transform(lambda x: zscore(x, nan_policy='omit'))
    df['tenure_percentile_in_region'] = df.groupby('region')['tenure_years'].transform(
        lambda x: x.rank(pct=True)
    )
    df['is_top_salary_percentile'] = (df['salary'] >= df['salary'].quantile(0.9)).astype(int)

    # === 6. Final Columns ===
    final_columns = [
        'employee_id', 'age', 'age_bucketed', 'salary', 'has_dependents', 'is_fulltime',

        # Engineered
        'salary_per_year_of_age', 'salary_per_tenure', 'age_tenure_ratio',
        'salary_dependents_ratio', 'is_senior_fulltime', 'is_young_with_dependents',
        'has_high_salary_and_long_tenure', 'log_salary',

        # Group-based
        'salary_zscore_by_region', 'tenure_percentile_in_region', 'is_top_salary_percentile',
        #target variable 
        'enrolled'
    ]
    df = df[final_columns]

    return df