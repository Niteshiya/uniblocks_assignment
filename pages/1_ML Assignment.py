import streamlit as st 
import pandas as pd 
from streamlit_utils import *
from data_preprocess_utils import *
from ml_utils import *



st.set_page_config(
    page_title="Uniblox",
    layout="wide",
    initial_sidebar_state="expanded"
)



st.title("Uniblox ML Assignment")

with st.expander("Dataset Introduction: Employee Benefits Enrollment Data",icon="📊",expanded=False):
    st.markdown('''
                
# 📊 Dataset Introduction: *Employee Benefits Enrollment Data*

The `employee_data.csv` file contains simulated records of approximately **10,000 employees**, curated to represent typical information collected during **group benefits enrollment** processes. Each row corresponds to a unique employee, capturing demographic, professional, and enrollment-specific attributes.

## 🔍 Key Features:
- **employee_id**: Unique identifier for each employee.
- **age**: Age of the employee.
- **gender**: Gender of the employee (e.g., Male, Female, Other).
- **marital_status**: Marital status (e.g., Single, Married).
- **salary**: Annual salary in base currency.
- **employment_type**: Employment classification (e.g., Full-Time, Part-Time, Contract).
- **region**: Geographic region or business unit location.
- **has_dependents**: Boolean flag indicating whether the employee has dependents.
- **tenure_years**: Number of years the employee has been with the organization.
- **enrolled** *(Target Variable)*: Indicates whether the employee has enrolled in benefits (1 = Enrolled, 0 = Not Enrolled).

## 🎯 Objective:
This dataset is well-suited for **predictive modeling** and **analytical tasks** such as:
- Understanding drivers of enrollment behavior.
- Identifying segments less likely to enroll.
- Building classification models to predict future enrollment.
- Informing policy or incentive design to improve benefit uptake.

                ''')
    
# Load data
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path) 

df = load_data("data/employee_data.csv")

with st.container(height=620):
    visualize_target_relationship(df)
    
    
st.markdown('''
### 🔍 Feature Relevance to Target (`enrolled`)

| Feature | Evaluation | Action |
|--------|------------|--------|
| `age` | Correlation = 0.27. Higher enrollment for employees aged 30+. Violin plot confirms visible trend. | ✅ **Selected** – Bucketed into ranges: 0–10, 10–20, ..., 80+ and label it ordinally . We also keep original age |
| `gender` | Equal enrollment proportions across categories. | ❌ **Not selected** – No discriminatory power |
| `marital_status` | Similar enrollment across groups. | ❌ **Not selected** |
| `salary` | Correlation = 0.37. Higher salary (>60k) linked with increased enrollment. | ✅ **Selected** Distritution is also normal so no need for normalisation |
| `employment_type` | Full-time employees have >70% enrollment. Contract and part-time similar and lower. | ✅ **Selected** – Binary feature `is_fulltime` created |
| `region` | Similar enrollment across regions. | ❌ **Not selected** |
| `has_dependents` | Stronger enrollment rates for those with dependents. | ✅ **Selected** – Binary encoded |
| `tenure_years` | Correlation = -0.01. No visible pattern in violin plot. | ❌ **Not selected** |

---

#### ✅ Final Base Feature Set for Modeling

| Feature | Notes |
|---------|-------|
| `age` | Continuous variable with positive correlation |
| `age_bucketed` | Derived from `age`, captures non-linear relationship |
| `salary` | Continuous variable with positive correlation |
| `is_fulltime` | Binary indicator from `employment_type` |
| `has_dependents` | Binary indicator; positively associated with enrollment |

---

#### 🧠 Engineered Features 

| Feature Name | Description |
|--------------|-------------|
| `salary_per_year_of_age` | `salary / age` – Normalizes salary by age to capture earning efficiency |
| `salary_per_tenure` | `salary / tenure_years` – Measures compensation growth relative to time at company |
| `age_tenure_ratio` | `age / tenure_years` – Indicates how early someone started in their career |
| `salary_dependents_ratio` | `salary / (1 + has_dependents)` – Adjusts salary based on dependent burden |
| `is_senior_fulltime` | 1 if `age > 45` and `is_fulltime == 1` – Highlights older full-time employees |
| `is_young_with_dependents` | 1 if `age < 30` and `has_dependents == 1` – Identifies younger employees with dependents |
| `has_high_salary_and_long_tenure` | 1 if `salary > threshold(median)` and `tenure_years > threshold(median)` – Flags loyal high earners |
| `log_salary` | `np.log1p(salary)` – Log transformation to handle salary skew(not present but might be in real data) |
| `age_squared` | `age ** 2` – Captures non-linear effects of age on enrollment likelihood |
| `salary_zscore_by_region` | Z-score of salary within `region` – Normalizes salary across geographies |
| `tenure_percentile_in_region` | Percentile rank of tenure within `region` – Contextualizes tenure duration |
| `is_top_salary_percentile` | 1 if salary is in the top 10% – Highlights high-income outliers |

---
            ''')

st.subheader("Processed Dataframe")

st.write("")

pre_processed_df = preprocess_employee_data(df)

st.dataframe(pre_processed_df,hide_index=True)

side1,side2=st.columns(2)


with side1:
    st.markdown("""
##### 📊 Why Data Splitting Matters

Before training a model, it's important to split the data into training and test sets to evaluate how well it generalizes to new data.

- Setting **`employee_id`** as the index preserves unique identity for each employee.
- Stratifying by **`enrolled`** ensures class balance, which is crucial for fair model evaluation.
""")
    
with side2:
    st.markdown("##### Test Train Split")
    with st.container(border=True):
        test_size = st.slider("Test Dataframe Ratio",min_value=0.01,max_value=0.99,value=0.2)
        stratify = st.checkbox("Stratify Target Variable",value=True)

train_df , test_df = split_employee_data(pre_processed_df,test_size=test_size,stratify=stratify)

train_col , test_col = st.columns(2)
with train_col:
    st.subheader("Train Dataframe")
    st.dataframe(train_df,use_container_width=True)
    
with test_col:
    st.subheader("Test Dataframe")
    st.dataframe(test_df,use_container_width=True)
    
    
st.markdown("""
### Model Introduction
We’ll use a **DecisionTreeClassifier** as our base model — it's simple, handles class imbalance and missing values (though we have none), and aligns with tree-based boosting methods like XGBoost, CatBoost, and LightGBM used in SOTA models for tabular data.
""")
    
hyper_params_intro , hyper_params_tool = st.columns(2)

with hyper_params_intro:
    st.markdown("""
##### 🧠 Model Pipeline Overview

A quick look at the 3 components in our model:

---
🔹 **VarianceThreshold (1.0)**  
Removes low-variance features that offer little value.  
*Set to 1.0 as feature engineering already handled weak features.*

---
🔹 **PolynomialFeatures (degree=2, interaction_only=True)**  
Generates pairwise interaction terms (e.g., `x₁ × x₂`),  
excluding squared terms like `x₁²`.

---
🔹 **DecisionTreeClassifier (max_depth=4)**  
Performs classification using decision trees.  
- `max_depth`: Prevents overfitting  
- `min_samples_split`: Minimum samples to split a node  
- `criterion`:  
  - `gini` (default)  
  - `entropy` (info gain)  
  - `log_loss` (probabilistic)

---
""")
with hyper_params_tool: 
    with st.container(border=True):
        hyper_params = get_pipeline_hyperparameters()


run_model = st.button("RUN Model and Show Results",use_container_width=True,type='primary')

if run_model:
    pipeline = build_pipeline_from_params(hyper_params)
    features = ['age', 'age_bucketed', 'salary', 'has_dependents', 'is_fulltime',
        # Engineered
        'salary_per_year_of_age', 'salary_per_tenure', 'age_tenure_ratio',
        'salary_dependents_ratio', 'is_senior_fulltime', 'is_young_with_dependents',
        'has_high_salary_and_long_tenure', 'log_salary',
        # Group-based
        'salary_zscore_by_region', 'tenure_percentile_in_region', 'is_top_salary_percentile',
    ]
    target = ['enrolled']
    train_and_evaluate_model(pipeline,train_df,test_df,features,target)
    
    st.markdown('''
                ### 📝 Exit Note

During my time on this project, I developed a baseline classification model using **Decision Trees**, prioritizing **explainability** and **quick inference**. This model serves as an initial framework to help the business understand key drivers behind employee enrollment behavior.

---

#### 🚀 Next Steps & Opportunities

- **Model Enhancement**  
  With more time, I would explore **ensemble methods** such as:
  - **Bagging** techniques (e.g., Random Forest)
  - **Boosting** techniques (e.g., XGBoost, LightGBM)

- **Model Deployment**  
  I would serve the model using a **Flask API**, enabling:
  - Real-time predictions
  - Seamless integration with business dashboards or applications

---

#### 🎯 Business Value

This approach ensures that stakeholders not only get **accurate predictions** but also understand the **"why"** behind each decision. With interpretable models at the core, the business can take **data-driven actions** with confidence.

---

*This is just the beginning — the foundation is set for building a powerful, explainable, and scalable decision-support system.*
                
                ''')


    




