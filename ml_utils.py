import sklearn
import graphviz
import pandas as pd
import dtreeviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from dtreeviz.utils import extract_params_from_pipeline
import streamlit as st
    
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit.components.v1 as components
import pathlib
import tempfile




def split_employee_data(df, test_size=0.2, stratify=False, random_state=42):
    """
    Splits the employee data into training and test sets.
    
    Parameters:
    - df (pd.DataFrame): Input dataframe containing all features and 'enrolled' as target.
    - test_size (float): Proportion of the dataset to include in the test split.
    - stratify (bool): Whether to stratify based on the target variable.
    - random_state (int): Seed for reproducibility.

    Returns:
    - train_df (pd.DataFrame): Training dataframe.
    - test_df (pd.DataFrame): Test dataframe.
    """
    df = df.copy()
    
    # Set employee_id as index
    df.set_index("employee_id", inplace=True)

    # Define target
    y = df["enrolled"]

    # Optional stratification
    stratify_vals = y if stratify else None

    # Train-test split
    train_idx, test_idx = train_test_split(
        df.index, 
        test_size=test_size, 
        stratify=stratify_vals, 
        random_state=random_state
    )

    # Slice the DataFrame based on the split indices
    train_df = df.loc[train_idx]
    test_df = df.loc[test_idx]

    return train_df, test_df


def get_pipeline_hyperparameters():
    with st.container():
        st.subheader("🛠️ Model Hyperparameter Tuning")

        # VarianceThreshold
        threshold = st.slider("Variance Threshold", min_value=0.0, max_value=1.0, value=0.99, step=0.01)

        # PolynomialFeatures
        degree = st.slider("Polynomial Degree", min_value=1, max_value=5, value=1)
        interaction_only = st.checkbox("Use Interaction Only (Polynomial)", value=True)

        # DecisionTreeClassifier
        st.markdown("#### Decision Tree Parameters")
        max_depth = st.slider("Max Depth", min_value=1, max_value=10, value=3)
        min_samples_split = st.slider("Min Samples Split", min_value=2, max_value=20, value=2)
        criterion = st.selectbox("Criterion", options=["gini", "entropy", "log_loss"], index=0)

    return {
        "threshold": threshold,
        "degree": degree,
        "interaction_only": interaction_only,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "criterion": criterion
    }
    


def build_pipeline_from_params(params):
    return make_pipeline(
        VarianceThreshold(threshold=params["threshold"]),
        PolynomialFeatures(degree=params["degree"], interaction_only=params["interaction_only"]),
        DecisionTreeClassifier(
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            criterion=params["criterion"]
        )
    )
    

def train_and_evaluate_model(pipeline, train_df, test_df, features, target):

    # ── 1 Split features / target ───────────────────────────────
    X_train, y_train = train_df[features], train_df[target]
    X_test,  y_test  =  test_df[features],  test_df[target]

    # ── 2 Train & predict ──────────────────────────────────────
    pipeline.fit(X_train, y_train)
    y_pred  = pipeline.predict(X_test)

    # ── 3 Streamlit layout  (matrix | metrics) ─────────────────
    col1, col2 = st.columns(2)

    # 3-a Confusion-matrix
    with col1:
        st.subheader("📊 Confusion Matrix")
        cm  = confusion_matrix(y_test, y_pred)
        fig = plt.figure(figsize=(3, 3))
        ConfusionMatrixDisplay(cm).plot(ax=plt.gca(), colorbar=False)
        st.pyplot(fig)

    # 3-b Scalar metrics
    with col2:
        st.subheader("📈 Evaluation Metrics")
        accuracy  = accuracy_score (y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall    = recall_score   (y_test, y_pred, zero_division=0)
        f1        = f1_score       (y_test, y_pred, zero_division=0)

        # AUC requires positive-class probabilities
        try:
            y_prob = pipeline.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
        except Exception:
            auc = None

        st.markdown(f"""
        - **Accuracy:** `{accuracy:.3f}`
        - **Precision:** `{precision:.3f}`
        - **Recall:** `{recall:.3f}`
        - **F1 Score:** `{f1:.3f}`
        - **AUC Score:** `{auc:.3f}`""" if auc is not None else
        "- **AUC Score:** `N/A (Only one class present)`")

        st.markdown("""
##### 🎯 Why Precision Matters
In the context of employee-benefit enrollment **precision** answers:

> *“Of all the people we predicted would enroll, how many actually did?”*

Because outreach (emails, calls, incentives) costs money, higher precision
means our **budget is spent on the right employees**, maximising ROI and
reducing wasted effort.
""")
    with st.expander("Understanding Decision Trees with `dtreeviz` (click for explanation)",icon="🌳",expanded=False):

        st.markdown("""
    ##### 🔍 What is `dtreeviz`?

    `dtreeviz` is a visualization tool for decision tree models that allows you to:
    - View how decisions are made at each node based on feature thresholds.
    - See how feature values guide samples to different parts of the tree.
    - Interpret model decisions in a way that's **explainable** and **transparent**.

    This is **crucial** when building employee benefit policies, as it shows **which features matter most**, and **how they influence the final decision** (enrolled or not enrolled).

    ---

    ##### 🧠 How to Read the Graph

    Each **split node** in the visualization shows:
    - The **feature** used to make the decision (e.g., salary, age, tenure).
    - The **threshold** value at which the split occurs.
    - A **histogram** of the distribution of samples in that node.
    - The **probability of enrollment** for samples that land in that node.

    Each **leaf node** provides:
    - The predicted class (✅ Enrolled / ❌ Not Enrolled).
    - The percentage distribution of samples.
    - The dominant class decision.

    This allows you to trace the **path from root to leaf**, understanding exactly:
    > “Why was this employee predicted to enroll (or not)?”

    ---

    ##### 🧩 Why It’s Useful for Policy Planning

    In this project, the dataset contains both **original and engineered features**, designed to reflect:
    - Demographic variables,
    - Employment details,
    - Behavioral signals for enrollment.

    By visualizing the decision tree:
    - **Teams** can identify what influences enrollment the most (e.g., salary bands, tenure levels).
    - **Policy designers** can tailor benefits or communication strategies toward features with the strongest influence.
    - **Stakeholders** gain a clear, visual reasoning behind each prediction, increasing trust in the model.

    ---


    ### ✅ Bottom Line
    Use `dtreeviz` to **open up your black-box model** and let stakeholders **see the story behind the prediction**. It's not just about accuracy – it's about **interpretability and impact**.

    """)
    try:
    # Extract decision tree and related info
        tree_clf, X_viz, post_names = extract_params_from_pipeline(
            pipeline=pipeline,
            X_train=X_train,
            feature_names=features
        )

        # Create visualization
        viz = dtreeviz.model(
            tree_clf,
            X_train=X_viz.squeeze(),
            y_train=y_train.squeeze(),
            feature_names=post_names,
            target_name=target,
            class_names=["Disenrolled", "Enrolled"]
        )

        # Save to a temporary SVG file
        tmp_svg_path = pathlib.Path(tempfile.mkstemp(suffix=".svg")[1])
        viz.view(scale=1.5).save(str(tmp_svg_path))

        # Read SVG content
        svg_code = tmp_svg_path.read_text()
        svg_code = svg_code.replace('width="1000pt"', 'width="1500pt"')  # adjust as needed
        svg_code = svg_code.replace('height="500pt"', 'height="750pt"')
        svg_code = svg_code.replace('font-size:10.00pt', 'font-size:14.00pt')  # tweak font size


        # Display in Streamlit
        st.markdown("##### Decision Tree Visualisation (dtreeviz)")
        st.components.v1.html(svg_code, height=600, scrolling=True)

    except Exception as e:
        st.warning(f"dtreeviz could not render the tree: {e}")