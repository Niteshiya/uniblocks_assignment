import seaborn as sns 
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st




# Check if column is categorical
def is_categorical(series):
    return series.dtype == "object" or series.nunique() < 15


def plot_numerical_feature(df, column, target):
    fig, axs = plt.subplots(1, 2, figsize=(7, 2.5))  # Compact layout

    # Histogram
    sns.histplot(df[column], kde=True, ax=axs[0])
    axs[0].set_title(f"{column} Distribution", fontsize=10)
    axs[0].tick_params(labelsize=8)

    # Violin Plot with Correlation
    sns.violinplot(x=target, y=column, data=df, ax=axs[1], inner="quartile", scale="width")
    axs[1].set_title(f"{column} by {target}", fontsize=10)
    axs[1].tick_params(labelsize=8)
    axs[1].set_xticklabels(["Not Enrolled", "Enrolled"], fontsize=8)

    # Compute and annotate correlation
    corr = df[[column, target]].corr().iloc[0, 1]
    axs[1].text(0.5, 0.95, f"Corr = {corr:.2f}", fontsize=9, transform=axs[1].transAxes,
                ha='center', va='top', bbox=dict(boxstyle="round,pad=0.2", edgecolor='gray', facecolor='white'))

    plt.tight_layout()
    st.pyplot(fig)


def plot_categorical_feature(df, column, target):
    prop_df = (
        df.groupby([column, target]).size()
        .unstack(fill_value=0)
        .apply(lambda x: x / x.sum(), axis=1)
    )

    fig, ax = plt.subplots(figsize=(6, 2.5))  # Smaller plot size
    prop_df.plot(kind='bar', stacked=True, ax=ax, width=0.6)

    ax.set_ylabel("Proportion", fontsize=9)
    ax.set_title(f"{column} vs {target}", fontsize=10)
    ax.legend(title=target, labels=["Not Enrolled", "Enrolled"], fontsize=7, title_fontsize=8)
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    st.pyplot(fig)

# Main Streamlit function
def visualize_target_relationship(df):
    st.subheader("Step 1 : EDA for relations, feature engineering and strategy")
    target_col = "enrolled"
    id_col = "employee_id"
    
    feature_cols = [col for col in df.columns if col != target_col and col!=id_col]
    selected_col = st.selectbox("Select a feature to visualize:", feature_cols)

    if is_categorical(df[selected_col]):
        plot_categorical_feature(df, selected_col, target_col)
    else:
        plot_numerical_feature(df, selected_col, target_col)