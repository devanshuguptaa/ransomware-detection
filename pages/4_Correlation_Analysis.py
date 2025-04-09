import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from utils.helpers import load_data

st.title("📊 Correlation Between Features")

df = load_data()

# Drop non-numeric and label columns
correlation_df = df.drop(['Name', 'md5', 'legitimate'], axis=1)
corr_matrix = correlation_df.corr()

# Display Heatmap
st.subheader("🔥 Correlation Heatmap")
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr_matrix, cmap='coolwarm', annot=False, linewidths=0.5, ax=ax)
st.pyplot(fig)

# Extract correlated pairs
st.subheader("📋 Pairwise Feature Correlation Table")
corr_pairs = corr_matrix.unstack().reset_index()
corr_pairs.columns = ['Feature 1', 'Feature 2', 'Correlation']
corr_pairs = corr_pairs[corr_pairs['Feature 1'] != corr_pairs['Feature 2']]  # Remove self-correlation
corr_pairs['Abs Correlation'] = corr_pairs['Correlation'].abs()
corr_pairs_sorted = corr_pairs.sort_values(by='Abs Correlation', ascending=False)

# Remove duplicate pairs
corr_pairs_unique = corr_pairs_sorted.drop_duplicates(subset=["Abs Correlation"], keep='first')
st.dataframe(corr_pairs_unique.head(20), use_container_width=True)
