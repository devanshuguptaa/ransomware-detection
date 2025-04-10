import streamlit as st
from utils.helpers import load_data, styled_dataframe

st.title("📊 Dataset Exploration")

df = load_data()

st.subheader("🔍 Preview of the Dataset")
st.dataframe(styled_dataframe(df.sample(5)), use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    st.metric("Rows", df.shape[0])
with col2:
    st.metric("Columns", df.shape[1])

st.subheader("🧹 Missing Values")
missing = df.isnull().sum()
missing_df = missing[missing > 0].sort_values(ascending=False).reset_index()
missing_df.columns = ["Column", "Missing Values"]
if not missing_df.empty:
    st.dataframe(missing_df)
else:
    st.success("No missing values found!")
