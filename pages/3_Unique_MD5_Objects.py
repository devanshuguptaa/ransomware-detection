import streamlit as st
from utils.helpers import load_data

st.title("🔐 Unique File Hashes (MD5)")

df = load_data()
unique_hashes = df['md5'].nunique()

st.metric("🔢 Unique MD5 Hashes", unique_hashes)
st.dataframe(df[['Name', 'md5']].drop_duplicates().head())
