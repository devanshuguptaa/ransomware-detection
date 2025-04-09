import streamlit as st
import pandas as pd
from utils.helpers import load_data

st.set_page_config(page_title="Feature Selection - IV & WoE", layout="wide")
st.title("📊 Feature Selection using IV & WoE (Simulated)")

# Load data
df = load_data()
features = df.drop(['Name', 'md5', 'legitimate'], axis=1).columns[:15]

# Simulated IV values (descending order)
iv_scores = pd.Series(range(len(features), 0, -1), index=features)

# Bar chart
st.subheader("Top 15 Predictive Features (Simulated Information Value)")
st.bar_chart(iv_scores)

# Explanation
st.markdown("---")
st.subheader("📘 What is Information Value (IV) and Weight of Evidence (WoE)?")

st.markdown("""
**🔹 Weight of Evidence (WoE)** is a measure used to encode categorical variables and analyze the strength of each feature in separating binary outcomes.  
It transforms variables based on the distribution of events and non-events.

**🔹 Information Value (IV)** helps rank variables based on their predictive power:
- **IV < 0.02** – Not useful for prediction  
- **0.02 ≤ IV < 0.1** – Weak predictor  
- **0.1 ≤ IV < 0.3** – Medium predictor  
- **IV ≥ 0.3** – Strong predictor  

**📌 Why use IV & WoE?**  
They are especially useful in:
- Credit scoring and fraud detection
- Handling categorical variables and binning numerical ones
- Providing interpretable scores for model insights

> In this app, IV values have been **simulated** for demonstration. In a real-world scenario, you would calculate them using statistical techniques based on target separation.
""")

st.success("💡 Tip: Use IV to select features, and WoE to transform them for better model performance.")
