import streamlit as st
from imblearn.combine import SMOTETomek
from utils.helpers import load_data, train_model, explain_instance

st.set_page_config(page_title=" SMOTE Tomek", layout="wide")
st.title("🧠 SMOTE Tomek")

# Introduction to SMOTE + Tomek
st.subheader("📌 What is SMOTE + Tomek?")
st.markdown("""
**SMOTE (Synthetic Minority Over-sampling Technique)** creates synthetic examples of the minority class to balance class distribution.  
**Tomek Links** help clean the boundary by removing overlapping instances between classes.

Combining both gives a better decision boundary and improves model generalization.
""")

# Show diagram image

try:
    st.image("assets\SMOTE.png", caption="Visual Representation: SMOTE + Tomek Technique", use_container_width=True)
except Exception as e:
    st.error(f"Failed to load image: {e}")




# Load data and preprocess

df = load_data()
features = df.drop(['Name', 'md5', 'legitimate'], axis=1).columns[:15]
X = df[features]
y = df['legitimate']

# Train model and apply SMOTE Tomek
X_train, X_test, y_train, y_test, model = train_model(X, y)
smote = SMOTETomek(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
model.fit(X_train_sm, y_train_sm)

# LIME explanations
st.subheader("🔍 LIME Explanation on SMOTE-enhanced Model")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📌 Instance 1")
    explain_instance(model, X_test, X_train_sm, 1)

with col2:
    st.markdown("### 📌 Instance 6")
    explain_instance(model, X_test, X_train_sm, 6)
