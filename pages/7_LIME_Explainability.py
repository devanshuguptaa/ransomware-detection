import streamlit as st
import lime.lime_tabular
import numpy as np
from utils.helpers import load_data, train_model, explain_instance

st.title("🧠 Explainability with LIME")

df = load_data()
features = df.drop(['Name', 'md5', 'legitimate'], axis=1).columns[:15]
X = df[features]
y = df['legitimate']

X_train, X_test, y_train, y_test, model = train_model(X, y)

st.subheader("🔍 Instance 1 Explanation")
explain_instance(model, X_test, X_train, 1)

st.subheader("🔍 Instance 6 Explanation")
explain_instance(model, X_test, X_train, 6)
