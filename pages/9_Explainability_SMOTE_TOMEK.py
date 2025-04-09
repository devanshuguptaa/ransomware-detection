import streamlit as st
from imblearn.combine import SMOTETomek
from utils.helpers import load_data, train_model, explain_instance

st.title("🧠 Explainability After SMOTE Tomek")

df = load_data()
features = df.drop(['Name', 'md5', 'legitimate'], axis=1).columns[:15]
X = df[features]
y = df['legitimate']

X_train, X_test, y_train, y_test, model = train_model(X, y)
smote = SMOTETomek(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

model.fit(X_train_sm, y_train_sm)

st.subheader("🔍 Instance 1 Explanation")
explain_instance(model, X_test, X_train_sm, 1)

st.subheader("🔍 Instance 6 Explanation")
explain_instance(model, X_test, X_train_sm, 6)
