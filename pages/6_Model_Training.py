import streamlit as st
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score
import pandas as pd
from utils.helpers import load_data, show_metrics

st.title("⚙️ Model Training")

df = load_data()
features = df.drop(['Name', 'md5', 'legitimate'], axis=1).columns[:15]
X = df[features]
y = df['legitimate']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)

progress = st.progress(0)
for i in range(100):
    time.sleep(0.005)
    progress.progress(i + 1)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

show_metrics(y_test, y_pred, y_proba)
