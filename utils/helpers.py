import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score, confusion_matrix
)
import lime.lime_tabular


# ------------------------------
# Load Default Dataset
# ------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/Ransomware.csv", sep="|")


# ------------------------------
# Load Uploaded or Default CSV
# ------------------------------
def get_uploaded_or_default(uploaded_file):
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, sep="|")
            st.success("✅ Uploaded file loaded successfully!")
        except Exception as e:
            st.error(f"❌ Failed to read uploaded file: {e}")
            return None
    else:
        st.info("📂 No file uploaded. Using default dataset.")
        df = load_data()
    return df


# ------------------------------
# Custom Styled Header
# ------------------------------
def custom_header(title):
    st.markdown(f"<h1 style='color:#2E86C1;'>{title}</h1>", unsafe_allow_html=True)


# ------------------------------
# Styled DataFrame
# ------------------------------
def styled_dataframe(df):
    return df.style.set_properties(**{
        'background-color': '#f7f7f7',
        'color': '#333',
        'border-color': 'black'
    })


# ------------------------------
# Train Model
# ------------------------------
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return X_train, X_test, y_train, y_test, model


# ------------------------------
# Show Evaluation Metrics
# ------------------------------
def show_metrics(y_true, y_pred, y_proba):
    col1, col2, col3 = st.columns(3)
    col1.metric("🎯 Accuracy", f"{accuracy_score(y_true, y_pred):.4f}")
    col2.metric("📌 Precision", f"{precision_score(y_true, y_pred):.4f}")
    col3.metric("🔁 Recall", f"{recall_score(y_true, y_pred):.4f}")

    st.markdown("#### 🧮 Additional Scores")
    st.write(f"**F1 Score:** `{f1_score(y_true, y_pred):.4f}`")
    st.write(f"**Matthews Corr Coef:** `{matthews_corrcoef(y_true, y_pred):.4f}`")
    st.write(f"**AUC Score:** `{roc_auc_score(y_true, y_proba):.4f}`")

    st.markdown("#### 📊 Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)
    st.dataframe(pd.DataFrame(
        cm,
        index=["Actual Legit", "Actual Malware"],
        columns=["Predicted Legit", "Predicted Malware"]
    ), use_container_width=True)


# ------------------------------
# LIME Explainability
# ------------------------------
def explain_instance(model, X_test, X_train, index):
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns,
        class_names=['Malware', 'Legitimate'],
        mode='classification'
    )
    instance = X_test.iloc[index]
    prediction = model.predict_proba([instance])[0]

    st.markdown(f"**Prediction Probabilities:** Malware: `{prediction[0]:.4f}` | Legitimate: `{prediction[1]:.4f}`")

    explanation = explainer.explain_instance(instance, model.predict_proba, num_features=5)

    st.pyplot(explanation.as_pyplot_figure())

    st.markdown("#### 📄 LIME Explanation Table")
    st.dataframe(pd.DataFrame(explanation.as_list(), columns=['Feature', 'Weight']), use_container_width=True)
