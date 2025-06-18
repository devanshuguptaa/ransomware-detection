import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTETomek

st.set_page_config(page_title="Ransomware Sample Prediction", layout="wide")
st.title("Section 11: Predict Ransomware or Legitimate")

@st.cache_data
def load_data():
    return pd.read_csv("data/Ransomware.csv", sep="|")

# Load and prepare data
df = load_data()
features = df.drop(columns=["Name", "md5", "legitimate"]).columns[:15]
X = df[features].apply(pd.to_numeric, errors='coerce').fillna(0)
y = df["legitimate"]

# Train-test split and resampling
X_train, _, y_train, _ = train_test_split(X, y, train_size=0.7, random_state=42)
smote = SMOTETomek(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_resampled, y_resampled)

# Input section
st.subheader("Enter Feature Values")
user_input = []
for feature in features:
    val = st.number_input(f"{feature}", value=float(X[feature].mean()))
    user_input.append(val)

# Optional actual class input
actual_class = st.radio("Optional: Actual Class (if known)", options=["Not Provided", "Legitimate", "Malware"])

# Prediction
if st.button("Predict"):
    prediction = model.predict([user_input])[0]
    probability = model.predict_proba([user_input])[0]

    predicted_label = "Legitimate" if prediction == 1 else "Malware"
    confidence = f"{np.max(probability) * 100:.2f}%"

    st.markdown("### 🧠 Prediction Result")
    st.write("**Predicted Class:**", predicted_label)
    st.write("**Confidence:**", confidence)

    if actual_class != "Not Provided":
        actual_binary = 1 if actual_class == "Legitimate" else 0
        st.write("**Actual Class:**", actual_class)

        if actual_binary == prediction:
            st.success("✅ Prediction matches the actual class!")
        else:
            st.error("❌ Prediction does NOT match the actual class.")
