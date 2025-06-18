import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTETomek

st.set_page_config(page_title="Predict Sample", layout="wide")
st.title("Section 11: Predict Ransomware or Legitimate")

@st.cache_data
def load_data():
    df = pd.read_csv("data/Ransomware.csv", sep="|")
    df = df.dropna()  # Remove rows with missing values
    df = df.apply(pd.to_numeric, errors='coerce')  # Ensure numeric
    df = df.dropna()  # Drop any rows that became NaN after coercion
    return df

# Load and clean data
df = load_data()

# Define features
all_features = df.drop(columns=['Name', 'md5', 'legitimate'], errors='ignore')
feature_names = all_features.columns[:15]
X = all_features[feature_names]
y = df['legitimate']

# Ensure input is valid for SMOTE
X = X.astype(float)
y = y.astype(int)

# Split and apply SMOTE-Tomek
X_train, _, y_train, _ = train_test_split(X, y, train_size=0.7, random_state=42)
smote = SMOTETomek(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_resampled, y_resampled)

# UI Input
st.subheader("Input Feature Values")
user_data = []
for col in feature_names:
    default_val = float(X[col].mean())
    user_val = st.number_input(f"{col}", value=default_val)
    user_data.append(user_val)

# Optional actual label
actual_class = st.radio("Optional: Actual Class", options=["Not Provided", "Legitimate", "Malware"])

# Predict button
if st.button("Predict"):
    prediction = model.predict([user_data])[0]
    prob = model.predict_proba([user_data])[0]
    pred_label = "Legitimate" if prediction == 1 else "Malware"

    st.markdown("### 🧠 Prediction Result")
    st.write("**Predicted Class:**", pred_label)
    st.write("**Confidence:**", f"{np.max(prob)*100:.2f}%")

    if actual_class != "Not Provided":
        actual_binary = 1 if actual_class == "Legitimate" else 0
        st.write("**Actual Class:**", actual_class)

        if actual_binary == prediction:
            st.success("✅ Prediction matches the actual class.")
        else:
            st.error("❌ Prediction does NOT match the actual class.")
