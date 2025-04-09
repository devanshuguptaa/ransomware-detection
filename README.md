# 🔐 Ransomware Detection using Machine Learning

![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-ff4b4b?style=for-the-badge&logo=streamlit)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

---

## 📘 Project Overview

**Ransomware** is a type of malware that encrypts users’ data and demands ransom to decrypt it. This project uses **machine learning** to detect ransomware based on static file features.

It is built using **Streamlit** for interactive UI and integrates explainability using **LIME**, class imbalance correction using **SMOTE Tomek**, and standard ML evaluation techniques.

---

## 🚀 Features

- 📊 **Dataset exploration** & imbalance visualization  
- 🔍 **Correlation heatmaps** and top predictive features (IV + WoE)  
- ⚙️ **Random Forest** classifier with real-time evaluation  
- 🎯 Metrics: Accuracy, Precision, Recall, F1, MCC, AUC  
- 🧠 **LIME Explainability** for instance-level predictions  
- ⚖️ **SMOTE Tomek** to balance data and improve generalization  
- 📈 Clean, professional Streamlit dashboard  

---

## 🗂️ Project Structure

```
ransomware_app/
│
├── Home.py                          # 1. Problem Statement
├── pages/
│   ├── 1_Dataset_Exploration.py     # 2. Dataset Exploration
│   ├── 2_Label_Distribution.py      # 3. Label Distribution (multiple charts)
│   ├── 3_Unique_MD5_Objects.py      # 4. Unique Files/MD5s
│   ├── 4_Correlation_Analysis.py    # 5. Feature Correlations (heatmap & table)
│   ├── 5_IV_WoE_Features.py         # 6. Simulated Feature Selection
│   ├── 6_Model_Training.py          # 7. Random Forest Training + Evaluation
│   ├── 7_LIME_Explainability.py     # 8. Explainability with LIME
│   ├── 8_SMOTE_TOMEK.py             # 9. SMOTE Tomek + Evaluation + Image
│   └── 9_Explainability_SMOTE_TOMEK.py # 10. Post-SMOTE LIME Explanations
│
├── utils/
│   └── helpers.py                   # Helper functions for loading, model, explainability
│
├── data/
│   ├── Ransomware.csv              # Dataset
│   └── SMOTE.png                   # Visual explanation of SMOTE + Tomek
│
└── requirements.txt                # All required dependencies
```

---

## 🛠️ Installation & Setup

> ⚠️ Python 3.9+ recommended

```bash
# Clone the repository
git clone https://github.com/yourusername/ransomware-detection-app.git
cd ransomware-detection-app

# (Optional) Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 🚦 How to Run the App

```bash
streamlit run Home.py
```

Then go to `http://localhost:8501` in your browser!

You can navigate between pages using the sidebar.

---

## 📊 Sample Output

- Interactive dashboards  
- Evaluation metrics  
- LIME explanations  
- Visualizations of feature correlation, class distribution, etc.  

---

## 📌 Dependencies

- `streamlit`
- `scikit-learn`
- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `imblearn`
- `lime`

See full list in `requirements.txt`.

---

## 🧠 Future Improvements

- Integration with real-time file monitoring  
- Deployment on Streamlit Cloud or HuggingFace Spaces  
- More classifiers (e.g., XGBoost, LightGBM)  
- Deep learning extension  

---

## 📄 License

This project is licensed under the MIT License.

---

## 🤝 Contributing

Pull requests and issues are welcome. If you'd like to improve this app or add a feature, feel free to fork and contribute!

---

## 🌐 Author

Developed by **Shivanya Gautam**  
🔗 [Your GitHub Profile](https://github.com/DejaVu-1729/)