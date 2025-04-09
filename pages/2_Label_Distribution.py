import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from utils.helpers import load_data

st.title("📈 Distribution of Legitimate vs Malware Files")

df = load_data()
label_counts = df['legitimate'].value_counts()
labels = ['Legitimate', 'Malware']
colors = ['#A3E4D7', '#F1948A']

# Pie Chart
st.subheader("🥧 Pie Chart")
fig1, ax1 = plt.subplots()
ax1.pie(label_counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
ax1.axis('equal')
st.pyplot(fig1)

# Bar Chart
st.subheader("📊 Bar Chart")
fig2, ax2 = plt.subplots()
sns.barplot(x=labels, y=label_counts.values, palette=colors, ax=ax2)
ax2.set_ylabel("Number of Files")
st.pyplot(fig2)

# Donut Chart
st.subheader("🍩 Donut Chart")
fig3, ax3 = plt.subplots()
wedges, texts, autotexts = ax3.pie(label_counts, autopct='%1.1f%%', startangle=90, colors=colors)
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
fig3.gca().add_artist(centre_circle)
ax3.axis('equal')
ax3.legend(labels, loc='upper right')
st.pyplot(fig3)

# Table Summary
st.subheader("📋 Numerical Summary")
summary_df = pd.DataFrame({
    "Label": labels,
    "Count": label_counts.values,
    "Percentage": [f"{x:.2f}%" for x in (label_counts.values / label_counts.sum() * 100)]
})
st.dataframe(summary_df, use_container_width=True)
