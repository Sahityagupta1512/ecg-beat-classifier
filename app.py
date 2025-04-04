import streamlit as st
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import joblib

st.set_page_config(page_title="ECG Beat Viewer", layout="centered")
st.title("📊 ECG Beat Viewer & Classifier")

# ---------- Load Data ----------
@st.cache_data
def load_data():
    normal = np.load(r"F:\2409029\Part 1 - Sem 2\FDS - Kalpesh Sir\ECG_Streamlitapp\Data\normal_beats.npy")
    abnormal = np.load(r"F:\2409029\Part 1 - Sem 2\FDS - Kalpesh Sir\ECG_Streamlitapp\Data\abnormal_beats.npy")
    return normal, abnormal

normal_beats, abnormal_beats = load_data()
all_beats = np.concatenate([normal_beats, abnormal_beats])
labels = np.concatenate([["Normal"] * len(normal_beats), ["Abnormal"] * len(abnormal_beats)])

# ---------- Load Model ----------
@st.cache_data
def load_model():
    return joblib.load(r"F:\2409029\Part 1 - Sem 2\FDS - Kalpesh Sir\ECG_Streamlitapp\model\ecg_classifier.pkl")

model = load_model()

# ---------- Random Beat Viewer ----------
st.header("🔀 View Random ECG Beat")

if st.button("Show Random Normal Beat"):
    idx = random.randint(0, len(normal_beats) - 1)
    st.line_chart(normal_beats[idx])
    st.caption("Class: Normal")

if st.button("Show Random Abnormal Beat"):
    idx = random.randint(0, len(abnormal_beats) - 1)
    st.line_chart(abnormal_beats[idx])
    st.caption("Class: Abnormal")

# ---------- Predict a Beat from Dataset ----------
st.header("💡 ECG Beat Classification")

selected_beat = st.selectbox("Select a Beat to Predict", list(range(len(all_beats))))
selected_data = all_beats[selected_beat].reshape(1, -1)

if st.button("Predict Class"):
    pred = model.predict(selected_data)
    st.success(f"✅ Predicted Class: **{pred[0]}**")
    st.line_chart(selected_data.flatten())

# ---------- Upload ECG Beat ----------
st.subheader("📤 Upload Your Own ECG Beat")

uploaded_file = st.file_uploader("Upload CSV with 1 row and 187 values", type=["csv"])
if uploaded_file is not None:
    try:
        uploaded_df = pd.read_csv(uploaded_file, header=None)
        if uploaded_df.shape[1] == 187:
            beat = uploaded_df.values.reshape(1, -1)
            pred = model.predict(beat)
            st.success(f"📊 Predicted Class: **{pred[0]}**")
            st.line_chart(beat.flatten())
        else:
            st.error("Uploaded beat must have 187 values.")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

# ---------- Summary Statistics ----------
st.header("📈 Summary Statistics")

st.write("**Normal Beats**")
st.dataframe(pd.DataFrame(normal_beats).describe())

st.write("**Abnormal Beats**")
st.dataframe(pd.DataFrame(abnormal_beats).describe())

# ---------- Pie Chart ----------
st.subheader("🧮 Beat Distribution")

labels_pie = ['Normal', 'Abnormal']
counts = [len(normal_beats), len(abnormal_beats)]

fig, ax = plt.subplots()
ax.pie(counts, labels=labels_pie, autopct='%1.1f%%', startangle=90)
ax.axis('equal')
st.pyplot(fig)

# ---------- Footer ----------
st.caption("Made with ❤️ by Sahitya Gupta")
