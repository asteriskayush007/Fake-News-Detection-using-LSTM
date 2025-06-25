import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
import string

import subprocess
import os

# Run download_assets.py to get Fake.csv and True.csv if not already present
if not (os.path.exists("Fake.csv") and os.path.exists("True.csv")):
    subprocess.run(["python", "download_assets.py"])


# Load model and tokenizer
model = load_model("lstm_fake_news_model.keras")
tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
max_len = 500

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub(r"\d+", "", text)
    return text.strip()

# Prediction function
def predict_news(text):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=max_len)
    prediction = model.predict(padded)[0][0]
    label = "REAL" if prediction > 0.5 else "FAKE"
    return label, float(prediction)

# UI
st.set_page_config(page_title="Fake News Detector", page_icon="🧠")
st.title("🧠 Fake News Detection using LSTM")
st.markdown("Enter a news article below to check if it's **REAL** or **FAKE**.")

user_input = st.text_area("📰 Paste your news content here:", height=200)

if st.button("🔎 Analyze"):
    if user_input.strip():
        label, score = predict_news(user_input)

        st.markdown(f"### 🔍 Prediction: **{label}**")
        st.markdown(f"**📊 Confidence Score:** `{score:.2%}`")
        st.progress(int(score * 100))

        if label == "REAL":
            st.success("✅ This appears to be a **genuine** news article.")
        else:
            st.error("🚫 This might be **fake** or misleading content.")
    else:
        st.warning("⚠️ Please enter a news article to analyze.")
