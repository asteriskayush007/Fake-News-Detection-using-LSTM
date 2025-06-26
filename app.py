import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

# Page setup
st.set_page_config(page_title="Fake News Detector", layout="centered")

st.title("📰 Fake News Detection App")
st.write("This app uses a Machine Learning model to detect whether a news article is **Real** or **Fake**.")

@st.cache_resource
def load_data():
    fake = pd.read_csv("Fake.csv")
    true = pd.read_csv("True.csv")
    fake["label"] = 0
    true["label"] = 1
    df = pd.concat([fake, true])
    df = df.sample(frac=1).reset_index(drop=True)
    return df

df = load_data()

@st.cache_resource
def train_model(data):
    x_train, x_test, y_train, y_test = train_test_split(
        data["text"], data["label"], test_size=0.2, random_state=42
    )
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)

    model = PassiveAggressiveClassifier(max_iter=1000)
    model.fit(x_train_vec, y_train)

    acc = accuracy_score(y_test, model.predict(x_test_vec))
    return model, vectorizer, acc

model, vectorizer, accuracy = train_model(df)

st.success(f"✅ Model trained with accuracy: {accuracy*100:.2f}%")

# Prediction input
st.subheader("📝 Enter News Content to Verify")
user_input = st.text_area("Paste the news text/article here:")

if st.button("Check"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some news content first.")
    else:
        input_vec = vectorizer.transform([user_input])
        pred = model.predict(input_vec)
        result = "✅ Real News" if pred[0] == 1 else "🚨 Fake News"
        st.subheader("Prediction Result:")
        st.info(result)