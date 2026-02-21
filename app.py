import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

@st.cache_resource
def train_model():
    df = pd.read_csv("https://raw.githubusercontent.com/lutzhamel/fake-news/master/data/fake_or_real_news.csv")
    df["content"] = df["title"] + " " + df["text"]
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    X = vectorizer.fit_transform(df["content"])
    Y = df["label"]
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, Y_train)
    return model, vectorizer

model, vectorizer = train_model()

st.title("📰 Fake News Detector")
st.markdown("**By Jad Mrad** · Computer Engineering Student · Lebanon 🇱🇧")
st.markdown("Model Accuracy: **91.48%**")
st.warning("⚠️ This model was trained on American political news from 2016. It works best with English political headlines and may be less accurate on other topics or recent news.")
st.markdown("---")

st.subheader("Enter a news headline or article")
news_input = st.text_area("Paste your news here:", height=150)
st.markdown("---")

if st.button("🔍 Check News", use_container_width=True):
    if news_input.strip() == "":
        st.warning("Please enter some news text first!")
    else:
        test = vectorizer.transform([news_input])
        result = model.predict(test)[0]
        probability = model.predict_proba(test)[0]

        if result == "FAKE":
            st.error("🚨 This news is likely FAKE!")
            st.metric("Fake Probability", f"{round(max(probability) * 100, 1)}%")
        else:
            st.success("✅ This news appears to be REAL!")
            st.metric("Real Probability", f"{round(max(probability) * 100, 1)}%")

st.markdown("---")
st.markdown("🔗 [GitHub](https://github.com/jad-mrad) · Built with Python & Streamlit")