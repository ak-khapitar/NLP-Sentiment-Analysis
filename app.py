import streamlit as st
import joblib

model = joblib.load("sentiment_model.pkl")
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")

st.title("🎬 Movie Review Sentiment Analysis")

review = st.text_area("Enter your movie review:")

if st.button("Predict Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        review_tfidf = tfidf_vectorizer.transform([review])
        prediction = model.predict(review_tfidf)[0]

        if prediction == 1:
            st.success("😊 Positive Sentiment")
        else:
            st.error("😞 Negative Sentiment")