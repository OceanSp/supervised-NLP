import streamlit as st
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
from transformers import pipeline

MODEL_PATH = "./logistic_regression_model.joblib"
SCALER_PATH = "./scaler.joblib"
SENTIMENT_MODEL_PATH = "./sentiment_classifier_with_embeddings.joblib"


@st.cache_resource
def load_models():
    classification_model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    sentiment_model = joblib.load(SENTIMENT_MODEL_PATH)

    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )  # A small, fast embedding model

    category_model = pipeline(
        "zero-shot-classification", model="facebook/bart-large-mnli"
    )

    return (
        classification_model,
        scaler,
        sentiment_model,
        embedding_model,
        category_model,
    )


classification_model, scaler, sentiment_model, embedding_model, category_model = (
    load_models()
)

st.title("Review Rating Predictor")
st.write(
    "This app predicts the rating (e.g., 1-5 stars) of a review. Enter a review in plain English, and the app will compute the embeddings and predict the rating."
)

st.header("Enter a Review")
review_input = st.text_area(
    "Write your review below:",
    placeholder="e.g., The insurance company exceeded my expectations!",
)

if st.button("Start Predictions"):
    if review_input.strip():
        try:
            embedding = embedding_model.encode(
                [review_input]
            )  # Embedding as a 2D array

            embedding_scaled = scaler.transform(embedding)

            predicted_rating = classification_model.predict(embedding_scaled)[0]

            predicted_sentiment = sentiment_model.predict(embedding_scaled)[0]
            sentiment_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}
            sentiment = sentiment_mapping.get(predicted_sentiment, "Unknown")

            predicted_category = category_model(
                review_input,
                [
                    "Pricing",
                    "Coverage",
                    "Enrollment",
                    "Customer Service",
                    "Claims Processing",
                    "Cancellation",
                    "Other",
                ],
            )

            st.success(f"The predicted rating is: {predicted_rating} stars")
            st.success(f"The predicted sentiment is: {sentiment}")
            st.success(f"The predicted category is: {predicted_category['labels'][0]}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a review before predicting.")
