import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langdetect import detect
import re
import pickle

# Load multilingual model
model = SentenceTransformer('distiluse-base-multilingual-cased')

# Load question-answer data
with open("qa_data_class9_12_2000.pkl", "rb") as f:
    data = pickle.load(f)

questions = [item["question"] for item in data]
answers = [item["answer"] for item in data]
embeddings = [item["embedding"] for item in data]

# App UI
st.set_page_config(page_title="🧠 Doubt Solver", page_icon="🧠")
st.title("🧠 Doubt Solver (Hindi + English)")
st.markdown("Ask your doubt below — Hinglish, Hindi, or English!")

# User input
user_question = st.text_input("Enter your question:")

if user_question:
    # Language detection
    lang = detect(user_question)
    st.markdown(f"**Language detected:** {lang}")

    # Preprocess input
    cleaned_q = re.sub(r'[^\w\s]', '', user_question).lower().strip()

    # Embed user query
    question_embedding = model.encode([cleaned_q])

    # Calculate cosine similarity
    similarities = cosine_similarity(question_embedding, embeddings).flatten()
    best_match_index = int(np.argmax(similarities))
    best_score = similarities[best_match_index]

    # Threshold logic — moved **inside** the block
    if best_score > 0.70:
        matched_q = data[best_match_index]["question"]
        answer = data[best_match_index]["answer"]
        st.markdown(f"**Matched Question:** {matched_q}")
        st.markdown(f"**Answer:** {answer}")
    else:
        st.warning("❌ Sorry, no relevant answer found in the database.")
