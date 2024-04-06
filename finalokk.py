"""Streamlit app to compute similarity between sentences or paragraphs."""

# Import from standard library
import logging
import random
import re

# Import from 3rd party libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util

# Configure Streamlit page and state
st.set_page_config(page_title="Sentence Similarity", page_icon="🤖")

def compute_similarity(sentence1: str, sentence2: str, model, approach) -> float:
    if approach == "Sentence Transformers":
        query_embeddings = model.encode([sentence1])
        passage_embeddings = model.encode([sentence2])
    elif approach == "TF-IDF":
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform([sentence1, sentence2])
        query_embeddings = tfidf_matrix[0].toarray()
        passage_embeddings = tfidf_matrix[1].toarray()
    elif approach == "Universal Sentence Encoder":
        # Load a pre-trained Universal Sentence Encoder model
        model = SentenceTransformer('stsb-roberta-base-v2')
        query_embeddings = model.encode([sentence1])
        passage_embeddings = model.encode([sentence2])

    scores = util.pytorch_cos_sim(query_embeddings, passage_embeddings)
    return scores[0][0].item()

# Render Streamlit page
st.title("RESEARCH WORK - VARSHA")
st.markdown("This app computes similarity between two sentences using different approaches")

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

sentence1 = st.text_input(label="Sentence 1", placeholder="Text goes here...")
sentence2 = st.text_input(label="Sentence 2", placeholder="Text goes here...")

if st.button(
    label="Compute similarity",
    type="primary",
):
    sentence_transformers_scores = []
    tfidf_scores = []
    use_scores = []

    for _ in range(10):  # Perform calculations multiple times for a more complex chart
        sentence_transformers_score = compute_similarity(sentence1, sentence2, model, "Sentence Transformers")
        tfidf_score = compute_similarity(sentence1, sentence2, model, "TF-IDF")
        use_score = compute_similarity(sentence1, sentence2, model, "Universal Sentence Encoder")

        sentence_transformers_scores.append(sentence_transformers_score)
        tfidf_scores.append(tfidf_score)
        use_scores.append(use_score)

    st.text(f'Sentence Transformers Similarity Scores: {sentence_transformers_scores}')
    st.text(f'TF-IDF Similarity Scores: {tfidf_scores}')
    st.text(f'Universal Sentence Encoder Similarity Scores: {use_scores}')

    # Calculate the mean for each approach
    sentence_transformers_mean = sum(sentence_transformers_scores) / len(sentence_transformers_scores)
    tfidf_mean = sum(tfidf_scores) / len(tfidf_scores)
    use_mean = sum(use_scores) / len(use_scores)

    # Calculate the standard deviation for each approach
    sentence_transformers_std = (sum((x - sentence_transformers_mean) ** 2 for x in sentence_transformers_scores) / (len(sentence_transformers_scores) - 1)) ** 0.5
    tfidf_std = (sum((x - tfidf_mean) ** 2 for x in tfidf_scores) / (len(tfidf_scores) - 1)) ** 0.5
    use_std = (sum((x - use_mean) ** 2 for x in use_scores) / (len(use_scores) - 1)) ** 0.5

    # Create a comparison chart with mean and standard deviation for all three approaches
    data = {
        'Approach': ["Sentence Transformers", "TF-IDF", "Universal Sentence Encoder"],
        'Mean Score': [round(sentence_transformers_mean, 4), round(tfidf_mean, 4), round(use_mean, 4)],
        'Standard Deviation': [round(sentence_transformers_std, 4), round(tfidf_std, 4), round(use_std, 4)]
    }
    df = pd.DataFrame(data)

    fig, ax = plt.subplots()
    ax.bar(df['Approach'], df['Mean Score'], yerr=df['Standard Deviation'])
    plt.ylabel('Similarity Score')
    plt.title('Comparison of Sentence Similarity Approaches')
    plt.xticks(rotation=15)
    st.pyplot(fig)
