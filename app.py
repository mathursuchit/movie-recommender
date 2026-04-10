import pickle
import torch
import torch.nn.functional as F
import pandas as pd
import streamlit as st
from model import NCF

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")
st.title("🎬 Movie Recommender — Neural Collaborative Filtering")
st.caption("Trained on MovieLens Latest Small (100K ratings, 9.7K movies)")

@st.cache_resource
def load_model_and_data():
    movies = pd.read_csv("data/movies.csv")
    with open("data/encodings.pkl", "rb") as f:
        enc = pickle.load(f)

    checkpoint = torch.load("data/model.pt", map_location="cpu")
    model = NCF(
        checkpoint["num_users"],
        checkpoint["num_items"],
        embedding_dim=64,
        layers=[128, 64, 32]
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    item_embeddings = model.get_item_embeddings()

    return model, movies, enc, item_embeddings

model, movies, enc, item_embeddings = load_model_and_data()

movie_titles = sorted(movies["title"].tolist())
title_to_idx = dict(zip(movies["title"], movies["movie_idx"]))

if "selected_movies" not in st.session_state:
    st.session_state.selected_movies = []

st.markdown("### Pick movies you liked")
st.caption("Add at least 3. The model will find movies with similar embedding patterns.")

search_query = st.text_input("Search for a movie:", placeholder="e.g. Toy Story, Matrix, Godfather")
filtered_titles = [t for t in movie_titles if search_query.lower() in t.lower()] if search_query else []

col1, col2 = st.columns([4, 1])
with col1:
    if filtered_titles:
        pick = st.selectbox("Select a match:", filtered_titles)
    elif search_query:
        st.caption("No movies found.")
        pick = None
    else:
        pick = None
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Add", type="primary"):
        if pick and pick not in st.session_state.selected_movies:
            st.session_state.selected_movies.append(pick)

if st.session_state.selected_movies:
    st.markdown("**Selected:**")
    for movie in st.session_state.selected_movies:
        col_a, col_b = st.columns([5, 1])
        col_a.markdown(f"- {movie}")
        if col_b.button("✕", key=f"remove_{movie}"):
            st.session_state.selected_movies.remove(movie)
            st.rerun()

selected = st.session_state.selected_movies
n_recommendations = st.slider("Number of recommendations", 5, 20, 10)

if st.button("Get Recommendations", disabled=len(selected) < 3):

    selected_indices = [title_to_idx[t] for t in selected if t in title_to_idx]
    selected_embs = item_embeddings[selected_indices]

    # average the selected movie embeddings into one taste vector
    taste_vector = selected_embs.mean(dim=0, keepdim=True)

    similarities = F.cosine_similarity(taste_vector, item_embeddings, dim=1)
    similarities[selected_indices] = -1  # don't recommend what they already picked

    top_indices = similarities.argsort(descending=True)[:n_recommendations]
    top_scores = similarities[top_indices]

    idx_to_title = dict(zip(movies["movie_idx"], movies["title"]))
    idx_to_genre = dict(zip(movies["movie_idx"], movies["genres"]))

    st.markdown("---")
    st.markdown("### Recommended for you")

    for rank, (idx, score) in enumerate(zip(top_indices.tolist(), top_scores.tolist()), 1):
        title = idx_to_title.get(idx, "Unknown")
        genre = idx_to_genre.get(idx, "")
        col1, col2, col3 = st.columns([3, 2, 1])
        col1.markdown(f"**{rank}. {title}**")
        col2.caption(genre.replace("|", " · "))
        col3.progress(float(score), text=f"{score:.2f}")

    with st.expander("How this works"):
        st.markdown("""
**Neural Collaborative Filtering** learns a 64-dimensional vector for every movie based on viewing patterns.

Movies that tend to be watched and rated together end up with similar vectors — similar to how word2vec works for words.

When you pick movies you liked:
1. Grab each movie's vector
2. Average them into a single taste vector
3. Find movies closest to that vector using cosine similarity
4. Return the top matches
        """)
