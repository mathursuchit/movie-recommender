# Movie Recommender — Neural Collaborative Filtering

A movie recommendation system built with Neural Collaborative Filtering (NCF), trained on the MovieLens Latest Small dataset.

**Live demo:** https://mathursuchit-movie-recommender.streamlit.app/

## How it works

Each user and movie is represented as a learned 64-dimensional embedding vector. Movies that tend to be watched and rated together end up with similar vectors — similar to how word2vec works for words.

When you pick movies you liked:
1. Grab each movie's embedding vector
2. Average them into a single taste vector
3. Find movies closest to that vector using cosine similarity
4. Return the top matches

## Dataset

[MovieLens Latest Small](https://grouplens.org/datasets/movielens/latest/) — 100,836 ratings across 9,724 movies, collected through 2018.

## Model

- Architecture: NCF with embedding dim=64, MLP layers [128, 64, 32]
- Loss: Binary Cross-Entropy
- Training: 20 epochs max with early stopping (patience=3)
- Best val loss: 0.5797

## Stack

- PyTorch — model training
- Streamlit — frontend
- MovieLens — dataset

## Run locally

```bash
git clone https://github.com/mathursuchit/movie-recommender.git
cd movie-recommender
pip install -r requirements.txt
streamlit run app.py
```

To retrain from scratch:
```bash
python download_data.py
python train.py
```

## Author

Suchit Mathur — [LinkedIn](https://www.linkedin.com/in/mathursuchit/)
