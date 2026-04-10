"""
Train NCF on MovieLens Latest Small and save the model.
Usage: python train.py
"""
import os
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from model import NCF

print("Loading data...")
ratings = pd.read_csv(
    "data/ml-latest-small/ratings.csv",
    names=["user_id", "movie_id", "rating", "timestamp"],
    skiprows=1
)
movies = pd.read_csv(
    "data/ml-latest-small/movies.csv",
    names=["movie_id", "title", "genres"],
    skiprows=1
)

# embeddings need 0-indexed integers, raw IDs won't work
user_ids = {uid: idx for idx, uid in enumerate(ratings["user_id"].unique())}
movie_ids = {mid: idx for idx, mid in enumerate(ratings["movie_id"].unique())}

ratings["user_idx"] = ratings["user_id"].map(user_ids)
ratings["movie_idx"] = ratings["movie_id"].map(movie_ids)
ratings["rating_norm"] = ratings["rating"] / 5.0  # sigmoid outputs 0-1

num_users = len(user_ids)
num_items = len(movie_ids)
print(f"Users: {num_users}, Movies: {num_items}, Ratings: {len(ratings)}")

# save movie metadata and encodings for the app
movies["movie_idx"] = movies["movie_id"].map(movie_ids)
movies = movies.dropna(subset=["movie_idx"])
movies["movie_idx"] = movies["movie_idx"].astype(int)
os.makedirs("data", exist_ok=True)
movies.to_csv("data/movies.csv", index=False)

with open("data/encodings.pkl", "wb") as f:
    pickle.dump({"user_ids": user_ids, "movie_ids": movie_ids}, f)


class RatingsDataset(Dataset):
    def __init__(self, df):
        self.users = torch.tensor(df["user_idx"].values, dtype=torch.long)
        self.items = torch.tensor(df["movie_idx"].values, dtype=torch.long)
        self.ratings = torch.tensor(df["rating_norm"].values, dtype=torch.float)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]


train_df, val_df = train_test_split(ratings, test_size=0.1, random_state=42)
train_loader = DataLoader(RatingsDataset(train_df), batch_size=1024, shuffle=True)
val_loader = DataLoader(RatingsDataset(val_df), batch_size=1024)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

model = NCF(num_users, num_items, embedding_dim=64, layers=[128, 64, 32]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

EPOCHS = 20
PATIENCE = 3  # stop if val loss plateaus

best_val_loss = float("inf")
epochs_no_improve = 0

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for users, items, ratings_batch in train_loader:
        users, items, ratings_batch = users.to(device), items.to(device), ratings_batch.to(device)

        optimizer.zero_grad()
        preds = model(users, items)
        loss = criterion(preds, ratings_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for users, items, ratings_batch in val_loader:
            users, items, ratings_batch = users.to(device), items.to(device), ratings_batch.to(device)
            preds = model(users, items)
            val_loss += criterion(preds, ratings_batch).item()

    avg_val = val_loss / len(val_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {avg_val:.4f}")

    if avg_val < best_val_loss:
        best_val_loss = avg_val
        epochs_no_improve = 0
        torch.save({
            "model_state": model.state_dict(),
            "num_users": num_users,
            "num_items": num_items,
        }, "data/model.pt")
        print(f"  -> Best model saved (val loss: {best_val_loss:.4f})")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1} — no improvement for {PATIENCE} epochs")
            break

print(f"Training complete. Best val loss: {best_val_loss:.4f}")
