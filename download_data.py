"""
Run this once to download MovieLens Latest Small dataset.
Usage: python download_data.py
~9,700 movies, 100K ratings, covers up to 2018.
"""
import os
import zipfile
import requests

URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
os.makedirs("data", exist_ok=True)

print("Downloading MovieLens Latest Small...")
r = requests.get(URL, stream=True)
with open("data/ml-latest-small.zip", "wb") as f:
    for chunk in r.iter_content(chunk_size=8192):
        f.write(chunk)

print("Extracting...")
with zipfile.ZipFile("data/ml-latest-small.zip", "r") as z:
    z.extractall("data/")

print("Done. Files saved to data/ml-latest-small/")
