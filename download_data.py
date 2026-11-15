"""Download MovieLens 25M, check it against GroupLens' MD5, and extract ratings.csv and movies.csv.

Usage: python download_data.py
The data may not be redistributed, so data/ stays out of git (see .gitignore).
"""
import hashlib
import os
import sys
import urllib.request
import zipfile

URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
EXPECTED_MD5 = "6b51fb2759a8657d3bfcbfc42b592ada"
RAW = "data/raw"

os.makedirs(RAW, exist_ok=True)
zip_path = f"{RAW}/ml-25m.zip"
if not os.path.exists(zip_path):
    print("downloading...")
    urllib.request.urlretrieve(URL, zip_path)

md5 = hashlib.md5()
with open(zip_path, "rb") as f:
    for block in iter(lambda: f.read(1 << 20), b""):
        md5.update(block)
if md5.hexdigest() != EXPECTED_MD5:
    sys.exit(f"CHECKSUM MISMATCH: got {md5.hexdigest()}, expected {EXPECTED_MD5}. "
             "Delete the zip and retry.")
print("checksum OK")

with zipfile.ZipFile(zip_path) as z:
    for name in ("ml-25m/ratings.csv", "ml-25m/movies.csv"):
        z.extract(name, RAW)
print("ready in", RAW + "/ml-25m")