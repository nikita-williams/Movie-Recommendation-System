# ENSF 612 Group 2 Course Project - Movie Recommendation System

## ENSF 612 Group 2 - Praise

    # Setup
    pip install -r requirements.txt

### Data Pipeline

    python test_pipeline.py     # sanity checks on fake data, no download needed
    python download_data.py     # downloads and verifies MovieLens 25M into data/raw
    python preprocess.py --ratings data/raw/ml-25m/ratings.csv --movies data/raw/ml-25m/movies.csv --out data/processed

- `preprocess.py`: time split, positive ratings, nested 10/30/100% user groups, movie x user tables, evaluation users
- `minhash.py`: our own MinHash (`minhash_signatures`), used by the LSH step
- `FORMAT.md`: the files in `data/processed/` for easy understanding

`data/` is git-ignored because GroupLens does not allow redistributing the dataset.
