
import sys
import os
import pandas as pd

# Add parent directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from recommender.data_loader import load_ratings, DATA_DIR
from recommender.data_splitter import temporal_train_validation_test_split

def create_splits():
    print("Loading ratings...")
    ratings = load_ratings()
    
    print("Splitting data temporally...")
    train, valid, test = temporal_train_validation_test_split(ratings, train_ratio=0.7, valid_ratio=0.15)
    
    print(f"Train size: {len(train)}")
    print(f"Validation size: {len(valid)}")
    print(f"Test size: {len(test)}")
    
    splits_dir = os.path.join(DATA_DIR, "splits")
    os.makedirs(splits_dir, exist_ok=True)
    
    train.to_csv(os.path.join(splits_dir, "train.csv"), index=False)
    valid.to_csv(os.path.join(splits_dir, "validation.csv"), index=False)
    test.to_csv(os.path.join(splits_dir, "test.csv"), index=False)
    
    print(f"Splits saved to {splits_dir}")

if __name__ == "__main__":
    create_splits()
