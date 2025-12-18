
import pandas as pd
import numpy as np


def temporal_train_validation_test_split(ratings, train_ratio=0.7, valid_ratio=0.15, min_ratings=5):
    """
    Splits the ratings dataframe into train, validation, and test sets based on timestamp PER USER.
    Ensures that every user in validation/test has some history in train.
    
    Args:
        ratings (pd.DataFrame): DataFrame containing 'userId', 'movieId', 'rating', 'timestamp'.
        train_ratio (float): Proportion of data for training.
        valid_ratio (float): Proportion of data for validation.
        min_ratings (int): Users with fewer than this many ratings are excluded from the split 
                           (or put entirely in train, but usually excluded from eval to avoid noise).
        
    Returns:
        train (pd.DataFrame)
        validation (pd.DataFrame)
        test (pd.DataFrame)
    """
    # Filter users with enough ratings
    user_counts = ratings['userId'].value_counts()
    valid_users = user_counts[user_counts >= min_ratings].index
    filtered_ratings = ratings[ratings['userId'].isin(valid_users)].copy()
    
    # Sort by user and timestamp
    filtered_ratings = filtered_ratings.sort_values(by=['userId', 'timestamp'])
    
    # Define a helper to split a single user's group
    def split_user_group(group):
        n = len(group)
        if n < min_ratings: 
            # Fallback if somehow a small group gets here
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            
        train_end = int(n * train_ratio)
        valid_end = int(n * (train_ratio + valid_ratio))
        
        # Ensure at least 1 item in train if we are to test
        if train_end == 0 and n > 0:
            train_end = 1
            
        # Ensure splits don't overlap boundaries due to small N
        if valid_end < train_end:
            valid_end = train_end
            
        t = group.iloc[:train_end]
        v = group.iloc[train_end:valid_end]
        s = group.iloc[valid_end:]
        
        # Add a column to identify split (optimization for concat)
        t = t.assign(split='train')
        v = v.assign(split='valid')
        s = s.assign(split='test')
        
        return pd.concat([t, v, s])

    # Apply split per user
    # Note: groupby().apply() can be slow on large datasets. 
    # For 100k ratings it's fine. For 20M it needs optimization.
    print("Grouping by user and splitting... this might take a moment.")
    
    # Faster approach than apply for simple splits:
    # Calculate indices
    # However, let's stick to a readable loop or apply for robustness first
    
    train_list = []
    valid_list = []
    test_list = []
    
    for user_id, group in filtered_ratings.groupby('userId'):
        # group is already sorted by timestamp due to earlier sort
        n = len(group)
        train_end = int(n * train_ratio)
        valid_end = int(n * (train_ratio + valid_ratio))
        
        # Enhancements for very small users (though min_ratings handles most)
        if train_end == 0 and n > 0: train_end = 1
        if valid_end < train_end: valid_end = train_end
        
        train_list.append(group.iloc[:train_end])
        valid_list.append(group.iloc[train_end:valid_end])
        test_list.append(group.iloc[valid_end:])
        
    train = pd.concat(train_list)
    validation = pd.concat(valid_list)
    test = pd.concat(test_list)
    
    return train, validation, test

