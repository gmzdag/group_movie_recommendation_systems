
"""
Movie Duplicate Cleanup Tool
----------------------------
Identifies and resolves duplicate movies in the dataset based on normalized titles.
Logic:
1. Normalize titles (lowercase, remove accents).
2. Group movies by normalized title.
3. Keep the one with the lowest ID (Canonical).
4. Remap all ratings from Duplicate IDs to Canonical ID.
5. Save cleaned movies and ratings files.
"""

import pandas as pd
import unicodedata
import os
import sys

# Paths (Relative to project root assumed)
MOVIES_PATH = "data/movies_tmdb.csv"
RATINGS_PATH = "data/ratings.csv"
REPORT_PATH = "data/cleanup_report.txt"

import re

def parse_title_year(title):
    """
    Returns (normalized_base_title, year)
    """
    try:
        if pd.isna(title): return ("", None)
        t = str(title).lower().strip()
        t = unicodedata.normalize('NFKD', t).encode('ASCII', 'ignore').decode('utf-8')
        
        # Extract Year
        match = re.search(r'\((\d{4})\)', t)
        year = match.group(1) if match else None
        
        # Remove Year from title for base comparison
        t = re.sub(r'\(\d{4}\)', '', t)
        
        # Remove noise (aka)
        t = re.sub(r'\(a\.k\.a\..*?\)', '', t)
        
        # Remove punctuation/spaces
        t = re.sub(r'[^\w\s]', '', t)
        t = " ".join(t.split())
        
        return t, year
    except Exception:
        return str(title), None

def cleanup_duplicates():
    if not os.path.exists(MOVIES_PATH) or not os.path.exists(RATINGS_PATH):
        print(f"Error: Data files not found.")
        return

    print("Loading data...")
    movies = pd.read_csv(MOVIES_PATH)
    ratings = pd.read_csv(RATINGS_PATH)
    
    initial_movies = len(movies)
    initial_ratings = len(ratings)
    
    # 1. Identify Duplicates
    print("Identifying duplicates (Title + Year Match)...")
    
    # helper wrapper
    def get_key(row):
        return parse_title_year(row['title'])
    
    movies['dedup_key'] = movies.apply(get_key, axis=1)
    
    movies.sort_values('movieId', inplace=True)
    
    canonical_map = {} # { (title, year): canon_id }
    id_map = {}        # { dup_id: canon_id }
    duplicates_info = [] 
    
    for _, row in movies.iterrows():
        key = row['dedup_key']
        mid = row['movieId']
        title_base, year = key
        
        if not title_base: continue 
        
        # Key includes YEAR, so distinct years won't merge
        if key not in canonical_map:
            canonical_map[key] = mid
        else:
            canon_id = canonical_map[key]
            if mid != canon_id:
                id_map[mid] = canon_id
                canon_title = movies[movies['movieId'] == canon_id]['title'].values[0]
                duplicates_info.append((mid, row['title'], canon_id, canon_title))
                
    if not id_map:
        print("✅ No duplicates found.")
        # Write Report for clean state
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            f.write("DATA CLEANUP REPORT\n")
            f.write("===================\n")
            f.write(f"Movies Checked: {len(movies)}\n")
            f.write(f"Ratings Checked: {len(ratings)}\n\n")
            f.write("STATUS: CLEAN. No duplicates found.\n")
        print(f" - Report saved to: {REPORT_PATH}")
        return

    print(f"Found {len(id_map)} duplicates to resolve.")
    
    # 2. Update Ratings
    print("Remapping ratings...")
    # Replace duplicate IDs with canonical IDs
    ratings['movieId'] = ratings['movieId'].replace(id_map)
    
    # Merge entries if a user rated both versions
    # We strip duplicates by taking the mean of the ratings (or just the latest, but mean is safer for mixed opinions)
    ratings = ratings.groupby(['userId', 'movieId'], as_index=False)['rating'].mean()
    
    # 3. Clean Movies
    print("Pruning movies file...")
    # Keep only rows where movieId is a canonical ID
    movies_clean = movies[movies['movieId'].isin(canonical_map.values())].drop(columns=['dedup_key'])
    
    # 4. Save
    print("Saving cleaned files...")
    movies_clean.to_csv(MOVIES_PATH, index=False)
    ratings.to_csv(RATINGS_PATH, index=False)
    
    # 5. Report
    print("Generating report...")
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("DATA CLEANUP REPORT\n")
        f.write("===================\n")
        f.write(f"Movies Reduced: {initial_movies} -> {len(movies_clean)}\n")
        f.write(f"Ratings Consolid.: {initial_ratings} -> {len(ratings)}\n\n")
        f.write("RESOLVED DUPLICATES:\n")
        f.write(f"{'DupID':<8} | {'Duplicate Title':<40} -> {'CanonID':<8} | {'Canonical Title'}\n")
        f.write("-" * 90 + "\n")
        for d in duplicates_info:
            f.write(f"{d[0]:<8} | {d[1][:38]:<40} -> {d[2]:<8} | {d[3]}\n")

    print(f"\n🎉 Cleanup Complete!")
    print(f" - Movies removed: {len(id_map)}")
    print(f" - Report saved to: {REPORT_PATH}")

if __name__ == "__main__":
    cleanup_duplicates()
