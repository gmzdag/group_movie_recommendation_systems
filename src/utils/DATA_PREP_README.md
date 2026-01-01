# Data Preparation Utilities

This directory contains utility scripts for data preparation and maintenance.

## 🧹 Data Cleanup Tool

**File:** `data_cleanup.py`

**Purpose:** Identifies and resolves duplicate movies in the dataset based on normalized titles.

**Usage:**
```bash
# Run from project root
cd group_movie_recommendation_systems
python src/utils/data_cleanup.py
```

**What it does:**
1. Normalizes movie titles (lowercase, removes accents)
2. Groups movies by normalized title + year
3. Keeps the movie with the lowest ID (canonical)
4. Remaps all ratings from duplicate IDs to canonical ID
5. Saves cleaned files and generates a report

**Output:**
- Updates `data/movies_tmdb.csv`
- Updates `data/ratings.csv`
- Creates `data/cleanup_report.txt`

---

## 📋 Watchlist Merger

**File:** `watchlist_merger.py`

**Purpose:** Merges individual watchlist CSV files into a single consolidated file.

**Usage:**
```bash
# Run from project root (where watchlist_*.csv files are located)
cd group_movie_recommendation_systems
python src/utils/watchlist_merger.py
```

**What it does:**
1. Finds all `watchlist_*.csv` files in the current directory
2. Extracts user IDs from filenames (e.g., `watchlist_618.csv` → userId=618)
3. Matches movie titles with `movies_tmdb.csv`
4. Consolidates all watchlists into a single file

**Input:** Individual watchlist files (e.g., `watchlist_618.csv`, `watchlist_623.csv`)

**Output:** `watchlist.csv` with columns: `userId`, `movieId`

---

## 📝 Notes

- These are **one-time data preparation tools**
- Run them only when you need to clean or prepare data
- They are not part of the main application flow
- Always backup your data before running cleanup tools
