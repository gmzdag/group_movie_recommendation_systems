import csv
import time
import os
import re
from src.user_manager import get_or_create_userid, user_has_ratings

MOVIES_FILE = "data/movies_tmdb.csv"
RATINGS_FILE = "data/ratings.csv"


def load_movie_mapping():
    """
    Creates a mapping from (clean_title, year) → movieId
    Title example in movies_tmdb.csv:
        "Toy Story (1995)"
    This function extracts:
        title_clean = "toy story"
        year = "1995"
    Final key format:
        "toy story_1995"
    """
    mapping = {}

    with open(MOVIES_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            full_title = row["title"].strip()

            # Extract year from "(1995)"
            match = re.search(r"\((\d{4})\)", full_title)
            if not match:
                continue

            year = match.group(1)

            # Clean title by removing "(1995)"
            title_clean = re.sub(r"\(\d{4}\)", "", full_title)
            title_clean = title_clean.strip().lower()

            key = f"{title_clean}_{year}"

            mapping[key] = int(row["movieId"])

    return mapping


def import_letterboxd_export(letterboxd_username: str, export_path: str) -> int:
    """
    Processes a Letterboxd ratings.csv export and adds the ratings
    to MovieLens-style ratings.csv using the user's MovieLens userId.
    """

    # Get or create a userId for this username
    user_id = get_or_create_userid(letterboxd_username)

    # If user already has ratings, skip import
    if user_has_ratings(user_id):
        print(f"[INFO] User already has ratings → userId={user_id}")
        print("[INFO] No import required.\n")
        return user_id

    print(f"[INFO] Importing Letterboxd export for new user → userId={user_id}")

    movie_map = load_movie_mapping()

    with open(export_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        write_header = not os.path.exists(RATINGS_FILE) or os.path.getsize(RATINGS_FILE) == 0

        with open(RATINGS_FILE, "a", encoding="utf-8", newline="") as out:
            writer = csv.writer(out)

            if write_header:
                writer.writerow(["userId", "movieId", "rating", "timestamp"])

            for row in reader:
                title = row["Name"].strip().lower()
                year = row["Year"].strip()
                rating = float(row["Rating"])
                key = f"{title}_{year}"

                if key not in movie_map:
                    print(f"[WARN] No match found → {row['Name']} ({year})")
                    continue

                movie_id = movie_map[key]
                timestamp = int(time.time())

                writer.writerow([user_id, movie_id, rating, timestamp])

    print("[SUCCESS] Ratings imported for user:", user_id)
    return user_id
