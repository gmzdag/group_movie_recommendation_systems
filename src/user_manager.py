import csv
import os
import time
import re
import zipfile
import io
import sys

# Define Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

USERS_FILE = os.path.join(DATA_DIR, "users.csv")
RATINGS_FILE = os.path.join(DATA_DIR, "ratings.csv")
WATCHLIST_FILE = os.path.join(DATA_DIR, "watchlist.csv")
MOVIES_FILE = os.path.join(DATA_DIR, "movies_tmdb.csv")

# ------------------------------------------------------
# User Check / Creation Logic
# ------------------------------------------------------

def load_users():
    """Loads all users from users.csv."""
    if not os.path.exists(USERS_FILE):
        return []

    with open(USERS_FILE, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def get_userid_if_exists(letterboxd_username: str) -> int | None:
    """
    Returns existing userId if username exists.
    Returns None if user doesn't exist.
    Does NOT create a new user.
    """
    users = load_users()
    
    for u in users:
        if u["letterboxd_username"].lower() == letterboxd_username.lower():
            return int(u["user_id"])
    
    return None


def create_user(letterboxd_username: str) -> int:
    """
    Creates a new user and adds them to users.csv.
    Returns the new user_id.
    """
    users = load_users()
    
    # Check if user already exists
    for u in users:
        if u["letterboxd_username"].lower() == letterboxd_username.lower():
            return int(u["user_id"])
    
    # Create new userId
    new_id = 1 if not users else max(int(u["user_id"]) for u in users) + 1
    
    write_header = not os.path.exists(USERS_FILE) or os.path.getsize(USERS_FILE) == 0
    
    with open(USERS_FILE, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["user_id", "letterboxd_username"])
        writer.writerow([new_id, letterboxd_username])
    
    print(f"[INFO] Created new user → userId={new_id}, username={letterboxd_username}")
    return new_id


def user_has_ratings(user_id: int) -> bool:
    """Checks if a user already appears inside ratings.csv."""
    if not os.path.exists(RATINGS_FILE):
        return False

    with open(RATINGS_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return False

        for row in reader:
            if int(row["userId"]) == user_id:
                return True

    return False


def get_user_rated_movie_ids(user_id: int) -> set:
    """Returns a set of movieIds that the user has already rated."""
    if not os.path.exists(RATINGS_FILE):
        return set()

    rated_movies = set()
    with open(RATINGS_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["userId"]) == user_id:
                rated_movies.add(int(row["movieId"]))
    
    return rated_movies


def get_user_watchlist_movie_ids(user_id: int) -> set:
    """Returns a set of movieIds that are already in the user's watchlist."""
    if not os.path.exists(WATCHLIST_FILE):
        return set()

    watchlist_movies = set()
    with open(WATCHLIST_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
             return set()
             
        for row in reader:
            if int(row["userId"]) == user_id:
                watchlist_movies.add(int(row["movieId"]))
    
    return watchlist_movies


# ------------------------------------------------------
# Import / Export Logic
# ------------------------------------------------------

def load_movie_mapping():
    """
    Creates a mapping from (clean_title, year) → movieId
    Title example in movies_tmdb.csv:
        "Toy Story (1995)"
    Final key format:
        "toy story_1995"
    """
    mapping = {}
    if not os.path.exists(MOVIES_FILE):
         print(f"[WARN] Movies file not found at {MOVIES_FILE}")
         return mapping
         
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


def import_letterboxd_export(letterboxd_username: str, export_path: str, update: bool = False) -> int:
    """
    Processes a Letterboxd export (CSV or ZIP) and adds the ratings AND/OR watchlist.
    Supports partial imports: only ratings.csv, only watchlist.csv, or both.
    Creates the user in users.csv ONLY if data is successfully imported.
    """
    # First check if user exists
    user_id = get_userid_if_exists(letterboxd_username)
    
    # If not updating and user exists with data, skip
    if user_id is not None and user_has_ratings(user_id) and not update:
        print(f"[INFO] User already has ratings → userId={user_id}")
        print("[INFO] No import required.\n")
        return user_id

    movie_map = load_movie_mapping()
    
    # Track if we successfully imported any data
    total_imported = 0

    # --- RATINGS IMPORT ---
    def import_ratings(source_rows):
        nonlocal total_imported
        
        if user_id is not None and update:
            print(f"[INFO] Updating ratings for user {user_id}...")
            existing_rated = get_user_rated_movie_ids(user_id)
            current_user_id = user_id
        else:
            print(f"[INFO] Preparing to import ratings for new user: {letterboxd_username}")
            existing_rated = set()
            current_user_id = None  # Will be created after successful import

        rows_to_write = []
        for row in source_rows:
            title = row["Name"].strip().lower()
            year = row["Year"].strip()
            rating = float(row["Rating"])
            key = f"{title}_{year}"

            if key not in movie_map:
                continue

            movie_id = movie_map[key]
            if movie_id in existing_rated:
                continue

            timestamp = int(time.time())
            rows_to_write.append([current_user_id, movie_id, rating, timestamp])
            existing_rated.add(movie_id)

        if rows_to_write:
            # Create user if this is a new import
            if current_user_id is None:
                current_user_id = create_user(letterboxd_username)
                # Update all rows with the new user_id
                rows_to_write = [[current_user_id, row[1], row[2], row[3]] for row in rows_to_write]
            
            write_header = not os.path.exists(RATINGS_FILE) or os.path.getsize(RATINGS_FILE) == 0
            with open(RATINGS_FILE, "a", encoding="utf-8", newline="") as out:
                writer = csv.writer(out)
                if write_header:
                    writer.writerow(["userId", "movieId", "rating", "timestamp"])
                writer.writerows(rows_to_write)
            
            total_imported += len(rows_to_write)
            print(f"[SUCCESS] Ratings processed. Added {len(rows_to_write)} new ratings.")
        
        return current_user_id

    # --- WATCHLIST IMPORT ---
    def import_watchlist(source_rows, current_user_id):
        nonlocal total_imported
        
        # If user doesn't exist yet, create them (watchlist-only scenario)
        if current_user_id is None:
            print(f"[INFO] Creating user for watchlist-only import: {letterboxd_username}")
            current_user_id = create_user(letterboxd_username)
        
        print(f"[INFO] Processing watchlist for user {current_user_id}...")
        existing_watchlist = get_user_watchlist_movie_ids(current_user_id)
        
        rows_to_write = []
        for row in source_rows:
            title = row["Name"].strip().lower()
            year = row["Year"].strip()
            key = f"{title}_{year}"

            if key not in movie_map:
                continue

            movie_id = movie_map[key]
            if movie_id in existing_watchlist:
                continue

            rows_to_write.append([current_user_id, movie_id])
            existing_watchlist.add(movie_id)

        if rows_to_write:
            write_header = not os.path.exists(WATCHLIST_FILE) or os.path.getsize(WATCHLIST_FILE) == 0
            with open(WATCHLIST_FILE, "a", encoding="utf-8", newline="") as out:
                writer = csv.writer(out)
                if write_header:
                    writer.writerow(["userId", "movieId"])
                writer.writerows(rows_to_write)
            
            total_imported += len(rows_to_write)
            print(f"[SUCCESS] Watchlist processed. Added {len(rows_to_write)} items.")
        
        return current_user_id
    
    # --- CSV TYPE DETECTION ---
    def detect_csv_type(csv_rows):
        """
        Detects if a CSV is ratings or watchlist based on column headers.
        Returns: 'ratings', 'watchlist', or None
        """
        if not csv_rows:
            return None
        
        # Check first row (headers are already parsed by DictReader)
        first_row = csv_rows[0]
        columns = set(first_row.keys())
        
        # Ratings CSV has 'Rating' column
        if 'Rating' in columns:
            return 'ratings'
        # Watchlist CSV doesn't have 'Rating' but has 'Name' and 'Year'
        elif 'Name' in columns and 'Year' in columns:
            return 'watchlist'
        
        return None

    # --- FILE PROCESSING ---
    final_user_id = user_id
    
    if export_path.lower().endswith(".zip"):
        print(f"[INFO] Detected ZIP file: {export_path}")
        try:
            with zipfile.ZipFile(export_path, 'r') as z:
                ratings_file = None
                watchlist_file = None
                
                # List all files in ZIP for debugging
                all_files = z.namelist()
                print(f"[DEBUG] Files in ZIP: {all_files}")
                
                for name in all_files:
                    name_lower = name.lower()
                    if 'ratings.csv' in name_lower:
                        ratings_file = name
                        print(f"[INFO] Found ratings file: {ratings_file}")
                    elif 'watchlist.csv' in name_lower:
                        watchlist_file = name
                        print(f"[INFO] Found watchlist file: {watchlist_file}")
                
                if not ratings_file and not watchlist_file:
                    print("[ERROR] No ratings.csv or watchlist.csv found in ZIP")
                    raise Exception("No valid data files found in ZIP")
                
                # Process ratings first (if exists)
                if ratings_file:
                    print(f"[INFO] Processing {ratings_file}...")
                    try:
                        with z.open(ratings_file) as f:
                            content = f.read().decode('utf-8-sig')  # Handle BOM
                            rows = list(csv.DictReader(io.StringIO(content)))
                            print(f"[DEBUG] Found {len(rows)} rows in ratings.csv")
                            final_user_id = import_ratings(rows)
                    except Exception as e:
                        print(f"[ERROR] Failed to process ratings: {e}")
                        raise
                
                # Process watchlist (if exists)
                if watchlist_file:
                    print(f"[INFO] Processing {watchlist_file}...")
                    try:
                        with z.open(watchlist_file) as f:
                            content = f.read().decode('utf-8-sig')  # Handle BOM
                            rows = list(csv.DictReader(io.StringIO(content)))
                            print(f"[DEBUG] Found {len(rows)} rows in watchlist.csv")
                            final_user_id = import_watchlist(rows, final_user_id)
                    except Exception as e:
                        print(f"[ERROR] Failed to process watchlist: {e}")
                        # Don't raise here if we already imported ratings
                        if not ratings_file:
                            raise
                        
        except zipfile.BadZipFile as e:
            print(f"[ERROR] Invalid ZIP file: {e}")
            raise Exception("Invalid ZIP file format")
        except Exception as e:
            print(f"[ERROR] Failed to process ZIP file: {e}")
            raise
    else:
        # Single CSV file - detect type automatically
        print(f"[INFO] Detected single CSV file: {export_path}")
        try:
            with open(export_path, "r", encoding="utf-8-sig") as f:
                rows = list(csv.DictReader(f))
                print(f"[DEBUG] Found {len(rows)} rows in CSV")
                
                # Detect CSV type
                csv_type = detect_csv_type(rows)
                
                if csv_type == 'ratings':
                    print("[INFO] Detected as ratings.csv (contains 'Rating' column)")
                    final_user_id = import_ratings(rows)
                elif csv_type == 'watchlist':
                    print("[INFO] Detected as watchlist.csv (no 'Rating' column)")
                    final_user_id = import_watchlist(rows, final_user_id)
                else:
                    raise Exception("Could not determine CSV type. Please ensure it's a valid Letterboxd export.")
                    
        except Exception as e:
            print(f"[ERROR] Failed to read CSV file: {e}")
            raise
    
    if total_imported == 0:
        raise Exception("No data was imported. Please check your export file.")
    
    print(f"[SUCCESS] Total items imported: {total_imported}")
    return final_user_id


# ------------------------------------------------------
# Cache Management Logic
# ------------------------------------------------------

def update_user_recommender_cache(user_id: int):
    """
    Incrementally updates the recommendation cache for the specific user.
    Uses lazy imports to prevent circular dependencies.
    """
    print("[INFO] Updating recommendation cache incrementally...")
    try:
        # Lazy imports to avoid slow startup and circular deps
        # We need to make sure 'src' is in path if not already
        src_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if src_path not in sys.path:
            sys.path.insert(0, src_path)

        from src.recommender.data_loader import load_all_data
        from src.recommender.UBCF.similarity_user import pearson_shrink
        from src.recommender.UBCF.neighbors_user import load_neighbors, update_neighbors_for_new_user, save_neighbors
        
        # Load fresh data
        print("   * Refetching data...")
        _, _, _, R_cf, _ = load_all_data()
        
        # Define cache path
        # Assuming cache is in the project root/cache
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # src/..
        # Check if we are in group_movie_recommendation_systems/src or similar
        # Since this file is in src/user_manager.py, project_root is group_movie_recommendation_systems
        
        cache_path = os.path.join(os.path.dirname(DATA_DIR), "cache", "user_neighbors_ubcf.pkl")
        
        if os.path.exists(cache_path):
                neighbors = load_neighbors(cache_path)
                
                # Define the sim function
                sim_fn = lambda u, v: pearson_shrink(u, v, MIN_OVERLAP=2, LAMBDA=20)
                
                print(f"   * Recalculating neighbors for user {user_id}...")
                update_neighbors_for_new_user(cache_path, neighbors, R_cf, sim_fn, user_id, K=75) 
                
                print(f"[SUCCESS] Cache updated for user {user_id}. Other users untouched.")
        else:
                print("[INFO] No existing cache found. A full compute will happen on next run.")
                
    except Exception as e:
        print(f"[ERROR] Could not update cache incrementally: {e}")
        print("Don't worry, the system will recompute it automatically next time if needed.")
