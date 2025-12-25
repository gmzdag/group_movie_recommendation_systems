import sys
import os

# Add src to path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, src_path)

try:
    import recommender
except ImportError as e:
    print(f"Error importing recommender: {e}")

from recommender.data_loader import load_movies, load_ratings, load_watchlists
from recommender.watchlist.watchlist_recommender import WatchlistRecommender

def print_recommendations(result):
    titles = result["movie_titles"]
    guaranteed = result["guaranteed_picks"]
    recommended = result["recommended_movies"]

    print("\n🔥 Everyone Wants to Watch (Common Watchlist)")
    if guaranteed:
        for mid in guaranteed:
            print(f"- {titles.get(mid, f'Movie {mid}')}")
    else:
        print("- (No common movies found)")

    print("\n🎯 Recommended for Your Group (Score Aggregation)")
    if not recommended:
        print("No recommendations generated.")
        return

    for idx, (mid, score, reason) in enumerate(recommended, start=1):
        title = titles.get(mid, f"Movie {mid}")
        print(f"{idx:2d}. {title:50.50s} ({score:.4f})  → {reason}")

def main():
    # Default Test Group
    group_users = [618, 623] 
    
    # Allow command line args
    if len(sys.argv) > 1:
        try:
            group_users = [int(arg) for arg in sys.argv[1:]]
        except ValueError:
            print("Please provide user IDs as integers.")
            return

    print(f"Running Group Watchlist Rec Demo for Users: {group_users}")
    
    print("[1] Loading Data...")
    movies = load_movies()
    ratings = load_ratings()
    watchlists = load_watchlists()
    
    print("[2] Initializing Recommender (Training Content Model)...")
    recommender = WatchlistRecommender(movies, ratings, watchlists)
    
    print("[3] Generating Recommendations...")
    try:
        result = recommender.recommend(group_users, top_n=10)
        print_recommendations(result)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
