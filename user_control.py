from src.user_manager import get_or_create_userid, user_has_ratings
from src.import_export_user import import_letterboxd_export

print("=== MovieLens + Letterboxd Integration ===\n")

username = input("Enter your Letterboxd username: ").strip()

# Assign or retrieve userId
user_id = get_or_create_userid(username)

# If user already has ratings, skip export
if user_has_ratings(user_id):
    print(f"[INFO] User already exists in ratings → userId={user_id}")
    print("Proceeding to recommendation engine...\n")
else:
    print(f"[INFO] User has no ratings yet (userId={user_id}).")
    print("""
To add your Letterboxd ratings:

1. Go to https://letterboxd.com/settings/data/
2. Scroll to 'Export Your Data'
3. Click 'Export'
4. Download the file named 'ratings.csv'
""")
    path = input("Enter full path to your Letterboxd 'ratings.csv' file: ").strip().replace('"', '').replace("'", "")
    import_letterboxd_export(username, path)

print("Ready for recommendation system!")
