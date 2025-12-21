from src.user_manager import (
    get_or_create_userid, 
    user_has_ratings, 
    import_letterboxd_export,
    update_user_recommender_cache
)

print("=== MovieLens + Letterboxd Integration ===\n")

username = input("Enter your Letterboxd username: ").strip()

# Assign or retrieve userId
user_id = get_or_create_userid(username)

# Check if user wants to update if they already exist
if user_has_ratings(user_id):
    choice = input(f"[INFO] User {username} (ID: {user_id}) already has ratings. Do you want to update/append new ratings? (y/n): ").strip().lower()
    update = (choice == 'y')

    if update:
        print("\nTo update your Letterboxd ratings:")
        print("1. Download your data export from Letterboxd (zip file).")
        path = input("Enter full path to your Letterboxd export file (zip or csv): ").strip().replace('"', '').replace("'", "")
        
        # Import (Update)
        import_letterboxd_export(username, path, update=True)
        
        # Incremental Cache Update
        update_user_recommender_cache(user_id)
        
    else:
        print(f"[INFO] User already exists in ratings → userId={user_id}")
        print("Proceeding to recommendation engine...\n")
else:
    print(f"[INFO] User has no ratings yet (userId={user_id}).")
    print("""
To add your Letterboxd ratings:

1. Go to https://letterboxd.com/settings/data/
2. Scroll to 'Export Your Data'
3. Click 'Export'
4. Download the 'letterboxd-username-date-utc.zip' file
""")
    path = input("Enter full path to your Letterboxd export file (zip or csv): ").strip().replace('"', '').replace("'", "")
    # Initial import (New User)
    import_letterboxd_export(username, path, update=False)

print("Ready for recommendation system!")
