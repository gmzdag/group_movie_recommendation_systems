import csv
import os

USERS_FILE = "data/users.csv"
RATINGS_FILE = "data/ratings.csv"


def load_users():
    """Loads all users from users.csv."""
    if not os.path.exists(USERS_FILE):
        return []

    with open(USERS_FILE, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def get_or_create_userid(letterboxd_username: str) -> int:
    """
    Returns existing userId if username exists.
    Otherwise creates a new userId and appends it to users.csv.
    """

    users = load_users()

    # 1) If user already exists → return userId
    for u in users:
        if u["letterboxd_username"].lower() == letterboxd_username.lower():
            return int(u["user_id"])

    # 2) Otherwise → create new userId
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
