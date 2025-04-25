# auth.py

import os
import json
import hashlib

def load_users():
    """
    Load the user database from a JSON file: {username: hashed_password}.
    """
    if os.path.exists("users.json"):
        with open("users.json", "r") as f:
            return json.load(f)
    return {}

def save_users(users_dict):
    """
    Save the user database to a JSON file.
    """
    with open("users.json", "w") as f:
        json.dump(users_dict, f)

def hash_password(password: str) -> str:
    """
    Return SHA256 hash of the given password.
    (For production, consider salted hashing with bcrypt or argon2.)
    """
    return hashlib.sha256(password.encode("utf-8")).hexdigest()