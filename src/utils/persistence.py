import os
import joblib
import time

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "cache", "models")

class PersistenceManager:
    """
    Minimal Persistence Manager using joblib.
    """
    
    @staticmethod
    def save(obj, filename):
        """Save object to cache directory."""
        os.makedirs(CACHE_DIR, exist_ok=True)
        path = os.path.join(CACHE_DIR, filename)
        joblib.dump(obj, path, compress=3)
        print(f"saved: {filename}")
        
    @staticmethod
    def load(filename):
        """Load object from cache directory. Returns None if missing."""
        path = os.path.join(CACHE_DIR, filename)
        if os.path.exists(path):
            start = time.time()
            obj = joblib.load(path)
            print(f"loaded: {filename} ({time.time() - start:.2f}s)")
            return obj
        return None

    @staticmethod
    def exists(filename):
        return os.path.exists(os.path.join(CACHE_DIR, filename))
