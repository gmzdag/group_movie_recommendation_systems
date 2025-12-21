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
    
    # Convenience methods for specific artifacts
    @staticmethod
    def save_item_neighbors(neighbors):
        """Save item neighbors."""
        PersistenceManager.save(neighbors, "item_neighbors.pkl")
    
    @staticmethod
    def save_user_neighbors(neighbors):
        """Save user neighbors."""
        PersistenceManager.save(neighbors, "user_neighbors.pkl")
    
    @staticmethod
    def save_cf_matrix(matrix):
        """Save CF matrix."""
        PersistenceManager.save(matrix, "cf_matrix.pkl")
    
    @staticmethod
    def save_norm_matrix(matrix):
        """Save normalized matrix."""
        PersistenceManager.save(matrix, "norm_matrix.pkl")
    
    @staticmethod
    def save_cbf_model(model):
        """Save CBF model."""
        PersistenceManager.save(model, "cbf_model.pkl")
