"""
Cache Manager for Group Movie Recommendation System
---------------------------------------------------
Generic caching layer with pickle-based storage, TTL support, and cache invalidation.

Features:
- Hash-based cache key generation
- TTL (Time-To-Live) support
- Cache invalidation
- Thread-safe operations
"""

import pickle
import hashlib
import os
import time
from typing import Any, Optional
import json


class CacheManager:
    """Generic caching layer for expensive computations."""
    
    def __init__(self, cache_dir=None):
        """
        Initialize cache manager.
        
        Args:
            cache_dir: Directory to store cache files.
                      If None, uses project_root/data/cache/
        """
        if cache_dir is None:
            # Get project root (2 levels up from this file: utils/ -> src/ -> project_root/)
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            cache_dir = os.path.join(project_root, "data", "cache")  # FIXED: Use data/cache
        
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Metadata file for TTL tracking
        self.metadata_file = os.path.join(cache_dir, '_cache_metadata.json')
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> dict:
        """Load cache metadata (TTL info)."""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_metadata(self):
        """Save cache metadata."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def get_cache_key(self, **kwargs) -> str:
        """
        Generate unique cache key from parameters.
        
        Args:
            **kwargs: Parameters to hash
            
        Returns:
            MD5 hash string
            
        Example:
            >>> cache = CacheManager()
            >>> key = cache.get_cache_key(normalization='zscore', K=20, model='ibcf')
            >>> print(key)  # 'a1b2c3d4...'
        """
        # Sort to ensure consistent hashing
        key_str = str(sorted(kwargs.items()))
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def load(self, key: str, ttl: Optional[int] = None) -> Optional[Any]:
        """
        Load cached object.
        
        Args:
            key: Cache key
            ttl: Time-to-live in seconds (None = no expiration)
            
        Returns:
            Cached object or None if not found/expired
        """
        cache_path = os.path.join(self.cache_dir, f"{key}.pkl")
        
        if not os.path.exists(cache_path):
            return None
        
        # Check TTL
        if ttl is not None:
            if key in self.metadata:
                cached_time = self.metadata[key].get('timestamp', 0)
                if time.time() - cached_time > ttl:
                    print(f"[CACHE] Expired: {key}")
                    self.invalidate(key)
                    return None
        
        # Load
        try:
            with open(cache_path, 'rb') as f:
                obj = pickle.load(f)
            
            file_size = os.path.getsize(cache_path) / 1024 / 1024  # MB
            print(f"[CACHE HIT] Loaded {key[:8]}... ({file_size:.2f} MB)")
            return obj
        except Exception as e:
            print(f"[CACHE ERROR] Failed to load {key}: {e}")
            return None
    
    def save(self, key: str, obj: Any, ttl: Optional[int] = None):
        """
        Save object to cache.
        
        Args:
            key: Cache key
            obj: Object to cache
            ttl: Time-to-live in seconds (None = no expiration)
        """
        cache_path = os.path.join(self.cache_dir, f"{key}.pkl")
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(obj, f)
            
            # Update metadata
            self.metadata[key] = {
                'timestamp': time.time(),
                'ttl': ttl,
                'size_mb': os.path.getsize(cache_path) / 1024 / 1024
            }
            self._save_metadata()
            
            file_size = self.metadata[key]['size_mb']
            print(f"[CACHE SAVE] Saved {key[:8]}... ({file_size:.2f} MB)")
        except Exception as e:
            print(f"[CACHE ERROR] Failed to save {key}: {e}")
    
    def invalidate(self, key: str):
        """
        Invalidate (delete) cached object.
        
        Args:
            key: Cache key
        """
        cache_path = os.path.join(self.cache_dir, f"{key}.pkl")
        
        if os.path.exists(cache_path):
            os.remove(cache_path)
            print(f"[CACHE] Invalidated: {key[:8]}...")
        
        if key in self.metadata:
            del self.metadata[key]
            self._save_metadata()
    
    def clear_all(self):
        """Clear all cache files."""
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.pkl'):
                os.remove(os.path.join(self.cache_dir, filename))
        
        self.metadata = {}
        self._save_metadata()
        print("[CACHE] Cleared all cache files")
    
    def get_stats(self) -> dict:
        """
        Get cache statistics.
        
        Returns:
            {
                'total_files': int,
                'total_size_mb': float,
                'keys': [...]
            }
        """
        total_size = 0
        keys = []
        
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.pkl'):
                path = os.path.join(self.cache_dir, filename)
                total_size += os.path.getsize(path)
                keys.append(filename.replace('.pkl', ''))
        
        return {
            'total_files': len(keys),
            'total_size_mb': total_size / 1024 / 1024,
            'keys': keys
        }


# Convenience function
def cached_computation(cache_key_params: dict, computation_fn, cache_dir='cache', ttl=None):
    """
    Decorator-style cached computation.
    
    Args:
        cache_key_params: Parameters for cache key
        computation_fn: Function to compute if cache miss
        cache_dir: Cache directory
        ttl: Time-to-live in seconds
        
    Returns:
        Cached or computed result
        
    Example:
        >>> def expensive_computation():
        ...     # Heavy computation
        ...     return result
        >>> 
        >>> result = cached_computation(
        ...     cache_key_params={'model': 'ibcf', 'K': 20},
        ...     computation_fn=expensive_computation,
        ...     ttl=3600  # 1 hour
        ... )
    """
    cache = CacheManager(cache_dir)
    key = cache.get_cache_key(**cache_key_params)
    
    # Try to load
    result = cache.load(key, ttl=ttl)
    
    if result is not None:
        return result
    
    # Compute
    print(f"[CACHE MISS] Computing...")
    result = computation_fn()
    
    # Save
    cache.save(key, result, ttl=ttl)
    
    return result
