"""
Configuration Manager for Group Movie Recommendation System
-----------------------------------------------------------
Loads and manages configuration from YAML files with environment variable override support.

Features:
- YAML config loading
- Nested value access
- Environment variable overrides
- Default values
"""

import yaml
import os
from typing import Any, Optional


class Config:
    """Configuration manager with YAML support."""
    
    def __init__(self, config_path='config/model_config.yaml'):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to YAML config file
        """
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        if not os.path.exists(self.config_path):
            print(f"[CONFIG WARNING] Config file not found: {self.config_path}")
            print(f"[CONFIG] Using default configuration")
            return self._get_default_config()
        
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"[CONFIG] Loaded configuration from: {self.config_path}")
            return config
        except Exception as e:
            print(f"[CONFIG ERROR] Failed to load config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> dict:
        """Return default configuration."""
        return {
            'models': {
                'hybrid_model_1': {
                    'C': 1.0,
                    'item_k': 20,
                    'normalization': 'zscore'
                },
                'hybrid_model_2': {
                    'user_k': 30,
                    'threshold': 0.5
                },
                'hybrid_model_3': {
                    'watchlist_weight': 0.8
                }
            },
            'evaluation': {
                'num_groups': 50,
                'min_group_size': 2,
                'max_group_size': 4,
                'k_values': [5, 10, 20],
                'ground_truth_threshold': 3.5
            },
            'data': {
                'recent_only': True,
                'recent_count': 50000,
                'train_ratio': 0.7,
                'valid_ratio': 0.15
            },
            'cache': {
                'enabled': True,
                'directory': 'cache',
                'ttl': 86400
            },
            'temporal': {
                'enabled': True,
                'recency_weight': 0.3
            },
            'logging': {
                'level': 'INFO',
                'file': 'logs/recommender.log'
            }
        }
    
    def get(self, *keys, default=None) -> Any:
        """
        Get nested config value.
        
        Args:
            *keys: Nested keys to traverse
            default: Default value if key not found
            
        Returns:
            Config value or default
            
        Example:
            >>> config = Config()
            >>> C = config.get('models', 'hybrid_model_1', 'C')
            >>> print(C)  # 1.0
        """
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        # Check for environment variable override
        env_key = '_'.join([str(k).upper() for k in keys])
        env_value = os.getenv(env_key)
        
        if env_value is not None:
            # Try to convert to appropriate type
            try:
                # Boolean
                if env_value.lower() in ['true', 'false']:
                    return env_value.lower() == 'true'
                # Integer
                if env_value.isdigit():
                    return int(env_value)
                # Float
                return float(env_value)
            except:
                return env_value
        
        return value
    
    def set(self, *keys, value):
        """
        Set nested config value.
        
        Args:
            *keys: Nested keys to traverse
            value: Value to set
            
        Example:
            >>> config = Config()
            >>> config.set('models', 'hybrid_model_1', 'C', value=1.5)
        """
        current = self.config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def save(self, path: Optional[str] = None):
        """
        Save configuration to YAML file.
        
        Args:
            path: Path to save (default: original config_path)
        """
        save_path = path or self.config_path
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
        
        print(f"[CONFIG] Saved configuration to: {save_path}")
    
    def print_config(self):
        """Print current configuration."""
        print("\n" + "="*60)
        print("CURRENT CONFIGURATION")
        print("="*60)
        print(yaml.dump(self.config, default_flow_style=False, indent=2))
        print("="*60 + "\n")


# Global config instance
_config = None

def get_config(config_path='config/model_config.yaml') -> Config:
    """
    Get global config instance (singleton pattern).
    
    Args:
        config_path: Path to config file
        
    Returns:
        Config instance
    """
    global _config
    
    if _config is None:
        _config = Config(config_path)
    
    return _config
