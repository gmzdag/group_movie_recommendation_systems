"""
Configuration and Secrets Manager
Loads API keys from environment-specific files with priority:
1. .env.local (highest priority, for local overrides)
2. .env (production secrets, gitignored)
3. .env.dev (development defaults, can be committed)
"""

import os
import json
from typing import Optional, Dict, Any
from pathlib import Path


class ConfigManager:
    """Manages configuration and API keys from multiple sources"""
    
    def __init__(self):
        """Initialize config manager"""
        self.config_data: Dict[str, Any] = {}
        self.environment = os.getenv('ENVIRONMENT', 'development')
        self._load_config()
    
    def _load_config(self):
        """Load configuration from available sources with priority"""
        try:
            from dotenv import load_dotenv
            
            # Get project root directory
            project_root = Path(__file__).parent.parent.parent
            
            # Priority order (highest to lowest):
            # 1. .env.local (local overrides, never committed)
            # 2. .env (production secrets, gitignored)
            # 3. .env.dev (development defaults, can be committed)
            
            env_files = [
                project_root / '.env.dev',      # Lowest priority
                project_root / '.env',          # Medium priority
                project_root / '.env.local',    # Highest priority
            ]
            
            loaded_files = []
            for env_file in env_files:
                if env_file.exists():
                    load_dotenv(env_file, override=True)
                    loaded_files.append(env_file.name)
            
            if loaded_files:
                print(f"[Config] Loaded environment from: {', '.join(loaded_files)}")
            else:
                print(f"[Config] Warning: No .env files found")
                
        except Exception as e:
            print(f"[Config] Warning: Failed to load environment: {e}")
    
    def get_api_key(self, service: str) -> Optional[str]:
        """
        Get API key for a service with priority:
        1. Environment variable
        2. config.json
        3. None
        
        Args:
            service: Service name (gemini, groq, huggingface)
            
        Returns:
            API key or None if not found
        """
        # Priority 1: Environment variables
        env_var_names = {
            'gemini': ['GEMINI_API_KEY', 'GOOGLE_API_KEY'],
            'groq': ['GROQ_API_KEY'],
            'huggingface': ['HUGGINGFACE_TOKEN', 'HF_TOKEN']
        }
        
        if service in env_var_names:
            for env_var in env_var_names[service]:
                value = os.getenv(env_var)
                if value and value not in ['your_api_key_here', f'your_{service}_api_key_here']:
                    return value
        
        # Priority 2: config.json
        if 'api_keys' in self.config_data:
            key = self.config_data['api_keys'].get(service)
            if key and key not in ['your_api_key_here', f'your_{service}_api_key_here']:
                return key
        
        return None
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """
        Get a setting value
        
        Args:
            key: Setting key
            default: Default value if not found
            
        Returns:
            Setting value or default
        """
        if 'settings' in self.config_data:
            return self.config_data['settings'].get(key, default)
        return default
    
    def get_preferred_llm(self) -> str:
        """Get preferred LLM provider"""
        return self.get_setting('preferred_llm', 'gemini')
    
    def is_debug_mode(self) -> bool:
        """Check if debug mode is enabled"""
        return self.get_setting('debug_mode', False)


# Singleton instance
_config_manager: Optional[ConfigManager] = None


def get_config() -> ConfigManager:
    """Get or create the config manager singleton"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


# Convenience functions
def get_api_key(service: str) -> Optional[str]:
    """Get API key for a service"""
    return get_config().get_api_key(service)


def get_setting(key: str, default: Any = None) -> Any:
    """Get a setting value"""
    return get_config().get_setting(key, default)
