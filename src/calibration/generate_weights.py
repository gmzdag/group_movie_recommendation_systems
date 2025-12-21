"""
Generate Model Weights
-----------------------
Calculates optimal model weights based on target user performance.
Saves weights to cache/model_weights.json
"""

import os
import sys
import json

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils.persistence import CACHE_DIR
from src.calibration.weight_calculator import calculate_production_weights

# Import config
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from evaluation_config import OFFLINE_EVAL_CONFIG


def generate_weights():
    """Generate and save production weights."""
    print("="*60)
    print("GENERATING PRODUCTION WEIGHTS")
    print("="*60)
    
    try:
        # Calculate weights using centralized calculator
        weights = calculate_production_weights(OFFLINE_EVAL_CONFIG)
        
        # Ensure directory exists
        os.makedirs(CACHE_DIR, exist_ok=True)
        
        # Save weights
        path = os.path.join(CACHE_DIR, "model_weights.json")
        with open(path, 'w') as f:
            json.dump(weights, f, indent=4)
            
        print(f"\n✅ Weights Saved Successfully to: {path}")
        print(f"Weights: {weights}")
        
    except Exception as e:
        print(f"\n❌ Weight Generation Failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    generate_weights()
