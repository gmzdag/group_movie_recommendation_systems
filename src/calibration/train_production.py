"""
Production Training Script
--------------------------
Run this script to RETRAIN the models on ALL available data.
It clears the cache, computes new similarity matrices, and saves them.
"""
import os
import sys
import shutil

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils.model_utils import quick_setup
from src.utils.persistence import CACHE_DIR, PersistenceManager
from src.experiments.evaluate_group_system import run_weight_optimization
import json

def train_production():
    print("="*60)
    print("PRODUCTION MODEL TRAINING")
    print("="*60)
    
    # 0. Clear Cache
    if os.path.exists(CACHE_DIR):
        print(f"Cleaning cache at {CACHE_DIR}...")
        shutil.rmtree(CACHE_DIR)
        
    # 1. OPTIMIZATION STEP: Calculate Dynamic Weights
    print("\n[STEP 1/2] Optimizing Model Weights...")
    try:
        weights = run_weight_optimization()
        # Save weights specifically to json (separate from pickle)
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(os.path.join(CACHE_DIR, "model_weights.json"), 'w') as f:
            json.dump(weights, f, indent=4)
        print(f"✅ Weights Saved: {weights}")
    except Exception as e:
        print(f"⚠️ Weight optimization failed: {e}. Using defaults.")
        
    # 2. TRAINING STEP: Train on Full Data
    print("\n[STEP 2/2] Training Final Models on ALL Data...")
    try:
        # Load everything (No sample_size limit)
        models = quick_setup(
            recent_only=False, 
            C=1.0 # Best parameter from experiments
        )
        print("\n✅ Training Complete. Models are cached.")
        print(f"Artifacts saved in: {CACHE_DIR}")
        
    except Exception as e:
        print(f"\n❌ Training Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    train_production()
