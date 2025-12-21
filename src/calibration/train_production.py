"""
Production Training Script
--------------------------
Complete production training pipeline:
1. Clear cache
2. Calculate optimal model weights
3. Train models on full data
4. Save all artifacts (weights, neighbors, matrices, models)
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
        
    # 1. OPTIMIZATION STEP: Calculate Dynamic Weights & Model Performance
    print("\n[STEP 1/3] Optimizing Model Weights & Calculating Performance...")
    try:
        # Calculate ensemble weights
        weights = run_weight_optimization()
        # Save weights
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(os.path.join(CACHE_DIR, "model_weights.json"), 'w') as f:
            json.dump(weights, f, indent=4)
        print(f"✅ Weights Saved: {weights}")
        
        # Calculate individual model performance for multi-model selection
        from src.calibration.weight_calculator import calculate_model_performance_metrics
        from evaluation_config import OFFLINE_EVAL_CONFIG
        
        performance = calculate_model_performance_metrics(OFFLINE_EVAL_CONFIG)
        print(f"✅ Model Performance Saved")
        
    except Exception as e:
        print(f"⚠️ Weight/Performance optimization failed: {e}. Using defaults.")
        
    # 2. TRAINING STEP: Train on Full Data
    print("\n[STEP 2/3] Training Final Models on ALL Data...")
    try:
        # Load everything (No sample_size limit)
        models = quick_setup(
            recent_only=False, 
            C=1.0  # Best parameter from experiments
        )
        print("✅ Models trained successfully")
        
    except Exception as e:
        print(f"\n❌ Training Failed: {e}")
        import traceback
        traceback.print_exc()
        return
        
    # 3. SAVE ARTIFACTS
    print("\n[STEP 3/3] Saving Model Artifacts...")
    try:
        pm = PersistenceManager()
        
        # Save neighbors
        if 'item_neighbors' in models:
            pm.save_item_neighbors(models['item_neighbors'])
            print("  ✓ Item neighbors saved")
            
        if 'user_neighbors' in models:
            pm.save_user_neighbors(models['user_neighbors'])
            print("  ✓ User neighbors saved")
        
        # Save matrices
        if 'cf_matrix' in models:
            pm.save_cf_matrix(models['cf_matrix'])
            print("  ✓ CF matrix saved")
            
        if 'norm_matrix' in models:
            pm.save_norm_matrix(models['norm_matrix'])
            print("  ✓ Normalized matrix saved")
        
        # Save CBF model
        if 'cbf' in models:
            pm.save_cbf_model(models['cbf'])
            print("  ✓ CBF model saved")
            
        print(f"\n✅ Training Complete. All artifacts saved to: {CACHE_DIR}")
        
    except Exception as e:
        print(f"⚠️ Artifact saving failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    train_production()
