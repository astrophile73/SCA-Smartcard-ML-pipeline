import sys
import os
import json
import argparse

# Add path for RSA src
sys.path.append("I:/freelance/Smartcard SCA ML Pipeline")

from src.feature_eng import perform_feature_extraction
from src.inference_rsa import attack_all_rsa_components

def run_rsa_pipeline(input_file_path):
    """Run RSA Feature Extraction and Inference."""
    print(f"\n[RSA Process] Processing {input_file_path}...")
    
    # 1. Feature Extraction
    input_dir = os.path.dirname(input_file_path)
    file_pattern = os.path.basename(input_file_path)
    output_dir = os.path.abspath("Processed_GreenVisa")
    
    try:
        print("[RSA Process] Extracting features (using Mastercard POIs)...")
        perform_feature_extraction(
            input_dir=input_dir,
            output_dir=output_dir,
            file_pattern=file_pattern,
            use_existing_pois=True,
            separate_sboxes=True,
            card_type="universal"
        )
    except Exception as e:
        print(f"[RSA Process] Feature extraction failed: {e}")
        return None

    # 2. Inference
    print("[RSA Process] Running inference...")
    model_dir = "I:/freelance/Smartcard SCA ML Pipeline/models/Ensemble_Final_Green"
    feat_path = os.path.join(output_dir, "X_features.npy")
    meta_path = os.path.join(output_dir, "Y_meta.csv")
    
    if not os.path.exists(feat_path):
        print(f"[RSA Process] Error: Features file {feat_path} not found.")
        return None
        
    results = attack_all_rsa_components(feat_path, meta_path, model_dir)
    return results

if __name__ == "__main__":
    raise SystemExit(
        "Deprecated entrypoint (legacy RSA wrapper; uses external paths/models).\n"
        "Use the supported pipeline instead:\n"
        "  python main.py --mode attack --input_dir <DIR> --processed_dir <DIR> --output_dir <DIR> --scan_type rsa\n"
    )
