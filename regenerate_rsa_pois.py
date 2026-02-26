import sys
import os
import numpy as np

# Add original pipeline to path to access src
sys.path.append("I:/freelance/Smartcard SCA ML Pipeline")

from src.feature_eng import perform_feature_extraction

def regenerate_pois():
    print("Regenerating RSA POIs using Mastercard data...")
    print("Optimization folder will be created in current directory.")
    
    # Input: Original Mastercard Data
    # Pattern: Single file to speed up (1000 traces is enough for POI)
    input_dir = "I:/freelance/Smartcard SCA ML Pipeline/Input/Mastercard"
    pattern = "traces_data_rsa_card1_1000T_1.npz"
    
    # Output: Temporary processed folder (we only care about Optimization/pois_global.npy)
    output_dir = "Processed_Mastercard_Temp"
    
    try:
        perform_feature_extraction(
            input_dir=input_dir,
            output_dir=output_dir,
            n_pois=200, # Default from feature_eng main
            file_pattern=pattern,
            use_existing_pois=False, # Force regeneration
            separate_sboxes=True,
            card_type="universal"
        )
        print("\nSUCCESS: POI Generation Complete.")
        
        if os.path.exists("Optimization/pois_global.npy"):
            print("Verified: Optimization/pois_global.npy exists.")
        else:
            print("ERROR: pois_global.npy was not created!")
            
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    regenerate_pois()
