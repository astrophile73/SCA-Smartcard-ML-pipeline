"""
Diagnostic Script for 3DES Key Recovery
Compares theoretical S-Box outputs with model predictions for a reference trace.
"""

import numpy as np
import tensorflow as tf
from des_crypto import DES
from data_loader import DataLoader
from label_generator import LabelGenerator
from pathlib import Path

def diagnose():
    # 1. Load data
    npz_path = "Input/Mastercard/traces_data_1000T_1.npz"
    data = np.load(npz_path, allow_pickle=True)
    trace = data['trace_data'][0]
    
    # Mastercard reference keys (from attack output)
    kenc_ref = "9E15204313F7318ACB79B90BD986AD29"
    
    # 2. Calculate Theoretical S-Box Outputs (Stage 1)
    k1_bytes = bytes.fromhex(kenc_ref)[:8]
    des = DES(k1_bytes)
    # Most training uses P=0
    plaintext = b'\x00' * 8
    theoretical_sbox = des.get_first_round_sbox_outputs(plaintext)
    
    print(f"\n--- Stage 1 Diagnosis (KENC) ---")
    print(f"Reference Key: {kenc_ref[:16].upper()}")
    print(f"Theoretical S-Box Outputs: {theoretical_sbox}")
    
    # 3. Get Model Predictions
    poi_indices = np.load("models/poi_indices_stage1.npy")
    normalized = DataLoader.normalize_trace(trace)
    X = normalized[poi_indices].reshape(1, -1, 1)
    
    preds = []
    print("\nModel Predictions (Stage 1):")
    for sbox_idx in range(8):
        model_path = f"models/sbox_{sbox_idx}_kenc_s1.keras"
        if not Path(model_path).exists():
            model_path = f"models/sbox_{sbox_idx}_kenc.keras"
            
        model = tf.keras.models.load_model(model_path)
        probs = model.predict(X, verbose=0)[0]
        pred = np.argmax(probs)
        conf = np.max(probs)
        preds.append(pred)
        print(f"  S-Box {sbox_idx}: Predicted={pred} (Conf={conf:.4f}), Expected={theoretical_sbox[sbox_idx]}")
    
    preds = np.array(preds)
    match = np.all(preds == theoretical_sbox)
    print(f"\nOverall Match: {'✅ YES' if match else '❌ NO'}")
    
    if not match:
        print("\nFAILURE: Theoretical outputs do not match model predictions.")
        print("Possible reasons:")
        print("1. Label generation logic in label_generator.py differs from des_crypto.py.")
        print("2. Models were trained on a different round or different key assumption.")
    else:
        print("\nSUCCESS: Models are predicting exactly what des_crypto expects.")
        print("Problem is in the RECONSTRUCTION logic (PC1/PC2 reversal).")

if __name__ == "__main__":
    diagnose()
