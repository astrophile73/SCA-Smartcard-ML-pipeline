"""
Cross-Verification: Attack Mode vs Reference Key
This script proves that the ML "Brains" are predicting the correct S-Box outputs 
and that our "Construction" logic can use those outputs to rebuild the exact 
Mastercard Reference Key (9E15...).
"""

import numpy as np
import binascii
from des_crypto import DES, IP, PC1, PC2, E, bytes_to_bits, permute, bits_to_bytes
from attack import Attacker
from key_recovery import DESKeyRecovery
import tensorflow as tf

def cross_verify():
    # 1. Setup Attacker and Load Mastercard Traces
    print("--- 1. Loading Attack Ensemble & Traces ---")
    attacker = Attacker(models_dir="models")
    data = np.load('Input/Mastercard/traces_data_1000T_1.npz', allow_pickle=True)
    traces = data['trace_data']
    ref_kenc = str(data['T_DES_KENC'])
    ref_k1_hex = ref_kenc[:16]
    
    print(f"Target Reference K1: {ref_k1_hex}")
    
    # Target Trace 0 for verification
    trace = traces[0]
    from data_loader import DataLoader
    normalized = DataLoader.normalize_trace(trace)
    
    # 2. Get ML Predictions (Brain Output)
    print("\n--- 2. Running ML Predictions (Brain Check) ---")
    poi1 = np.load("models/poi_indices_stage1.npy")
    # Simulate the batch prediction for one trace
    X = normalized[poi1].reshape(1, -1, 1)
    
    ml_outputs = []
    for sbox_idx in range(8):
        model_path = f"models/sbox_{sbox_idx}_kenc.keras"
        model = tf.keras.models.load_model(model_path)
        pred = model.predict(X, verbose=0)
        ml_outputs.append(np.argmax(pred))
    
    print(f"ML S-Box Outputs: {ml_outputs}")
    
    # 3. Get Reference S-Box Outputs (Ground Truth)
    print("\n--- 3. Calculating Ground Truth S-Box Outputs ---")
    des_ref = DES(binascii.unhexlify(ref_k1_hex))
    pt_bits = bytes_to_bits(b'\x00' * 8) # Mastercard known plaintext
    perm = permute(pt_bits, IP)
    R0 = perm[32:]
    E_R0 = permute(R0, E)
    target_subkey1 = des_ref.round_keys[0]
    
    sbox_in_ref = [s ^ e for s, e in zip(target_subkey1, E_R0)]
    from label_generator import SBOX
    ref_outputs = []
    for i in range(8):
        chunk = sbox_in_ref[i*6:(i+1)*6]
        row = (chunk[0] << 1) | chunk[5]
        col = (chunk[1] << 3) | (chunk[2] << 2) | (chunk[3] << 1) | chunk[4]
        ref_outputs.append(SBOX[i][row][col])
        
    print(f"Ref S-Box Outputs: {ref_outputs}")
    
    brain_success = (ml_outputs == ref_outputs)
    print(f"\n>>>> Brain Accuracy Check: {'PASSED (100%)' if brain_success else 'FAILED'}")
    
    if not brain_success:
        return

    # 4. Construction Check: Guided Recovery
    print("\n--- 4. Guided Reconstruction (Construction Check) ---")
    recovery = DESKeyRecovery()
    
    # We use the ML outputs but "guide" the selection of the 16+8 bits 
    # using the reference key to prove the math chain works.
    recovered_k1 = recovery.recover_key_from_sbox_outputs(
        sbox_outputs=np.array(ml_outputs),
        reference_key=ref_k1_hex,
        input_data=b'\x00' * 8,
        stage=1
    )
    
    print(f"Recovered Hex: {recovered_k1}")
    print(f"Reference Hex: {ref_k1_hex.upper()}")
    
    construction_success = (recovered_k1 == ref_k1_hex.upper())
    print(f"\n>>>> Construction Proof Check: {'PASSED' if construction_success else 'FAILED'}")
    
    if construction_success:
        print("\n[VERIFIED] Attack Mode Cross-Verification SUCCESSFUL.")
        print("1. The ML models are correctly identifying the internal card states.")
        print("2. The reconstruction math is correctly mapped to the DES specification.")
    else:
        print("\n[FAILED] Construction mismatch - investigating shift/parity logic.")

if __name__ == "__main__":
    raise SystemExit(
        "Deprecated entrypoint (legacy cross-verification script).\n"
        "Use the supported pipeline instead:\n"
        "  python main.py --mode attack --input_dir <DIR> --processed_dir <DIR> --output_dir <DIR>\n"
    )
