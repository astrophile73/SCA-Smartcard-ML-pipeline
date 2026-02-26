
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from src.inference_ensemble import run_ensemble_attack, load_ensemble_models, predict_ensemble
from src.crypto import apply_permutation, IP, E_TABLE, des_sbox_output, generate_round_keys
from src.utils import setup_logger

logger = setup_logger("Ensemble_Attack")

def to_hex(val, length=16):
    return f"{val:0{length}X}"

def attack_sbox_fragment(sbox_idx, probs, metadata_df, num_traces=2000):
    """
    Recovers the 6-bit key fragment for a specific S-Box using CPA/MLE attack.
    """
    # Parse ATC for the first N traces
    subset = metadata_df.head(num_traces)
    
    # Pre-calculate R0 Expanded for these traces
    r0_expanded_list = []
    
    for _, row in subset.iterrows():
        # (Same ATC parsing logic as gen_labels)
        atc_bytes = bytes([int(row.get(f'ATC_{i}', 0)) for i in range(8)])
        atc_int = int.from_bytes(atc_bytes, 'big')
        atc_permuted = apply_permutation(atc_int, IP, width=64)
        r0 = atc_permuted & 0xFFFFFFFF
        r0_expanded = apply_permutation(r0, E_TABLE, width=32)
        r0_expanded_list.append(r0_expanded)
        
    r0_expanded_arr = np.array(r0_expanded_list, dtype=np.uint64)
    probs_subset = probs[:num_traces] # (N, 16)
    
    # MLE Attack (maximize log-likelihood)
    scores = np.zeros(64)
    
    # Which bits of R0_expanded correspond to this S-Box?
    # S1: bits 42-47 (indices 1-6 in 1-based, or 0-5 in 0-based from left?)
    # Standard DES numbering 1..48.
    # The gen_labels logic: shift = 42 - (sbox_idx * 6) -> (xor >> shift) & 0x3F
    
    # Vectorized attack
    for k_guess in range(64):
        # XOR R0 with Key Guess
        # We only care about the 6 bits relevant to this S-Box
        # But R0_expanded is 48 bits.
        # Conceptually: (R0_exp ^ K_round) 
        # We can just extract the 6 bits from R0_exp first
        
        shift = 42 - (sbox_idx * 6)
        r0_input_bits = (r0_expanded_arr >> shift) & 0x3F
        
        # XOR with 6-bit key guess
        sbox_in = r0_input_bits ^ k_guess
        
        # Compute expected S-Box output (0-15)
        sbox_out = np.array([des_sbox_output(sbox_idx, x) for x in sbox_in])
        
        # Score = Sum of log(Prob(predicted_class == expected_out))
        # We index into probs with (trace_idx, sbox_out)
        
        # Advanced indexing to get probabilities of the expected outputs
        trace_indices = np.arange(num_traces)
        relevant_probs = probs_subset[trace_indices, sbox_out]
        
        # Avoid log(0)
        relevant_probs = np.clip(relevant_probs, 1e-15, 1.0)
        
        scores[k_guess] = np.sum(np.log(relevant_probs))
        
    best_k = np.argmax(scores)
    return best_k

def full_ensemble_attack(processed_dir="Processed/Mastercard", model_dir="Models/Ensemble_ZaidNet"):
    
    logger.info("1. Generating Ensemble Predictions...")
    sbox_probs = run_ensemble_attack(processed_dir, model_dir)
    
    logger.info("2. Loading Metadata for Key Recovery...")
    meta_path = os.path.join(processed_dir, "Y_meta.csv")
    if not os.path.exists(meta_path):
        logger.error(f"Metadata not found at {meta_path}")
        return None
        
    df = pd.read_csv(meta_path)
    
    logger.info("3. Reconstructing Round Key 1...")
    rk1_parts = []
    
    # Iterate S-Boxes 0-7 (Internal index) matching standard S1-S8
    recovered_rk1 = 0
    
    for sbox_i in range(8):
        # Inference used 1-based indexing for files/logging (sbox1..8)
        if (sbox_i + 1) not in sbox_probs:
            logger.error(f"Missing probabilities for S-Box {sbox_i+1}")
            continue
            
        probs = sbox_probs[sbox_i + 1]
        
        best_6bit = attack_sbox_fragment(sbox_i, probs, df)
        logger.info(f"  S-Box {sbox_i+1} Best Guess: {best_6bit:02X} ({best_6bit:06b})")
        
        # Construct 48-bit Round Key 1
        # The 6 bits correspond to the chunk at 'shift'
        shift = 42 - (sbox_i * 6)
        recovered_rk1 |= (best_6bit << shift)
        
    logger.info(f"Recovering Round Key 1 (48-bit): {recovered_rk1:012X}")
    
    # Return the key as a hex string for main.py integration
    recovered_key_hex = f"{recovered_rk1:012X}"
    
    # Validation against Ground Truth (if known)
    if df is not None and not df.empty and 'T_DES_KENC' in df.columns:
        true_key_full = str(df.iloc[0]['T_DES_KENC']).strip()
        
        # Determine K1 (first 16 hex chars)
        true_k1 = ""
        if len(true_key_full) == 32: # 16 bytes (K1+K2)
             true_k1 = true_key_full[:16]
        elif len(true_key_full) == 16: # 8 bytes (K1 only)
             true_k1 = true_key_full
             
        if len(true_k1) == 16:
            try:
                true_key_val = int(true_k1, 16)
                # Generate Round Keys from K1
                true_rks = generate_round_keys(bytes.fromhex(true_k1))
                true_rk1 = true_rks[0]
                
                logger.info(f"Ground Truth K1:                {true_k1}")
                logger.info(f"Reference Round Key 1:          {true_rk1:012X}")
                logger.info(f"Match: {recovered_rk1 == true_rk1}")
                
                if recovered_rk1 == true_rk1:
                    logger.info("SUCCESS: ENSEMBLE RECOVERED CORRECT KEY!")
                else:
                    logger.warning("FAILURE: Key Mismatch.")
            except Exception as e:
                logger.error(f"Validation Error: {e}")
    
    return recovered_key_hex
    
    return recovered_key_hex

if __name__ == "__main__":
    raise SystemExit(
        "Deprecated entrypoint (legacy ensemble attack runner).\n"
        "Use the supported pipeline instead:\n"
        "  python main.py --mode attack --input_dir <DIR> --processed_dir <DIR> --output_dir <DIR> --scan_type 3des\n"
    )
