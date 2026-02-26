
"""
Validation Script for Key Extraction Logic.
This script proves that the key extraction logic in key_recovery.py is mathematically correct.
It simulates the process with a known key and checks if the recovered key matches.
"""

import numpy as np
import binascii
from des_crypto import DES, bytes_to_bits, bits_to_bytes, PC1, PC2, IP, E, permute
from key_recovery import DESKeyRecovery

def test_extraction_logic():
    print("--- Validate Key Extraction Logic ---")
    
    # 1. Setup Known Key and Plaintext
    known_key_hex = "0123456789ABCDEF"
    known_key = bytes.fromhex(known_key_hex)
    plaintext = bytes.fromhex("0000000000000000") # All zeros for simplicity
    
    print(f"Original Key: {known_key_hex}")
    
    # 2. Generate Theoretical S-Box Outputs (Ground Truth)
    des = DES(known_key)
    # Debug: Print internal state
    print(f"DEBUG: C0 (Forward): {des.round_keys_cd[0][:28]}") # Actually round_keys_cd[0] is C1D1
    # We need C0D0? des_crypto doesn't store C0D0 explicitly in list, but generates it.
    
    # Re-calculate C0 D0 manually for debug
    k_bits = bytes_to_bits(known_key)
    from des_crypto import permute, PC1, left_shift, SHIFT_SCHEDULE, PC2
    k56 = permute(k_bits, PC1)
    C0 = k56[:28]
    D0 = k56[28:]
    print(f"DEBUG: C0 (True): {C0[:10]}... Hex: {bits_to_bytes(C0+[0]*4).hex()}")
    print(f"DEBUG: D0 (True): {D0[:10]}... Hex: {bits_to_bytes(D0+[0]*4).hex()}")
    
    # C1 D1
    C1 = left_shift(C0, 1)
    D1 = left_shift(D0, 1)
    # K1 (48 bits)
    K1_bits = permute(C1 + D1, PC2)
    print(f"DEBUG: K1 (True): {K1_bits[:10]}... Hex: {bits_to_bytes(K1_bits).hex()}")
    
    # S-Box Outputs
    sbox_outputs = des.get_first_round_sbox_outputs(plaintext)
    print(f"S-Box Outputs: {sbox_outputs}")
    
    # 3. Simulate Attack
    print("--- Attempting Stage 1 Recovery (Full Search) ---")
    recovery = DESKeyRecovery()
    
    # Use FULL candidate generation (handling 4^8 * 256 ambiguities)
    # We increase limit to ensure we find it (16.7M total).
    print("Generating all ~17M candidates (this may take a moment)...")
    candidates = recovery.get_candidate_keys_full(sbox_outputs, input_data=plaintext, stage=1, limit=20000000)
    
    # DEBUG: Check K48 reconstruction inside recovery
    # We copy the logic here to see what k48 it produces
    # Imports already at top level
    
    bits = bytes_to_bits(plaintext)
    perm = permute(bits, IP)
    R0 = perm[32:] 
    R0_expanded = permute(R0, E)
    sbox_cand_list = [recovery.reverse_sbox_lookup(i, int(out)) for i, out in enumerate(sbox_outputs)]
    
    # Reconstruct K48 using the first candidate for each S-Box (blindly)
    k48_bits_rec = []
    for sbox_idx in range(8):
        # We need to pick the CORRECT input, not just the first one, for validation
        # Find which input matches the true key?
        
        # True K48 segment
        true_k48_seg = K1_bits[sbox_idx*6:(sbox_idx+1)*6]
        # True er0
        er0_seg = R0_expanded[sbox_idx*6:(sbox_idx+1)*6]
        # True input
        true_input = 0
        for b in [k^e for k,e in zip(true_k48_seg, er0_seg)]:
             true_input = (true_input << 1) | b
             
        # Check if true_input is in candidates
        if true_input in sbox_cand_list[sbox_idx]:
            # Use this for reconstruction to test PC2/Shift logic
            best_in_val = true_input
        else:
            print(f"❌ CRITICAL: S-Box {sbox_idx} reverse lookup failed to include true input {true_input}!")
            best_in_val = sbox_cand_list[sbox_idx][0]
            
        er0_bits = R0_expanded[sbox_idx*6:(sbox_idx+1)*6]
        for bit_pos in range(6):
            in_bit = (best_in_val >> (5 - bit_pos)) & 0x01
            k48_bits_rec.append(in_bit ^ er0_bits[bit_pos])
            
    print(f"DEBUG: K1 (Rec):  {k48_bits_rec[:10]}... Hex: {bits_to_bytes(k48_bits_rec).hex()}")
    
    if k48_bits_rec != K1_bits:
        print("❌ K1 Mismatch! Error in S-Box/XOR/K48 logic.")
    else:
        print("✅ K1 Matches! Error is in PC2/Shift/PC1.")
        
        # Check C1/D1 Reconstruction (Inverse PC2)
        # rec_c1_d1, rec_mask = recovery.reverse_pc2_debug(k48_bits_rec) 
        # We'll just copy the logic effectively
        
        from des_crypto import PC2
        c1_d1_rec = [0] * 56
        mask_rec = [False] * 56
        for i, src in enumerate(PC2):
            c1_d1_rec[src - 1] = k48_bits_rec[i]
            mask_rec[src - 1] = True
            
        rec_C1 = c1_d1_rec[:28]
        rec_D1 = c1_d1_rec[28:]
        
        # Ground Truth C1/D1
        # C1 = left_shift(C0, 1)
        # D1 = left_shift(D0, 1)
        # Assuming C0, D0 calculated earlier are correct (True)
        # We need to calculate TRUE C1 D1
        # C1, D1 variable names from earlier (lines 35-36)
        
        print("\n--- C1 Comparison ---")
        print(f"True C1: {C1}")
        print(f"Rec C1:  {rec_C1}")
        
        diff = [i for i in range(28) if C1[i] != rec_C1[i] and mask_rec[i]]
        if diff:
            print(f"❌ C1 Mismatch at known indices: {diff}")
        else:
            print("✅ C1 Matches (at known positions)!")
            
        # Check Shift Logic
        # C0_rec = right_shift(C1_rec, 1)
        C0_rec = recovery.circular_right_shift(rec_C1, 1)
        print("\n--- C0 Shift Comparison ---")
        print(f"True C0: {C0}")
        print(f"Rec C0:  {C0_rec}")
    
    print(f"DEBUG: Missing Indices (C/D joined): {recovery.reverse_pc2(k48_bits_rec)[2]}") 
    # rec_c1_d1, rec_mask = recovery.reverse_pc2_debug(k48_bits_rec) 
    
    # Re-impl mask logic
    from des_crypto import PC2
    c1_d1_mask = [False] * 56
    for pc2_src_bit in PC2:
        c1_d1_mask[pc2_src_bit - 1] = True
    missing_indices = [i for i, m in enumerate(c1_d1_mask) if not m]
    
    # Find the "True" bits for these missing indices from C1/D1 (True)
    # Note: C1/D1 in validation are separate lists.
    # C1 is 0-27. D1 is 0-27 (which is 28-55).
    # We need to construct C1+D1 list?
    # Actually K1 reconstruction (Stage 1) uses K1 = PC2(C1, D1).
    # Reverse PC2 gives C1, D1.
    # So "c_round" in recovery corresponds to C1.
    true_bits_8 = []
    for idx in missing_indices:
        if idx < 28:
            true_bits_8.append(C1[idx])
        else:
            true_bits_8.append(D1[idx-28])
            
    print(f"DEBUG: True Missing Bits Tuple: {tuple(true_bits_8)}")
    
    # Now check if any candidate matches these bits?
    # Candidates are hex strings. We'd have to reverse engineer them.
    # Instead, let's just assert that the loop logic works by manually constructing 
    # the matching C0/D0 and checking if it equals True C0/D0.
    
    # Manual Reconstruct using True Bits
    c_temp_manual = list(rec_C1)
    d_temp_manual = list(rec_D1)
    for i, val in zip(missing_indices, true_bits_8):
         if i < 28: c_temp_manual[i] = val
         else: d_temp_manual[i-28] = val
         
    # Shift
    c0_manual = recovery.circular_right_shift(c_temp_manual, 1)
    d0_manual = recovery.circular_right_shift(d_temp_manual, 1)
    
    print("\n--- Manual Reconstruction Check ---")
    if c0_manual == C0:
        print("✅ Manual Reconstruction C0 Matches True C0!")
    else:
        print("❌ Manual Reconstruction C0 Mismatch!")
        # Print Diff
        print(f"True C0: {C0}")
        print(f"Man  C0: {c0_manual}")
        
    print(f"Generated {len(candidates)} candidates.")
    
    # 4. Verify
    found = False
    c0_matches = 0
    full_matches = 0
    
    for i, cand in enumerate(candidates):
        cand_bytes = bytes.fromhex(cand)
        cand_bits = bytes_to_bits(cand_bytes)
        
        # Check C0/D0 match for this candidate
        cand_k56 = permute(cand_bits, PC1)
        cand_c0 = cand_k56[:28]
        cand_d0 = cand_k56[28:]
        
        if cand_c0 == C0 and cand_d0 == D0:
            c0_matches += 1
            # If C0/D0 match, then K56 matches.
            # PC1 inverse should map K56 -> K64.
            # So cand bits should match K bits (ignoring parity).
            
            # Why did full match fail?
            # Check ignored parity
            def ignore_parity(k_hex):
                k = int(k_hex, 16)
                mask = 0xFEFEFEFEFEFEFEFE
                return k & mask
        
            if ignore_parity(cand) == ignore_parity(known_key_hex):
                print(f"✅ EXACT MATCH (ignoring parity) at index {i}: {cand}")
                found = True
                full_matches += 1
                break
            else:
                print(f"⚠️ C0/D0 matched but parity/PC1 inverse failed! Cand: {cand}")
                # Print mismatch details
                k_val = int(known_key_hex, 16)
                c_val = int(cand, 16)
                print(f"  True: {bin(k_val)}")
                print(f"  Cand: {bin(c_val)}")
                
    print(f"Total C0/D0 matches found: {c0_matches}")

    if not found:
        print("❌ FAILURE: Original key not found in candidates!")
        print(f" First candidate: {candidates[0]}")
    else:
        print("Logic is verified correct.")

if __name__ == "__main__":
    test_extraction_logic()
