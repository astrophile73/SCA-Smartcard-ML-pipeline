
import numpy as np
import binascii
from des_crypto import DES, IP, E, bytes_to_bits, permute
from key_recovery import DESKeyRecovery
from attack import Attacker
from Crypto.Cipher import DES3
import time

def solve_green_visa():
    # 1. Setup Data
    npz_path = 'i:/freelance/Smartcard SCA ML Pipeline/Capture_trace_green_card_1T/traces_data_3DES_green_card_1T_260201.npz'
    log_challenge = "0102030405060708"
    log_cryptogram = "49709CD7822B197D5BA99733726B6171"
    
    print(f"--- Solving GreenVisa KENC ---")
    print(f"Target NPZ: {npz_path}")
    print(f"Reference Challenge:  {log_challenge}")
    print(f"Reference Cryptogram: {log_cryptogram}")
    
    # 2. Get S-Box Predictions for Trace 0
    attacker = Attacker(models_dir="models")
    data = np.load(npz_path, allow_pickle=True)
    traces = data['trace_data']
    trace = traces[0]
    
    from data_loader import DataLoader
    normalized_trace = DataLoader.normalize_trace(trace)
    
    # We target KENC
    key_type = "KENC"
    
    poi1 = np.load("models/poi_indices_stage1.npy")
    poi2 = np.load("models/poi_indices_stage2.npy")
    
    t1 = normalized_trace[poi1].reshape(1, -1, 1)
    t2 = normalized_trace[poi2].reshape(1, -1, 1)
    
    print("Predicting S-Box Outputs...")
    preds1 = attacker._predict_batch_for_key(t1, key_type, stage=1)
    preds2 = attacker._predict_batch_for_key(t2, key_type, stage=2)
    
    sbox_outs_st1 = np.argmax(preds1[0], axis=1)
    sbox_outs_st2 = np.argmax(preds2[0], axis=1)
    
    print(f"S1: {sbox_outs_st1}")
    print(f"S2: {sbox_outs_st2}")
    
    # 3. Generate Candidates and Brute-Force
    recovery = DESKeyRecovery()
    
    # To be fast, we use a custom loop here instead of the full 16M space if possible.
    # Actually, 16M checks of DES3(K, P) is about 20-30 seconds.
    
    # Step A: Get all K1 candidates (65k from SBox ambiguity * 256 from PC2 = 16.7M)
    # No, wait. S-Box ambiguity is 4^8 = 65536. PC2 is 256. 
    # Total is 16,777,216.
    
    print(f"Starting Brute-Force Search (16.7M candidates)...")
    start_time = time.time()
    
    # We'll use the 8482 cryptogram check. 
    # Plaintxet: challenge. 
    # Many cards use padding: challenge + 8000... or just challenge + 00...
    challenge_bytes = binascii.unhexlify(log_challenge)
    target_bytes = binascii.unhexlify(log_cryptogram)[:8] # Check first 8 bytes first
    
    # We re-implement the candidate generator to avoid huge lists
    from des_crypto import PC1, PC2, bits_to_bytes
    import itertools
    
    # Precompute S-Box candidates
    sbox_cand_list = [recovery.reverse_sbox_lookup(i, int(out)) for i, out in enumerate(sbox_outs_st1)]
    
    # PC2 structure
    c1_d1_mask = [False] * 56
    for pc2_src_bit in PC2:
        c1_d1_mask[pc2_src_bit - 1] = True
    missing_indices = [i for i, m in enumerate(c1_d1_mask) if not m]
    
    # Preparation for Stage 1 Bits
    bits_input = bytes_to_bits(b'\x00' * 8)
    perm = permute(bits_input, IP)
    R0 = perm[32:]
    R0_expanded = permute(R0, E)
    
    found_k1 = None
    
    print("Searching K1...")
    # Heuristic: search PC2 bits first for the "Blind Candidate 0" and see if it hits.
    # If not, search all S-Box combos.
    
    sbox_combos = itertools.product(*sbox_cand_list)
    total_checked = 0
    for sbox_bits_6_list in sbox_combos:
        # Map S-Box inputs to K48
        k48_bits = []
        for sbox_idx in range(8):
            in_val = sbox_bits_6_list[sbox_idx]
            er0_bits = R0_expanded[sbox_idx*6:(sbox_idx+1)*6]
            for bit_pos in range(6):
                in_bit = (in_val >> (5 - bit_pos)) & 0x01
                k48_bits.append(in_bit ^ er0_bits[bit_pos])
        
        c_round, d_round, mask = recovery.reverse_pc2(k48_bits)
        
        for bits_8 in itertools.product([0, 1], repeat=8):
            c_temp = list(c_round)
            d_temp = list(d_round)
            for i, val in zip(missing_indices, bits_8):
                if i < 28: c_temp[i] = val
                else: d_temp[i-28] = val
            
            c0 = recovery.circular_right_shift(c_temp, 1)
            d0 = recovery.circular_right_shift(d_temp, 1)
            
            k56 = c0 + d0
            k64 = [0] * 64
            for i, src in enumerate(PC1):
                k64[src - 1] = k56[i]
            
            for i in range(0, 64, 8):
                byte = k64[i:i+7]
                k64[i+7] = 1 if (sum(byte) % 2 == 0) else 0
            
            k1_candidate = bits_to_bytes(k64)
            
            # Since 8482 External Auth often uses 3DES(K, P), and for 16-byte keys it's DES(K1, P) XOR DES_Dec(K2, DES(K1, P)) ...
            # We can't verify K1 alone unless we know K1 is enough.
            # BUT: 8482 cryptogram is 16 bytes. 
            # Often it's Two-Key 3DES: 
            # Block 1 = DES3(K1||K2, Challenge)
            # Block 2 = DES3(K1||K2, Challenge XOR Block 1)
            
            # Actually, most cards for 8482 use the KENC but with a simplified derivation.
            # Let's try to see if ANY K1 candidate produces the first 8 bytes of the cryptogram.
            # This is only possible if K2 is not involved in block 1 (unlikely).
            
            # BETTER: Search K1 and K2 together? No, too slow.
            # BUT: We can use Stage 2!
            # Stage 2 S-Box outputs are predicted correctly.
            # They depend on C1 = Enc(K1, 00..00).
            # So for each K1 candidate, we calculate C1, then we can uniquely identify the 24-bit ambiguity space for K2.
            
            # BUT we only have ONE trace. If we search 16M K1s, and for each we do 16M K2s... no.
            
            # TRAPDOOR:
            # For each K1 candidate:
            # 1. Calc C1 = Enc(K1, 00..00)
            # 2. Re-calculate S-Box outputs for Stage 2 (ML targets these).
            # 3. If the K1 is right, then the SBox_Outs_K2 we predicted MUST correspond to the leakage of K2 on C1.
            # This still doesn't verify K1.
            
            # Wait, there is a simpler way.
            # The client says "GenAC cryptographic error". 
            # I have the GenAC ARQC: `1787472F63DC2F73`.
            # I have the data: `00000000200000000000000000012408A08010000124260216007530FD`
            # Data length is 29 bytes.
            # Visa GenAC ARQC calculation:
            # 1. Padding: Data + 80 + 00... to multiple of 8.
            # 2. Diversification: MK -> SK (using ATC).
            # 3. MACing: SK on Padded Data using DES3-MAC (ISO 9797-1 Algorithm 3).
            
            # This is the gold standard.
            # Let's write a loop that brute-forces K1 (16.7M) assuming K2=K1 (Single DES mode) just to see.
            # Then try K1 || K1.
            
            pass

    print("Heuristic check failed to finish in dry run. Implementing full solver in verify_green_visa_apdu.py")

if __name__ == "__main__":
    raise SystemExit(
        "Deprecated entrypoint (legacy GreenVisa solver).\n"
        "Use the supported pipeline instead:\n"
        "  python main.py --mode attack --input_dir <DIR> --processed_dir <DIR> --output_dir <DIR> --scan_type 3des\n"
    )
