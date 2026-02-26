"""
Diagnostic: Full Candidate Search
Check if the REAL Mastercard key is within the 16M possible candidates.
"""

from des_crypto import DES, bits_to_bytes, bytes_to_bits, permute, PC2, IP, E
from key_recovery import DESKeyRecovery
import binascii
import numpy as np

def diag_full():
    data = np.load('Input/Mastercard/traces_data_1000T_1.npz', allow_pickle=True)
    ref_k1_hex = str(data['T_DES_KENC'])[:16]
    print(f"Target Key: {ref_k1_hex}")
    
    # Simulate S-Box outputs from Ref Key
    des_ref = DES(binascii.unhexlify(ref_k1_hex))
    pt_bits = bytes_to_bits(b'\x00' * 8)
    perm = permute(pt_bits, IP)
    R0 = perm[32:]
    E_R0 = permute(R0, E)
    subkey1 = des_ref.round_keys[0]
    sbox_in = [s ^ e for s, e in zip(subkey1, E_R0)]
    from label_generator import SBOX
    sbox_outputs = []
    for i in range(8):
        chunk = sbox_in[i*6:(i+1)*6]
        row = (chunk[0] << 1) | chunk[5]
        col = (chunk[1] << 3) | (chunk[2] << 2) | (chunk[3] << 1) | chunk[4]
        sbox_outputs.append(SBOX[i][row][col])
    
    print(f"S-Box Outputs: {sbox_outputs}")
    
    recovery = DESKeyRecovery()
    # Increase limit to check more space
    print("Searching first 100,000 candidates...")
    candidates = recovery.get_candidate_keys_full(
        sbox_outputs=np.array(sbox_outputs),
        input_data=b'\x00' * 8,
        stage=1,
        limit=100000
    )
    
    found = ref_k1_hex.upper() in candidates
    print(f"Found in first 100k? {found}")
    
    if not found:
        # Check if the correct S-Box combo is even in the first few combos
        sbox_cand_list = [recovery.reverse_sbox_lookup(i, int(out)) for i, out in enumerate(sbox_outputs)]
        print(f"Possible inputs per S-Box: {[len(c) for c in sbox_cand_list]}")

if __name__ == "__main__":
    diag_full()
