"""
Demonstration: Why the Hex Mismatch happens

This script shows that the "incorrect" key in the report and the "correct" reference key 
actually share the EXACT same subkeys and produce the EXACT same S-Box outputs.
"""

from des_crypto import DES, bits_to_bytes
from key_recovery import DESKeyRecovery
import binascii
import numpy as np

def demo_hex_mismatch():
    # 1. Load Mastercard Reference Key
    data = np.load('Input/Mastercard/traces_data_1000T_1.npz', allow_pickle=True)
    ref_kenc = str(data['T_DES_KENC'])
    print(f"Mastercard Reference Key: {ref_kenc}")
    
    # 2. Get the "Incorrect" Key from the Blind Report Result
    # (The one where my script guessed the missing 8 bits as 0)
    # Based on the previous output, the KENC recovered was: 8007234A01402CC15B9808EC3458918C
    # Let's take just the first 8 bytes (K1)
    recovered_k1 = "8007234A01402CC1" 
    print(f"Report Key (Blind Guess): {recovered_k1}")
    
    # 3. Prove they have the SAME 48-bit subkey in Round 1
    des_ref = DES(binascii.unhexlify(ref_kenc[:16]))
    des_rpt = DES(binascii.unhexlify(recovered_k1))
    
    sub_ref = des_ref.round_keys[0]
    sub_rpt = des_rpt.round_keys[0]
    
    matches = (sub_ref == sub_rpt)
    print(f"\nDo they produce the same Round 1 subkey? {matches}")
    
    # 4. Show the 256 Candidates
    # This is the "Construction" part. From the 48 subkey bits, we can build 256 keys.
    recovery = DESKeyRecovery()
    
    # We'll simulate the S-Box outputs from the reference key
    # (We know the ML models are 100% accurate, so they give these outputs)
    from des_crypto import permute, IP, E, bytes_to_bits
    des_ref = DES(binascii.unhexlify(ref_kenc[:16]))
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
    
    candidates = recovery.get_candidate_keys(
        sbox_outputs=np.array(sbox_outputs),
        input_data=b'\x00' * 8,
        stage=1
    )
    
    print(f"\nNumber of possible keys (Candidates): {len(candidates)}")
    found = ref_kenc[:16].upper() in candidates
    print(f"Is the REAL Mastercard key in that list? {found}")
    
    if found:
        idx = candidates.index(ref_kenc[:16].upper())
        print(f"The real key is Candidate #{idx} out of 256.")
        print(f"Our report used Candidate #0.")

if __name__ == "__main__":
    demo_hex_mismatch()
