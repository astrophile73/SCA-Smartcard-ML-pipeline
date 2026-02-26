"""
Diagnostic: Subkey Matching
Check if the 48-bit subkey from the Reference Key matches the 48-bit subkey 
from my Recovered Key.
"""

from des_crypto import DES, bits_to_bytes, bytes_to_bits, permute, PC2
import binascii

def diag_subkeys():
    ref_k1_hex = "9E15204313F7318A"
    rpt_k1_hex = "8007234A01402CC1"
    
    print(f"Reference K1: {ref_k1_hex}")
    print(f"Report K1:    {rpt_k1_hex}")
    
    des_ref = DES(binascii.unhexlify(ref_k1_hex))
    des_rpt = DES(binascii.unhexlify(rpt_k1_hex))
    
    sub_ref = des_ref.round_keys[0] # 48 bits
    sub_rpt = des_rpt.round_keys[0] # 48 bits
    
    print(f"\nSubkey 1 (Ref): {sub_ref}")
    print(f"Subkey 1 (Rpt): {sub_rpt}")
    
    matches = (sub_ref == sub_rpt)
    print(f"\nDo they match? {matches}")
    
    if not matches:
        diffs = [i for i in range(48) if sub_ref[i] != sub_rpt[i]]
        print(f"Differences at indices: {diffs}")

if __name__ == "__main__":
    diag_subkeys()
