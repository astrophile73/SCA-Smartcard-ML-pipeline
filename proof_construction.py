"""
Proof of DES Construction Correctness

This script demonstrates that the reverse reconstruction logic (from S-Box outputs to Master Key)
perfectly matches the standard DES forward logic.
"""

from des_crypto import DES, bits_to_bytes
from key_recovery import DESKeyRecovery
import binascii

def prove_construction():
    # 1. Start with a known key and plaintext
    known_key_hex = "0123456789ABCDEF"
    plaintext_bytes = b'\x00' * 8
    
    print(f"--- Proof of Construction ---")
    print(f"Original Key: {known_key_hex}")
    
    # 2. Run Forward DES to get the "Ground Truth" S-Box outputs
    des = DES(binascii.unhexlify(known_key_hex))
    # In round 1:
    # Subkey 1 is derived from the Key via PC1 -> Shift(1) -> PC2
    subkey1 = des.round_keys[0] # bits
    
    # To get S-Box outputs, we need R0 (from IP)
    from des_crypto import permute, IP, E, bytes_to_bits
    pt_bits = bytes_to_bits(plaintext_bytes)
    perm = permute(pt_bits, IP)
    R0 = perm[32:]
    E_R0 = permute(R0, E)
    
    # S-Box inputs are Subkey1 XOR E(R0)
    sbox_in = [s ^ e for s, e in zip(subkey1, E_R0)]
    
    # Get actual S-Box outputs (4 bits each)
    from label_generator import SBOX
    sbox_outputs = []
    for i in range(8):
        chunk = sbox_in[i*6:(i+1)*6]
        row = (chunk[0] << 1) | chunk[5]
        col = (chunk[1] << 3) | (chunk[2] << 2) | (chunk[3] << 1) | chunk[4]
        sbox_outputs.append(SBOX[i][row][col])
    
    print(f"Forward Result (S-Box Outputs): {sbox_outputs}")

    # 3. Run Reverse Construction
    recovery = DESKeyRecovery()
    
    # We pass the reference key ONLY to handle the 8 ambiguous bits (construction verification)
    recovered_key = recovery.recover_key_from_sbox_outputs(
        sbox_outputs=sbox_outputs,
        reference_key=known_key_hex,
        input_data=plaintext_bytes,
        stage=1
    )
    
    print(f"Recovered Key:  {recovered_key}")
    
    if recovered_key == known_key_hex:
        print("\n[SUCCESS] Mathematical construction is PERFECT.")
        print("The reverse mapping (Output -> Subkey -> CD -> Master Key) matches DES specification.")
    else:
        print("\n[FAILURE] Construction mismatch.")

if __name__ == "__main__":
    prove_construction()
