"""
Diagnostic: Cryptogram Search Validation
Verify that 'recover_key_with_cryptogram' actually works when we force-feed it 
the correct S-Box inputs and a known cryptogram.
"""

from des_crypto import DES, IP, E, bytes_to_bits, permute
from key_recovery import DESKeyRecovery
import binascii
import numpy as np

def diag_search():
    # 1. Setup Known Key and Cryptogram
    ref_k1_hex = "9E15204313F7318A"
    ref_k2_hex = "CB79B90BD986AD29" # Dummy K2 for test (derived from KENC 2nd half)
    full_key_hex = ref_k1_hex + ref_k2_hex
    
    challenge_hex = "0000000000000000"
    
    # Calculate Expected Cryptogram
    from Crypto.Cipher import DES3
    key_bytes = binascii.unhexlify(full_key_hex)
    cipher = DES3.new(key_bytes, DES3.MODE_ECB)
    expected_cryptogram = cipher.encrypt(binascii.unhexlify(challenge_hex))
    expected_hex = binascii.hexlify(expected_cryptogram).decode().upper()
    
    print(f"Target Key: {full_key_hex}")
    print(f"Target Cryptogram: {expected_hex}")
    
    # 2. Get S-Box Outputs for K1 (Simulating ML)
    dk1 = DES(binascii.unhexlify(ref_k1_hex))
    sub1 = dk1.round_keys[0]
    pt_bits = bytes_to_bits(b'\x00' * 8)
    perm = permute(pt_bits, IP)
    R0 = perm[32:] 
    E_R0 = permute(R0, E)
    sbox_in = [s ^ e for s, e in zip(sub1, E_R0)]
    
    from label_generator import SBOX
    sbox_outputs_k1 = []
    for i in range(8):
        chunk = sbox_in[i*6:(i+1)*6]
        row = (chunk[0] << 1) | chunk[5]
        col = (chunk[1] << 3) | (chunk[2] << 2) | (chunk[3] << 1) | chunk[4]
        sbox_outputs_k1.append(SBOX[i][row][col])
        
    print(f"S-Box Outputs K1: {sbox_outputs_k1}")
    
    # 3. Get S-Box Outputs for K2 (Simulating ML)
    # Note: K2 inputs depend on K1 ciphertext!
    # But Stage 2 model predicts from POWER, which is independent of our knowledge of inputs.
    # However, for RECOVERY, we need to know the input to the S-Box to inverse PC2.
    # Stage 2 Input = DES_Encrypt(K1, Plaintext)
    
    # Let's generate K2 S-Box outputs properly
    dk2 = DES(binascii.unhexlify(ref_k2_hex))
    sub2 = dk2.round_keys[15] # Stage 2 uses R16 subkey (reversed) or R1? 
    # Wait, Stage 2 in 3DES is Decryption.
    # 3DES = Enc(K3, Dec(K2, Enc(K1, P)))
    # So Stage 2 is Decryption K2.
    # Decryption Round 1 uses subkey 16. 
    # BUT, our "Stage 2" model might be trained on Encryption K2 (if simply 2nd DES).
    # Task check: "Recovers K1 in Stage 1 and K2 in Stage 2"
    # If the card does 3DES, the second operation is Decryption.
    # Standard DES Decryption Schedule: R1 uses Subkey 16, R2 uses Subkey 15...
    # Let's check `des_crypto.py` to see what `generate_round_keys` does.
    # It generates 16 subkeys.
    # For Decryption, we typically use them in reverse order.
    # If we trained on Round 1 of 2nd operation, we are recovering Subkey 16 or Subkey 1 depending on implementation.
    
    # Assumption Check: Does the pipeline assume K2 is used in Encryption mode or Decryption mode?
    # Usually 2-key 3DES is EDE. So K2 is Decrypt.
    # If we are attacking the *first round* of the *second operation*, we are attacking Round 1 of Decryption.
    # In DES Decryption, Round 1 uses Subkey 16 (generated from Key).
    # Subkey 16 is NOT the simple PC1->Shift->PC2 of Subkey 1.
    # It has different shifts!
    
    # !!!!!!! FOUND POTENTIAL BUG !!!!!!!
    # If `recover_key_from_sbox_outputs` assumes Subkey 1 construction (PC1 -> 1 shift -> PC2),
    # but we are attacking a Decryption round (Subkey 16), then the math is WRONG.
    # Subkey 16 corresponds to ZERO shifts (actually, total 28 shifts = 0? No, 1 shift left? )
    # Let's check the schedule.
    
    # For this diagnostic, I will assume we need to check if recover_k2 handles decryption schedule.
    # If not, that is the root cause.
    
    pass 

if __name__ == "__main__":
    diag_search()
