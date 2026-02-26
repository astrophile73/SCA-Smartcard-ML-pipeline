"""
Diagnostic: Stage 2 (Decryption) Key Schedule Analysis
Check how Round Key 16 relates to the Original 56-bit Key.
Does it involve shifts? Or is it just a permutation?
"""

from des_crypto import DES, bits_to_bytes, bytes_to_bits, permute, PC1, PC2, SHIFT_SCHEDULE
import binascii

def analyze_stage2_schedule():
    key_hex = "0123456789ABCDEF"
    print(f"Original Key: {key_hex}")
    
    des = DES(binascii.unhexlify(key_hex))
    
    # K16 is used in Stage 2 (Decryption Round 1)
    k16 = des.round_keys[15] 
    
    # Let's trace the shifts
    # Total shifts in full 16 rounds = 28 (which wraps around to 0)
    # So K16 should be derived from C16, D16
    # C16 = LeftShift(C0, 28) = C0
    # D16 = LeftShift(D0, 28) = D0
    # WAIT! 
    # SHIFT_SCHEDULE = [1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1]
    # Sum = 28.
    # So after 16 rounds, the registers C and D are back to their original state (C0, D0).
    # BUT! 
    # des._generate_round_keys loop:
    #   for round_num in range(16):
    #       Shift...
    #       key = PC2(C+D)
    
    # So round_keys[15] (K16) is generated from (C16, D16).
    # And C16 is indeed C0 rotated by 28 positions (which is 0 mod 28).
    # So C16 == C0 !!!
    
    # If C16 == C0, then K16 = PC2(C0 + D0).
    # And K1 (Round 1) = PC2(LeftRotate(C0, 1) + LeftRotate(D0, 1)).
    
    # IMPLICATION:
    # For Stage 1 (Encryption), we recover K1, so we must reverse the 1-bit shift.
    # For Stage 2 (Decryption), we recover K16. 
    # Since K16 is derived from C16/D16 which equals C0/D0...
    # WE DO NOT NEED TO REVERSE ANY SHIFTS FOR STAGE 2!
    
    # My current code in `recover_key_from_sbox_outputs`:
    # if stage == 1: reverse shift
    # else: no shift (c0=c_round, d0=d_round)
    
    # Let's verify this matches my code:
    # Line 128: 
    # if stage == 1:
    #     c0 = self.circular_right_shift(final_c, 1)
    # else:
    #     c0 = final_c
    
    # IT SEEMS MY CODE IS ALREADY CORRECT ABOUT THE SHIFTS?
    # "else: c0 = final_c" means for Stage 2, we assume no shift.
    # If C16 == C0, this is correct.
    
    # Let's verify experimentally.
    key_bits = bytes_to_bits(binascii.unhexlify(key_hex))
    key_56 = permute(key_bits, PC1)
    C0, D0 = key_56[:28], key_56[28:]
    
    print(f"C0: {C0[:10]}...")
    
    # Check K16 internal state
    k16_cd = des.round_keys_cd[15]
    C16_actual = k16_cd[:28]
    
    print(f"C16: {C16_actual[:10]}...")
    
    if C0 == C16_actual:
        print("[CONFIRMED] C16 == C0. Stage 2 recovery needs NO SHIFTS.")
    else:
        print("[SURPRISE] C16 != C0. Shift logic is wrong.")

if __name__ == "__main__":
    analyze_stage2_schedule()
