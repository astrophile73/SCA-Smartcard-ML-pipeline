"""
Final 3DES Key Verification for APDU Logs
Verifies if a recovered 16-byte key matches the cryptogram in a 8482 command.
"""

from Crypto.Cipher import DES3
import binascii

def verify_16byte_key_against_apdu(key_hex, plaintext_hex, expected_cryptogram_hex):
    """
    Simultes the 8482 host cryptogram calculation.
    EMV 3DES Mode (Keying Option 2): 16-byte key K1 || K2.
    """
    key = binascii.unhexlify(key_hex)
    plaintext = binascii.unhexlify(plaintext_hex)
    expected = binascii.unhexlify(expected_cryptogram_hex)
    
    # 3DES Keying Option 2 (16 bytes) is internally handled as K1 || K2 || K1
    # which matches pycryptodome's DES3 implementation for 16-byte keys.
    cipher = DES3.new(key, DES3.MODE_ECB)
    calculated = cipher.encrypt(plaintext)
    
    print("\n" + "=" * 40)
    print("APDU Cryptogram Verification")
    print("=" * 40)
    print(f"Key:        {key_hex.upper()}")
    print(f"Plaintext:  {plaintext_hex.upper()}")
    print(f"Expected:   {expected_cryptogram_hex.upper()}")
    print(f"Calculated: {calculated.hex().upper()}")
    
    if calculated == expected:
        print("\n✅ MATCH SUCCESSFUL! The key is mathematically proven.")
        return True
    else:
        print("\n❌ MISMATCH FAILURE. The key is incorrect.")
        return False

if __name__ == "__main__":
    # Data from User's LogExport_20260216_115141.md:
    # [11:51:38.892] -> 8482 0000 10 A18D2ECB1BD33BA52E2DA739E8FDE9BC
    # [11:51:38.745] <- 10 177AD9B7FF800000 0110 ... (Challenge from 8050)
    
    # Normally, host cryptogram for 8482 is calculated on 8-byte challenge + 8 padding OR other data.
    # However, since the client's 8482 data is 16 bytes (A18D...), 
    # we will test if our recovered keys produce this exact 16-byte output 
    # given the input challenge or a zero string (depending on the card profile).
    
    test_key = "00112233445566778899AABBCCDDEEFF" # Placeholder
    test_plaintext = "00" * 16 # Placeholder
    test_expected = "A18D2ECB1BD33BA52E2DA739E8FDE9BC"
    
    # print("Usage: Modify this script with the recovered key to verify.")
    # verify_16byte_key_against_apdu(test_key, test_plaintext, test_expected)
