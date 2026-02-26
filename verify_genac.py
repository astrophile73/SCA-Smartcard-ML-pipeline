
import binascii
from Crypto.Cipher import DES3

def verify_genac():
    # Data from APDU Log
    # GenAC Command Data (29 bytes)
    # CDOL1 Data: 00000000200000000000000000012408A08010000124260216007530FD20
    # Note: Pad with 80 00 ... to 8-byte boundary.
    # 29 bytes + 80 + 00 00 -> 32 bytes (4 blocks).
    
    cdol_data_hex = "00000000200000000000000000012408A08010000124260216007530FD20"
    atc_hex = "0001" # Extracted from IAD or log context (ATC is usually in 9F36 tag in response)
    # Response: 77 1E 9F 27 01 80 9F 36 02 00 01 ... -> 9F36 (ATC) = 0001
    
    expected_arqc = "1787472F63DC2F73"
    
    # Placeholder for the Recovered Key (from solve_green_visa_fuzzy.py)
    # Once the solver finds the key, paste it here.
    master_key_hex = "00000000000000000000000000000000" 
    
    if master_key_hex == "00000000000000000000000000000000":
        print("⚠️  WARNING: Using placeholder Master Key. Please update 'master_key_hex' with the recovered key.")
    
    print("--- Visa GenAC Verification ---")
    print(f"Master Key: {master_key_hex}")
    print(f"ATC:        {atc_hex}")
    print(f"ARQC (Exp): {expected_arqc}")
    
    # 1. Prepare Data
    data_bytes = binascii.unhexlify(cdol_data_hex)
    padded_data = data_bytes + b'\x80' + b'\x00' * 2 # 29+1+2 = 32 bytes
    
    print(f"Padded Data: {padded_data.hex().upper()}")
    
    # 2. Derive Session Key (SK) - Try Methods
    atc_bytes = binascii.unhexlify(atc_hex)
    mk_bytes = binascii.unhexlify(master_key_hex)
    
    # Method A: SK = MK (Static)
    sk_a = mk_bytes
    
    # Method B: EMV Common Session Key Derivation
    # R = Left(MK) if F0, Right(MK) if 0F? No.
    # R = Pad(ATC)
    # Sk = 3DES(MK, R)
    # Usually: 
    #   SK_L = 3DES(MK, ATC || F0 || 00..)
    #   SK_R = 3DES(MK, ATC || 0F || 00..)
    #   SK = SK_L || SK_R
    
    def derive_common_sk(mk, atc):
        # Method B: EMV Common Session Key Derivation (ATC-based)
        # SK_L = 3DES(MK, ATC || F0 || 00..)
        # SK_R = 3DES(MK, ATC || 0F || 00..)
        d1 = atc + b'\xF0' + b'\x00'*5
        d2 = atc + b'\x0F' + b'\x00'*5
        
        cipher = DES3.new(mk, DES3.MODE_ECB)
        sk_l = cipher.encrypt(d1)
        sk_r = cipher.encrypt(d2)
        return sk_l + sk_r

    def derive_visa_sk(mk, atc):
        # Method C: Visa Proprietary Derivations (common in older cards)
        # Option 1: XOR with ATC? No, usually not for 3DES.
        # Option 2: Height-map?
        # Option 3: Tree-based (EMV Option A) - Unlikely for Visa.
        
        # Let's try a variation where only 8 bytes are derived?
        # Or maybe just using ATC as IV?
        return None # Placeholder for now

    methods = [
        ("Static (SK=MK)", sk_a),
        ("EMV Common (SK=Derived)", sk_b)
    ]
    
    # Try Visa variation: SK = 3DES(MK, ATC_padded) ?
    # Let's add a brute-force check on SK if MK is correct?
    # No, focus on finding MK first.
    
    for name, sk in methods:
        if sk is None: continue
        print(f"\nTesting Session Key Method: {name}")
        print(f"  SK: {sk.hex().upper()}")
        
        try:
            # 3. Calculate ARQC (MAC)
            # ISO/IEC 9797-1 Algorithm 3 (Retail MAC)
            # Signature: Padded Data -> MAC (8 bytes)
            
            # Step 1: DES(K_L) on blocks using CBC (ISO/IEC 9797-1 Alg 1)
            # Initial Vector = 0
            k_l = sk[:8]
            k_r = sk[8:16]
            
            # Manual CBC with Single DES for Block 1..N-1
            from Crypto.Cipher import DES
            des_l = DES.new(k_l, DES.MODE_ECB)
            
            current_block = b'\x00'*8
            # Split padded data into 8-byte blocks
            blocks = [padded_data[i:i+8] for i in range(0, len(padded_data), 8)]
            
            for i, blk in enumerate(blocks):
                to_enc = bytes(x ^ y for x, y in zip(current_block, blk))
                current_block = des_l.encrypt(to_enc)
            
            # Final Step: Decrypt with K_R, Encrypt with K_L (Retail MAC)
            des_r = DES.new(k_r, DES.MODE_ECB)
            d_block = des_r.decrypt(current_block)
            final_mac = des_l.encrypt(d_block)
            
            print(f"  Calculated ARQC: {final_mac.hex().upper()}")
            
            if final_mac.hex().upper() == expected_arqc:
                print(f"  ✅ MATCH! Session Derivation Logic Confirmed: {name}")
                return True
            else:
                print(f"  ❌ Mismatch (Expected: {expected_arqc})")
                
        except Exception as e:
            print(f"  Error: {e}")
            
    print("\n[Optional] Direct Key Check against Log 8482 Cryptogram (Static Key Assumption)")
    
    # Check if MK directly encrypts Challenge -> Cryptogram
    # 3DES(MK, Challenge) ==? Cryptogram
    # Challenge: 0102030405060708
    # Cryptogram: 49709CD7822B197D5BA99733726B6171
    
    challenge_bin = binascii.unhexlify("0102030405060708")
    expected_crypto_bin = binascii.unhexlify("49709CD7822B197D5BA99733726B6171")[:8] # First block check? Or full?
    # 3DES ECB usually used for simple check.
    
    try:
        cipher_static = DES3.new(mk_bytes, DES3.MODE_ECB)
        ct = cipher_static.encrypt(challenge_bin)
        print(f"  Calculated Cryptogram (3DES ECB): {ct.hex().upper()}")
        
        if ct.hex().upper() == "49709CD7822B197D": # First 8 bytes
            print("  ✅ MATCH! Master Key is static and matches 8482 Cryptogram.")
            return True
    except Exception as e:
        print(f"  Error in static check: {e}")

    print("\nNo match found for GenAC or Static logic.")
    return False

if __name__ == "__main__":
    verify_genac()
