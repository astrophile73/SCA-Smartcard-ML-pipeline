from key_recovery import DESKeyRecovery
import numpy as np
from des_crypto import bytes_to_bits, bits_to_bytes, PC1, DES, permute

def debug():
    print("Debugging Key Recovery Logic...")
    
    # Setup - from verified debug_key.py results
    ref_hex = "9E15204313F7318ACB79B90BD986AD29"
    predicted_sbox_outputs = np.array([3, 0, 3, 1, 4, 2, 11, 12])
    
    # 1. Verify S-Box outputs match reference (Sanity check)
    from label_generator import LabelGenerator
    gen = LabelGenerator()
    labels = gen.generate_sbox_labels_for_key(ref_hex)
    print(f"Ref Key: {ref_hex}")
    print(f"Generated Labels: {labels}")
    print(f"Predicted Labels: {predicted_sbox_outputs}")
    if not np.array_equal(labels, predicted_sbox_outputs):
        print("MISMATCH! Aborting debug as model prediction is wrong.")
        return

    # 2. Run Recovery
    rec = DESKeyRecovery()
    recovered_hex = rec.recover_key_from_sbox_outputs(
        predicted_sbox_outputs, 
        plaintext=b'\x00'*8,
        # reference_key=ref_hex
    )
    print(f"Recovered Key: {recovered_hex}")
    
    if recovered_hex[:16] == ref_hex[:16]:
        print("SUCCESS: Matches first 8 bytes of reference.")
    else:
        print("FAILURE: Does not match.")
        
        # 3. Deep Dive into bits
        ref_bytes = bytes.fromhex(ref_hex[:16])
        ref_bits = bytes_to_bits(ref_bytes)
        
        # Manually trace PC1 and Round Keys
        print(f"\n--- Bit Layout Analysis ---")
        
        # Check PC1 mapping
        des = DES(ref_bytes)
        print(f"Reference K (64 bits): {ref_bits}")
        
        # Reconstruct what the recovery got
        rec_bytes = bytes.fromhex(recovered_hex)
        rec_bits = bytes_to_bits(rec_bytes)
        print(f"Recovered K (64 bits): {rec_bits}")
        
        # Compare bits
        diff =  [i for i in range(64) if ref_bits[i] != rec_bits[i]]
        print(f"Bit differences at indices: {diff}")
        
        # Check PC1 inversion
        # recovery uses: recovered_k64[pc1_src_bit - 1] = k56[i]
        # This seems correct (PC1 table maps input index to output index).
        # PC1[i] is the Source Bit Index (1-based) for Output Bit i.
        
        # Check Shift
        # Recovery does right shift 1.
        # Encryption does left shift 1.
        # This is correct.
        
if __name__ == "__main__":
    debug()
