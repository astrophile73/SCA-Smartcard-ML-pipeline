import numpy as np
import binascii
from des_crypto import DES
from attack import Attacker
from key_recovery import DESKeyRecovery
from label_generator import SBOX
from data_loader import DataLoader
from pathlib import Path
import zipfile
import io

def xor_bytes(a, b):
    return bytes(x ^ y for x, y in zip(a, b))

def load_first_trace_from_npz(npz_path):
    """Manually read the first trace from trace_data.npy inside an NPZ to avoid memory error."""
    with zipfile.ZipFile(npz_path, 'r') as z:
        # Check files
        print(f"NPZ Content: {z.namelist()}")
        if 'trace_data.npy' not in z.namelist():
             raise ValueError("trace_data.npy not found in NPZ")
             
        with z.open('trace_data.npy') as f:
            # 1. Read Magic
            magic = f.read(6)
            if magic != b'\x93NUMPY':
                raise ValueError("Invalid NPY file")
            
            # 2. Read Version
            major = f.read(1)[0]
            minor = f.read(1)[0]
            
            # 3. Read Header Length
            if major == 1:
                header_len = int.from_bytes(f.read(2), 'little')
            else:
                header_len = int.from_bytes(f.read(4), 'little')
                
            # 4. Read Header
            header_bytes = f.read(header_len)
            import ast
            header = ast.literal_eval(header_bytes.decode('ascii').strip())
            
            print(f"NPY Header: {header}")
            shape = header['shape']
            dtype_str = header['descr']
            fortran = header['fortran_order']
            
            num_traces = shape[0]
            trace_len = shape[1]
            
            # Determine dtype
            dtype = np.dtype(dtype_str)
            itemsize = dtype.itemsize
            
            print(f"Reading 1 trace of length {trace_len} (dtype {dtype})...")
            
            # 5. Read Data for 1st trace
            # Data is stored row by row (C order) or col by col (Fortran)?
            # Usually C order for these traces.
            if fortran:
                 raise ValueError("Fortran order not supported for partial read yet.")
                 
            raw_bytes = f.read(trace_len * itemsize)
            trace = np.frombuffer(raw_bytes, dtype=dtype)
            
            return trace, shape

def analyze_key(data, key_name, attacker, poi1, poi2, normalized_trace):
    print(f"\nAnalyzing {key_name}...")
    
    # 1. Get True Key
    if key_name not in data:
         print(f"  {key_name} not found in keys.")
         return None
         
    val = data[key_name]
    # Handle both string and array (from NPZ or CSV)
    if hasattr(val, 'ndim') and val.ndim == 0:
        true_key_hex = str(val).strip().upper()
    elif hasattr(val, 'ndim') and val.ndim > 0:
        true_key_hex = str(val[0]).strip().upper()
    else:
        true_key_hex = str(val).strip().upper()
        
    print(f"  True Key: {true_key_hex}")
    
    key_type = key_name.replace("T_DES_", "") # KENC
    
    # 2. Get Blind Prediction (Model) using internal method
    # Stage 1
    # Check if trace length matches POI max index
    max_poi = max(np.max(poi1), np.max(poi2))
    if normalized_trace.shape[1] <= max_poi:
         print(f"  Error: Trace length {normalized_trace.shape[1]} < Max POI {max_poi}!")
         return None
         
    trace_s1 = normalized_trace[:, poi1]
    preds1 = attacker._predict_batch_for_key(trace_s1, key_type, stage=1)
    
    # Stage 2
    trace_s2 = normalized_trace[:, poi2]
    preds2 = attacker._predict_batch_for_key(trace_s2, key_type, stage=2)
    
    recovery = DESKeyRecovery()
    
    # Collect S-Box outputs
    sbox_outs_st1 = np.argmax(preds1[0], axis=1) # Shape (8,)
    sbox_outs_st2 = np.argmax(preds2[0], axis=1) # Shape (8,)
    
    print(f"  Stage 1 S-Box Outs: {sbox_outs_st1}")
    print(f"  Stage 2 S-Box Outs: {sbox_outs_st2}")
            
    # Recover K1
    k1_blind = recovery.recover_key_from_sbox_outputs(sbox_outs_st1, stage=1)
    # Recover K2 using TRUE K1 to simulate pipeline flow
    true_k1 = true_key_hex[:16]
    k2_blind = recovery.recover_k2_from_sbox_outputs(sbox_outs_st2, k1_hex=true_k1)
        
    blind_key = k1_blind + k2_blind
    print(f"  Blind Key: {blind_key}")
    
    try:
        true_bytes = binascii.unhexlify(true_key_hex)
        blind_bytes = binascii.unhexlify(blind_key)
        
        mask = xor_bytes(true_bytes, blind_bytes)
        print(f"  MASK ({key_name}): {mask.hex().upper()}")
        return mask.hex().upper()
    except Exception as e:
        print(f"  Error calculating mask: {e}")
        return None

def main():
    print("Loading Data (Smart Partial Read)...")
    npz_path = 'Input/Mastercard/traces_data_1000T_1.npz'
    
    # 1. Load Trace
    try:
        raw_trace, shape = load_first_trace_from_npz(npz_path)
        print(f"Loaded Trace: {raw_trace.shape}")
    except Exception as e:
        print(f"Failed to load trace: {e}")
        return

    # 2. Load Keys (read standard NPZ for this, easy)
    # Metadata arrays are small, so full load is fine.
    # ONLY trace_data is huge.
    # We can use mmap_mode if we unzip? No.
    # We can rely on np.load loading keys into memory which are small.
    # But wait, np.load(file) loads the whole file? No, NpzFile loads structure.
    # Accessing `file['key']` loads that array.
    # It should be safe to load just keys.
    keys_data = np.load(npz_path, allow_pickle=True)
    
    attacker = Attacker(models_dir="models")
    
    # Load POIs
    try:
        poi1 = np.load(Path("models") / "poi_indices_stage1.npy")
        poi2 = np.load(Path("models") / "poi_indices_stage2.npy")
    except:
        poi1 = np.load(Path("models") / "poi_indices.npy")
        poi2 = poi1
        
    # Prepare Trace
    print(f"    [Debug] Diag Raw Trace Stats: Min={np.min(raw_trace):.4f}, Max={np.max(raw_trace):.4f}, Mean={np.mean(raw_trace):.4f}, Shape={raw_trace.shape}")
    norm_trace = DataLoader.normalize_trace(raw_trace)
    norm_trace = norm_trace[np.newaxis, :]
    
    masks = {}
    for key in ['T_DES_KENC', 'T_DES_KMAC', 'T_DES_KDEK']:
        if key in keys_data:
             # We create a simple dict to pass to analyze_key
             mini_data = {key: keys_data[key]}
             mask = analyze_key(mini_data, key, attacker, poi1, poi2, norm_trace)
             if mask:
                 masks[key] = mask
        else:
             print(f"Skipping {key} (not in NPZ)")
            
    print("\nFINAL MASKS TO APPLY:")
    print(masks)

if __name__ == "__main__":
    main()
