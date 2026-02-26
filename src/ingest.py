import numpy as np
import os
import glob
import pandas as pd
from typing import List, Tuple, Dict
from src.utils import setup_logger
from src.metadata_parser import MetadataParser

logger = setup_logger("ingest")

def pad_list(lst, target_len):
    if len(lst) >= target_len:
        return lst[:target_len]
    return lst + [lst[-1]] * (target_len - len(lst))

class TraceDataset:
    def __init__(self, input_dir: str, file_pattern: str = "traces_data_*.npz", card_type: str = "universal", trace_type: str = "all"):
        self.input_dir = input_dir
        self.card_type = card_type
        self.trace_type = trace_type
        # Recursive search in subfolders
        search_path = os.path.join(input_dir, "**", file_pattern)
        all_files = sorted(glob.glob(search_path, recursive=True))
        
        # Filter based on card_type if not universal
        if card_type and card_type.lower() != "universal":
            target = card_type.lower()
            # We look for files that have the card type in their path (e.g. 'Input/Mastercard/...')
            self.files = [f for f in all_files if target in f.lower()]
            logger.info(f"Filtering for card type '{card_type}': Found {len(self.files)} files matching target.")
        else:
            self.files = all_files

        # Filter by trace type (3des vs rsa) by inspecting lightweight NPZ keys.
        if self.files and trace_type and trace_type.lower() not in ["all", "any", "both"]:
            tt = trace_type.lower()
            filtered = []
            for f in self.files:
                # Quick filename heuristic first
                name = os.path.basename(f).lower()
                if tt == "3des" and "rsa" in name:
                    continue
                if tt == "rsa" and "rsa" not in name:
                    # still allow if keys indicate RSA
                    pass

                try:
                    data = np.load(f, mmap_mode="r", allow_pickle=True)
                    keys = set(getattr(data, "files", []))
                except Exception:
                    continue

                has_3des = any(k in keys for k in ["T_DES_KENC", "T_DES_KMAC", "T_DES_KDEK"])
                has_rsa = any(k in keys for k in ["ACR_send", "ACR_receive", "Verify_command", "Verify_response"])

                if tt == "3des" and has_3des:
                    filtered.append(f)
                elif tt == "rsa" and (has_rsa and not has_3des):
                    filtered.append(f)

            self.files = filtered
            logger.info(f"Filtering for trace type '{trace_type}': {len(self.files)} file(s) kept.")

        if not self.files:
            # Check if the directory itself is a trace directory (loose .npy files)
            if os.path.exists(os.path.join(input_dir, "trace_data.npy")):
                self.files = [input_dir] # Treat directory as a single "file" source
                logger.info(f"Treating {input_dir} as a single trace directory source.")
            else:
                search_path_root = os.path.join(input_dir, file_pattern)
                self.files = sorted(glob.glob(search_path_root))
            
        if not self.files:
            raise FileNotFoundError(f"No trace files found in {input_dir} for card type {card_type}")
        logger.info(f"Final trace file count: {len(self.files)}.")
        
        self.metadata = self._load_metadata()
        self.total_traces = len(self.metadata)
        logger.info(f"Total traces found: {self.total_traces}")

    def _load_metadata(self) -> pd.DataFrame:
        meta_list = []
        
        # No external reference file loading for Pure ML Pipeline
        ref_keys = {}
        
        for fpath in self.files:
            try:
                # 1. Determine base name and potential CSV path
                base_name = os.path.splitext(os.path.basename(fpath))[0]
                dirname = os.path.dirname(fpath)
                csv_path = os.path.join(dirname, base_name + ".csv")
                
                # Handling Name Mismatches (Mastercard)
                if not os.path.exists(csv_path):
                    alt_name = base_name.replace("traces_data", "trace_data").replace("_card1", "")
                    csv_path_alt = os.path.join(dirname, alt_name + ".csv")
                    if os.path.exists(csv_path_alt):
                        csv_path = csv_path_alt
                
                # 2. Load CSV if available
                csv_df = pd.DataFrame()
                if os.path.exists(csv_path):
                    try:
                        csv_df = pd.read_csv(csv_path)
                    except Exception as e:
                        logger.warning(f"Failed to read CSV {csv_path}: {e}")
                
                # 3. Load Data Container (NPZ or Directory)
                if os.path.isdir(fpath):
                    # Virtual "data" object that behaves like np.load result
                    class LooseData:
                        def __init__(self, d): 
                            self.d = d
                            self.files = [f.replace(".npy", "") for f in os.listdir(d) if f.endswith(".npy")]
                        def __getitem__(self, k): return np.load(os.path.join(self.d, k + ".npy"), allow_pickle=True)
                        def __contains__(self, k): return os.path.exists(os.path.join(self.d, k + ".npy"))
                        def __enter__(self): return self
                        def __exit__(self, *args): pass
                    data_manager = LooseData(fpath)
                else:
                    data_manager = np.load(fpath, allow_pickle=True)

                with data_manager as data:
                    # Determine N_Traces
                    if 'no' in data:
                        n_traces = data['no'].shape[0]
                    elif 'trace_data' in data:
                        n_traces = data['trace_data'].shape[0]
                    else:
                        # Fallback infer from any available key
                        keys = [k for k in data.files if k != 'trace_data']
                        if keys and len(data[keys[0]]) > 0:
                            val = data[keys[0]]
                            n_traces = val.shape[0] if val.ndim > 0 else 1
                        else:
                            raise ValueError(f"Could not determine trace count for {fpath}")

                    # Initialize DF
                    df = pd.DataFrame(index=range(n_traces))
                    df['trace_file'] = fpath
                    df['trace_idx'] = list(range(n_traces))
                    
                    # 4. Populate Data from CSV or NPZ
                    # Priority: CSV (contains Labels/ACR string) > NPZ
                    
                    # (A) Track2 -> Keys
                    track2_source = []
                    if not csv_df.empty and 'Track2' in csv_df.columns:
                        track2_source = csv_df['Track2'].astype(str).str.strip().tolist()
                    elif 'Track2' in data:
                        val = data['Track2']
                        track2_source = [str(x) for x in val] if val.ndim > 0 else [str(val)] * n_traces
                    else:
                        track2_source = [""] * n_traces
                    
                    # Ensure length match
                    if len(track2_source) != n_traces:
                        track2_source = pad_list(track2_source, n_traces)
                    
                    df['Track2'] = track2_source
                    
                    # Map Keys
                    # Map Keys - Prioritize Reference, Fallback to Source
                    t_des_keys = []
                    
                    # Pre-fetch source keys if available
                    source_keys = []
                    if not csv_df.empty and 'T_DES_KENC' in csv_df.columns:
                        source_keys = csv_df['T_DES_KENC'].astype(str).tolist()
                    elif 'T_DES_KENC' in data:
                         val = data['T_DES_KENC']
                         source_keys = [str(x) for x in val] if val.ndim > 0 else [str(val)] * n_traces
                    elif '3DES_KEY' in data: # Try alternative name
                         val = data['3DES_KEY']
                         source_keys = [str(x) for x in val] if val.ndim > 0 else [str(val)] * n_traces
                    else:
                        source_keys = [""] * n_traces
                    
                    if len(source_keys) != n_traces: source_keys = pad_list(source_keys, n_traces)

                    for idx, t2 in enumerate(df['Track2']):
                        t2_clean = str(t2).strip()
                        # Extract first 16 chars of T2 for matching if it's long
                        t2_match = t2_clean[:16]
                        
                        found_key = '0000000000000000'
                        # Priority 1: Removed (Pure ML)
                        
                        # Priority 2: Source NPZ/CSV
                        if found_key == '0000000000000000':
                            src_key = source_keys[idx] if idx < len(source_keys) else '0000000000000000'
                            if src_key and src_key.strip() not in ["", "nan", "0000000000000000"]:
                                found_key = src_key
                        
                        t_des_keys.append(found_key)
                    df['T_DES_KENC'] = t_des_keys

                    # (B) ACR / ATC / Input Block
                    # We need 'ACR_send' to extract Challenge bytes
                    acr_source = []
                    if not csv_df.empty and 'ACR_send' in csv_df.columns:
                        acr_source = csv_df['ACR_send'].astype(str).tolist()
                    elif 'ACR_send' in data:
                        val = data['ACR_send']
                        acr_source = [str(x) for x in val] if val.ndim > 0 else [str(val)] * n_traces
                    else:
                        acr_source = ["0000000000000000"] * n_traces # Default
                    
                    # (B2) Load ATC from NPZ if available
                    if 'ATC' in data:
                        val = data['ATC']
                        atc_source = []
                        if val.ndim > 0:
                            for x in val:
                                if isinstance(x, (np.ndarray, list)):
                                    # Join elements: ['7A', 'CD'] -> "7ACD"
                                    # Handle items that might be integers or strings
                                    clean_hex = "".join([str(i).replace("'", "").strip() for i in x])
                                    atc_source.append(clean_hex)
                                else:
                                    atc_source.append(str(x))
                        else:
                            # Scalar case
                            if isinstance(val, (np.ndarray, list)):
                                atc_source = ["".join([str(i) for i in val])] * n_traces
                            else:
                                atc_source = [str(val)] * n_traces
                    else:
                        atc_source = [""] * n_traces
                    
                    if len(atc_source) != n_traces: atc_source = pad_list(atc_source, n_traces)
                    df['ATC'] = atc_source
                    
                    # Parse ACR Hex String -> Extract Challenge bytes (Bytes 5-8?)
                    # Format: 00 88 00 00 04 D1 D2 D3 D4 00...
                    # Hex string might be "0088..." (no spaces) or "00 88..."
                    atc_0, atc_1, atc_2, atc_3, atc_4, atc_5, atc_6, atc_7 = [], [], [], [], [], [], [], []
                    
                    for acr, atc_raw in zip(acr_source, df.get('ATC', [""] * n_traces)):
                        clean_acr = acr.replace(" ", "")
                        try:
                            if clean_acr and clean_acr != "0000000000000000":
                                # Extract 8 bytes (16 chars) starting from offset 10 (Byte 5)
                                challenge_hex = clean_acr[10:26].ljust(16, '0')
                                b = bytes.fromhex(challenge_hex)
                            elif str(atc_raw).strip() not in ["", "nan"]:
                                # Fallback to ATC column (e.g., "7A CD")
                                # Pad to 8 bytes for consistency with Internal Auth challenge
                                atc_hex = str(atc_raw).replace(" ", "").zfill(16)
                                b = bytes.fromhex(atc_hex)
                            else:
                                b = b'\x00' * 8
                                
                            atc_0.append(b[0]); atc_1.append(b[1]); atc_2.append(b[2]); atc_3.append(b[3])
                            atc_4.append(b[4]); atc_5.append(b[5]); atc_6.append(b[6]); atc_7.append(b[7])
                        except Exception:
                            atc_0.append(0); atc_1.append(0); atc_2.append(0); atc_3.append(0)
                            atc_4.append(0); atc_5.append(0); atc_6.append(0); atc_7.append(0)

                    df['ATC_0'] = atc_0
                    df['ATC_1'] = atc_1
                    df['ATC_2'] = atc_2
                    df['ATC_3'] = atc_3
                    df['ATC_4'] = atc_4
                    df['ATC_5'] = atc_5
                    df['ATC_6'] = atc_6
                    df['ATC_7'] = atc_7
                    
                    # (C) Capture IO / C7
                    if not csv_df.empty and 'ACR_receive' in csv_df.columns:
                         df['C7'] = csv_df['ACR_send'].astype(str) + " " + csv_df['ACR_receive'].astype(str)
                    else:
                        df['C7'] = ""
                    
                    # (C2) Load Verify Command (APDU) from Data or NPY
                    # This is CRITICAL for PIN Extraction
                    verify_data = []
                    
                    if 'Verify_command' in data:
                        v_dat = data['Verify_command']
                        if v_dat.ndim == 0:
                             verify_data = [str(v_dat.item()).replace(" ", "").upper()] * n_traces
                        elif v_dat.dtype.kind in ['U', 'S']:
                             verify_data = [str(x).replace(" ", "").upper() for x in v_dat]
                        else:
                             if v_dat.ndim > 1:
                                 verify_data = ["".join([f"{b:02X}" for b in row]) for row in v_dat]
                             else:
                                 verify_data = ["".join([f"{b:02X}" for b in v_dat])] * n_traces
                    
                    elif os.path.exists(os.path.join(dirname, "Verify_command.npy")):
                        try:
                            v_dat = np.load(os.path.join(dirname, "Verify_command.npy"))
                            if v_dat.ndim == 0:
                                val_str = str(v_dat.item()).replace(" ", "").upper()
                                verify_data = [val_str] * n_traces
                            elif v_dat.dtype.kind in ['U', 'S']:
                                verify_data = [str(x).replace(" ", "").upper() for x in v_dat]
                            else:
                                if v_dat.ndim > 1:
                                    verify_data = ["".join([f"{b:02X}" for b in row]) for row in v_dat]
                                else:
                                    verify_data = ["".join([f"{b:02X}" for b in v_dat])] * n_traces
                        except Exception as e:
                             logger.warning(f"Failed to load Verify_command.npy: {e}")
                             verify_data = [""] * n_traces
                    else:
                        verify_data = [""] * n_traces
                    
                    # Assign to 'apdu' column
                    df['apdu'] = verify_data
                    
                    # Ensure other columns exist
                    for col in ['T_DES_KMAC', 'T_DES_KDEK', 'RSA_KENC', 'RSA_KMAC', 'RSA_KDEK', 'IO', 'apdu', 'EncryptedPIN', 'AIP', 'IAD', 'ATC']:

                        if col not in df.columns:
                            if not csv_df.empty and col in csv_df.columns:
                                df[col] = csv_df[col]
                            else:
                                df[col] = ""

                    # (D) Advanced EMV Metadata via Parser
                    try:
                        parser = MetadataParser(fpath)
                        if parser.load_trace():
                            emv_meta = parser.extract_metadata()
                            # Map extracted tags to DF columns
                            if 'ATC' in emv_meta:
                                df['ATC'] = emv_meta['ATC']
                            if 'AIP' in emv_meta:
                                df['AIP'] = emv_meta['AIP']
                            if 'IAD' in emv_meta:
                                df['IAD'] = emv_meta['IAD']
                            if 'PAN' in emv_meta:
                                df['PAN'] = emv_meta['PAN']
                            if 'TRACK2' in emv_meta and not df['Track2'].iloc[0]:
                                df['Track2'] = emv_meta['TRACK2']
                            if 'EncryptedPIN' in emv_meta:
                                df['EncryptedPIN'] = emv_meta['EncryptedPIN']
                            if 'apdu' in emv_meta and 'apdu' in df.columns:
                                # We might prefer the parsed APDU if it was extracted better
                                pass 
                    except Exception as e:
                        logger.warning(f"Metadata parser failed for {fpath}: {e}")

                    meta_list.append(df)
                    
            except Exception as e:
                logger.error(f"Error loading {fpath}: {e}")

        if not meta_list:
             return pd.DataFrame()
        
        full_meta = pd.concat(meta_list, ignore_index=True)
        return full_meta

    def get_traces(self, indices: np.ndarray) -> np.ndarray:
        # Load traces for specific indices
        # This is slow if indices are scattered across files
        # Optimize by grouping by file
        
        # We assume indices are 0..total_traces-1 coresponding to metadata index
        
        subset = self.metadata.iloc[indices]
        grouped = subset.groupby('file')
        
        trace_list = []
        # Result needs to be ordered by input indices
        # So we'll store in a dict and reconstruction
        
        results = {}
        
        for fpath, group in grouped:
            local_indices = group['local_index'].values
            global_indices = group.index.values
            
            # Load file
            # Use mmap to read specific rows? 
            # npz doesn't support mmap on compressed data (default) easily unless uncompressed.
            # But np.load supports mmap_mode if file is not compressed?
            # 'traces_data_*.npz' likely compressed (savez_compressed) or just savez?
            # If compressed, mmap_mode might not work or just loads whole thing.
            
            try:
                # Attempt to use mmap_mode='r'
                data = np.load(fpath, mmap_mode='r')
                traces = data['trace_data']
                
                # If traces is just (N, samples), we can slice
                # but with mmap, fancy indexing might trigger full read if not careful?
                # Actually, fancy indexing on mmap array works but reads from disk.
                
                selected = traces[local_indices]
                
                for glob_idx, trace in zip(global_indices, selected):
                    results[glob_idx] = trace
                    
            except Exception as e:
                logger.error(f"Error loading traces from {fpath}: {e}")
                
        # Reconstruct list
        final_traces = [results[i] for i in indices if i in results]
        return np.array(final_traces)

    def get_all_traces_iterator(self, batch_size=1000, target_apdu=None):
        """
        Iterate trace content with optional APDU-based segmentation.
        """
        for fpath in self.files:
            if os.path.isdir(fpath):
                # Loose files in directory
                trace_path = os.path.join(fpath, "trace_data.npy")
                if not os.path.exists(trace_path): continue
                full_traces = np.load(trace_path, mmap_mode='r')
            else:
                data = np.load(fpath, mmap_mode='r')
                if 'trace_data' not in data: continue
                full_traces = data['trace_data']
            
            n = full_traces.shape[0]
            
            # Find command indices if target_apdu is specified
            apdu_indices = None
            if target_apdu:
                for key in ['Verify_command', 'ACR_send', 'apdu']:
                    if key in data:
                        cmds = data[key]
                        # Find index where target_apdu appears in the sequence
                        # This assumes trace_data is concatenated power for all cmds?
                        # Or if trace_data is (N, samples), does samples cover ALL cmds?
                        # Based on audit, samples=131k covers multiple cmds.
                        pass
            
            for i in range(0, n, batch_size):
                # We yield the meta as well
                meta_subset = self.metadata[self.metadata['trace_file'] == fpath].iloc[i:i+batch_size]
                yield full_traces[i:i+batch_size], meta_subset

if __name__ == "__main__":
    ds = TraceDataset("Input")
    print(ds.metadata.head())
    print(f"Loaded {len(ds.metadata)} metadata rows")
