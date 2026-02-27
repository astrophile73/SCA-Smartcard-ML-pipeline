"""
Data Loader for 3DES Key Extraction Pipeline

This module handles loading and preprocessing of power traces from NPZ files.
Focus on generalization through Z-Score normalization.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Union
from sklearn.model_selection import train_test_split


class DataLoader:
    """Load and preprocess 3DES power traces from Mastercard dataset."""
    
    def __init__(
        self,
        data_dir: Union[str, List[str]] = None,
        mode: str = "3des"
    ):
        """
        Initialize data loader.
        
        Args:
            data_dir: Directory or list of directories containing NPZ files.
                     If None, scans both Mastercard and Visa folders.
            mode: File selection mode.
                  - "3des": keep only files containing 3DES keys
                  - "rsa": keep only files containing RSA I/O fields
                  - "all": keep all NPZ files matching pattern
        """
        if data_dir is None:
            self.data_dirs = [Path("Input/MasterCard"), Path("Input/Visa")]
        elif isinstance(data_dir, (list, tuple)):
            self.data_dirs = [Path(d) for d in data_dir]
        else:
            self.data_dirs = [Path(data_dir)]
        
        self.mode = mode.lower()
        if self.mode not in {"3des", "rsa", "all"}:
            raise ValueError("mode must be one of: '3des', 'rsa', 'all'")
        
        all_npz_files: List[Path] = []
        for directory in self.data_dirs:
            all_npz_files.extend(sorted(directory.glob("traces_data_*.npz")))
        
        # De-duplicate while preserving deterministic order
        all_npz_files = sorted(set(all_npz_files), key=lambda p: str(p).lower())
        
        filtered_files = self._filter_by_mode(all_npz_files)
        
        # Limit to first 4 files (7000 traces) to avoid memory issues
        # Skip the large 3000T_5.npz file that causes memory allocation errors
       # self.npz_files = filtered_files[:4]  # Only load 1000T + 2000T + 2000T + 2000T = 7000 traces
        self.npz_files = filtered_files
        if not self.npz_files:
            searched = ", ".join(str(d) for d in self.data_dirs)
            raise FileNotFoundError(
                f"No NPZ files found for mode '{self.mode}' in: {searched}"
            )
        
        print(f"Found {len(self.npz_files)} NPZ files:")
        for f in self.npz_files:
            print(f"  - {f.name}")

    def _filter_by_mode(self, files: List[Path]) -> List[Path]:
        """Filter files based on actual NPZ keys rather than filename heuristics."""
        if self.mode == "all":
            return files
        
        selected = []
        for f in files:
            try:
                with np.load(f, mmap_mode="r") as data:
                    keys = set(data.files)
            except Exception:
                # Skip unreadable/corrupt files
                continue
            
            if self.mode == "3des":
                # 3DES training files contain DES key fields + ATC
                required = {"T_DES_KENC", "T_DES_KMAC", "T_DES_KDEK", "trace_data"}
                if required.issubset(keys) and "white" not in f.name.lower():
                    selected.append(f)
            elif self.mode == "rsa":
                # RSA files contain ACR send/receive metadata
                required = {"ACR_send", "ACR_receive", "trace_data"}
                if required.issubset(keys):
                    selected.append(f)
        
        return selected
    
    def trace_generator(self):
        """Generator that yields one trace at a time for memory-efficient processing."""
        for npz_file in self.npz_files:
            data = np.load(npz_file, mmap_mode='r')
            raw_traces = data['trace_data']
            for i in range(len(raw_traces)):
                yield raw_traces[i].astype(np.float32)
            del data
    
    def load_all_traces(self, poi_indices: np.ndarray = None) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Load traces from NPZ files. If poi_indices is provided, only those samples are loaded
        into the final array to save memory.
        """
        all_traces = []
        all_kenc, all_kmac, all_kdek, all_atc = [], [], [], []
        
        for npz_file in self.npz_files:
            print(f"\nLoading {npz_file.name}...")
            data = np.load(npz_file, allow_pickle=True)
            raw_traces = data['trace_data']
            num_traces = raw_traces.shape[0]
            
            if poi_indices is not None:
                # Reduced load
                traces = raw_traces[:, poi_indices].astype(np.float32)
            else:
                # FULL load (WARNING: High RAM usage)
                traces = raw_traces.astype(np.float32)
            
            all_traces.append(traces)
            all_kenc.extend([str(data['T_DES_KENC'])] * num_traces)
            all_kmac.extend([str(data['T_DES_KMAC'])] * num_traces)
            all_kdek.extend([str(data['T_DES_KDEK'])] * num_traces)
            if 'ATC' in data: all_atc.extend(data['ATC'])
            del raw_traces
            del data
        
        return np.vstack(all_traces), {
            'KENC': np.array(all_kenc), 'KMAC': np.array(all_kmac), 
            'KDEK': np.array(all_kdek), 'ATC': np.array(all_atc) if all_atc else None
        }
    
    @staticmethod
    def normalize_trace(trace: np.ndarray) -> np.ndarray:
        """
        Apply Z-Score normalization to a single trace.
        
        This is CRITICAL for generalization - it removes absolute power levels
        and focuses on relative patterns, preventing memorization of specific
        acquisition conditions.
        
        Args:
            trace: Raw power trace
            
        Returns:
            Normalized trace with mean=0, std=1
        """
        mean = np.mean(trace)
        std = np.std(trace)
        
        # Add epsilon to avoid division by zero
        return ((trace - mean) / (std + 1e-8)).astype(np.float32)
    
    def normalize_traces(self, traces: np.ndarray) -> np.ndarray:
        """
        Apply Z-Score normalization to all traces (per-trace, in-place).
        
        Args:
            traces: Array of shape (N, trace_length)
            
        Returns:
            Normalized traces of same shape (modifies in-place)
        """
        print("\nNormalizing traces (Z-Score per trace, in-place)...")
        
        # Normalize in-place to save memory
        for i in range(len(traces)):
            if (i + 1) % 1000 == 0:
                print(f"  Normalized {i + 1}/{len(traces)} traces...")
            
            mean = np.mean(traces[i])
            std = np.std(traces[i])
            traces[i] = (traces[i] - mean) / (std + 1e-8)
        
        print(f"[OK] Normalized {len(traces)} traces")
        print(f"  Mean of means: {np.mean([np.mean(t) for t in traces[:100]]):.6f} (should be ~0)")
        print(f"  Mean of stds: {np.mean([np.std(t) for t in traces[:100]]):.6f} (should be ~1)")
        
        return traces
    
    def train_val_split(
        self, 
        traces: np.ndarray, 
        labels: Dict[str, np.ndarray],
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Split data into training and validation sets.
        
        Args:
            traces: Normalized traces
            labels: Dictionary of labels
            test_size: Fraction for validation (default 0.2 = 20%)
            random_state: Random seed for reproducibility
            
        Returns:
            X_train, X_val, y_train, y_val
        """
        # Create indices for splitting
        indices = np.arange(len(traces))
        train_idx, val_idx = train_test_split(
            indices, 
            test_size=test_size, 
            random_state=random_state
        )
        
        X_train = traces[train_idx]
        X_val = traces[val_idx]
        
        y_train = {k: v[train_idx] if v is not None else None for k, v in labels.items()}
        y_val = {k: v[val_idx] if v is not None else None for k, v in labels.items()}
        
        print(f"\n✓ Train/Val split:")
        print(f"  Training: {len(X_train)} traces ({(1-test_size)*100:.0f}%)")
        print(f"  Validation: {len(X_val)} traces ({test_size*100:.0f}%)")
        
        return X_train, X_val, y_train, y_val
    
    def prepare_data(
        self, 
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Complete data preparation pipeline.
        
        Returns:
            X_train, X_val, y_train, y_val
        """
        # Load all traces
        traces, labels = self.load_all_traces()
        
        # Normalize traces (CRITICAL for generalization)
        traces_normalized = self.normalize_traces(traces)
        
        # Split into train/val
        X_train, X_val, y_train, y_val = self.train_val_split(
            traces_normalized, 
            labels, 
            test_size=test_size,
            random_state=random_state
        )
        
        return X_train, X_val, y_train, y_val


if __name__ == "__main__":
    # Test data loader
    print("=" * 60)
    print("Testing Data Loader")
    print("=" * 60)
    
    loader = DataLoader()
    X_train, X_val, y_train, y_val = loader.prepare_data()
    
    print("\n" + "=" * 60)
    print("Data Preparation Complete!")
    print("=" * 60)
    print(f"\nTraining set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"\nSample normalized trace stats:")
    print(f"  Mean: {np.mean(X_train[0]):.6f}")
    print(f"  Std: {np.std(X_train[0]):.6f}")
    print(f"  Min: {np.min(X_train[0]):.6f}")
    print(f"  Max: {np.max(X_train[0]):.6f}")
