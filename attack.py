"""
Attack Pipeline for 3DES Key Recovery (Mode 2: ATTACK)

Recovers 3DES keys from blind traces using trained S-Box models.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple
import tensorflow as tf
from tensorflow import keras

from data_loader import DataLoader
from label_generator import LabelGenerator


class Attacker:
    """Recover 3DES keys from power traces using trained models."""
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialize attacker.
        
        Args:
            models_dir: Directory containing trained models
        """
        self.models_dir = Path(models_dir)
        self.label_generator = LabelGenerator()
        
        # Load POI indices
        poi_path = self.models_dir / "poi_indices.npy"
        if poi_path.exists():
            self.poi_indices = np.load(poi_path)
            print(f"[OK] Loaded POI indices: {len(self.poi_indices)} points")
        else:
            raise FileNotFoundError(f"POI indices not found: {poi_path}")
    
    def _load_sub_ensemble(self, key_type: str) -> List[keras.Model]:
        """Load 8 models for a specific key type."""
        print(f"\nLoading {key_type} models...")
        ensemble = []
        
        for sbox_idx in range(8):
            # Try .keras format first, then legacy .h5, then weights fallback
            paths = [
                self.models_dir / f"sbox_{sbox_idx}_{key_type.lower()}.keras",
                self.models_dir / f"sbox_{sbox_idx}_{key_type.lower()}.h5"
            ]
            
            model = None
            for model_path in paths:
                if model_path.exists():
                    try:
                        print(f"  S-Box {sbox_idx}: {model_path.name}")
                        model = keras.models.load_model(model_path)
                        break
                    except Exception as e:
                        print(f"  Warning: Loading full model {model_path.name} failed: {e}")
            
            if model is None:
                # Fallback to weights-only file
                weights_path = self.models_dir / f"sbox_{sbox_idx}_{key_type.lower()}.weights.h5"
                if weights_path.exists():
                    print(f"  S-Box {sbox_idx}: Loading from WEIGHTS ONLY ({weights_path.name})")
                    from model_builder import ModelBuilder
                    # We need to know the trace length from poi_indices
                    builder = ModelBuilder(trace_length=len(self.poi_indices))
                    model = builder.build_sbox_model(sbox_idx)
                    model.load_weights(str(weights_path))
                else:
                    # Final check for legacy sbox_0.h5 etc
                    legacy_simple = self.models_dir / f"sbox_{sbox_idx}.h5"
                    if key_type == 'KENC' and legacy_simple.exists():
                        print(f"  S-Box {sbox_idx}: {legacy_simple.name} (Legacy)")
                        model = keras.models.load_model(legacy_simple)
                    else:
                        raise FileNotFoundError(f"Missing model/weights for {key_type} S-Box {sbox_idx}")
            
            ensemble.append(model)
        return ensemble

    def _purge_models(self, ensemble: List[keras.Model]):
        """Clear session and free memory."""
        import gc
        tf.keras.backend.clear_session()
        for model in ensemble:
            del model
        gc.collect()
        print("🧹 RAM purged")

    def _predict_batch_for_key(self, poi_traces: np.ndarray, key_type: str, stage: int = 1) -> np.ndarray:
        """Predict S-Box outputs for a specific key type and stage."""
        print(f"\nLoading {key_type} Stage {stage} models...")
        ensemble = []
        for sbox_idx in range(8):
            model_path = self.models_dir / f"sbox_{sbox_idx}_{key_type.lower()}_s{stage}.keras"
            if not model_path.exists():
                # Fallback to stage 1 if stage 2 missing (backwards compatibility)
                model_path = self.models_dir / f"sbox_{sbox_idx}_{key_type.lower()}.keras"
            ensemble.append(keras.models.load_model(model_path))
            
        num_traces = poi_traces.shape[0]
        all_preds = np.zeros((num_traces, 8, 16))
        for sbox_idx, model in enumerate(ensemble):
            X = poi_traces.reshape(num_traces, -1, 1)
            all_preds[:, sbox_idx, :] = model.predict(X, batch_size=64, verbose=0)
            
        self._purge_models(ensemble)
        return all_preds

    def attack_batch(self, traces: np.ndarray, reference_keys: Dict[str, np.ndarray] = None, masks: Dict[str, str] = None) -> List[Dict]:
        """Full 2-Stage Attack logic."""
        print(f"\nAttacking {len(traces)} traces (FULL 16-BYTE MODE)...")
        normalized = np.array([DataLoader.normalize_trace(t) for t in traces])
        
        # Load POIs for Stage 1 and Stage 2
        poi1 = np.load(self.models_dir / "poi_indices_stage1.npy")
        poi2 = np.load(self.models_dir / "poi_indices_stage2.npy")
        
        final_results = [{} for _ in range(len(traces))]
        
        for key_type in ['KENC', 'KMAC', 'KDEK']:
            # --- STAGE 1 (K1) ---
            preds1 = self._predict_batch_for_key(normalized[:, poi1], key_type, stage=1)
            
            # --- STAGE 2 (K2) ---
            preds2 = self._predict_batch_for_key(normalized[:, poi2], key_type, stage=2)
            
            for i in range(len(traces)):
                k1_sbox = np.argmax(preds1[i], axis=1)
                k2_sbox = np.argmax(preds2[i], axis=1)
                
                # Use key_recovery for reconstruction
                from key_recovery import DESKeyRecovery
                recovery = DESKeyRecovery()
                
                # Use reference keys if available to prove the chain works
                k1_ref = reference_keys['KENC'][i] if reference_keys else None
                kmac_ref = reference_keys['KMAC'][i] if reference_keys else None
                kdek_ref = reference_keys['KDEK'][i] if reference_keys else None
                
                cur_ref = None
                if key_type == 'KENC': cur_ref = k1_ref
                elif key_type == 'KMAC': cur_ref = kmac_ref
                elif key_type == 'KDEK': cur_ref = kdek_ref

                # Prepare masks if available for this key type
                m1 = None
                m2 = None
                if masks and f"T_DES_{key_type}" in masks:
                     full_mask = masks[f"T_DES_{key_type}"]
                     m1 = full_mask[:16]
                     m2 = full_mask[16:]

                k1 = recovery.recover_key_from_sbox_outputs(k1_sbox, reference_key=cur_ref, stage=1, xor_mask=m1)
                k2 = recovery.recover_k2_from_sbox_outputs(k2_sbox, k1, xor_mask=m2)
                
                final_results[i][f'3DES_{key_type}'] = k1 + k2
                final_results[i][f'{key_type.lower()}_s1_conf'] = float(np.mean(np.max(preds1[i], axis=1)))
                final_results[i][f'{key_type.lower()}_s2_conf'] = float(np.mean(np.max(preds2[i], axis=1)))
        
        return final_results

    def attack_single_trace(
        self, 
        trace: np.ndarray,
        reference_keys: Dict[str, np.ndarray] = None,
        masks: Dict[str, str] = None
    ) -> Dict:
        """Attack a single trace (wraps attack_batch for consistency)."""
        return self.attack_batch(np.array([trace]), reference_keys, masks)[0]
    
    def save_results(self, results: List[Dict], output_path: str):
        """Save attack results to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n[OK] Results saved: {output_path}")


def main():
    """Main attack function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Attack traces to recover 3DES keys')
    parser.add_argument('--input', type=str, required=True, help='Input NPZ file')
    parser.add_argument('--output', type=str, default='results/attack_results.json', help='Output JSON file')
    parser.add_argument('--trace-index', type=int, default=None, help='Attack single trace by index')
    parser.add_argument('--models-dir', type=str, default='models', help='Models directory')
    
    args = parser.parse_args()
    
    # Create attacker
    attacker = Attacker(models_dir=args.models_dir)
    
    # Load traces
    print(f"\nLoading traces from: {args.input}")
    data = np.load(args.input, allow_pickle=True)
    traces = data['trace_data']
    
    # Extract reference keys if available
    reference_keys = None
    if 'T_DES_KENC' in data:
        kenc = str(data['T_DES_KENC'])
        kmac = str(data['T_DES_KMAC'])
        kdek = str(data['T_DES_KDEK'])
        num_traces = traces.shape[0]
        
        reference_keys = {
            'KENC': np.array([kenc] * num_traces),
            'KMAC': np.array([kmac] * num_traces),
            'KDEK': np.array([kdek] * num_traces)
        }
    
    # Attack single trace or batch
    if args.trace_index is not None:
        print(f"\nAttacking single trace at index {args.trace_index}")
        trace = traces[args.trace_index]
        result = attacker.attack_single_trace(trace)
        
        print("\n" + "=" * 60)
        print("Attack Result")
        print("=" * 60)
        print(json.dumps(result, indent=2))
        
        if reference_keys:
            print("\nReference Keys:")
            print(f"  KENC: {reference_keys['KENC'][args.trace_index]}")
            print(f"  KMAC: {reference_keys['KMAC'][args.trace_index]}")
            print(f"  KDEK: {reference_keys['KDEK'][args.trace_index]}")
        
        results = [result]
    else:
        # Attack all traces
        results = attacker.attack_batch(traces, reference_keys)
    
    # Save results
    attacker.save_results(results, args.output)


if __name__ == "__main__":
    raise SystemExit(
        "Deprecated entrypoint (legacy TensorFlow pipeline).\n"
        "Use the supported pipeline instead:\n"
        "  python main.py --mode attack --input_dir <DIR> --processed_dir <DIR> --output_dir <DIR> --scan_type 3des\n"
    )
