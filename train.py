"""
Training Pipeline for 3DES S-Box Models (Mode 1: TRAIN)

Trains 8 decoupled S-Box models with strong regularization to learn
general SCA attack methodology rather than memorizing specific traces.
"""

import numpy as np
import os
import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import time
import shutil

import tensorflow as tf
from data_loader import DataLoader
from label_generator import LabelGenerator
from model_builder import ModelBuilder


class Trainer:
    """Train 8 S-Box models for 3DES key recovery."""
    
    def __init__(
        self,
        data_dir: str = "Input/Mastercard",
        models_dir: str = "models",
        results_dir: str = "results",
        force_train: bool = False
    ):
        """
        Initialize trainer.
        
        Args:
            data_dir: Directory containing NPZ files
            models_dir: Directory to save trained models
            results_dir: Directory to save training results
            force_train: Whether to overwrite existing models
        """
        self.data_dir = data_dir
        self.models_dir = Path(models_dir)
        self.results_dir = Path(results_dir)
        self.force_train = force_train
        
        # Create directories
        self.models_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.data_loader = DataLoader(data_dir)
        self.label_generator = LabelGenerator()
        self.model_builder = None  # Will be initialized after loading data
        
        # Training history
        self.histories = []
    
    def prepare_data(self, stage: int = 1):
        """Load and prepare training data with extreme memory efficiency."""
        print("\n" + "=" * 60)
        print(f"STEP 1: Data Preparation (Stage {stage})")
        print("=" * 60)
        
        from poi_selector import POISelector
        import gc
        
        # Use stage-specific POIs if they exist
        poi_path = self.models_dir / f"poi_indices_stage{stage}.npy"
        
        # Fallback to general POIs if stage-specific doesn't exist
        if not poi_path.exists() and stage == 1:
            poi_path = self.models_dir / "poi_indices.npy"

        poi_selector = POISelector(num_poi=5000)
        
        if poi_path.exists():
            print(f"Reading existing POI indices from {poi_path}...")
            poi_selector.load_poi(str(poi_path))
            poi_indices = poi_selector.poi_indices
        else:
            print(f"No POI indices found for Stage {stage}. Discovering POIs (Incremental)...")
            # Use incremental fit to save RAM
            poi_selector.fit_incremental(self.data_loader.trace_generator())
            poi_indices = poi_selector.poi_indices
            poi_selector.save_poi(str(poi_path))
            gc.collect()

        X_all, y_all_raw = self.data_loader.load_all_traces(poi_indices=poi_indices)
        self.data_loader.normalize_traces(X_all)
        
        X_train, X_val, y_train, y_val = self.data_loader.train_val_split(
            X_all, y_all_raw, test_size=0.2, random_state=42
        )
        del X_all
        gc.collect()
        
        self.model_builder = ModelBuilder(trace_length=len(poi_indices))
        
        print(f"\nSTEP 3: Label Generation (Stage {stage})")
        all_train_labels = self.label_generator.generate_labels_for_all_keys(
            y_train['KENC'], y_train['KMAC'], y_train['KDEK'], stage=stage
        )
        all_val_labels = self.label_generator.generate_labels_for_all_keys(
            y_val['KENC'], y_val['KMAC'], y_val['KDEK'], stage=stage
        )
        
        train_labels_cat = {}
        val_labels_cat = {}
        for key_type in ['KENC', 'KMAC', 'KDEK']:
            train_labels_cat[key_type] = self.label_generator.labels_to_categorical(all_train_labels[key_type])
            val_labels_cat[key_type] = self.label_generator.labels_to_categorical(all_val_labels[key_type])
        
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
        
        return X_train, X_val, train_labels_cat, val_labels_cat

    def train_sbox_model(self, sbox_idx, key_type, model, X_train, y_train, X_val, y_val, stage, epochs, batch_size):
        """Train a single S-Box model for a specific key and stage."""
        name = f"{key_type.lower()}_sbox_{sbox_idx}_s{stage}"
        callbacks = self.model_builder.get_callbacks(name, patience=10)
        
        history = model.fit(
            X_train, y_train, validation_data=(X_val, y_val),
            epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=1
        )
        
        model_name = f"sbox_{sbox_idx}_{key_type.lower()}_s{stage}.keras"
        self.save_model_safe(model, self.models_dir / model_name)
        return history

    def save_model_safe(self, model, filepath: Path):
        """Save model with retries to handle Windows file locking."""
        temp_path = filepath.with_suffix('.temp.keras')
        for attempt in range(3):
            try:
                model.save(str(temp_path))
                if filepath.exists():
                    os.remove(filepath)
                shutil.move(str(temp_path), str(filepath))
                print(f"✓ Saved: {filepath.name}")
                return
            except Exception as e:
                print(f"⚠️ Save attempt {attempt+1} failed: {e}")
                time.sleep(2)
        print(f"❌ Failed to save model: {filepath}")
    
    def train_all_models(self, stage: int = 1, epochs: int = 100, batch_size: int = 128):
        """Train all 24 S-Box models for a specific stage."""
        X_train, X_val, train_labels_cat, val_labels_cat = self.prepare_data(stage=stage)
        
        force_train = getattr(self, 'force_train', False)
        total = 0
        for key_type in ['KENC', 'KMAC', 'KDEK']:
            for sbox_idx in range(8):
                total += 1
                model_name = f"sbox_{sbox_idx}_{key_type.lower()}_s{stage}.keras"
                if (self.models_dir / model_name).exists() and not force_train:
                    print(f"⏩ Skipping {model_name}")
                    continue
                
                print(f"\n[Model {total}/24] Training Stage {stage} {key_type} S-Box {sbox_idx}")
                model = self.model_builder.build_sbox_model(sbox_idx)
                history = self.train_sbox_model(sbox_idx, key_type, model, X_train, train_labels_cat[key_type][sbox_idx], X_val, val_labels_cat[key_type][sbox_idx], stage, epochs, batch_size)
                
                self.histories.append({'key_type': key_type, 'sbox_idx': sbox_idx, 'stage': stage, 'history': history.history})
                
                import gc
                tf.keras.backend.clear_session()
                del model
                gc.collect()

        self.save_training_metadata(stage)
        self.plot_training_history(stage)

    def save_training_metadata(self, stage=1):
        """Save training metadata."""
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'stage': stage,
            'total_models': len(self.histories),
            'metrics': [
                {
                    'key': h['key_type'], 'sbox': h['sbox_idx'], 
                    'acc': float(h['history']['val_accuracy'][-1])
                } for h in self.histories
            ]
        }
        with open(self.results_dir / f'metadata_s{stage}.json', 'w') as f:
            json.dump(metadata, f, indent=2)

    def plot_training_history(self, stage=1):
        """Plot training history."""
        # Simplified plot logic
        plt.figure(figsize=(10, 6))
        for h in self.histories:
            plt.plot(h['history']['val_accuracy'], label=f"{h['key_type']} S{h['sbox_idx']}")
        plt.title(f"Stage {stage} Validation Accuracy")
        plt.savefig(self.results_dir / f"history_s{stage}.png")
        plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--stage', type=int, default=1, help='1 for K1, 2 for K2')
    parser.add_argument('--data-dir', type=str, default='Input/Mastercard')
    parser.add_argument('--force', action='store_true')
    
    args = parser.parse_args()
    trainer = Trainer(data_dir=args.data_dir, force_train=args.force)
    trainer.train_all_models(stage=args.stage, epochs=args.epochs, batch_size=args.batch_size)

if __name__ == "__main__":
    raise SystemExit(
        "Deprecated entrypoint (legacy TensorFlow pipeline).\n"
        "Use the supported pipeline instead:\n"
        "  python main.py --mode train --input_dir <DIR> --processed_dir <DIR> --opt_dir <DIR> --scan_type 3des\n"
    )
