"""
Validation Pipeline for 3DES Models (Mode 3: VALIDATE)

Validates trained models on test set and generates accuracy metrics.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime

from data_loader import DataLoader
from label_generator import LabelGenerator
from attack import Attacker


class Validator:
    """Validate trained S-Box models."""
    
    def __init__(
        self,
        data_dir: str = "Input/Mastercard",
        models_dir: str = "models",
        results_dir: str = "results"
    ):
        """
        Initialize validator.
        
        Args:
            data_dir: Directory containing NPZ files
            models_dir: Directory containing trained models
            results_dir: Directory to save validation results
        """
        self.data_dir = data_dir
        self.models_dir = Path(models_dir)
        self.results_dir = Path(results_dir)
        
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.data_loader = DataLoader(data_dir)
        self.label_generator = LabelGenerator()
        self.attacker = Attacker(models_dir=str(models_dir))
    
    def validate(self) -> Dict:
        """
        Run validation on test set for all 3 keys.
        """
        print("\n" + "=" * 60)
        print("VALIDATION PIPELINE (3DES ENSEMBLE)")
        print("=" * 60)
        
        # 1. Load data
        poi_path = self.models_dir / "poi_indices.npy"
        poi_indices = np.load(poi_path)
        
        print("\nLoading validation data...")
        X_all, y_all_raw = self.data_loader.load_all_traces(poi_indices=poi_indices)
        self.data_loader.normalize_traces(X_all)
        
        _, X_val, _, y_val = self.data_loader.train_val_split(
            X_all, y_all_raw, test_size=0.2, random_state=42
        )
        
        print(f"Validation set size: {len(X_val)} traces")
        
        # 2. Attack validation traces
        print("\n" + "=" * 60)
        print("Running Attack on Validation Set")
        print("=" * 60)
        
        attack_results = self.attacker.attack_batch(X_val, reference_keys=y_val)
        
        # 3. Calculate metrics for each key
        all_metrics = {
            'timestamp': datetime.now().isoformat(),
            'num_traces': len(X_val),
            'composite_success': all(r['rank_0_success'] for r in attack_results),
            'keys': {}
        }
        
        for key_type in ['KENC', 'KMAC', 'KDEK']:
            print(f"\nEvaluating {key_type} models...")
            # Generate reference labels for this key
            reference_labels = self.label_generator.generate_labels_for_dataset(y_val[key_type], key_type)
            key_metrics = self.calculate_metrics(attack_results, reference_labels, key_type)
            all_metrics['keys'][key_type] = key_metrics
        
        # 4. Global Summary Metrics
        all_metrics['global_sbox_acc'] = float(np.mean([k['mean_sbox_acc'] for k in all_metrics['keys'].values()]))
        all_metrics['rank_0_rate'] = float(np.mean([1.0 if r['rank_0_success'] else 0.0 for r in attack_results]))
        
        # 5. Generate report
        self.generate_report(all_metrics)
        
        return all_metrics
    
    def calculate_metrics(
        self,
        attack_results: List[Dict],
        reference_labels: np.ndarray,
        key_type: str
    ) -> Dict:
        """
        Calculate accuracy per S-Box for a specific key.
        """
        num_traces = len(attack_results)
        sbox_accs = []
        
        pred_key = f'{key_type.lower()}_sbox_outputs'
        conf_key = f'{key_type.lower()}_mean_conf'
        
        for sbox_idx in range(8):
            matches = 0
            for i in range(num_traces):
                predicted = attack_results[i][pred_key][sbox_idx]
                actual = reference_labels[i, sbox_idx]
                if predicted == actual:
                    matches += 1
            
            acc = matches / num_traces
            sbox_accs.append(acc)
            print(f"  S-Box {sbox_idx}: {acc*100:.2f}% accuracy")
            
        return {
            'sbox_accuracies': [float(a) for a in sbox_accs],
            'mean_sbox_acc': float(np.mean(sbox_accs)),
            'mean_confidence': float(np.mean([r[conf_key] for r in attack_results]))
        }
    
    def generate_report(self, metrics: Dict):
        """
        Produce a comprehensive validation report.
        """
        report_lines = [
            "# 3DES Model Ensemble Validation Report",
            f"**Generated**: {metrics['timestamp']}",
            "",
            "## 1. Overall Performance",
            f"- **Validation Set Size**: {metrics['num_traces']} traces",
            f"- **Rank-0 Key Recovery Rate**: {metrics['rank_0_rate']*100:.2f}%",
            f"- **Global S-Box Accuracy**: {metrics['global_sbox_acc']*100:.2f}%",
            "",
            "## 2. Per-Key Accuracy Details",
            ""
        ]
        
        for key_type, data in metrics['keys'].items():
            report_lines.extend([
                f"### {key_type}",
                f"- **Mean S-Box Accuracy**: {data['mean_sbox_acc']*100:.2f}%",
                f"- **Mean Model Confidence**: {data['mean_confidence']:.4f}",
                "",
                "| S-Box | Accuracy | Status |",
                "|:---|:---|:---|"
            ])
            for i, acc in enumerate(data['sbox_accuracies']):
                status = "✅ PASS" if acc > 0.99 else "❌ FAIL"
                report_lines.append(f"| S-Box {i} | {acc*100:.2f}% | {status} |")
            report_lines.append("")
            
        # Save files
        report_path = self.results_dir / 'validation_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
            
        metrics_path = self.results_dir / 'validation_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
            
        print(f"\n[OK] Validation report: {report_path}")
        print(f"[OK] Validation metrics: {metrics_path}")


def main():
    """Main validation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate 3DES S-Box models')
    parser.add_argument('--data-dir', type=str, default='Input/Mastercard', help='Data directory')
    parser.add_argument('--models-dir', type=str, default='models', help='Models directory')
    parser.add_argument('--results-dir', type=str, default='results', help='Results directory')
    
    args = parser.parse_args()
    
    # Create validator
    validator = Validator(
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        results_dir=args.results_dir
    )
    
    # Run validation
    metrics = validator.validate()
    
    print("\n" + "=" * 60)
    print("[OK] Validation Complete!")
    print("=" * 60)


if __name__ == "__main__":
    raise SystemExit(
        "Deprecated entrypoint (legacy TensorFlow pipeline).\n"
        "Use the supported pipeline instead (attack with a withheld set and compare externally):\n"
        "  python main.py --mode attack --input_dir <DIR> --processed_dir <DIR> --output_dir <DIR>\n"
    )
