"""
Output Generator for 3DES Pipeline (Mode 4: OUTPUT)

Generates client-ready CSV files with recovered 3DES keys.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import List, Dict


class OutputGenerator:
    """Generate CSV output files from attack results."""
    
    def __init__(self, results_dir: str = "results"):
        """
        Initialize output generator.
        
        Args:
            results_dir: Directory to save output files
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
    
    def load_attack_results(self, results_path: str) -> List[Dict]:
        """Load attack results from JSON file."""
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        print(f"✓ Loaded {len(results)} attack results from {results_path}")
        
        return results
    
    def format_output_data(
        self,
        attack_results: List[Dict],
        card_type: str = "Mastercard",
        default_aip: str = "1980",
        default_iad: str = "06010A03A00000"
    ) -> pd.DataFrame:
        """
        Format attack results into output DataFrame.
        
        Args:
            attack_results: List of attack result dictionaries
            card_type: Card type (Mastercard/Visa)
            default_aip: Default AIP value if missing
            default_iad: Default IAD value if missing
            
        Returns:
            DataFrame with formatted output
        """
        print("\nFormatting output data...")
        
        rows = []
        
        for i, result in enumerate(attack_results):
            row = {
                'Card_Type': card_type,
                'PAN': '5413330337554966',  # Placeholder - extract from metadata if available
                'Expiry': '1225',  # Placeholder
                'ATC': f'{i+1:04d}',  # Decimal format, zero-padded
                'AIP': default_aip,
                'IAD': default_iad,
                '3DES_KENC': result['3DES_KENC'],
                '3DES_KMAC': result['3DES_KMAC'],
                '3DES_KDEK': result['3DES_KDEK']
            }
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        print(f"✓ Formatted {len(df)} rows")
        
        return df
    
    def generate_clean_csv(self, df: pd.DataFrame, output_path: str):
        """
        Generate clean CSV (no quotes, for EMV tools).
        
        Args:
            df: DataFrame with output data
            output_path: Path to save CSV file
        """
        output_path = Path(output_path)
        
        # Save without quotes
        df.to_csv(output_path, index=False, quoting=0)
        
        print(f"✓ Clean CSV saved: {output_path}")
    
    def generate_excel_safe_csv(self, df: pd.DataFrame, output_path: str):
        """
        Generate Excel-safe CSV (with ' prefix to prevent scientific notation).
        
        Args:
            df: DataFrame with output data
            output_path: Path to save CSV file
        """
        output_path = Path(output_path)
        
        # Create copy and add ' prefix to hex columns
        df_excel = df.copy()
        
        hex_columns = ['AIP', 'IAD', '3DES_KENC', '3DES_KMAC', '3DES_KDEK']
        
        for col in hex_columns:
            if col in df_excel.columns:
                df_excel[col] = "'" + df_excel[col].astype(str)
        
        # Save without quotes
        df_excel.to_csv(output_path, index=False, quoting=0)
        
        print(f"✓ Excel-safe CSV saved: {output_path}")
    
    def generate_outputs(
        self,
        attack_results_path: str,
        card_type: str = "Mastercard"
    ):
        """
        Generate both clean and Excel-safe CSV outputs.
        
        Args:
            attack_results_path: Path to attack results JSON
            card_type: Card type
        """
        print("\n" + "=" * 60)
        print("OUTPUT GENERATION")
        print("=" * 60)
        
        # Load attack results
        attack_results = self.load_attack_results(attack_results_path)
        
        # Format data
        df = self.format_output_data(attack_results, card_type=card_type)
        
        # Generate outputs
        print("\nGenerating output files...")
        
        clean_path = self.results_dir / "output_clean.csv"
        excel_path = self.results_dir / "output_excel_safe.csv"
        
        self.generate_clean_csv(df, clean_path)
        self.generate_excel_safe_csv(df, excel_path)
        
        # Print sample
        print("\n" + "=" * 60)
        print("Sample Output (first 3 rows)")
        print("=" * 60)
        print(df.head(3).to_string(index=False))
        
        print("\n" + "=" * 60)
        print("✓ Output Generation Complete!")
        print("=" * 60)
        print(f"\nFiles saved in: {self.results_dir}")
        print(f"  - {clean_path.name} (for EMV tools)")
        print(f"  - {excel_path.name} (for Excel viewing)")


def main():
    """Main output generation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate CSV output from attack results')
    parser.add_argument('--input', type=str, required=True, help='Attack results JSON file')
    parser.add_argument('--card-type', type=str, default='Mastercard', help='Card type')
    parser.add_argument('--results-dir', type=str, default='results', help='Results directory')
    
    args = parser.parse_args()
    
    # Create output generator
    generator = OutputGenerator(results_dir=args.results_dir)
    
    # Generate outputs
    generator.generate_outputs(
        attack_results_path=args.input,
        card_type=args.card_type
    )


if __name__ == "__main__":
    raise SystemExit(
        "Deprecated entrypoint (legacy output generator).\n"
        "Use the supported pipeline report generator instead:\n"
        "  python main.py --mode attack --output_dir <DIR>\n"
    )
