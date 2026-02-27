"""
Label Generator for 3DES S-Box Targeting

This module generates S-Box labels for first-round DES attack.
Implements proper DES S-Box tables to avoid collisions.
"""

import numpy as np
from typing import List, Tuple


# DES S-Box tables (8 S-Boxes, each 4x16)
# Reference: https://gist.github.com/BlackRabbit-github/2924939
SBOX = [
    # S-Box 1
    [
        [14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7],
        [0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8],
        [4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0],
        [15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13]
    ],
    # S-Box 2
    [
        [15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10],
        [3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11, 5],
        [0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15],
        [13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9]
    ],
    # S-Box 3
    [
        [10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8],
        [13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11, 15, 1],
        [13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7],
        [1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12]
    ],
    # S-Box 4
    [
        [7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12, 4, 15],
        [13, 8, 11, 5, 6, 15, 0, 3, 4, 7, 2, 12, 1, 10, 14, 9],
        [10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2, 8, 4],
        [3, 15, 0, 6, 10, 1, 13, 8, 9, 4, 5, 11, 12, 7, 2, 14]
    ],
    # S-Box 5
    [
        [2, 12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0, 14, 9],
        [14, 11, 2, 12, 4, 7, 13, 1, 5, 0, 15, 10, 3, 9, 8, 6],
        [4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3, 0, 14],
        [11, 8, 12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4, 5, 3]
    ],
    # S-Box 6
    [
        [12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7, 5, 11],
        [10, 15, 4, 2, 7, 12, 9, 5, 6, 1, 13, 14, 0, 11, 3, 8],
        [9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13, 11, 6],
        [4, 3, 2, 12, 9, 5, 15, 10, 11, 14, 1, 7, 6, 0, 8, 13]
    ],
    # S-Box 7
    [
        [4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1],
        [13, 0, 11, 7, 4, 9, 1, 10, 14, 3, 5, 12, 2, 15, 8, 6],
        [1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5, 9, 2],
        [6, 11, 13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2, 3, 12]
    ],
    # S-Box 8
    [
        [13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7],
        [1, 15, 13, 8, 10, 3, 7, 4, 12, 5, 6, 11, 0, 14, 9, 2],
        [7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3, 5, 8],
        [2, 1, 14, 7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5, 6, 11]
    ]
]


class LabelGenerator:
    """Generate S-Box labels for 3DES key recovery."""
    
    def __init__(self):
        """Initialize label generator with DES S-Box tables."""
        self.sbox = SBOX
        self.num_sboxes = 8
        self.num_classes = 16  # 4-bit output per S-Box
    
    @staticmethod
    def hex_to_bytes(hex_string: str) -> bytes:
        """Convert hex string to bytes."""
        return bytes.fromhex(hex_string)
    
    def sbox_output(self, sbox_idx: int, input_6bit: int) -> int:
        """
        Compute S-Box output for given 6-bit input.
        
        Args:
            sbox_idx: S-Box index (0-7)
            input_6bit: 6-bit input value (0-63)
            
        Returns:
            4-bit output value (0-15)
        """
        # Extract row (bits 0 and 5) and column (bits 1-4)
        row = ((input_6bit & 0x20) >> 4) | (input_6bit & 0x01)
        col = (input_6bit & 0x1E) >> 1
        
        return self.sbox[sbox_idx][row][col]
    
    # def generate_sbox_labels_for_key(
    #     self, 
    #     key_hex: str, 
    #     plaintext: bytes = None
    # ) -> np.ndarray:
    #     """Stage 1 Labels (K1)."""
    #     from des_crypto import DES
    #     key_bytes = self.hex_to_bytes(key_hex)
    #     if plaintext is None: plaintext = b'\x00' * 8
    #     des_key = key_bytes[:8]
    #     des = DES(des_key)
    #     return des.get_first_round_sbox_outputs(plaintext)

    def generate_sbox_labels_for_key(
        self, 
        key_hex: str, 
        plaintext: bytes
    ) -> np.ndarray:
        """Stage 1 Labels (K1)."""
        from des_crypto import DES
        if plaintext is None:
            raise ValueError("Plaintext must be provided per trace. It cannot be None.")
        key_bytes = self.hex_to_bytes(key_hex)
        # 3DES K1
        des_key = key_bytes[:8]
        des = DES(des_key)
        return des.get_first_round_sbox_outputs(plaintext)

    # def generate_stage2_labels_for_key(
    #     self,
    #     key_hex: str,
    #     plaintext: bytes = None
    # ) -> np.ndarray:
    #     """
    #     Stage 2 Labels (K2).
        
    #     Calculates C1 = DES_Enc(K1, P), then targets DES_Dec(K2, C1).
    #     """
    #     from des_crypto import DES
    #     key_bytes = self.hex_to_bytes(key_hex)
    #     #if plaintext is None: plaintext = b'\x00' * 8
        
    #     # 1. Recover K1 and K2
    #     k1_bytes = key_bytes[:8]
    #     k2_bytes = key_bytes[8:16]
        
    #     # 2. Calculate Intermediate State C1
    #     des1 = DES(k1_bytes)
    #     c1 = des1.encrypt(plaintext)
        
    #     # 3. Target Stage 2 S-Boxes
    #     des2 = DES(k2_bytes)
    #     return des2.get_stage2_sbox_outputs(c1)

    def generate_stage2_labels_for_key(
        self,
        key_hex: str,
        plaintext: bytes
    ) -> np.ndarray:
        """Stage 2 Labels (K2).Calculates C1 = DES_Enc(K1, P), then targets DES_Dec(K2, C1)."""
        from des_crypto import DES

        if plaintext is None:
            raise ValueError("Plaintext must be provided per trace for Stage 2.")

        key_bytes = self.hex_to_bytes(key_hex)

        # Recover K1 and K2
        k1_bytes = key_bytes[:8]
        k2_bytes = key_bytes[8:16]

        # Intermediate state C1 = DES_Enc(K1, P)
        des1 = DES(k1_bytes)
        c1 = des1.encrypt(plaintext)

        # Target Stage 2 S-Boxes (DES_Dec(K2, C1))
        des2 = DES(k2_bytes)

        return des2.get_stage2_sbox_outputs(c1)
    
    def generate_labels_for_dataset(
        self,
        keys: np.ndarray,
        plaintexts: np.ndarray,
        key_type: str = 'KENC',
        stage: int = 1
    ) -> np.ndarray:
        """Generate S-Box labels for entire dataset for a specific stage."""
        print(f"\nGenerating Stage {stage} S-Box labels for {key_type}...")
        all_labels = []
        for key_hex, pt in zip(keys, plaintexts):
            if isinstance(pt, str):
                pt_bytes = bytes.fromhex(pt.zfill(16))
            else:
                pt_bytes = pt

            if stage == 1:
                labels = self.generate_sbox_labels_for_key(
                    key_hex,
                    plaintext=pt_bytes
                )
            else:
                labels = self.generate_stage2_labels_for_key(
                    key_hex,
                    plaintext=pt_bytes
                )
            all_labels.append(labels)

        return np.array(all_labels)
    
    # def generate_labels_for_all_keys(
    #     self,
    #     kenc_keys: np.ndarray,
    #     kmac_keys: np.ndarray,
    #     kdek_keys: np.ndarray,
    #     stage: int = 1
    # ) -> dict:
    #     """Generate S-Box labels for all 3 keys for a specific stage."""
    #     print("\n" + "=" * 60)
    #     print(f"Generating Stage {stage} Labels (Proper DES)")
    #     print("=" * 60)
        
    #     return {
    #         'KENC': self.generate_labels_for_dataset(kenc_keys, 'KENC', stage),
    #         'KMAC': self.generate_labels_for_dataset(kmac_keys, 'KMAC', stage),
    #         'KDEK': self.generate_labels_for_dataset(kdek_keys, 'KDEK', stage)
    #     }
    def generate_labels_for_all_keys(
        self,
        kenc_keys: np.ndarray,
        kmac_keys: np.ndarray,
        kdek_keys: np.ndarray,
        plaintexts: np.ndarray,   # ← ADDED
        stage: int = 1
    ) -> dict:
        """Generate S-Box labels for all 3 keys for a specific stage."""
        print("\n" + "=" * 60)
        print(f"Generating Stage {stage} Labels (Proper DES)")
        print("=" * 60)

        return {
            'KENC': self.generate_labels_for_dataset(
                kenc_keys,
                plaintexts,              # ← PASS PLAINTEXTS
                key_type='KENC',
                stage=stage
            ),
        'KMAC': self.generate_labels_for_dataset(
            kmac_keys,
            plaintexts,              # ← PASS PLAINTEXTS
            key_type='KMAC',
            stage=stage
        ),
        'KDEK': self.generate_labels_for_dataset(
            kdek_keys,
            plaintexts,              # ← PASS PLAINTEXTS
            key_type='KDEK',
            stage=stage
        )
    }
    def labels_to_categorical(self, labels: np.ndarray) -> List[np.ndarray]:
        """
        Convert S-Box labels to categorical format for training.
        
        Args:
            labels: Array of shape (N, 8) with S-Box outputs
            
        Returns:
            List of 8 arrays, each of shape (N, 16) for categorical training
        """
        import tensorflow as tf
        categorical_labels = []
        
        for sbox_idx in range(8):
            # Extract labels for this S-Box
            sbox_labels = labels[:, sbox_idx]
            
            # Convert to categorical (one-hot encoding)
            categorical = tf.keras.utils.to_categorical(sbox_labels, num_classes=16)
            categorical_labels.append(categorical)
        
        print(f"\n[OK] Converted to categorical format:")
        print(f"  {len(categorical_labels)} S-Boxes")
        print(f"  Each with shape: {categorical_labels[0].shape}")
        
        return categorical_labels


if __name__ == "__main__":
    # Test label generator
    print("=" * 60)
    print("Testing Label Generator")
    print("=" * 60)
    
    generator = LabelGenerator()
    
    # Test with sample key
    test_key = "9E15204313F7318ACB79B90BD986AD29"
    print(f"\nTest key: {test_key}")
    
    labels = generator.generate_sbox_labels_for_key(test_key)
    print(f"\nS-Box outputs: {labels}")
    print(f"Shape: {labels.shape}")
    
    # Test with dataset
    test_keys = np.array([test_key] * 100)
    dataset_labels = generator.generate_labels_for_dataset(test_keys)
    print(f"\nDataset labels shape: {dataset_labels.shape}")
    
    # Test categorical conversion
    categorical = generator.labels_to_categorical(dataset_labels)
    print(f"\nCategorical labels:")
    for i, cat in enumerate(categorical):
        print(f"  S-Box {i}: {cat.shape}")
    
    print("\n" + "=" * 60)
    print("Label Generation Complete!")
    print("=" * 60)
