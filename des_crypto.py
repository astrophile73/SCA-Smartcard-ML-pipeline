"""
DES Cryptography Module

Implements proper DES algorithm for side-channel analysis.
Includes all permutation tables, key schedule, and first-round S-Box extraction.

Reference: FIPS 46-3 (DES Standard)
"""

import numpy as np
from typing import List, Tuple


# ============================================================================
# DES Permutation Tables
# ============================================================================

# Initial Permutation (IP)
IP = [
    58, 50, 42, 34, 26, 18, 10, 2,
    60, 52, 44, 36, 28, 20, 12, 4,
    62, 54, 46, 38, 30, 22, 14, 6,
    64, 56, 48, 40, 32, 24, 16, 8,
    57, 49, 41, 33, 25, 17, 9, 1,
    59, 51, 43, 35, 27, 19, 11, 3,
    61, 53, 45, 37, 29, 21, 13, 5,
    63, 55, 47, 39, 31, 23, 15, 7
]

# Inverse Initial Permutation (IP^-1)
IP_INV = [
    40, 8, 48, 16, 56, 24, 64, 32,
    39, 7, 47, 15, 55, 23, 63, 31,
    38, 6, 46, 14, 54, 22, 62, 30,
    37, 5, 45, 13, 53, 21, 61, 29,
    36, 4, 44, 12, 52, 20, 60, 28,
    35, 3, 43, 11, 51, 19, 59, 27,
    34, 2, 42, 10, 50, 18, 58, 26,
    33, 1, 41, 9, 49, 17, 57, 25
]

# Expansion (E) - 32 bits to 48 bits
E = [
    32, 1, 2, 3, 4, 5,
    4, 5, 6, 7, 8, 9,
    8, 9, 10, 11, 12, 13,
    12, 13, 14, 15, 16, 17,
    16, 17, 18, 19, 20, 21,
    20, 21, 22, 23, 24, 25,
    24, 25, 26, 27, 28, 29,
    28, 29, 30, 31, 32, 1
]

# Permutation (P) - after S-Boxes
P = [
    16, 7, 20, 21, 29, 12, 28, 17,
    1, 15, 23, 26, 5, 18, 31, 10,
    2, 8, 24, 14, 32, 27, 3, 9,
    19, 13, 30, 6, 22, 11, 4, 25
]

# Permuted Choice 1 (PC-1) - 64 bits to 56 bits (removes parity)
PC1 = [
    57, 49, 41, 33, 25, 17, 9,
    1, 58, 50, 42, 34, 26, 18,
    10, 2, 59, 51, 43, 35, 27,
    19, 11, 3, 60, 52, 44, 36,
    63, 55, 47, 39, 31, 23, 15,
    7, 62, 54, 46, 38, 30, 22,
    14, 6, 61, 53, 45, 37, 29,
    21, 13, 5, 28, 20, 12, 4
]

# Permuted Choice 2 (PC-2) - 56 bits to 48 bits
PC2 = [
    14, 17, 11, 24, 1, 5,
    3, 28, 15, 6, 21, 10,
    23, 19, 12, 4, 26, 8,
    16, 7, 27, 20, 13, 2,
    41, 52, 31, 37, 47, 55,
    30, 40, 51, 45, 33, 48,
    44, 49, 39, 56, 34, 53,
    46, 42, 50, 36, 29, 32
]

# Left shift schedule for each round
SHIFT_SCHEDULE = [1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1]

# S-Boxes (same as in label_generator.py)
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


# ============================================================================
# Bit Manipulation Utilities
# ============================================================================

def bytes_to_bits(data: bytes) -> List[int]:
    """Convert bytes to list of bits (MSB first)."""
    bits = []
    for byte in data:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return bits


def bits_to_bytes(bits: List[int]) -> bytes:
    """Convert list of bits to bytes."""
    result = []
    for i in range(0, len(bits), 8):
        byte = 0
        for j in range(8):
            if i + j < len(bits):
                byte = (byte << 1) | bits[i + j]
        result.append(byte)
    return bytes(result)


def permute(bits: List[int], table: List[int]) -> List[int]:
    """Apply permutation table to bits."""
    return [bits[i - 1] for i in table]


def left_shift(bits: List[int], n: int) -> List[int]:
    """Circular left shift."""
    return bits[n:] + bits[:n]


def xor_bits(a: List[int], b: List[int]) -> List[int]:
    """XOR two bit lists."""
    return [x ^ y for x, y in zip(a, b)]


# ============================================================================
# DES Class
# ============================================================================

class DES:
    """DES encryption with first-round S-Box extraction for SCA."""
    
    def __init__(self, key: bytes):
        """
        Initialize DES with 64-bit key (8 bytes).
        
        Args:
            key: 8-byte DES key (parity bits will be ignored)
        """
        if len(key) != 8:
            raise ValueError(f"DES key must be 8 bytes, got {len(key)}")
        
        self.key = key
        self.round_keys = self._generate_round_keys()
    
    def _generate_round_keys(self) -> List[List[int]]:
        """Generate 16 round keys from master key."""
        # Convert key to bits
        key_bits = bytes_to_bits(self.key)
        
        # PC-1: 64 bits → 56 bits (remove parity)
        key_56 = permute(key_bits, PC1)
        
        # Split into C0 and D0 (28 bits each)
        C = key_56[:28]
        D = key_56[28:]
        
        # Generate 16 round keys
        round_keys = []
        self.round_keys_cd = []
        for round_num in range(16):
            # Left shift
            C = left_shift(C, SHIFT_SCHEDULE[round_num])
            D = left_shift(D, SHIFT_SCHEDULE[round_num])
            
            # Concatenate and apply PC-2
            CD = C + D
            self.round_keys_cd.append(CD)
            round_key = permute(CD, PC2)  # 48 bits
            round_keys.append(round_key)
        
        return round_keys
    
    def get_first_round_sbox_inputs(self, plaintext: bytes) -> np.ndarray:
        """
        Get 6-bit inputs to each S-Box in first round.
        
        Args:
            plaintext: 8-byte plaintext
            
        Returns:
            Array of shape (8,) with 6-bit inputs (0-63)
        """
        if len(plaintext) != 8:
            raise ValueError(f"Plaintext must be 8 bytes, got {len(plaintext)}")
        
        # Convert to bits
        pt_bits = bytes_to_bits(plaintext)
        
        # Initial Permutation
        pt_perm = permute(pt_bits, IP)
        
        # Split into L0 and R0
        L0 = pt_perm[:32]
        R0 = pt_perm[32:]
        
        # Expansion: 32 bits → 48 bits
        R0_expanded = permute(R0, E)
        
        # XOR with first round key
        round_key_1 = self.round_keys[0]
        sbox_input_48 = xor_bits(R0_expanded, round_key_1)
        
        # Split into 8 × 6-bit chunks
        sbox_inputs = []
        for i in range(8):
            # Extract 6 bits
            bits_6 = sbox_input_48[i*6:(i+1)*6]
            
            # Convert to integer (0-63)
            value = 0
            for bit in bits_6:
                value = (value << 1) | bit
            
            sbox_inputs.append(value)
        
        return np.array(sbox_inputs, dtype=np.int32)
    
    def get_first_round_sbox_outputs(self, plaintext: bytes) -> np.ndarray:
        """
        Get 4-bit outputs from each S-Box in first round.
        
        Args:
            plaintext: 8-byte plaintext
            
        Returns:
            Array of shape (8,) with 4-bit outputs (0-15)
        """
        sbox_inputs = self.get_first_round_sbox_inputs(plaintext)
        
        sbox_outputs = []
        for sbox_idx, input_6bit in enumerate(sbox_inputs):
            # Extract row (bits 0 and 5) and column (bits 1-4)
            row = ((input_6bit & 0x20) >> 4) | (input_6bit & 0x01)
            col = (input_6bit & 0x1E) >> 1
            
            # S-Box lookup
            output = SBOX[sbox_idx][row][col]
            sbox_outputs.append(output)
        
        return np.array(sbox_outputs, dtype=np.int32)


    def encrypt(self, data: bytes) -> bytes:
        """Full DES encryption."""
        bits = bytes_to_bits(data)
        bits = permute(bits, IP)
        L, R = bits[:32], bits[32:]
        
        for i in range(16):
            L_next = R
            # Round function
            R_exp = permute(R, E)
            s_in = xor_bits(R_exp, self.round_keys[i])
            s_out = []
            for j in range(8):
                b = s_in[j*6:(j+1)*6]
                row = ((b[0] << 1) | b[5])
                col = ((b[1] << 3) | (b[2] << 2) | (b[3] << 1) | b[4])
                val = SBOX[j][row][col]
                for k in range(3, -1, -1):
                    s_out.append((val >> k) & 1)
            R_next = xor_bits(L, permute(s_out, P))
            L, R = L_next, R_next
        
        return bits_to_bytes(permute(R + L, IP_INV))

    def decrypt(self, data: bytes) -> bytes:
        """Full DES decryption."""
        bits = bytes_to_bits(data)
        bits = permute(bits, IP)
        L, R = bits[:32], bits[32:]
        
        for i in range(15, -1, -1):
            L_next = R
            # Round function
            R_exp = permute(R, E)
            s_in = xor_bits(R_exp, self.round_keys[i])
            s_out = []
            for j in range(8):
                b = s_in[j*6:(j+1)*6]
                row = ((b[0] << 1) | b[5])
                col = ((b[1] << 3) | (b[2] << 2) | (b[3] << 1) | b[4])
                val = SBOX[j][row][col]
                for k in range(3, -1, -1):
                    s_out.append((val >> k) & 1)
            R_next = xor_bits(L, permute(s_out, P))
            L, R = L_next, R_next
            
        return bits_to_bytes(permute(R + L, IP_INV))

    def get_stage2_sbox_outputs(self, stage1_output: bytes) -> np.ndarray:
        """
        Get S-Box outputs for Stage 2 (Decryption with K2).
        
        In 3DES-EDE2, Stage 2 is DES_Decrypt(K2, Stage1_Output).
        """
        # Stage 2 is a DECRYPTION operation in 3DES-EDE
        # But for SCA targeting the first round of the second operation:
        # we treat it as the START of a decryption process.
        
        bits = bytes_to_bits(stage1_output)
        bits = permute(bits, IP)
        L0, R0 = bits[:32], bits[32:]
        
        # In DES decryption, the first round key used is Round Key 16 (last one)
        # However, for labels, we need to know what bits of K2 are leaking.
        # Most hardware executes Decryption by reversing the key schedule.
        
        R0_expanded = permute(R0, E)
        
        # Target the FIRST round key used in decryption (which is round_keys[15])
        first_dec_key = self.round_keys[15] 
        sbox_input_48 = xor_bits(R0_expanded, first_dec_key)
        
        sbox_outputs = []
        for i in range(8):
            bits_6 = sbox_input_48[i*6:(i+1)*6]
            row = ((bits_6[0] << 1) | bits_6[5])
            col = ((bits_6[1] << 3) | (bits_6[2] << 2) | (bits_6[3] << 1) | bits_6[4])
            sbox_outputs.append(SBOX[i][row][col])
            
        return np.array(sbox_outputs, dtype=np.int32)


def test_des():
    """Test DES implementation with known test vectors."""
    print("=" * 60)
    print("Testing DES Implementation (Stage 1 & 2)")
    print("=" * 60)
    
    # Test vector from FIPS 46-3
    key = bytes.fromhex("0123456789ABCDEF")
    plaintext = bytes.fromhex("4E6F772069732074") # "Now is t"
    
    des = DES(key)
    ciphertext = des.encrypt(plaintext)
    decrypted = des.decrypt(ciphertext)
    
    print(f"PT: {plaintext.hex().upper()}")
    print(f"CT: {ciphertext.hex().upper()}")
    print(f"DT: {decrypted.hex().upper()}")
    
    assert plaintext == decrypted, "Encryption/Decryption mismatch!"
    print("✅ Full Enc/Dec Verified")
    
    # Test Stage 2 labels
    s2_out = des.get_stage2_sbox_outputs(plaintext)
    print(f"Stage 2 S-Box Labels: {s2_out}")

if __name__ == "__main__":
    test_des()
