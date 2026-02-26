import numpy as np
from label_generator import LabelGenerator

def check_structure():
    print("Checking NPZ structure...")
    data = np.load('Input/Mastercard/traces_data_1000T_1.npz', allow_pickle=True)
    
    if 'T_DES_KENC' in data:
        kenc_raw = data['T_DES_KENC']
        print(f"Type: {type(kenc_raw)}")
        print(f"Shape: {kenc_raw.shape}")
        print(f"Value: {kenc_raw}")
        
        kenc_str = str(kenc_raw)
        print(f"Str conversion: '{kenc_str}'")
        
        # Try generating labels
        gen = LabelGenerator()
        try:
            labels = gen.generate_sbox_labels_for_key(kenc_str)
            print(f"Labels from '{kenc_str}': {labels}")
        except Exception as e:
            print(f"Label generation failed: {e}")
            
    else:
        print("T_DES_KENC not found")

if __name__ == "__main__":
    check_structure()
