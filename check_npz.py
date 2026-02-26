import numpy as np

data = np.load('Input/Mastercard/traces_data_1000T_1.npz', allow_pickle=True)
print("NPZ keys:", list(data.keys()))
print("\nData structure:")
for k in data.keys():
    if hasattr(data[k], 'shape'):
        print(f"  {k}: shape={data[k].shape}, dtype={data[k].dtype}")
    else:
        print(f"  {k}: {type(data[k])}, value={data[k]}")
