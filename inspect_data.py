"""Inspect available data files."""
import numpy as np
import os

os.chdir('D:/系统辨识作业/sindy_bicycle')

# Check sindy model
print("=== meijaard_sindy_v35.npz ===")
d = np.load('meijaard_sindy_v35.npz', allow_pickle=True)
for k in d.keys():
    v = d[k]
    if hasattr(v, 'shape'):
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        if v.size < 20:
            print(f"    values: {v}")
    else:
        print(f"  {k}: {type(v).__name__} = {v}")

# Check openloop data
print("\n=== meijaard_openloop_data_v35.npz ===")
d2 = np.load('meijaard_openloop_data_v35.npz', allow_pickle=True)
for k in d2.keys():
    v = d2[k]
    if hasattr(v, 'shape'):
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
    else:
        print(f"  {k}: {type(v).__name__}")
