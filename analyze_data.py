import numpy as np
import os

os.chdir('D:/系统辨识作业/sindy_bicycle')

print('=' * 60)
print('1. bicycle_data.npz')
print('=' * 60)
data = np.load('bicycle_data.npz', allow_pickle=True)
for key in data.files:
    arr = data[key]
    print(f'  {key}: shape={arr.shape}, dtype={arr.dtype}')

print()
print('=' * 60)
print('2. meijaard_openloop_data_v35.npz')
print('=' * 60)
data = np.load('meijaard_openloop_data_v35.npz', allow_pickle=True)
for key in data.files:
    arr = data[key]
    print(f'  {key}: shape={arr.shape}, dtype={arr.dtype}')

print()
print('=' * 60)
print('3. meijaard_openloop_data.npz')
print('=' * 60)
data = np.load('meijaard_openloop_data.npz', allow_pickle=True)
for key in data.files:
    arr = data[key]
    print(f'  {key}: shape={arr.shape}, dtype={arr.dtype}')

print()
print('=' * 60)
print('4. meijaard_openloop_data_v5.npz')
print('=' * 60)
data = np.load('meijaard_openloop_data_v5.npz', allow_pickle=True)
for key in data.files:
    arr = data[key]
    print(f'  {key}: shape={arr.shape}, dtype={arr.dtype}')

print()
print('=' * 60)
print('5. sindy_model.npz')
print('=' * 60)
data = np.load('sindy_model.npz', allow_pickle=True)
for key in data.files:
    arr = data[key]
    print(f'  {key}: shape={arr.shape}, dtype={arr.dtype}')

print()
print('=' * 60)
print('6. sindy_full_model.npz')
print('=' * 60)
data = np.load('sindy_full_model.npz', allow_pickle=True)
for key in data.files:
    arr = data[key]
    print(f'  {key}: shape={arr.shape}, dtype={arr.dtype}')

print()
print('=' * 60)
print('7. meijaard_sindy_v35.npz')
print('=' * 60)
data = np.load('meijaard_sindy_v35.npz', allow_pickle=True)
for key in data.files:
    arr = data[key]
    print(f'  {key}: shape={arr.shape}, dtype={arr.dtype}')
