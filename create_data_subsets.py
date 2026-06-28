"""Create data subsets from the 750k dataset for progressive training."""
import sys, os
import numpy as np

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, 'data')

src = os.path.join(data_dir, 'stanley_ref_dataset_750k.npz')
print(f"Loading: {src}")
data = np.load(src)

obs = data['obs']
action = data['action']
next_obs = data['next_obs']
reward = data['reward']
done = data['done']

print(f"Total: {len(obs)} steps")

for size in [30000, 150000, 450000]:
    filename = f'stanley_ref_dataset_{size//1000}k.npz'
    filepath = os.path.join(data_dir, filename)

    np.savez_compressed(filepath,
        obs=obs[:size], action=action[:size],
        next_obs=next_obs[:size], reward=reward[:size], done=done[:size],
        injection_mode='stanley_ref',
        epsilon_range='[-0.1, 0.1]',
        timestamp=str(data.get('timestamp', '')),
        n_episodes=int(data.get('n_episodes', 0)),
        n_steps=size,
        lqr_label=str(data.get('lqr_label', 'Q500_D50_R02')),
        lqr_Q=data.get('lqr_Q', np.array([500, 50, 10, 1])),
        lqr_R=data.get('lqr_R', np.array([0.2])),
    )
    print(f"  Saved: {filepath} ({size} steps)")

print("Done.")
