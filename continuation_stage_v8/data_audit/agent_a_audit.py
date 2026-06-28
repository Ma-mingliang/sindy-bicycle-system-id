"""Agent A: Data Audit for 7D real bicycle dataset (stage2_dataset_150k.npz)."""
import sys
sys.path.insert(0, 'D:/系统辨识作业/sindy_bicycle/continuation_stage_v8')

import numpy as np
import json
from collections import OrderedDict

# --- Constants ---
DATA_PATH = 'D:/系统辨识作业/sindy_bicycle/data/stage2_dataset_150k.npz'
OUTPUT_PATH = 'D:/系统辨识作业/sindy_bicycle/continuation_stage_v8/raw_results/AGENT_A_DATA_AUDIT.json'
IDX_7D_FROM_8D = [0, 1, 2, 3, 4, 6, 7]
STATE_NAMES_8D = ['e_y', 'e_psi', 'v', 'theta', 'theta_dot', 'kappa', 'delta', 'delta_dot']
STATE_NAMES_7D = ['e_y', 'e_psi', 'v', 'theta', 'theta_dot', 'delta', 'delta_dot']

results = OrderedDict()

# ===================== 1. Load raw 8D data =====================
print("=" * 70)
print("AGENT A: DATA AUDIT - 7D Real Bicycle Dataset")
print("=" * 70)

data = np.load(DATA_PATH, allow_pickle=True)
print(f"\nLoaded: {DATA_PATH}")
print(f"Keys: {list(data.keys())}")

# Store raw array info
raw_info = OrderedDict()
for key in data.keys():
    arr = data[key]
    raw_info[key] = {
        'shape': list(arr.shape),
        'dtype': str(arr.dtype),
    }
results['raw_file_info'] = raw_info
results['raw_file_info_summary'] = (
    f"150000 timesteps, 8D obs, 1D action, "
    f"{int(data['n_episodes'])} episodes, "
    f"label='{data['lqr_label']}', seed={int(data['lqr_R'][0])}"
)

# ===================== 2. Extract 7D =====================
obs_8d = data['obs']           # (150000, 8) float32
next_obs_8d = data['next_obs'] # (150000, 8) float32
actions_raw = data['action']   # (150000, 1) float32
done = data['done']            # (150000,) bool
n_episodes_declared = int(data['n_episodes'])

obs_7d = obs_8d[:, IDX_7D_FROM_8D].astype(np.float64)
next_obs_7d = next_obs_8d[:, IDX_7D_FROM_8D].astype(np.float64)
actions = actions_raw.flatten().astype(np.float64)

print(f"\n8D obs shape: {obs_8d.shape}")
print(f"7D obs shape: {obs_7d.shape}")
print(f"Actions shape: {actions.shape}")
print(f"Done shape: {done.shape}")

results['extraction'] = {
    'idx_7d_from_8d': IDX_7D_FROM_8D,
    'dropped_dim': 5,
    'dropped_dim_name': 'kappa',
    'obs_7d_shape': list(obs_7d.shape),
    'next_obs_7d_shape': list(next_obs_7d.shape),
    'actions_shape': list(actions.shape),
}

# ===================== 3. Per-dimension stats (8D) =====================
def compute_dim_stats(arr, names):
    """Compute per-dimension statistics."""
    stats = OrderedDict()
    for i, name in enumerate(names):
        col = arr[:, i] if arr.ndim == 2 else arr
        stats[name] = {
            'min': float(np.min(col)),
            'max': float(np.max(col)),
            'mean': float(np.mean(col)),
            'std': float(np.std(col)),
            'median': float(np.median(col)),
            'q01': float(np.percentile(col, 1)),
            'q99': float(np.percentile(col, 99)),
        }
    return stats

print("\n" + "-" * 70)
print("8D STATE STATISTICS (obs)")
print("-" * 70)
stats_8d = compute_dim_stats(obs_8d, STATE_NAMES_8D)
for name, s in stats_8d.items():
    print(f"  {name:12s}: min={s['min']:+.6f}  max={s['max']:+.6f}  "
          f"mean={s['mean']:+.6f}  std={s['std']:.6f}")
results['stats_8d_obs'] = stats_8d

print("\n" + "-" * 70)
print("7D STATE STATISTICS (obs)")
print("-" * 70)
stats_7d = compute_dim_stats(obs_7d, STATE_NAMES_7D)
for name, s in stats_7d.items():
    print(f"  {name:12s}: min={s['min']:+.6f}  max={s['max']:+.6f}  "
          f"mean={s['mean']:+.6f}  std={s['std']:.6f}")
results['stats_7d_obs'] = stats_7d

# next_obs stats
print("\n" + "-" * 70)
print("7D NEXT-OBS STATISTICS")
print("-" * 70)
stats_7d_next = compute_dim_stats(next_obs_7d, STATE_NAMES_7D)
for name, s in stats_7d_next.items():
    print(f"  {name:12s}: min={s['min']:+.6f}  max={s['max']:+.6f}  "
          f"mean={s['mean']:+.6f}  std={s['std']:.6f}")
results['stats_7d_next_obs'] = stats_7d_next

# ===================== 4. NaN / Inf check =====================
print("\n" + "-" * 70)
print("NaN / Inf CHECK")
print("-" * 70)

nan_inf_report = OrderedDict()
for name, arr in [('obs_8d', obs_8d), ('next_obs_8d', next_obs_8d),
                   ('obs_7d', obs_7d), ('next_obs_7d', next_obs_7d),
                   ('actions', actions), ('done', done.astype(float))]:
    n_nan = int(np.sum(np.isnan(arr)))
    n_inf = int(np.sum(np.isinf(arr)))
    nan_inf_report[name] = {'n_nan': n_nan, 'n_inf': n_inf}
    status = "OK" if n_nan == 0 and n_inf == 0 else "PROBLEM"
    print(f"  {name:15s}: NaN={n_nan}, Inf={n_inf}  [{status}]")

results['nan_inf_check'] = nan_inf_report

# ===================== 5. Episode boundaries =====================
print("\n" + "-" * 70)
print("EPISODE BOUNDARY ANALYSIS")
print("-" * 70)

ep_starts = []
ep_ends = []
start = 0
for i in range(len(done)):
    if done[i]:
        ep_starts.append(start)
        ep_ends.append(i + 1)
        start = i + 1
if start < len(done):
    ep_starts.append(start)
    ep_ends.append(len(done))

actual_n_episodes = len(ep_starts)
ep_lengths = [ep_ends[i] - ep_starts[i] for i in range(actual_n_episodes)]

print(f"  Declared n_episodes: {n_episodes_declared}")
print(f"  Actual n_episodes (from done flags): {actual_n_episodes}")
print(f"  Match: {'YES' if n_episodes_declared == actual_n_episodes else 'NO - MISMATCH!'}")
print(f"  Episode lengths: min={min(ep_lengths)}, max={max(ep_lengths)}, "
      f"mean={np.mean(ep_lengths):.1f}, median={np.median(ep_lengths):.1f}, "
      f"std={np.std(ep_lengths):.1f}")

# Check if done flags are at expected positions
done_count = int(np.sum(done))
print(f"  Total done=True count: {done_count}")
print(f"  Last step done: {bool(done[-1])}")

ep_boundary_info = {
    'declared_n_episodes': n_episodes_declared,
    'actual_n_episodes': actual_n_episodes,
    'match': n_episodes_declared == actual_n_episodes,
    'done_true_count': done_count,
    'last_step_done': bool(done[-1]),
    'episode_length_stats': {
        'min': int(min(ep_lengths)),
        'max': int(max(ep_lengths)),
        'mean': float(np.mean(ep_lengths)),
        'median': float(np.median(ep_lengths)),
        'std': float(np.std(ep_lengths)),
    },
    'episode_lengths': ep_lengths,
}
results['episode_boundaries'] = ep_boundary_info

# ===================== 6. Train/test split =====================
print("\n" + "-" * 70)
print("TRAIN/TEST SPLIT (via canonical_7d.data_loader)")
print("-" * 70)

from canonical_7d.data_loader import load_7d_data
from canonical_7d.config import DataConfig

cfg = DataConfig(data_path=DATA_PATH)
loader_data = load_7d_data(cfg)

train_states = loader_data['train_states']
test_states = loader_data['test_states']
train_actions = loader_data['train_actions']
test_actions = loader_data['test_actions']
train_deltas = loader_data['train_deltas']
test_deltas = loader_data['test_deltas']

split_info = {
    'train_ratio': cfg.train_ratio,
    'seed': cfg.seed,
    'train_samples': int(train_states.shape[0]),
    'test_samples': int(test_states.shape[0]),
    'total_samples': int(train_states.shape[0] + test_states.shape[0]),
    'train_states_shape': list(train_states.shape),
    'test_states_shape': list(test_states.shape),
    'train_actions_shape': list(train_actions.shape),
    'test_actions_shape': list(test_actions.shape),
    'train_deltas_shape': list(train_deltas.shape),
    'test_deltas_shape': list(test_deltas.shape),
    'actual_train_ratio': float(train_states.shape[0] / (train_states.shape[0] + test_states.shape[0])),
}

print(f"  Train samples: {split_info['train_samples']}")
print(f"  Test samples:  {split_info['test_samples']}")
print(f"  Actual train ratio: {split_info['actual_train_ratio']:.4f}")

# Per-dim stats for train/test
split_info['train_state_stats'] = compute_dim_stats(train_states, STATE_NAMES_7D)
split_info['test_state_stats'] = compute_dim_stats(test_states, STATE_NAMES_7D)
split_info['train_action_stats'] = {
    'min': float(np.min(train_actions)),
    'max': float(np.max(train_actions)),
    'mean': float(np.mean(train_actions)),
    'std': float(np.std(train_actions)),
}
split_info['test_action_stats'] = {
    'min': float(np.min(test_actions)),
    'max': float(np.max(test_actions)),
    'mean': float(np.mean(test_actions)),
    'std': float(np.std(test_actions)),
}

print("\n  Train state stats:")
for name in STATE_NAMES_7D:
    i = STATE_NAMES_7D.index(name)
    print(f"    {name:12s}: mean={split_info['train_state_stats'][name]['mean']:+.6f}  "
          f"std={split_info['train_state_stats'][name]['std']:.6f}")

print("\n  Test state stats:")
for name in STATE_NAMES_7D:
    print(f"    {name:12s}: mean={split_info['test_state_stats'][name]['mean']:+.6f}  "
          f"std={split_info['test_state_stats'][name]['std']:.6f}")

print(f"\n  Train action: mean={split_info['train_action_stats']['mean']:+.6f}  "
      f"std={split_info['train_action_stats']['std']:.6f}")
print(f"  Test action:  mean={split_info['test_action_stats']['mean']:+.6f}  "
      f"std={split_info['test_action_stats']['std']:.6f}")

results['train_test_split'] = split_info

# ===================== 7. Variance check =====================
print("\n" + "-" * 70)
print("VARIANCE ANALYSIS (near-zero / constant dimensions)")
print("-" * 70)

variance_report = OrderedDict()
state_stds_7d = np.std(obs_7d, axis=0)
for i, name in enumerate(STATE_NAMES_7D):
    std_val = float(state_stds_7d[i])
    var_val = std_val ** 2
    is_constant = std_val < 1e-10
    is_near_zero = std_val < 1e-3
    variance_report[name] = {
        'std': std_val,
        'variance': var_val,
        'is_constant': is_constant,
        'is_near_zero_var': is_near_zero,
    }
    flag = ""
    if is_constant:
        flag = " *** CONSTANT ***"
    elif is_near_zero:
        flag = " * near-zero *"
    print(f"  {name:12s}: std={std_val:.8f}  var={var_val:.2e}{flag}")

# Also check kappa (index 5) from 8D
kappa_std = float(np.std(obs_8d[:, 5]))
print(f"\n  kappa (dim 5, dropped): std={kappa_std:.8f}")
variance_report['kappa (dropped)'] = {
    'std': kappa_std,
    'variance': kappa_std ** 2,
    'is_constant': kappa_std < 1e-10,
    'is_near_zero_var': kappa_std < 1e-3,
}

results['variance_analysis'] = variance_report

# ===================== 8. Action distribution =====================
print("\n" + "-" * 70)
print("ACTION DISTRIBUTION")
print("-" * 70)

action_stats = {
    'min': float(np.min(actions)),
    'max': float(np.max(actions)),
    'mean': float(np.mean(actions)),
    'std': float(np.std(actions)),
    'median': float(np.median(actions)),
    'q01': float(np.percentile(actions, 1)),
    'q25': float(np.percentile(actions, 25)),
    'q75': float(np.percentile(actions, 75)),
    'q99': float(np.percentile(actions, 99)),
    'skewness': float(np.mean(((actions - np.mean(actions)) / (np.std(actions) + 1e-12))**3)),
    'kurtosis': float(np.mean(((actions - np.mean(actions)) / (np.std(actions) + 1e-12))**4) - 3),
    'n_unique': int(len(np.unique(actions))),
}

# Histogram bins
hist, bin_edges = np.histogram(actions, bins=20)
action_histogram = []
for j in range(len(hist)):
    action_histogram.append({
        'bin_lo': float(bin_edges[j]),
        'bin_hi': float(bin_edges[j + 1]),
        'count': int(hist[j]),
        'fraction': float(hist[j] / len(actions)),
    })
action_stats['histogram'] = action_histogram

print(f"  Range: [{action_stats['min']:.6f}, {action_stats['max']:.6f}]")
print(f"  Mean: {action_stats['mean']:+.6f}, Std: {action_stats['std']:.6f}")
print(f"  Median: {action_stats['median']:+.6f}")
print(f"  Skewness: {action_stats['skewness']:.4f}, Kurtosis: {action_stats['kurtosis']:.4f}")
print(f"  Unique values: {action_stats['n_unique']}")
print(f"\n  Histogram:")
for h in action_histogram:
    bar = '#' * int(h['fraction'] * 200)
    print(f"    [{h['bin_lo']:+.4f}, {h['bin_hi']:+.4f}): {h['count']:6d} ({h['fraction']:.4f}) {bar}")

results['action_distribution'] = action_stats

# ===================== 9. Delta (state change) statistics =====================
print("\n" + "-" * 70)
print("DELTA (STATE CHANGE) STATISTICS: next_state - state")
print("-" * 70)

deltas_7d = next_obs_7d - obs_7d
delta_stats = compute_dim_stats(deltas_7d, STATE_NAMES_7D)

for name, s in delta_stats.items():
    print(f"  {name:12s}: min={s['min']:+.8f}  max={s['max']:+.8f}  "
          f"mean={s['mean']:+.8f}  std={s['std']:.8f}")

# Delta correlation with action
print("\n  Delta vs action correlation:")
delta_action_corr = {}
for i, name in enumerate(STATE_NAMES_7D):
    corr = float(np.corrcoef(deltas_7d[:, i], actions)[0, 1])
    delta_action_corr[name] = corr
    print(f"    {name:12s}: r={corr:+.6f}")

results['delta_statistics'] = {
    'per_dimension': delta_stats,
    'delta_action_correlation': delta_action_corr,
    'delta_mean_norm': float(np.linalg.norm(np.mean(deltas_7d, axis=0))),
    'delta_std_mean': float(np.mean(np.std(deltas_7d, axis=0))),
}

# ===================== 10. Additional checks =====================
print("\n" + "-" * 70)
print("ADDITIONAL INTEGRITY CHECKS")
print("-" * 70)

# Check continuity: next_obs[t] should relate to obs[t+1]
continuity_errors = 0
max_continuity_diff = 0.0
for i in range(min(10000, len(obs_8d) - 1)):
    if not done[i]:
        diff = np.max(np.abs(next_obs_8d[i] - obs_8d[i + 1]))
        if diff > 1e-6:
            continuity_errors += 1
        max_continuity_diff = max(max_continuity_diff, diff)

print(f"  Continuity check (next_obs[t] vs obs[t+1], first 10k non-done):")
print(f"    Max difference: {max_continuity_diff:.2e}")
print(f"    Errors (>1e-6): {continuity_errors}")

# Check if data is sorted by episode
episode_order_ok = True
last_done_idx = -1
for i in range(len(done)):
    if done[i]:
        if i <= last_done_idx:
            episode_order_ok = False
            break
        last_done_idx = i

print(f"  Episodes in order: {'YES' if episode_order_ok else 'NO'}")

# Check for duplicate rows
n_unique_rows_7d = len(np.unique(obs_7d, axis=0))
print(f"  Unique 7D obs rows: {n_unique_rows_7d} / {len(obs_7d)} ({n_unique_rows_7d/len(obs_7d)*100:.2f}%)")

# Reward distribution
rewards = data['reward']
reward_stats = {
    'min': float(np.min(rewards)),
    'max': float(np.max(rewards)),
    'mean': float(np.mean(rewards)),
    'std': float(np.std(rewards)),
    'sum': float(np.sum(rewards)),
}
print(f"  Reward: min={reward_stats['min']:.4f}, max={reward_stats['max']:.4f}, "
      f"mean={reward_stats['mean']:.4f}, sum={reward_stats['sum']:.2f}")

results['additional_checks'] = {
    'continuity_max_diff': max_continuity_diff,
    'continuity_errors': continuity_errors,
    'episodes_in_order': episode_order_ok,
    'unique_7d_rows': n_unique_rows_7d,
    'unique_7d_fraction': float(n_unique_rows_7d / len(obs_7d)),
    'reward_stats': reward_stats,
}

# ===================== SAVE =====================
print("\n" + "=" * 70)
print(f"Saving results to: {OUTPUT_PATH}")
print("=" * 70)

import os
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, 'w') as f:
    json.dump(results, f, indent=2, default=str)

print("DONE. Audit complete.")
