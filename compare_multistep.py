"""Multi-step rollout comparison using CONTIGUOUS trajectories.

The previous test used test-set indices as sequential steps, which is wrong
because train_test_split shuffles the data. This script uses actual
contiguous trajectories from the original data.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
import os

os.chdir('D:/系统辨识作业/sindy_bicycle')

STATE_NAMES = ['theta', 'delta', 'theta_dot', 'delta_dot']


# ============================================================
# 1. Load data & models
# ============================================================
print("=" * 70)
print("Loading data...")
data = np.load('meijaard_openloop_data_v35.npz')
states = data['states']       # (29478, 4) - contiguous in time
actions = data['taus']        # (29478, 1)
next_states = data['next_states']  # (29478, 4)
dt = float(data['dt'])

print(f"  Samples: {len(states)}, dt={dt:.4f}")

# Load SINDy model
sindy_data = np.load('meijaard_sindy_v35.npz')
Xi_full = sindy_data['coefficients']  # (21, 4)
labels = sindy_data['feature_names']

# Full SINDy model (best threshold from previous comparison)
sindy_full = np.load('sindy_full_model.npz')
best_Xi = sindy_full['coefficients']
state_std = sindy_full['state_std']
action_std = sindy_full['action_std']


# ============================================================
# 2. Define models
# ============================================================

# Linear model (current MBPO-SAC)
A_d = np.eye(4) + Xi_full[1:5, :].T
B_d = Xi_full[5:6, :].T

def linear_predict(s, a):
    return (A_d @ s.T + B_d @ a.T).T


# SINDy library builder
def build_library(s_norm, a_norm):
    n = s_norm.shape[0]
    x = np.hstack([s_norm, a_norm])
    n_feat = x.shape[1]
    features = [np.ones((n, 1))]
    for i in range(n_feat):
        features.append(x[:, i:i+1])
    for i in range(n_feat):
        for j in range(i, n_feat):
            features.append((x[:, i:i+1] * x[:, j:j+1]))
    return np.hstack(features)


# NN models (load saved weights)
class ResidualNet(nn.Module):
    def __init__(self, state_dim=4, action_dim=1, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, state_dim),
        )
    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=-1))


class PureDynamicsNet(nn.Module):
    def __init__(self, state_dim=4, action_dim=1, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, state_dim),
        )
    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=-1))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

res_net = ResidualNet().to(device)
res_net.load_state_dict(torch.load('sindy_nn_residual.pt', map_location=device))
res_net.eval()

pure_net = PureDynamicsNet().to(device)
pure_net.load_state_dict(torch.load('pure_dynamics_nn.pt', map_location=device))
pure_net.eval()


# ============================================================
# 3. Multi-step rollout on contiguous trajectories
# ============================================================
print("\n" + "=" * 70)
print("Multi-step Rollout on Contiguous Trajectories")
print("=" * 70)

# Use multiple trajectory segments from the data
# Each segment is a contiguous sequence of states/actions
total_samples = len(states)
n_segments = 5
seg_length = 500  # steps per segment
max_display_steps = [1, 5, 10, 20, 50, 100, 200, 500]

# Pick evenly spaced segments
segment_starts = np.linspace(0, total_samples - seg_length - 1, n_segments, dtype=int)

all_results = {
    'linear': {s: [] for s in max_display_steps},
    'sindy': {s: [] for s in max_display_steps},
    'sindy_nn': {s: [] for s in max_display_steps},
    'pure_nn': {s: [] for s in max_display_steps},
}

# Per-step RMSE for detailed curves
max_steps = min(seg_length, 500)
step_rmses = {
    'linear': np.zeros((n_segments, max_steps)),
    'sindy': np.zeros((n_segments, max_steps)),
    'sindy_nn': np.zeros((n_segments, max_steps)),
    'pure_nn': np.zeros((n_segments, max_steps)),
}

for seg_idx, start in enumerate(segment_starts):
    print(f"\n  Segment {seg_idx+1}/{n_segments} (start={start})")

    # Ground truth trajectory
    true_traj = states[start:start + seg_length + 1]  # (seg_length+1, 4)
    act_seq = actions[start:start + seg_length]        # (seg_length, 1)

    s0 = true_traj[0:1]  # (1, 4)

    # --- Linear rollout ---
    linear_traj = np.zeros_like(true_traj)
    linear_traj[0] = s0[0]
    for i in range(seg_length):
        linear_traj[i+1] = linear_predict(linear_traj[i:i+1], act_seq[i:i+1])[0]

    # --- SINDy rollout ---
    sindy_traj = np.zeros_like(true_traj)
    sindy_traj[0] = s0[0]
    for i in range(seg_length):
        s_n = sindy_traj[i:i+1] / state_std
        a_n = act_seq[i:i+1] / action_std
        lib = build_library(s_n, a_n)
        delta_n = lib @ best_Xi
        sindy_traj[i+1] = sindy_traj[i] + (delta_n * state_std)[0]

    # --- SINDy+NN rollout ---
    sindy_nn_traj = np.zeros_like(true_traj)
    sindy_nn_traj[0] = s0[0]
    for i in range(seg_length):
        s_n = sindy_nn_traj[i:i+1] / state_std
        a_n = act_seq[i:i+1] / action_std
        lib = build_library(s_n, a_n)
        delta_sindy_n = lib @ best_Xi
        with torch.no_grad():
            s_t = torch.FloatTensor(s_n).to(device)
            a_t = torch.FloatTensor(a_n).to(device)
            delta_nn_n = res_net(s_t, a_t).cpu().numpy()
        delta_n = delta_sindy_n + delta_nn_n
        sindy_nn_traj[i+1] = sindy_nn_traj[i] + (delta_n * state_std)[0]

    # --- Pure NN rollout ---
    pure_nn_traj = np.zeros_like(true_traj)
    pure_nn_traj[0] = s0[0]
    for i in range(seg_length):
        s_n = pure_nn_traj[i:i+1] / state_std
        a_n = act_seq[i:i+1] / action_std
        with torch.no_grad():
            s_t = torch.FloatTensor(s_n).to(device)
            a_t = torch.FloatTensor(a_n).to(device)
            delta_n = pure_net(s_t, a_t).cpu().numpy()
        pure_nn_traj[i+1] = pure_nn_traj[i] + (delta_n * state_std)[0]

    # Per-step RMSE (Euclidean distance in state space)
    for i in range(max_steps):
        step_rmses['linear'][seg_idx, i] = np.sqrt(np.mean((linear_traj[i+1] - true_traj[i+1])**2))
        step_rmses['sindy'][seg_idx, i] = np.sqrt(np.mean((sindy_traj[i+1] - true_traj[i+1])**2))
        step_rmses['sindy_nn'][seg_idx, i] = np.sqrt(np.mean((sindy_nn_traj[i+1] - true_traj[i+1])**2))
        step_rmses['pure_nn'][seg_idx, i] = np.sqrt(np.mean((pure_nn_traj[i+1] - true_traj[i+1])**2))

    # Collect at specific steps
    for s in max_display_steps:
        if s < seg_length:
            all_results['linear'][s].append(np.sqrt(np.mean((linear_traj[s] - true_traj[s])**2)))
            all_results['sindy'][s].append(np.sqrt(np.mean((sindy_traj[s] - true_traj[s])**2)))
            all_results['sindy_nn'][s].append(np.sqrt(np.mean((sindy_nn_traj[s] - true_traj[s])**2)))
            all_results['pure_nn'][s].append(np.sqrt(np.mean((pure_nn_traj[s] - true_traj[s])**2)))

    # Per-dimension at step 100 for this segment
    if seg_length >= 100:
        print(f"    Step 100 per-dim RMSE:")
        for d, name in enumerate(STATE_NAMES):
            e_l = abs(linear_traj[100, d] - true_traj[100, d])
            e_s = abs(sindy_traj[100, d] - true_traj[100, d])
            e_sn = abs(sindy_nn_traj[100, d] - true_traj[100, d])
            e_pn = abs(pure_nn_traj[100, d] - true_traj[100, d])
            print(f"      {name:12s}: L={e_l:.4f}  S={e_s:.4f}  S+NN={e_sn:.4f}  NN={e_pn:.4f}")


# ============================================================
# 4. Results summary
# ============================================================
print("\n" + "=" * 70)
print("RESULTS: Multi-step RMSE (averaged over segments)")
print("=" * 70)

header = f"  {'Step':>6} | {'Linear':>10} | {'SINDy':>10} | {'SINDy+NN':>10} | {'Pure NN':>10} | {'S+NN/Lin':>10} | {'NN/Lin':>10}"
print(header)
print(f"  {'-'*6} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*10}")

for s in max_display_steps:
    if len(all_results['linear'][s]) > 0:
        m_l = np.mean(all_results['linear'][s])
        m_s = np.mean(all_results['sindy'][s])
        m_sn = np.mean(all_results['sindy_nn'][s])
        m_pn = np.mean(all_results['pure_nn'][s])
        ratio_sn = m_l / m_sn if m_sn > 0 else float('inf')
        ratio_pn = m_l / m_pn if m_pn > 0 else float('inf')
        print(f"  {s:>6} | {m_l:>10.4f} | {m_s:>10.4f} | {m_sn:>10.4f} | {m_pn:>10.4f} | {ratio_sn:>9.2f}x | {ratio_pn:>9.2f}x")


# ============================================================
# 5. Per-dimension breakdown at key steps
# ============================================================
print("\n" + "=" * 70)
print("Per-Dimension RMSE at Key Steps (averaged over segments)")
print("=" * 70)

for target_step in [10, 50, 100, 200]:
    if target_step >= max_steps:
        continue
    print(f"\n  Step {target_step}:")
    print(f"  {'Dim':>12} | {'Linear':>10} | {'SINDy':>10} | {'SINDy+NN':>10} | {'Pure NN':>10} | {'S+NN/Lin':>10}")
    print(f"  {'-'*12} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*10}")

    for d, name in enumerate(STATE_NAMES):
        vals_l, vals_s, vals_sn, vals_pn = [], [], [], []
        for seg_idx, start in enumerate(segment_starts):
            true_traj = states[start:start + seg_length + 1]
            act_seq = actions[start:start + seg_length]
            s0 = true_traj[0:1]

            # Recompute (could cache but keeping it clear)
            traj_l = np.zeros((target_step + 1, 4)); traj_l[0] = s0[0]
            traj_s = np.zeros((target_step + 1, 4)); traj_s[0] = s0[0]
            traj_sn = np.zeros((target_step + 1, 4)); traj_sn[0] = s0[0]
            traj_pn = np.zeros((target_step + 1, 4)); traj_pn[0] = s0[0]

            for i in range(target_step):
                traj_l[i+1] = linear_predict(traj_l[i:i+1], act_seq[i:i+1])[0]

                s_n = traj_s[i:i+1] / state_std
                a_n = act_seq[i:i+1] / action_std
                lib = build_library(s_n, a_n)
                traj_s[i+1] = traj_s[i] + ((lib @ best_Xi) * state_std)[0]

                s_n = traj_sn[i:i+1] / state_std
                a_n = act_seq[i:i+1] / action_std
                lib = build_library(s_n, a_n)
                d_sindy = lib @ best_Xi
                with torch.no_grad():
                    d_nn = res_net(torch.FloatTensor(s_n).to(device),
                                   torch.FloatTensor(a_n).to(device)).cpu().numpy()
                traj_sn[i+1] = traj_sn[i] + ((d_sindy + d_nn) * state_std)[0]

                s_n = traj_pn[i:i+1] / state_std
                a_n = act_seq[i:i+1] / action_std
                with torch.no_grad():
                    d_pn = pure_net(torch.FloatTensor(s_n).to(device),
                                    torch.FloatTensor(a_n).to(device)).cpu().numpy()
                traj_pn[i+1] = traj_pn[i] + (d_pn * state_std)[0]

            gt = true_traj[target_step]
            vals_l.append(abs(traj_l[target_step, d] - gt[d]))
            vals_s.append(abs(traj_s[target_step, d] - gt[d]))
            vals_sn.append(abs(traj_sn[target_step, d] - gt[d]))
            vals_pn.append(abs(traj_pn[target_step, d] - gt[d]))

        m_l = np.mean(vals_l)
        m_s = np.mean(vals_s)
        m_sn = np.mean(vals_sn)
        m_pn = np.mean(vals_pn)
        ratio = m_l / m_sn if m_sn > 0 else float('inf')
        print(f"  {name:>12} | {m_l:>10.4f} | {m_s:>10.4f} | {m_sn:>10.4f} | {m_pn:>10.4f} | {ratio:>9.2f}x")


# ============================================================
# 6. RMSE curve over time (averaged across segments)
# ============================================================
print("\n" + "=" * 70)
print("RMSE Curve (sampled points, averaged over segments)")
print("=" * 70)

avg_rmses = {}
for model in ['linear', 'sindy', 'sindy_nn', 'pure_nn']:
    avg_rmses[model] = np.mean(step_rmses[model], axis=0)

print(f"\n  {'Step':>6} | {'Linear':>10} | {'SINDy':>10} | {'SINDy+NN':>10} | {'Pure NN':>10}")
print(f"  {'-'*6} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*10}")

for i in [0, 1, 2, 4, 9, 19, 49, 99, 149, 199, 299, 399, 499]:
    if i < max_steps:
        print(f"  {i+1:>6} | {avg_rmses['linear'][i]:>10.4f} | {avg_rmses['sindy'][i]:>10.4f} | "
              f"{avg_rmses['sindy_nn'][i]:>10.4f} | {avg_rmses['pure_nn'][i]:>10.4f}")

print("\nDone.")
