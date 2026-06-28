"""Compare dynamics models: Linear vs SINDy vs SINDy+NN residual.

Uses Meijaard openloop data (29478 samples, 4D state: theta, delta, theta_dot, delta_dot).
Evaluates:
  1. Current linear model (A_d @ s + B_d * tau) - what MBPO-SAC v6 uses
  2. Full SINDy (polynomial library + STLSQ)
  3. SINDy + NN residual
  4. Pure NN

Metrics: single-step RMSE per dimension, multi-step rollout RMSE.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import os

os.chdir('D:/系统辨识作业/sindy_bicycle')

STATE_NAMES = ['theta', 'delta', 'theta_dot', 'delta_dot']


# ============================================================
# 1. Load data
# ============================================================
print("=" * 70)
print("Loading data...")
data = np.load('meijaard_openloop_data_v35.npz')
states = data['states']       # (29478, 4)
actions = data['taus']        # (29478, 1)
next_states = data['next_states']  # (29478, 4)
dt = float(data['dt'])        # 1/30
v_speed = float(data['v'])    # 3.5 m/s

print(f"  Samples: {len(states)}, dt={dt:.4f}, v={v_speed}")
print(f"  State range:")
for i, name in enumerate(STATE_NAMES):
    print(f"    {name}: [{states[:, i].min():.4f}, {states[:, i].max():.4f}]")

# Train/test split
(idx_train, idx_test) = train_test_split(
    np.arange(len(states)), test_size=0.2, random_state=42
)
s_train, s_test = states[idx_train], states[idx_test]
a_train, a_test = actions[idx_train], actions[idx_test]
ns_train, ns_test = next_states[idx_train], next_states[idx_test]

# Normalization scales
state_std = np.std(s_train, axis=0)
state_std[state_std < 1e-6] = 1.0
action_std = max(np.std(a_train), 1.0)

s_train_norm = s_train / state_std
s_test_norm = s_test / state_std
a_train_norm = a_train / action_std
a_test_norm = a_test / action_std
ns_train_norm = ns_train / state_std
ns_test_norm = ns_test / state_std

delta_train_norm = ns_train_norm - s_train_norm
delta_test_norm = ns_test_norm - s_test_norm


# ============================================================
# 2. Model 1: Linear (current MBPO-SAC)
# ============================================================
print("\n" + "=" * 70)
print("Model 1: Linear (A_d @ s + B_d * tau)")
print("=" * 70)

# Load existing SINDy coefficients
sindy_data = np.load('meijaard_sindy_v35.npz')
Xi_full = sindy_data['coefficients']  # (21, 4)
labels = sindy_data['feature_names']

# Current approach: extract linear terms only
A_d = np.eye(4) + Xi_full[1:5, :].T
B_d = Xi_full[5:6, :].T

print(f"  A_d:\n{A_d}")
print(f"  B_d:\n{B_d.T}")

# Predict on test set
def linear_predict(s, a):
    """Current MBPO-SAC model: s_next = A_d @ s + B_d * tau."""
    return (A_d @ s.T + B_d @ a.T).T

ns_pred_linear = linear_predict(s_test, a_test)
delta_pred_linear = (ns_pred_linear - s_test) / state_std

print(f"\n  Single-step RMSE (normalized):")
for i, name in enumerate(STATE_NAMES):
    rmse = np.sqrt(np.mean((delta_test_norm[:, i] - delta_pred_linear[:, i]) ** 2))
    r2 = r2_score(delta_test_norm[:, i], delta_pred_linear[:, i])
    print(f"    {name:12s}: RMSE={rmse:.6f}, R²={r2:.4f}")

avg_rmse_linear = np.sqrt(np.mean((delta_test_norm - delta_pred_linear) ** 2))
print(f"  Average RMSE: {avg_rmse_linear:.6f}")


# ============================================================
# 3. Model 2: Full SINDy (polynomial library)
# ============================================================
print("\n" + "=" * 70)
print("Model 2: Full SINDy (polynomial degree=2)")
print("=" * 70)


def build_library(s_norm, a_norm):
    """Build polynomial library up to degree 2."""
    n = s_norm.shape[0]
    x = np.hstack([s_norm, a_norm])  # (n, 5)
    n_feat = x.shape[1]
    features = [np.ones((n, 1))]  # constant
    for i in range(n_feat):
        features.append(x[:, i:i+1])
    for i in range(n_feat):
        for j in range(i, n_feat):
            features.append((x[:, i:i+1] * x[:, j:j+1]))
    return np.hstack(features)


def stlsq(Theta, Y, threshold, max_iter=50, alpha=0.01):
    """STLSQ with ridge regularization."""
    n_lib = Theta.shape[1]
    n_states = Y.shape[1]
    Xi = np.zeros((n_lib, n_states))
    for dim in range(n_states):
        y = Y[:, dim]
        coef = np.linalg.lstsq(Theta, y, rcond=None)[0]
        for _ in range(max_iter):
            coef_old = coef.copy()
            small = np.abs(coef) < threshold
            coef[small] = 0
            active = ~small
            if not np.any(active):
                break
            Theta_a = Theta[:, active]
            ATA = Theta_a.T @ Theta_a + alpha * np.eye(Theta_a.shape[1])
            ATy = Theta_a.T @ y
            coef[active] = np.linalg.solve(ATA, ATy)
            if np.allclose(coef, coef_old, atol=1e-10):
                break
        Xi[:, dim] = coef
    return Xi


# Build libraries
Theta_train = build_library(s_train_norm, a_train_norm)
Theta_test = build_library(s_test_norm, a_test_norm)

# Threshold sweep
best_Xi = None
best_score = -np.inf
best_thresh = None

for thresh in [0.005, 0.01, 0.02, 0.05, 0.1]:
    Xi = stlsq(Theta_train, delta_train_norm, threshold=thresh)
    pred = Theta_test @ Xi
    rmse = np.sqrt(np.mean((delta_test_norm - pred) ** 2))
    n_terms = np.sum(np.abs(Xi) > 1e-6)
    r2_avg = np.mean([r2_score(delta_test_norm[:, i], pred[:, i]) for i in range(4)])
    score = r2_avg / (1 + n_terms / 30)
    print(f"  thresh={thresh:.3f}: RMSE={rmse:.6f}, R²={r2_avg:.4f}, terms={n_terms}, score={score:.4f}")
    if score > best_score:
        best_score = score
        best_Xi = Xi
        best_thresh = thresh

print(f"\n  Best threshold: {best_thresh}")
delta_pred_sindy = Theta_test @ best_Xi

print(f"\n  Single-step RMSE (normalized):")
for i, name in enumerate(STATE_NAMES):
    rmse = np.sqrt(np.mean((delta_test_norm[:, i] - delta_pred_sindy[:, i]) ** 2))
    r2 = r2_score(delta_test_norm[:, i], delta_pred_sindy[:, i])
    print(f"    {name:12s}: RMSE={rmse:.6f}, R²={r2:.4f}")

avg_rmse_sindy = np.sqrt(np.mean((delta_test_norm - delta_pred_sindy) ** 2))
print(f"  Average RMSE: {avg_rmse_sindy:.6f}")


# ============================================================
# 4. Model 3: SINDy + NN Residual
# ============================================================
print("\n" + "=" * 70)
print("Model 3: SINDy + NN Residual")
print("=" * 70)


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


# Compute SINDy residuals on training data
sindy_delta_train_norm = Theta_train @ best_Xi
residual_train = delta_train_norm - sindy_delta_train_norm

sindy_delta_test_norm = Theta_test @ best_Xi
residual_test = delta_test_norm - sindy_delta_test_norm

print(f"  Residual statistics (train):")
for i, name in enumerate(STATE_NAMES):
    print(f"    {name:12s}: mean={residual_train[:, i].mean():.6f}, "
          f"std={residual_train[:, i].std():.6f}, "
          f"max={np.abs(residual_train[:, i]).max():.6f}")

# Train NN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  Device: {device}")

res_net = ResidualNet().to(device)
optimizer = optim.Adam(res_net.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

state_t = torch.FloatTensor(s_train_norm).to(device)
action_t = torch.FloatTensor(a_train_norm).to(device)
residual_t = torch.FloatTensor(residual_train).to(device)

dataset = TensorDataset(state_t, action_t, residual_t)
loader = DataLoader(dataset, batch_size=512, shuffle=True)

print(f"  Training NN residual (200 epochs)...")
for epoch in range(200):
    total_loss = 0
    n_batches = 0
    for s_b, a_b, r_b in loader:
        pred = res_net(s_b, a_b)
        loss = nn.MSELoss()(pred, r_b)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    scheduler.step()
    if (epoch + 1) % 50 == 0:
        print(f"    Epoch {epoch+1}: loss={total_loss/n_batches:.8f}")

res_net.eval()

# Evaluate SINDy + NN
with torch.no_grad():
    s_test_t = torch.FloatTensor(s_test_norm).to(device)
    a_test_t = torch.FloatTensor(a_test_norm).to(device)
    nn_res_pred = res_net(s_test_t, a_test_t).cpu().numpy()

delta_pred_sindy_nn = sindy_delta_test_norm + nn_res_pred

print(f"\n  Single-step RMSE (normalized):")
for i, name in enumerate(STATE_NAMES):
    rmse = np.sqrt(np.mean((delta_test_norm[:, i] - delta_pred_sindy_nn[:, i]) ** 2))
    r2 = r2_score(delta_test_norm[:, i], delta_pred_sindy_nn[:, i])
    print(f"    {name:12s}: RMSE={rmse:.6f}, R²={r2:.4f}")

avg_rmse_sindy_nn = np.sqrt(np.mean((delta_test_norm - delta_pred_sindy_nn) ** 2))
print(f"  Average RMSE: {avg_rmse_sindy_nn:.6f}")


# ============================================================
# 5. Model 4: Pure NN (no SINDy prior)
# ============================================================
print("\n" + "=" * 70)
print("Model 4: Pure NN (no SINDy prior)")
print("=" * 70)


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


pure_net = PureDynamicsNet().to(device)
optimizer2 = optim.Adam(pure_net.parameters(), lr=1e-3)
scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=200)

delta_t = torch.FloatTensor(delta_train_norm).to(device)

dataset2 = TensorDataset(state_t, action_t, delta_t)
loader2 = DataLoader(dataset2, batch_size=512, shuffle=True)

print(f"  Training pure NN (200 epochs)...")
for epoch in range(200):
    total_loss = 0
    n_batches = 0
    for s_b, a_b, d_b in loader2:
        pred = pure_net(s_b, a_b)
        loss = nn.MSELoss()(pred, d_b)
        optimizer2.zero_grad()
        loss.backward()
        optimizer2.step()
        total_loss += loss.item()
        n_batches += 1
    scheduler2.step()
    if (epoch + 1) % 50 == 0:
        print(f"    Epoch {epoch+1}: loss={total_loss/n_batches:.8f}")

pure_net.eval()

with torch.no_grad():
    nn_pred = pure_net(s_test_t, a_test_t).cpu().numpy()

print(f"\n  Single-step RMSE (normalized):")
for i, name in enumerate(STATE_NAMES):
    rmse = np.sqrt(np.mean((delta_test_norm[:, i] - nn_pred[:, i]) ** 2))
    r2 = r2_score(delta_test_norm[:, i], nn_pred[:, i])
    print(f"    {name:12s}: RMSE={rmse:.6f}, R²={r2:.4f}")

avg_rmse_pure_nn = np.sqrt(np.mean((delta_test_norm - nn_pred) ** 2))
print(f"  Average RMSE: {avg_rmse_pure_nn:.6f}")


# ============================================================
# 6. Multi-step rollout comparison
# ============================================================
print("\n" + "=" * 70)
print("Multi-step Rollout Comparison (100 steps)")
print("=" * 70)

# Pick a test trajectory
rollout_len = 100
s0 = s_test[0]
a_seq = a_test[:rollout_len]

# True trajectory
true_traj = np.zeros((rollout_len + 1, 4))
true_traj[0] = s0
for i in range(rollout_len):
    true_traj[i + 1] = ns_test[i] if i < len(ns_test) else true_traj[i]

# Linear rollout
linear_traj = np.zeros((rollout_len + 1, 4))
linear_traj[0] = s0
for i in range(rollout_len):
    s = linear_traj[i:i+1]
    a = a_seq[i:i+1]
    linear_traj[i+1] = linear_predict(s, a)[0]

# SINDy rollout
sindy_traj = np.zeros((rollout_len + 1, 4))
sindy_traj[0] = s0
for i in range(rollout_len):
    s_n = sindy_traj[i:i+1] / state_std
    a_n = a_seq[i:i+1] / action_std
    lib = build_library(s_n, a_n)
    delta_n = lib @ best_Xi
    sindy_traj[i+1] = sindy_traj[i] + (delta_n * state_std)[0]

# SINDy+NN rollout
sindy_nn_traj = np.zeros((rollout_len + 1, 4))
sindy_nn_traj[0] = s0
for i in range(rollout_len):
    s_n = sindy_nn_traj[i:i+1] / state_std
    a_n = a_seq[i:i+1] / action_std
    lib = build_library(s_n, a_n)
    delta_sindy_n = lib @ best_Xi
    with torch.no_grad():
        s_t = torch.FloatTensor(s_n).to(device)
        a_t = torch.FloatTensor(a_n).to(device)
        delta_nn_n = res_net(s_t, a_t).cpu().numpy()
    delta_n = delta_sindy_n + delta_nn_n
    sindy_nn_traj[i+1] = sindy_nn_traj[i] + (delta_n * state_std)[0]

# Pure NN rollout
pure_nn_traj = np.zeros((rollout_len + 1, 4))
pure_nn_traj[0] = s0
for i in range(rollout_len):
    s_n = pure_nn_traj[i:i+1] / state_std
    a_n = a_seq[i:i+1] / action_std
    with torch.no_grad():
        s_t = torch.FloatTensor(s_n).to(device)
        a_t = torch.FloatTensor(a_n).to(device)
        delta_n = pure_net(s_t, a_t).cpu().numpy()
    pure_nn_traj[i+1] = pure_nn_traj[i] + (delta_n * state_std)[0]


def rollout_rmse(pred, true):
    return np.sqrt(np.mean((pred - true) ** 2, axis=1))


print(f"\n  {'Step':>6} | {'Linear':>10} | {'SINDy':>10} | {'SINDy+NN':>10} | {'Pure NN':>10}")
print(f"  {'-'*6} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*10}")

for step in [1, 5, 10, 20, 50, 100]:
    if step > rollout_len:
        continue
    rmse_l = np.sqrt(np.mean((linear_traj[step] - true_traj[step]) ** 2))
    rmse_s = np.sqrt(np.mean((sindy_traj[step] - true_traj[step]) ** 2))
    rmse_sn = np.sqrt(np.mean((sindy_nn_traj[step] - true_traj[step]) ** 2))
    rmse_pn = np.sqrt(np.mean((pure_nn_traj[step] - true_traj[step]) ** 2))
    print(f"  {step:>6} | {rmse_l:>10.6f} | {rmse_s:>10.6f} | {rmse_sn:>10.6f} | {rmse_pn:>10.6f}")

# Overall trajectory RMSE
print(f"\n  Trajectory RMSE (avg over {rollout_len} steps):")
print(f"  Linear:    {np.sqrt(np.mean((linear_traj[:rollout_len+1] - true_traj[:rollout_len+1])**2)):.6f}")
print(f"  SINDy:     {np.sqrt(np.mean((sindy_traj[:rollout_len+1] - true_traj[:rollout_len+1])**2)):.6f}")
print(f"  SINDy+NN:  {np.sqrt(np.mean((sindy_nn_traj[:rollout_len+1] - true_traj[:rollout_len+1])**2)):.6f}")
print(f"  Pure NN:   {np.sqrt(np.mean((pure_nn_traj[:rollout_len+1] - true_traj[:rollout_len+1])**2)):.6f}")


# ============================================================
# 7. Summary
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY: Single-step Prediction Accuracy")
print("=" * 70)
print(f"  {'Model':>15} | {'Avg RMSE':>10} | {'vs Linear':>10}")
print(f"  {'-'*15} | {'-'*10} | {'-'*10}")
print(f"  {'Linear':>15} | {avg_rmse_linear:>10.6f} | {'baseline':>10}")
print(f"  {'SINDy':>15} | {avg_rmse_sindy:>10.6f} | {(avg_rmse_linear/avg_rmse_sindy-1)*100:>+9.1f}%")
print(f"  {'SINDy+NN':>15} | {avg_rmse_sindy_nn:>10.6f} | {(avg_rmse_linear/avg_rmse_sindy_nn-1)*100:>+9.1f}%")
print(f"  {'Pure NN':>15} | {avg_rmse_pure_nn:>10.6f} | {(avg_rmse_linear/avg_rmse_pure_nn-1)*100:>+9.1f}%")

# Save models
print("\nSaving models...")
torch.save(res_net.state_dict(), 'sindy_nn_residual.pt')
torch.save(pure_net.state_dict(), 'pure_dynamics_nn.pt')
np.savez('sindy_full_model.npz',
         coefficients=best_Xi,
         threshold=best_thresh,
         state_std=state_std,
         action_std=action_std)
print("  Saved: sindy_nn_residual.pt, pure_dynamics_nn.pt, sindy_full_model.npz")
print("Done.")
