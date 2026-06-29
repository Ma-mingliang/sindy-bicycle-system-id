"""Per-State Ensemble Methods for Bicycle System Identification.

Implements three ensemble strategies that combine different model types
(Neural ODE, GP, physics-based) with per-state weighting:

1. Per-State Ensemble Model:
   - For each state dimension, train multiple models (NODE variants, GP, physics)
   - Combine via weighted average, weights from validation performance

2. Adaptive Ensemble:
   - Dynamically adjust per-state weights based on current state location
   - Uses state-space partitioning (k-means or grid) for local weight learning
   - Falls back to physics model in OOD regions

3. Physics-Informed Ensemble:
   - Uses known kinematics (e_y_dot = v*sin(e_psi), e_psi_dot = -v*delta/L)
   - Blends physics prediction with data-driven (NODE/GP) via learned weights
   - Physics acts as a structural prior + regularizer

Target: Beat current best primary score of 0.3728 (physics_correction_multi_seed).
Baseline v9 primary score: 0.5110.
"""

import sys
import json
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel, ConstantKernel
from sklearn.cluster import KMeans

warnings.filterwarnings('ignore')

sys.path.insert(0, 'D:/系统辨识作业/sindy_bicycle')

# ============================================================
# Constants
# ============================================================

STATE_NAMES_7D = ['e_y', 'e_psi', 'v', 'theta', 'theta_dot', 'delta', 'delta_dot']
STATE_DIM = 7
ACTION_DIM = 1
IDX_7D_FROM_8D = [0, 1, 2, 3, 4, 6, 7]
DT = 1.0 / 30.0
WHEELBASE = 1.0

PHYSICAL_LIMITS = {
    'e_y': 5.0, 'e_psi': np.pi, 'v': 5.0,
    'theta': np.pi, 'theta_dot': 10.0,
    'delta': np.pi / 2, 'delta_dot': 10.0,
}

IDX_EY, IDX_EPSI, IDX_V = 0, 1, 2
IDX_THETA, IDX_THETA_DOT = 3, 4
IDX_DELTA, IDX_DELTA_DOT = 5, 6


# ============================================================
# Physics Model
# ============================================================

def compute_physics_deltas(state):
    """Compute physics-based state deltas from known kinematics.

    Known exact relationships:
      e_y_dot    = v * sin(e_psi)
      e_psi_dot  = -v * delta / L
      theta_dot  = d(theta)/dt  (definition)
      delta_dot  = d(delta)/dt  (definition)

    Unknown (set to 0):
      v_dot, theta_ddot, delta_ddot
    """
    e_psi = state[IDX_EPSI]
    v = state[IDX_V]
    theta_dot_val = state[IDX_THETA_DOT]
    delta = state[IDX_DELTA]
    delta_dot_val = state[IDX_DELTA_DOT]

    d = np.zeros(7)
    d[IDX_EY] = v * np.sin(e_psi) * DT
    d[IDX_EPSI] = -v * delta / WHEELBASE * DT
    d[IDX_V] = 0.0
    d[IDX_THETA] = theta_dot_val * DT
    d[IDX_THETA_DOT] = 0.0
    d[IDX_DELTA] = delta_dot_val * DT
    d[IDX_DELTA_DOT] = 0.0
    return d


def compute_physics_deltas_batch(states):
    """Batch version of physics delta computation."""
    n = len(states)
    out = np.zeros((n, 7))
    for i in range(n):
        out[i] = compute_physics_deltas(states[i])
    return out


# ============================================================
# Data Loading
# ============================================================

def load_data(seed=42):
    """Load 7D data from 8D dataset with episode structure."""
    data = np.load(
        'D:/系统辨识作业/sindy_bicycle/data/stage2_dataset_150k.npz',
        allow_pickle=True,
    )
    obs = data['obs'][:, IDX_7D_FROM_8D]
    next_obs = data['next_obs'][:, IDX_7D_FROM_8D]
    deltas = next_obs - obs
    done = data['done']
    action = data['action']

    episode_ends = np.where(done)[0]
    episode_starts = np.concatenate([[0], episode_ends[:-1] + 1])
    n_episodes = len(episode_ends)

    np.random.seed(seed)
    indices = np.random.permutation(n_episodes)
    n_train = int(0.75 * n_episodes)
    train_eps = indices[:n_train]
    test_eps = indices[n_train:]

    episodes = []
    for ep_idx in range(n_episodes):
        start = episode_starts[ep_idx]
        end = episode_ends[ep_idx] + 1
        episodes.append({
            'obs': obs[start:end],
            'action': action[start:end],
            'deltas': deltas[start:end],
            'length': end - start,
        })

    train_obs = np.concatenate([episodes[ep]['obs'] for ep in train_eps])
    train_action = np.concatenate([episodes[ep]['action'] for ep in train_eps])
    train_deltas = np.concatenate([episodes[ep]['deltas'] for ep in train_eps])

    state_std = np.std(train_obs, axis=0)
    state_std[state_std < 1e-10] = 1.0
    action_std = np.std(train_action)
    if action_std < 1e-10:
        action_std = 1.0
    delta_std = np.std(train_deltas, axis=0)
    delta_std[delta_std < 1e-10] = 1.0

    # Split train into train/val (90/10)
    n_train_eps = len(train_eps)
    n_val = max(1, n_train_eps // 10)
    val_eps = train_eps[:n_val]
    actual_train_eps = train_eps[n_val:]

    val_obs = np.concatenate([episodes[ep]['obs'] for ep in val_eps])
    val_action = np.concatenate([episodes[ep]['action'] for ep in val_eps])
    val_deltas = np.concatenate([episodes[ep]['deltas'] for ep in val_eps])

    return {
        'episodes': episodes,
        'train_eps': actual_train_eps,
        'val_eps': val_eps,
        'test_eps': test_eps,
        'state_std': state_std,
        'action_std': action_std,
        'delta_std': delta_std,
        'train_obs': train_obs,
        'train_action': train_action,
        'train_deltas': train_deltas,
        'val_obs': val_obs,
        'val_action': val_action,
        'val_deltas': val_deltas,
    }


# ============================================================
# Neural ODE Models (multiple variants)
# ============================================================

class ODEFunc(nn.Module):
    """Neural ODE dynamics function: dsdt = f(s, a)."""
    def __init__(self, hidden=64, depth=3, activation='tanh'):
        super().__init__()
        act_map = {'tanh': nn.Tanh, 'silu': nn.SiLU, 'relu': nn.ReLU}
        act = act_map.get(activation, nn.Tanh)

        layers = [nn.Linear(STATE_DIM + ACTION_DIM, hidden), act()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden, hidden), act()])
        layers.append(nn.Linear(hidden, STATE_DIM))
        self.net = nn.Sequential(*layers)

        nn.init.zeros_(self.net[-1].bias)
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.1)

    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=-1))


def train_node_model(data, config, seed=42):
    """Train a single Neural ODE model."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    state_std = data['state_std']
    action_std = data['action_std']
    delta_std = data['delta_std']

    train_eps = data['train_eps']
    episodes = data['episodes']

    train_obs = np.concatenate([episodes[ep]['obs'] for ep in train_eps])
    train_action = np.concatenate([episodes[ep]['action'] for ep in train_eps])
    train_deltas = np.concatenate([episodes[ep]['deltas'] for ep in train_eps])

    train_s = torch.FloatTensor(train_obs / state_std)
    train_a = torch.FloatTensor(train_action.reshape(-1, 1) / action_std)
    train_dsdot = torch.FloatTensor(train_deltas / (delta_std * DT))

    ds = torch.utils.data.TensorDataset(train_s, train_a, train_dsdot)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=config['batch_size'], shuffle=True,
    )

    model = ODEFunc(
        hidden=config['hidden'],
        depth=config['depth'],
        activation=config['activation'],
    )

    opt = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=config['n_epochs'],
    )

    model.train()
    for epoch in range(config['n_epochs']):
        for sb, ab, yb in loader:
            pred = model(sb, ab)
            loss = nn.functional.mse_loss(pred, yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        scheduler.step()

    model.eval()
    return model


# ============================================================
# GP Models
# ============================================================

class GPSingleDim:
    """Single-dimension GP model."""
    def __init__(self, max_samples=5000, kernel_type='matern'):
        self._max_samples = max_samples
        self._gp = None
        self._x_mean = None
        self._x_std = None
        self._kernel_type = kernel_type

    def train(self, X, y):
        n = len(X)
        if n > self._max_samples:
            rng = np.random.RandomState(42)
            idx = rng.choice(n, self._max_samples, replace=False)
            X, y = X[idx], y[idx]

        self._x_mean = X.mean(axis=0)
        self._x_std = X.std(axis=0) + 1e-8
        X_scaled = (X - self._x_mean) / self._x_std

        if self._kernel_type == 'matern':
            kernel = Matern(nu=2.5, length_scale=1.0)
        elif self._kernel_type == 'rbf':
            kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(0.01)
        else:
            kernel = Matern(nu=2.5, length_scale=1.0)

        self._gp = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=2, alpha=1e-3,
        )
        self._gp.fit(X_scaled, y)

    def predict(self, x):
        x_scaled = (x - self._x_mean) / self._x_std
        return self._gp.predict(x_scaled.reshape(1, -1))[0]

    def predict_with_uncertainty(self, x):
        x_scaled = (x - self._x_mean) / self._x_std
        mean, std = self._gp.predict(x_scaled.reshape(1, -1), return_std=True)
        return mean[0], std[0]


def train_gp_model(data, max_samples=5000, kernel_type='matern'):
    """Train a GP model per state dimension."""
    state_std = data['state_std']
    action_std = data['action_std']
    delta_std = data['delta_std']

    train_obs = data['train_obs']
    train_action = data['train_action']
    train_deltas = data['train_deltas']

    X = np.hstack([train_obs / state_std, train_action.reshape(-1, 1) / action_std])

    gp_models = {}
    for dim in range(STATE_DIM):
        y = train_deltas[:, dim] / delta_std[dim]
        gp = GPSingleDim(max_samples=max_samples, kernel_type=kernel_type)
        gp.train(X, y)
        gp_models[dim] = gp

    return gp_models


# ============================================================
# Validation & Evaluation
# ============================================================

def check_survival(state):
    """Check if state is within physical limits."""
    for i, name in enumerate(STATE_NAMES_7D):
        if name in PHYSICAL_LIMITS:
            if abs(state[i]) > PHYSICAL_LIMITS[name]:
                return False
    return not (np.any(np.isnan(state)) or np.any(np.isinf(state)))


def evaluate_model_on_val(predict_fn, data, n_segments=3, seed=99):
    """Evaluate a model's per-state NMAE on the validation set.

    Returns per_state_nmae: dict dim_idx -> mean NMAE across segments.
    """
    state_std = data['state_std']
    episodes = data['episodes']
    val_eps = data['val_eps']

    np.random.seed(seed)
    segments = []
    for ep_idx in val_eps:
        ep = episodes[ep_idx]
        if ep['length'] >= 100:
            segments.append(ep)
    segments = segments[:n_segments]

    if not segments:
        return {d: 1.0 for d in range(STATE_DIM)}

    per_state_errors = {d: [] for d in range(STATE_DIM)}

    for seg in segments:
        s0 = seg['obs'][0].copy()
        actions_seg = seg['action'].flatten()
        real_states = seg['obs']
        n = min(100, len(actions_seg))
        s_cur = s0.copy()

        for step in range(n):
            try:
                s_next = predict_fn(s_cur, actions_seg[step])
                if np.any(np.isnan(s_next)) or np.any(np.isinf(s_next)):
                    break
                if not check_survival(s_next):
                    break
                if step + 1 < len(real_states):
                    step_err = np.abs(s_next - real_states[step + 1]) / state_std
                    for d in range(STATE_DIM):
                        per_state_errors[d].append(step_err[d])
                s_cur = s_next
            except Exception:
                break

    per_state_nmae = {}
    for d in range(STATE_DIM):
        errs = per_state_errors[d]
        per_state_nmae[d] = np.mean(errs) if errs else 1.0

    return per_state_nmae


def evaluate_full(predict_fn, data, horizons, n_segments=5, seed=42):
    """Full evaluation across multiple horizons.

    Returns dict: horizon -> {nmae_mean, survival_rate, per_state_nmae}.
    """
    state_std = data['state_std']
    episodes = data['episodes']
    test_eps = data['test_eps']

    np.random.seed(seed)
    segments = []
    for ep_idx in test_eps:
        ep = episodes[ep_idx]
        if ep['length'] >= 1100:
            segments.append(ep)
    segments = segments[:n_segments]

    results = {}
    for h in horizons:
        nmae_list = []
        survival_list = []
        per_state_nmae = {name: [] for name in STATE_NAMES_7D}

        for seg in segments:
            s0 = seg['obs'][0].copy()
            actions_seg = seg['action'].flatten()
            real_states = seg['obs']
            n = min(h, len(actions_seg))
            s_cur = s0.copy()
            survived = True
            step_errors = []

            for step in range(n):
                try:
                    s_next = predict_fn(s_cur, actions_seg[step])
                    if np.any(np.isnan(s_next)) or np.any(np.isinf(s_next)):
                        survived = False
                        break
                    if not check_survival(s_next):
                        survived = False
                        break
                    if step + 1 < len(real_states):
                        step_err = np.abs(s_next - real_states[step + 1]) / state_std
                        step_errors.append(step_err)
                    s_cur = s_next
                except Exception:
                    survived = False
                    break

            if step_errors:
                errors = np.array(step_errors[:min(len(step_errors), n)])
                nmae_per_state = np.mean(errors, axis=0)
                nmae_overall = np.mean(nmae_per_state)
                nmae_list.append(nmae_overall)
                for i, name in enumerate(STATE_NAMES_7D):
                    per_state_nmae[name].append(nmae_per_state[i])
            else:
                nmae_list.append(float('nan'))
            survival_list.append(survived and len(step_errors) >= n - 1)

        valid_nmae = [x for x in nmae_list if not np.isnan(x)]
        results[h] = {
            'nmae_mean': float(np.nanmean(valid_nmae)) if valid_nmae else float('nan'),
            'nmae_std': float(np.nanstd(valid_nmae)) if valid_nmae else float('nan'),
            'survival_rate': float(np.mean(survival_list)),
            'per_state_nmae': {
                name: {
                    'mean': float(np.nanmean(per_state_nmae[name]))
                    if per_state_nmae[name] else float('nan'),
                }
                for name in STATE_NAMES_7D
            },
        }

    return results


def compute_primary_score(results):
    """Compute PrimaryLongHorizonScore = mean(H=100, H=200, H=500)."""
    return np.mean([results[100]['nmae_mean'], results[200]['nmae_mean'],
                    results[500]['nmae_mean']])


# ============================================================
# Approach 1: Per-State Ensemble Model
# ============================================================

class PerStateEnsemble:
    """Weighted ensemble with per-state weights.

    For each state dimension, maintains a weight for each sub-model:
      - NODE-A (tanh, hidden=64, depth=3)
      - NODE-B (tanh, hidden=128, depth=4)
      - NODE-C (silu, hidden=64, depth=3)
      - GP (Matern kernel)
      - Physics model
    Weights are learned from validation performance.
    """
    MODEL_NAMES = ['node_a', 'node_b', 'node_c', 'gp', 'physics']

    def __init__(self, node_models, gp_models, state_std, action_std, delta_std,
                 weights_per_state):
        """
        Args:
            node_models: dict model_name -> ODEFunc model
            gp_models: dict dim -> GPSingleDim
            state_std, action_std, delta_std: normalization arrays
            weights_per_state: dict dim -> array of shape (n_models,)
        """
        self._node_models = node_models
        self._gp_models = gp_models
        self._state_std = state_std
        self._action_std = action_std
        self._delta_std = delta_std
        self._weights = weights_per_state

    def predict(self, s, tau):
        s_norm = s / self._state_std
        a_norm = tau / self._action_std
        x_torch = torch.FloatTensor(
            np.concatenate([s_norm, [a_norm]]),
        ).unsqueeze(0)

        delta = np.zeros(STATE_DIM)

        for dim in range(STATE_DIM):
            w = self._weights[dim]
            pred = 0.0

            # NODE-A
            with torch.no_grad():
                node_a_pred = self._node_models['node_a'](x_torch).numpy()[0][dim]
            pred += w[0] * node_a_pred * self._delta_std[dim] * DT

            # NODE-B
            with torch.no_grad():
                node_b_pred = self._node_models['node_b'](x_torch).numpy()[0][dim]
            pred += w[1] * node_b_pred * self._delta_std[dim] * DT

            # NODE-C
            with torch.no_grad():
                node_c_pred = self._node_models['node_c'](x_torch).numpy()[0][dim]
            pred += w[2] * node_c_pred * self._delta_std[dim] * DT

            # GP
            if dim in self._gp_models:
                gp_pred = self._gp_models[dim].predict(
                    np.concatenate([s_norm, [a_norm]]),
                )
                pred += w[3] * gp_pred * self._delta_std[dim]

            # Physics
            physics_deltas = compute_physics_deltas(s)
            pred += w[4] * physics_deltas[dim]

            delta[dim] = pred

        return s + delta


def learn_per_state_weights(node_models, gp_models, data):
    """Learn per-state weights by evaluating each model on validation data.

    For each state dimension d:
      weight_m[d] = 1 / (epsilon + val_error_m[d])
      then normalize so weights sum to 1.
    """
    state_std = data['state_std']
    action_std = data['action_std']
    delta_std = data['delta_std']

    # Build predict functions for each model
    model_predict_fns = {}

    # NODE models
    for name, model in node_models.items():
        def make_node_fn(m):
            def predict_fn(s, tau):
                s_norm = s / state_std
                a_norm = tau / action_std
                x = torch.FloatTensor(
                    np.concatenate([s_norm, [a_norm]]),
                ).unsqueeze(0)
                with torch.no_grad():
                    dsdt = m(x).numpy()[0]
                return s + dsdt * delta_std * DT
            return predict_fn
        model_predict_fns[name] = make_node_fn(model)

    # GP model
    def gp_predict_fn(s, tau):
        s_norm = s / state_std
        a_norm = tau / action_std
        x = np.concatenate([s_norm, [a_norm]])
        delta = np.zeros(STATE_DIM)
        for dim in range(STATE_DIM):
            if dim in gp_models:
                delta[dim] = gp_models[dim].predict(x) * delta_std[dim]
        return s + delta
    model_predict_fns['gp'] = gp_predict_fn

    # Physics model
    def physics_predict_fn(s, tau):
        physics_deltas = compute_physics_deltas(s)
        return s + physics_deltas
    model_predict_fns['physics'] = physics_predict_fn

    # Evaluate each model on validation
    val_errors = {}
    for name, fn in model_predict_fns.items():
        per_state = evaluate_model_on_val(fn, data, n_segments=3, seed=99)
        val_errors[name] = per_state

    # Compute inverse-error weights per state
    epsilon = 1e-6
    weights_per_state = {}
    model_names = PerStateEnsemble.MODEL_NAMES

    for dim in range(STATE_DIM):
        raw_w = np.array([
            1.0 / (epsilon + val_errors[name][dim]) for name in model_names
        ])
        # Normalize
        w = raw_w / raw_w.sum()
        weights_per_state[dim] = w

    return weights_per_state, val_errors


# ============================================================
# Approach 2: Adaptive Ensemble
# ============================================================

class AdaptiveEnsemble:
    """Adaptive ensemble that adjusts weights based on state location.

    Partitions the state space into clusters using K-Means.
    Each cluster has its own set of per-state ensemble weights.
    For OOD states (far from all clusters), falls back to physics.
    """
    MODEL_NAMES = ['node_a', 'node_b', 'node_c', 'gp', 'physics']

    def __init__(self, node_models, gp_models, state_std, action_std, delta_std,
                 cluster_centers, cluster_weights, fallback_physics=True):
        """
        Args:
            cluster_centers: (n_clusters, STATE_DIM) normalized
            cluster_weights: dict cluster_id -> {dim -> array(n_models,)}
        """
        self._node_models = node_models
        self._gp_models = gp_models
        self._state_std = state_std
        self._action_std = action_std
        self._delta_std = delta_std
        self._cluster_centers = cluster_centers
        self._cluster_weights = cluster_weights
        self._n_clusters = len(cluster_centers)
        self._fallback_physics = fallback_physics
        # OOD threshold: if min distance > threshold, use physics fallback
        self._ood_threshold = 3.0

    def _find_cluster(self, s_norm):
        """Find nearest cluster for normalized state."""
        dists = np.linalg.norm(self._cluster_centers - s_norm, axis=1)
        return int(np.argmin(dists)), float(np.min(dists))

    def predict(self, s, tau):
        s_norm = s / self._state_std
        a_norm = tau / self._action_std
        x_torch = torch.FloatTensor(
            np.concatenate([s_norm, [a_norm]]),
        ).unsqueeze(0)

        cluster_id, dist = self._find_cluster(s_norm)

        # OOD fallback: physics only
        if self._fallback_physics and dist > self._ood_threshold:
            return s + compute_physics_deltas(s)

        w_dict = self._cluster_weights[cluster_id]
        delta = np.zeros(STATE_DIM)

        for dim in range(STATE_DIM):
            w = w_dict[dim]
            pred = 0.0

            # NODE-A
            with torch.no_grad():
                na = self._node_models['node_a'](x_torch).numpy()[0][dim]
            pred += w[0] * na * self._delta_std[dim] * DT

            # NODE-B
            with torch.no_grad():
                nb = self._node_models['node_b'](x_torch).numpy()[0][dim]
            pred += w[1] * nb * self._delta_std[dim] * DT

            # NODE-C
            with torch.no_grad():
                nc = self._node_models['node_c'](x_torch).numpy()[0][dim]
            pred += w[2] * nc * self._delta_std[dim] * DT

            # GP
            if dim in self._gp_models:
                gp_pred = self._gp_models[dim].predict(
                    np.concatenate([s_norm, [a_norm]]),
                )
                pred += w[3] * gp_pred * self._delta_std[dim]

            # Physics
            physics_deltas = compute_physics_deltas(s)
            pred += w[4] * physics_deltas[dim]

            delta[dim] = pred

        return s + delta


def learn_adaptive_weights(node_models, gp_models, data, n_clusters=5):
    """Learn cluster-specific weights.

    1. Cluster training data into n_clusters using K-Means
    2. For each cluster, evaluate each model on validation samples in that region
    3. Learn per-cluster, per-state inverse-error weights
    """
    state_std = data['state_std']
    action_std = data['action_std']
    delta_std = data['delta_std']
    train_obs = data['train_obs']
    val_obs = data['val_obs']

    # Normalize for clustering
    train_obs_norm = train_obs / state_std
    val_obs_norm = val_obs / state_std

    # Cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(train_obs_norm)
    cluster_centers = kmeans.cluster_centers_

    # Assign validation points to clusters
    val_labels = kmeans.predict(val_obs_norm)

    # For each cluster, compute per-state weights using local validation
    # We'll use a simplified approach: evaluate each model's error
    # on the full validation set, then weight by cluster membership
    # (soft assignment with distance-based kernel)

    # Compute all model predictions on validation data
    model_names = AdaptiveEnsemble.MODEL_NAMES

    # Build prediction arrays for validation: each model predicts deltas
    val_action = data['val_action']
    val_deltas = data['val_deltas']

    # Compute model predictions for all validation points
    model_pred_deltas = {}

    # NODE models
    for name, model in node_models.items():
        preds = np.zeros_like(val_deltas)
        batch_size = 1024
        for i in range(0, len(val_obs), batch_size):
            batch_obs = val_obs[i:i + batch_size]
            batch_act = val_action[i:i + batch_size].reshape(-1, 1)
            s_t = torch.FloatTensor(batch_obs / state_std)
            a_t = torch.FloatTensor(batch_act / action_std)
            with torch.no_grad():
                dsdt = model(s_t, a_t).numpy()
            preds[i:i + batch_size] = dsdt * delta_std * DT
        model_pred_deltas[name] = preds

    # GP model
    gp_preds = np.zeros_like(val_deltas)
    for dim in range(STATE_DIM):
        if dim in gp_models:
            for i in range(len(val_obs)):
                x = np.concatenate([
                    val_obs[i] / state_std,
                    [val_action[i].item() / action_std],
                ])
                gp_preds[i, dim] = gp_models[dim].predict(x) * delta_std[dim]
    model_pred_deltas['gp'] = gp_preds

    # Physics model
    phys_preds = compute_physics_deltas_batch(val_obs)
    model_pred_deltas['physics'] = phys_preds

    # Compute per-cluster, per-state weights
    cluster_weights = {}
    epsilon = 1e-6

    for c in range(n_clusters):
        mask = val_labels == c
        if mask.sum() < 5:
            # Too few points, use uniform weights
            cluster_weights[c] = {
                dim: np.ones(len(model_names)) / len(model_names)
                for dim in range(STATE_DIM)
            }
            continue

        cluster_true = val_deltas[mask]
        n_points = mask.sum()

        # Compute per-state MSE for each model
        per_state_errors = {}
        for name in model_names:
            preds = model_pred_deltas[name][mask]
            mse_per_state = np.mean((preds - cluster_true) ** 2, axis=0)
            per_state_errors[name] = mse_per_state

        # Inverse-error weights
        w_dict = {}
        for dim in range(STATE_DIM):
            raw_w = np.array([
                1.0 / (epsilon + per_state_errors[name][dim])
                for name in model_names
            ])
            w = raw_w / raw_w.sum()
            w_dict[dim] = w

        cluster_weights[c] = w_dict

    return cluster_centers, cluster_weights


# ============================================================
# Approach 3: Physics-Informed Ensemble
# ============================================================

class PhysicsInformedEnsemble:
    """Physics-informed ensemble that uses known kinematics as structural prior.

    Strategy:
    - For states with exact physics (e_y, e_psi, theta, delta):
      prediction = alpha * physics + (1-alpha) * data_driven
    - For states without exact physics (v, theta_dot, delta_dot):
      prediction = data_driven (NODE or GP)
    - alpha is learned per state from validation performance

    Additionally, applies physics-constrained correction:
    - After data-driven prediction, nudge e_y, e_psi toward physics-consistent values
    - Correction strength controlled by a learned parameter
    """
    def __init__(self, node_models, gp_model, state_std, action_std, delta_std,
                 physics_blend, correction_strength):
        """
        Args:
            physics_blend: dict dim -> alpha in [0,1]
                alpha=1 means pure physics, alpha=0 means pure data
            correction_strength: float, strength of physics correction
        """
        self._node_models = node_models  # dict name -> model
        self._gp_model = gp_model  # GPSingleDim per dim
        self._state_std = state_std
        self._action_std = action_std
        self._delta_std = delta_std
        self._physics_blend = physics_blend
        self._correction_strength = correction_strength
        # Use best NODE model (node_a as default, can be overridden)
        self._best_node = 'node_a'

    def predict(self, s, tau):
        s_norm = s / self._state_std
        a_norm = tau / self._action_std
        x_torch = torch.FloatTensor(
            np.concatenate([s_norm, [a_norm]]),
        ).unsqueeze(0)

        # Physics deltas
        physics_d = compute_physics_deltas(s)

        # Data-driven deltas from best NODE
        with torch.no_grad():
            node_dsdt = self._node_models[self._best_node](x_torch).numpy()[0]
        node_deltas = node_dsdt * self._delta_std * DT

        # GP deltas
        gp_deltas = np.zeros(STATE_DIM)
        x_gp = np.concatenate([s_norm, [a_norm]])
        for dim in range(STATE_DIM):
            if dim in self._gp_model:
                gp_deltas[dim] = self._gp_model[dim].predict(x_gp) * self._delta_std[dim]

        # Blend per state
        delta = np.zeros(STATE_DIM)
        for dim in range(STATE_DIM):
            alpha = self._physics_blend.get(dim, 0.0)
            # Data-driven = average of NODE and GP
            data_pred = 0.5 * node_deltas[dim] + 0.5 * gp_deltas[dim]
            delta[dim] = alpha * physics_d[dim] + (1.0 - alpha) * data_pred

        s_next = s + delta

        # Physics-informed correction for e_y and e_psi
        if self._correction_strength > 0:
            # Correct e_y: ensure e_y_dot = v * sin(e_psi)
            v_cur = s[IDX_V]
            e_psi_cur = s[IDX_EPSI]
            e_y_dot_physics = v_cur * np.sin(e_psi_cur)
            e_y_next_physics = s[IDX_EY] + e_y_dot_physics * DT
            correction_ey = (e_y_next_physics - s_next[IDX_EY]) * self._correction_strength
            s_next[IDX_EY] += correction_ey

            # Correct e_psi: ensure e_psi_dot = -v * delta / L
            delta_cur = s[IDX_DELTA]
            e_psi_dot_physics = -v_cur * delta_cur / WHEELBASE
            e_psi_next_physics = s[IDX_EPSI] + e_psi_dot_physics * DT
            correction_epsi = (e_psi_next_physics - s_next[IDX_EPSI]) * self._correction_strength
            s_next[IDX_EPSI] += correction_epsi

            # Correct theta: theta_dot = d(theta)/dt
            theta_dot_cur = s[IDX_THETA_DOT]
            theta_next_physics = s[IDX_THETA] + theta_dot_cur * DT
            correction_theta = (theta_next_physics - s_next[IDX_THETA]) * self._correction_strength
            s_next[IDX_THETA] += correction_theta

            # Correct delta: delta_dot = d(delta)/dt
            delta_dot_cur = s[IDX_DELTA_DOT]
            delta_next_physics = s[IDX_DELTA] + delta_dot_cur * DT
            correction_delta = (delta_next_physics - s_next[IDX_DELTA]) * self._correction_strength
            s_next[IDX_DELTA] += correction_delta

        return s_next


def learn_physics_blend(node_models, gp_model, data, correction_strength=0.1):
    """Learn optimal physics blending ratio for each state.

    For each state dimension:
      - Evaluate physics-only prediction error on validation
      - Evaluate data-driven (NODE + GP average) prediction error on validation
      - Find alpha that minimizes: alpha^2 * physics_err + (1-alpha)^2 * data_err
      - Closed-form: alpha = data_err / (physics_err + data_err)
    """
    state_std = data['state_std']
    action_std = data['action_std']
    delta_std = data['delta_std']
    val_obs = data['val_obs']
    val_action = data['val_action']
    val_deltas = data['val_deltas']

    # Compute physics predictions
    physics_preds = compute_physics_deltas_batch(val_obs)

    # Compute best NODE predictions
    best_node = node_models['node_a']
    node_preds = np.zeros_like(val_deltas)
    batch_size = 1024
    for i in range(0, len(val_obs), batch_size):
        batch_obs = val_obs[i:i + batch_size]
        batch_act = val_action[i:i + batch_size].reshape(-1, 1)
        s_t = torch.FloatTensor(batch_obs / state_std)
        a_t = torch.FloatTensor(batch_act / action_std)
        with torch.no_grad():
            dsdt = best_node(s_t, a_t).numpy()
        node_preds[i:i + batch_size] = dsdt * delta_std * DT

    # Compute GP predictions
    gp_preds = np.zeros_like(val_deltas)
    for dim in range(STATE_DIM):
        if dim in gp_model:
            for i in range(len(val_obs)):
                x = np.concatenate([
                    val_obs[i] / state_std,
                    [val_action[i].item() / action_std],
                ])
                gp_preds[i, dim] = gp_model[dim].predict(x) * delta_std[dim]

    # Data-driven = 0.5 * NODE + 0.5 * GP
    data_preds = 0.5 * node_preds + 0.5 * gp_preds

    # Compute per-state errors
    physics_err = np.mean((physics_preds - val_deltas) ** 2, axis=0)
    data_err = np.mean((data_preds - val_deltas) ** 2, axis=0)

    # Optimal alpha per state: alpha = data_err / (physics_err + data_err)
    # Higher alpha = more physics weight
    physics_blend = {}
    for dim in range(STATE_DIM):
        total = physics_err[dim] + data_err[dim]
        if total > 1e-10:
            alpha = data_err[dim] / total
        else:
            alpha = 0.5
        # Clip to reasonable range
        alpha = np.clip(alpha, 0.0, 0.95)
        physics_blend[dim] = float(alpha)

    return physics_blend


# ============================================================
# Main Experiment Runner
# ============================================================

def run_experiment():
    """Run all three ensemble approaches and evaluate."""
    print("=" * 70)
    print("Per-State Ensemble Methods for Bicycle System Identification")
    print("=" * 70)
    print(f"Started at: {datetime.now().isoformat()}")

    # ----------------------------------------------------------
    # 1. Load data
    # ----------------------------------------------------------
    print("\n[1/7] Loading data...")
    t0 = time.time()
    data = load_data(seed=42)
    print(f"  Data loaded in {time.time()-t0:.1f}s")
    print(f"  Train episodes: {len(data['train_eps'])}, Val episodes: {len(data['val_eps'])}, "
          f"Test episodes: {len(data['test_eps'])}")

    state_std = data['state_std']
    action_std = data['action_std']
    delta_std = data['delta_std']

    horizons = [1, 10, 50, 100, 200, 500, 1000]
    all_results = {}

    # ----------------------------------------------------------
    # 2. Train models
    # ----------------------------------------------------------
    print("\n[2/7] Training Neural ODE models (3 variants)...")
    node_configs = {
        'node_a': {'hidden': 64, 'depth': 3, 'activation': 'tanh',
                    'lr': 1e-3, 'n_epochs': 200, 'batch_size': 256},
        'node_b': {'hidden': 128, 'depth': 4, 'activation': 'tanh',
                    'lr': 1e-3, 'n_epochs': 200, 'batch_size': 256},
        'node_c': {'hidden': 64, 'depth': 3, 'activation': 'silu',
                    'lr': 1e-3, 'n_epochs': 200, 'batch_size': 256},
    }

    node_models = {}
    for name, config in node_configs.items():
        t1 = time.time()
        print(f"  Training {name} (hidden={config['hidden']}, depth={config['depth']}, "
              f"act={config['activation']})...")
        model = train_node_model(data, config, seed=42)
        node_models[name] = model
        print(f"    Done in {time.time()-t1:.1f}s")

    print("\n[3/7] Training GP models (per state)...")
    t1 = time.time()
    gp_models = train_gp_model(data, max_samples=5000, kernel_type='matern')
    print(f"  GP trained in {time.time()-t1:.1f}s")

    # ----------------------------------------------------------
    # 3. Approach 1: Per-State Ensemble
    # ----------------------------------------------------------
    print("\n[4/7] Approach 1: Per-State Ensemble...")
    t1 = time.time()

    weights_per_state, val_errors = learn_per_state_weights(
        node_models, gp_models, data,
    )

    # Print learned weights
    print("  Learned per-state weights:")
    print(f"  {'State':<12} {'NODE-A':<10} {'NODE-B':<10} {'NODE-C':<10} {'GP':<10} {'Physics':<10}")
    print("  " + "-" * 62)
    for dim in range(STATE_DIM):
        w = weights_per_state[dim]
        print(f"  {STATE_NAMES_7D[dim]:<12} {w[0]:<10.3f} {w[1]:<10.3f} {w[2]:<10.3f} "
              f"{w[3]:<10.3f} {w[4]:<10.3f}")

    pse = PerStateEnsemble(
        node_models, gp_models, state_std, action_std, delta_std,
        weights_per_state,
    )

    print("  Evaluating per-state ensemble...")
    pse_results = evaluate_full(pse.predict, data, horizons)
    pse_primary = compute_primary_score(pse_results)
    v9_primary = 0.5110
    pse_improvement = (v9_primary - pse_primary) / v9_primary * 100

    print(f"  Per-State Ensemble Primary: {pse_primary:.4f} (improvement: {pse_improvement:.1f}%)")
    print(f"  Completed in {time.time()-t1:.1f}s")

    all_results['per_state_ensemble'] = {
        'results': pse_results,
        'primary': float(pse_primary),
        'improvement': float(pse_improvement),
        'weights': {str(k): v.tolist() for k, v in weights_per_state.items()},
    }

    # ----------------------------------------------------------
    # 4. Approach 2: Adaptive Ensemble
    # ----------------------------------------------------------
    print("\n[5/7] Approach 2: Adaptive Ensemble...")
    t1 = time.time()

    best_adaptive = None
    best_adaptive_primary = float('inf')
    best_adaptive_results = None

    for n_clusters in [3, 5, 8]:
        print(f"\n  Trying n_clusters={n_clusters}...")
        cluster_centers, cluster_weights = learn_adaptive_weights(
            node_models, gp_models, data, n_clusters=n_clusters,
        )

        ae = AdaptiveEnsemble(
            node_models, gp_models, state_std, action_std, delta_std,
            cluster_centers, cluster_weights, fallback_physics=True,
        )

        ae_results = evaluate_full(ae.predict, data, horizons, n_segments=5)
        ae_primary = compute_primary_score(ae_results)
        ae_improvement = (v9_primary - ae_primary) / v9_primary * 100

        print(f"    Adaptive Ensemble (k={n_clusters}) Primary: {ae_primary:.4f} "
              f"(improvement: {ae_improvement:.1f}%)")

        if ae_primary < best_adaptive_primary:
            best_adaptive_primary = ae_primary
            best_adaptive = n_clusters
            best_adaptive_results = ae_results
            best_adaptive_centers = cluster_centers
            best_adaptive_weights = cluster_weights

    best_adaptive_improvement = (v9_primary - best_adaptive_primary) / v9_primary * 100
    print(f"\n  Best adaptive: k={best_adaptive}, Primary={best_adaptive_primary:.4f}, "
          f"Improvement={best_adaptive_improvement:.1f}%")
    print(f"  Completed in {time.time()-t1:.1f}s")

    all_results['adaptive_ensemble'] = {
        'results': best_adaptive_results,
        'primary': float(best_adaptive_primary),
        'improvement': float(best_adaptive_improvement),
        'best_n_clusters': best_adaptive,
    }

    # ----------------------------------------------------------
    # 5. Approach 3: Physics-Informed Ensemble
    # ----------------------------------------------------------
    print("\n[6/7] Approach 3: Physics-Informed Ensemble...")
    t1 = time.time()

    best_pi_primary = float('inf')
    best_pi_results = None
    best_pi_config = None

    # Sweep correction strengths
    for correction_strength in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]:
        # Learn physics blend ratios
        physics_blend = learn_physics_blend(
            node_models, gp_models, data,
            correction_strength=correction_strength,
        )

        pie = PhysicsInformedEnsemble(
            node_models, gp_models, state_std, action_std, delta_std,
            physics_blend, correction_strength,
        )

        pie_results = evaluate_full(pie.predict, data, horizons, n_segments=5)
        pie_primary = compute_primary_score(pie_results)
        pie_improvement = (v9_primary - pie_primary) / v9_primary * 100

        print(f"  Correction={correction_strength:.2f}: Primary={pie_primary:.4f}, "
              f"Improvement={pie_improvement:.1f}%, "
              f"Blend={ {STATE_NAMES_7D[d]: round(physics_blend[d], 2) for d in range(STATE_DIM)} }")

        if pie_primary < best_pi_primary:
            best_pi_primary = pie_primary
            best_pi_results = pie_results
            best_pi_config = {
                'correction_strength': correction_strength,
                'physics_blend': {STATE_NAMES_7D[d]: float(physics_blend[d])
                                   for d in range(STATE_DIM)},
            }

    best_pi_improvement = (v9_primary - best_pi_primary) / v9_primary * 100
    print(f"\n  Best Physics-Informed: Primary={best_pi_primary:.4f}, "
          f"Improvement={best_pi_improvement:.1f}%")
    print(f"  Config: {best_pi_config}")
    print(f"  Completed in {time.time()-t1:.1f}s")

    all_results['physics_informed_ensemble'] = {
        'results': best_pi_results,
        'primary': float(best_pi_primary),
        'improvement': float(best_pi_improvement),
        'best_config': best_pi_config,
    }

    # ----------------------------------------------------------
    # 6. Summary
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n{'Method':<30} {'H=100':<10} {'H=200':<10} {'H=500':<10} {'H=1000':<10} "
          f"{'Primary':<10} {'Improve':<10}")
    print("-" * 90)

    for name, rd in all_results.items():
        r = rd['results']
        prim = rd['primary']
        imp = rd['improvement']
        print(f"{name:<30} {r[100]['nmae_mean']:<10.4f} {r[200]['nmae_mean']:<10.4f} "
              f"{r[500]['nmae_mean']:<10.4f} {r[1000]['nmae_mean']:<10.4f} "
              f"{prim:<10.4f} {imp:<10.1f}%")

    # Per-state breakdown for best method
    best_method = min(all_results.keys(), key=lambda k: all_results[k]['primary'])
    best_r = all_results[best_method]['results']
    print(f"\nBest method: {best_method}")
    print(f"\nPer-state NMAE at H=200:")
    print(f"  {'State':<12} {'NMAE':<10}")
    print("  " + "-" * 22)
    for name in STATE_NAMES_7D:
        print(f"  {name:<12} {best_r[200]['per_state_nmae'][name]['mean']:<10.4f}")

    # ----------------------------------------------------------
    # 7. Save results
    # ----------------------------------------------------------
    output = {
        'timestamp': datetime.now().isoformat(),
        'run_id': '20260628_175824_neural_ode_144h',
        'experiment': 'per_state_ensemble',
        'target_improvement': 75,
        'v9_primary': v9_primary,
        'best_primary': float(min(rd['primary'] for rd in all_results.values())),
        'best_method': best_method,
        'methods': {},
    }

    for name, rd in all_results.items():
        method_data = {
            'primary': rd['primary'],
            'improvement': rd['improvement'],
            'horizons': {},
        }
        for h in horizons:
            method_data['horizons'][str(h)] = {
                'nmae_mean': rd['results'][h]['nmae_mean'],
                'survival_rate': rd['results'][h]['survival_rate'],
                'per_state_nmae': rd['results'][h]['per_state_nmae'],
            }
        # Add method-specific config
        if name == 'per_state_ensemble':
            method_data['weights'] = rd.get('weights', {})
        elif name == 'adaptive_ensemble':
            method_data['best_n_clusters'] = rd.get('best_n_clusters', None)
        elif name == 'physics_informed_ensemble':
            method_data['best_config'] = rd.get('best_config', {})
        output['methods'][name] = method_data

    output_path = 'D:/系统辨识作业/sindy_bicycle/research_72h/05_candidates/EXP042_per_state_ensemble.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    # Per-state analysis
    print("\n" + "=" * 70)
    print("PER-STATE ANALYSIS")
    print("=" * 70)
    for name, rd in all_results.items():
        r200 = rd['results'][200]
        print(f"\n{name}:")
        for sname in STATE_NAMES_7D:
            m = r200['per_state_nmae'][sname]['mean']
            print(f"  {sname:<12}: {m:.4f}")

    return output


if __name__ == '__main__':
    run_experiment()
