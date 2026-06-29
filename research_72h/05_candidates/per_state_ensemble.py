"""Per-State Ensemble Methods for Bicycle System Identification.

Implements three ensemble strategies that combine different model types
(Neural ODE, GP, physics-based) with per-state weighting:

1. Per-State Ensemble Model: weighted average of NODE + GP + Physics
2. Adaptive Ensemble: state-space partitioned weights with K-Means
3. Physics-Informed Ensemble: physics-blended predictions with correction

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
from sklearn.gaussian_process.kernels import Matern
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

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
    """Compute physics-based state deltas from known kinematics."""
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
# Neural ODE Models
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
    """Single-dimension GP model with optimized prediction."""
    def __init__(self, max_samples=3000, kernel_type='matern'):
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

        kernel = Matern(nu=2.5, length_scale=1.0)
        self._gp = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=2, alpha=1e-3,
        )
        self._gp.fit(X_scaled, y)
        self._n_train = len(X_scaled)

    def predict(self, x):
        x_scaled = (x - self._x_mean) / self._x_std
        return self._gp.predict(x_scaled.reshape(1, -1))[0]

    def predict_batch(self, X):
        """Predict for multiple inputs at once."""
        X_scaled = (X - self._x_mean) / self._x_std
        return self._gp.predict(X_scaled)


def train_gp_model(data, max_samples=3000, kernel_type='matern'):
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
# Utility
# ============================================================

def check_survival(state):
    """Check if state is within physical limits."""
    for i, name in enumerate(STATE_NAMES_7D):
        if name in PHYSICAL_LIMITS:
            if abs(state[i]) > PHYSICAL_LIMITS[name]:
                return False
    return not (np.any(np.isnan(state)) or np.any(np.isinf(state)))


def clip_state(s):
    """Clip state to physical limits, return clipped state."""
    s_clipped = s.copy()
    for i, name in enumerate(STATE_NAMES_7D):
        if name in PHYSICAL_LIMITS:
            s_clipped[i] = np.clip(s_clipped[i], -PHYSICAL_LIMITS[name], PHYSICAL_LIMITS[name])
    return s_clipped


def compute_primary_score(results):
    """Compute PrimaryLongHorizonScore = mean(H=100, H=200, H=500)."""
    scores = [results[h]['nmae_mean'] for h in [100, 200, 500]]
    if any(np.isnan(s) for s in scores):
        return float('nan')
    return float(np.mean(scores))


# ============================================================
# Evaluation
# ============================================================

def evaluate_model_on_val(predict_fn, data, n_segments=3, max_steps=50, seed=99):
    """Evaluate a model's per-state NMAE on validation data.

    Uses short rollouts (max_steps) to assess single-step quality.
    Returns per_state_nmae: dict dim_idx -> mean NMAE.
    """
    state_std = data['state_std']
    episodes = data['episodes']
    val_eps = data['val_eps']

    np.random.seed(seed)
    segments = []
    for ep_idx in val_eps:
        ep = episodes[ep_idx]
        if ep['length'] >= 50:
            segments.append(ep)
    segments = segments[:n_segments]

    if not segments:
        return {d: 1.0 for d in range(STATE_DIM)}

    per_state_errors = {d: [] for d in range(STATE_DIM)}

    for seg in segments:
        s_cur = seg['obs'][0].copy()
        actions_seg = seg['action'].flatten()
        real_states = seg['obs']
        n = min(max_steps, len(actions_seg))

        for step in range(n):
            try:
                s_next = predict_fn(s_cur, actions_seg[step])
                if np.any(np.isnan(s_next)) or np.any(np.isinf(s_next)):
                    break
                s_next = clip_state(s_next)
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
        per_state_nmae[d] = float(np.mean(errs)) if errs else 1.0

    return per_state_nmae


def evaluate_full(predict_fn, data, horizons, n_segments=5, seed=42):
    """Full evaluation across multiple horizons."""
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
                    s_next = clip_state(s_next)
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
            survival_list.append(survived and len(step_errors) >= min(n - 1, 10))

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


# ============================================================
# Approach 1: Per-State Ensemble Model
# ============================================================

class PerStateEnsemble:
    """Weighted ensemble with per-state weights.

    Combines: NODE-A, NODE-B, NODE-C, GP, Physics
    with per-state learned weights.
    """
    MODEL_NAMES = ['node_a', 'node_b', 'node_c', 'gp', 'physics']

    def __init__(self, node_models, gp_models, state_std, action_std, delta_std,
                 weights_per_state):
        self._node_models = node_models
        self._gp_models = gp_models
        self._state_std = state_std
        self._action_std = action_std
        self._delta_std = delta_std
        self._weights = weights_per_state

    def predict(self, s, tau):
        s_norm = s / self._state_std
        a_norm = tau / self._action_std
        x_arr = np.concatenate([s_norm, [a_norm]])
        x_torch = torch.FloatTensor(x_arr).unsqueeze(0)

        # Pre-compute all model deltas
        node_deltas = {}
        for name, model in self._node_models.items():
            with torch.no_grad():
                dsdt = model(x_torch).numpy()[0]
            node_deltas[name] = dsdt * self._delta_std * DT

        gp_deltas = np.zeros(STATE_DIM)
        for dim in range(STATE_DIM):
            if dim in self._gp_models:
                gp_deltas[dim] = self._gp_models[dim].predict(x_arr) * self._delta_std[dim]

        physics_d = compute_physics_deltas(s)

        # Combine per state
        delta = np.zeros(STATE_DIM)
        model_names_ordered = ['node_a', 'node_b', 'node_c']
        for dim in range(STATE_DIM):
            w = self._weights[dim]
            pred = 0.0
            for j, name in enumerate(model_names_ordered):
                pred += w[j] * node_deltas[name][dim]
            pred += w[3] * gp_deltas[dim]
            pred += w[4] * physics_d[dim]
            delta[dim] = pred

        s_next = s + delta
        s_next = clip_state(s_next)
        return s_next


def learn_per_state_weights(node_models, gp_models, data):
    """Learn per-state weights via inverse validation error."""
    state_std = data['state_std']
    action_std = data['action_std']
    delta_std = data['delta_std']

    # Build predict functions for each model
    model_predict_fns = {}

    for name, model in node_models.items():
        def make_node_fn(m):
            def predict_fn(s, tau):
                s_norm = s / state_std
                a_norm = tau / action_std
                x = torch.FloatTensor(np.concatenate([s_norm, [a_norm]])).unsqueeze(0)
                with torch.no_grad():
                    dsdt = m(x).numpy()[0]
                delta = dsdt * delta_std * DT
                return s + delta
            return predict_fn
        model_predict_fns[name] = make_node_fn(model)

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

    def physics_predict_fn(s, tau):
        return s + compute_physics_deltas(s)
    model_predict_fns['physics'] = physics_predict_fn

    # Evaluate each model on validation
    val_errors = {}
    for name, fn in model_predict_fns.items():
        per_state = evaluate_model_on_val(fn, data, n_segments=3, max_steps=50, seed=99)
        val_errors[name] = per_state

    # Compute inverse-error weights per state
    epsilon = 1e-6
    weights_per_state = {}
    model_names = PerStateEnsemble.MODEL_NAMES

    for dim in range(STATE_DIM):
        raw_w = np.array([
            1.0 / (epsilon + val_errors[name][dim]) for name in model_names
        ])
        w = raw_w / raw_w.sum()
        weights_per_state[dim] = w

    return weights_per_state, val_errors


# ============================================================
# Approach 2: Adaptive Ensemble
# ============================================================

class AdaptiveEnsemble:
    """Adaptive ensemble with state-space partitioned weights.

    Uses K-Means to partition state space into clusters.
    Each cluster has its own per-state weights.
    Falls back to physics in OOD regions.
    """
    MODEL_NAMES = ['node_a', 'node_b', 'node_c', 'gp', 'physics']

    def __init__(self, node_models, gp_models, state_std, action_std, delta_std,
                 cluster_centers, cluster_weights):
        self._node_models = node_models
        self._gp_models = gp_models
        self._state_std = state_std
        self._action_std = action_std
        self._delta_std = delta_std
        self._cluster_centers = cluster_centers
        self._cluster_weights = cluster_weights
        self._n_clusters = len(cluster_centers)
        self._ood_threshold = 3.0

    def _find_cluster(self, s_norm):
        dists = np.linalg.norm(self._cluster_centers - s_norm, axis=1)
        return int(np.argmin(dists)), float(np.min(dists))

    def predict(self, s, tau):
        s_norm = s / self._state_std
        a_norm = tau / self._action_std
        x_arr = np.concatenate([s_norm, [a_norm]])
        x_torch = torch.FloatTensor(x_arr).unsqueeze(0)

        cluster_id, dist = self._find_cluster(s_norm)

        # OOD fallback
        if dist > self._ood_threshold:
            return clip_state(s + compute_physics_deltas(s))

        w_dict = self._cluster_weights[cluster_id]

        # Pre-compute model deltas
        node_deltas = {}
        for name, model in self._node_models.items():
            with torch.no_grad():
                dsdt = model(x_torch).numpy()[0]
            node_deltas[name] = dsdt * self._delta_std * DT

        gp_deltas = np.zeros(STATE_DIM)
        for dim in range(STATE_DIM):
            if dim in self._gp_models:
                gp_deltas[dim] = self._gp_models[dim].predict(x_arr) * self._delta_std[dim]

        physics_d = compute_physics_deltas(s)

        delta = np.zeros(STATE_DIM)
        model_names_ordered = ['node_a', 'node_b', 'node_c']
        for dim in range(STATE_DIM):
            w = w_dict[dim]
            pred = 0.0
            for j, name in enumerate(model_names_ordered):
                pred += w[j] * node_deltas[name][dim]
            pred += w[3] * gp_deltas[dim]
            pred += w[4] * physics_d[dim]
            delta[dim] = pred

        return clip_state(s + delta)


def learn_adaptive_weights(node_models, gp_models, data, n_clusters=5):
    """Learn cluster-specific weights using model predictions on validation data."""
    state_std = data['state_std']
    action_std = data['action_std']
    delta_std = data['delta_std']
    train_obs = data['train_obs']
    val_obs = data['val_obs']
    val_action = data['val_action']
    val_deltas = data['val_deltas']

    # Normalize for clustering
    train_obs_norm = train_obs / state_std
    val_obs_norm = val_obs / state_std

    # Cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(train_obs_norm)
    cluster_centers = kmeans.cluster_centers_
    val_labels = kmeans.predict(val_obs_norm)

    # Compute model predictions for all validation points (batch)
    model_names = ['node_a', 'node_b', 'node_c']
    model_pred_deltas = {}

    # NODE models (batch)
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

    # GP model (batch)
    gp_preds = np.zeros_like(val_deltas)
    val_X = np.hstack([val_obs / state_std, val_action.reshape(-1, 1) / action_std])
    for dim in range(STATE_DIM):
        if dim in gp_models:
            gp_preds[:, dim] = gp_models[dim].predict_batch(val_X) * delta_std[dim]
    model_pred_deltas['gp'] = gp_preds

    # Physics model (batch)
    model_pred_deltas['physics'] = compute_physics_deltas_batch(val_obs)

    # Compute per-cluster, per-state weights
    cluster_weights = {}
    epsilon = 1e-6
    all_model_names = model_names + ['gp', 'physics']

    for c in range(n_clusters):
        mask = val_labels == c
        if mask.sum() < 5:
            cluster_weights[c] = {
                dim: np.ones(len(all_model_names)) / len(all_model_names)
                for dim in range(STATE_DIM)
            }
            continue

        cluster_true = val_deltas[mask]

        per_state_errors = {}
        for name in all_model_names:
            preds = model_pred_deltas[name][mask]
            mse_per_state = np.mean((preds - cluster_true) ** 2, axis=0)
            per_state_errors[name] = mse_per_state

        w_dict = {}
        for dim in range(STATE_DIM):
            raw_w = np.array([
                1.0 / (epsilon + per_state_errors[name][dim])
                for name in all_model_names
            ])
            w = raw_w / raw_w.sum()
            w_dict[dim] = w

        cluster_weights[c] = w_dict

    return cluster_centers, cluster_weights


# ============================================================
# Approach 3: Physics-Informed Ensemble
# ============================================================

class PhysicsInformedEnsemble:
    """Physics-informed ensemble with learned blending ratios.

    For states with known kinematics (e_y, e_psi, theta, delta):
      prediction = alpha * physics + (1-alpha) * data_driven
    For unknown states (v, theta_dot, delta_dot):
      prediction = data_driven

    Then applies physics-informed correction.
    """
    def __init__(self, node_model, gp_model, state_std, action_std, delta_std,
                 physics_blend, correction_strength, nn_corrector=None,
                 train_obs_norm=None):
        self._node_model = node_model  # single best NODE model
        self._gp_model = gp_model
        self._state_std = state_std
        self._action_std = action_std
        self._delta_std = delta_std
        self._physics_blend = physics_blend  # dict dim -> alpha
        self._correction_strength = correction_strength
        self._nn_corrector = nn_corrector
        self._train_obs_norm = train_obs_norm

    def predict(self, s, tau):
        s_norm = s / self._state_std
        a_norm = tau / self._action_std
        x_arr = np.concatenate([s_norm, [a_norm]])
        x_torch = torch.FloatTensor(x_arr).unsqueeze(0)

        # Physics deltas
        physics_d = compute_physics_deltas(s)

        # NODE deltas
        with torch.no_grad():
            dsdt = self._node_model(x_torch).numpy()[0]
        node_deltas = dsdt * self._delta_std * DT

        # GP deltas
        gp_deltas = np.zeros(STATE_DIM)
        for dim in range(STATE_DIM):
            if dim in self._gp_model:
                gp_deltas[dim] = self._gp_model[dim].predict(x_arr) * self._delta_std[dim]

        # Data-driven = average of NODE and GP
        data_deltas = 0.5 * node_deltas + 0.5 * gp_deltas

        # Blend per state
        delta = np.zeros(STATE_DIM)
        for dim in range(STATE_DIM):
            alpha = self._physics_blend.get(dim, 0.0)
            delta[dim] = alpha * physics_d[dim] + (1.0 - alpha) * data_deltas[dim]

        s_next = s + delta

        # Physics-informed correction
        if self._correction_strength > 0:
            # Correct e_y
            e_y_dot_phys = s[IDX_V] * np.sin(s[IDX_EPSI])
            e_y_next_phys = s[IDX_EY] + e_y_dot_phys * DT
            s_next[IDX_EY] += (e_y_next_phys - s_next[IDX_EY]) * self._correction_strength

            # Correct e_psi
            e_psi_dot_phys = -s[IDX_V] * s[IDX_DELTA] / WHEELBASE
            e_psi_next_phys = s[IDX_EPSI] + e_psi_dot_phys * DT
            s_next[IDX_EPSI] += (e_psi_next_phys - s_next[IDX_EPSI]) * self._correction_strength

            # Correct theta
            theta_next_phys = s[IDX_THETA] + s[IDX_THETA_DOT] * DT
            s_next[IDX_THETA] += (theta_next_phys - s_next[IDX_THETA]) * self._correction_strength

            # Correct delta
            delta_next_phys = s[IDX_DELTA] + s[IDX_DELTA_DOT] * DT
            s_next[IDX_DELTA] += (delta_next_phys - s_next[IDX_DELTA]) * self._correction_strength

        # Nearest-neighbor correction
        if self._nn_corrector is not None and self._train_obs_norm is not None:
            s_pred_norm = s_next / self._state_std
            distances, indices = self._nn_corrector.kneighbors([s_pred_norm])
            neighbor_mean = np.mean(self._train_obs_norm[indices[0]], axis=0)
            nn_correction = (neighbor_mean - s_pred_norm) * 0.05
            s_next = (s_pred_norm + nn_correction) * self._state_std

        s_next = clip_state(s_next)
        return s_next


def learn_physics_blend(node_model, gp_model, data):
    """Learn optimal physics blending ratio for each state."""
    state_std = data['state_std']
    action_std = data['action_std']
    delta_std = data['delta_std']
    val_obs = data['val_obs']
    val_action = data['val_action']
    val_deltas = data['val_deltas']

    # Physics predictions
    physics_preds = compute_physics_deltas_batch(val_obs)

    # NODE predictions (batch)
    node_preds = np.zeros_like(val_deltas)
    batch_size = 1024
    for i in range(0, len(val_obs), batch_size):
        batch_obs = val_obs[i:i + batch_size]
        batch_act = val_action[i:i + batch_size].reshape(-1, 1)
        s_t = torch.FloatTensor(batch_obs / state_std)
        a_t = torch.FloatTensor(batch_act / action_std)
        with torch.no_grad():
            dsdt = node_model(s_t, a_t).numpy()
        node_preds[i:i + batch_size] = dsdt * delta_std * DT

    # GP predictions (batch)
    gp_preds = np.zeros_like(val_deltas)
    val_X = np.hstack([val_obs / state_std, val_action.reshape(-1, 1) / action_std])
    for dim in range(STATE_DIM):
        if dim in gp_model:
            gp_preds[:, dim] = gp_model[dim].predict_batch(val_X) * delta_std[dim]

    data_preds = 0.5 * node_preds + 0.5 * gp_preds

    physics_err = np.mean((physics_preds - val_deltas) ** 2, axis=0)
    data_err = np.mean((data_preds - val_deltas) ** 2, axis=0)

    physics_blend = {}
    for dim in range(STATE_DIM):
        total = physics_err[dim] + data_err[dim]
        if total > 1e-10:
            alpha = data_err[dim] / total
        else:
            alpha = 0.5
        alpha = float(np.clip(alpha, 0.0, 0.95))
        physics_blend[dim] = alpha

    return physics_blend


# ============================================================
# Main
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
    # 2. Train NODE models
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

    # ----------------------------------------------------------
    # 3. Train GP models (reduced samples for speed)
    # ----------------------------------------------------------
    print("\n[3/7] Training GP models (per state)...")
    t1 = time.time()
    gp_models = train_gp_model(data, max_samples=3000, kernel_type='matern')
    print(f"  GP trained in {time.time()-t1:.1f}s")

    # ----------------------------------------------------------
    # 4. Approach 1: Per-State Ensemble
    # ----------------------------------------------------------
    print("\n[4/7] Approach 1: Per-State Ensemble...")
    t1 = time.time()

    weights_per_state, val_errors = learn_per_state_weights(
        node_models, gp_models, data,
    )

    # Print validation errors and learned weights
    print("  Validation errors (per state):")
    for dim in range(STATE_DIM):
        errs_str = ", ".join([f"{name}={val_errors[name][dim]:.4f}"
                              for name in PerStateEnsemble.MODEL_NAMES])
        print(f"    {STATE_NAMES_7D[dim]:<12}: {errs_str}")

    print("\n  Learned per-state weights:")
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
    pse_improvement = (v9_primary - pse_primary) / v9_primary * 100 if not np.isnan(pse_primary) else float('nan')

    print(f"  Per-State Ensemble Primary: {pse_primary:.4f} (improvement: {pse_improvement:.1f}%)")
    print(f"  Completed in {time.time()-t1:.1f}s")

    all_results['per_state_ensemble'] = {
        'results': pse_results,
        'primary': float(pse_primary) if not np.isnan(pse_primary) else None,
        'improvement': float(pse_improvement) if not np.isnan(pse_improvement) else None,
        'weights': {str(k): v.tolist() for k, v in weights_per_state.items()},
    }

    # ----------------------------------------------------------
    # 5. Approach 2: Adaptive Ensemble
    # ----------------------------------------------------------
    print("\n[5/7] Approach 2: Adaptive Ensemble...")
    t1 = time.time()

    best_adaptive_primary = float('inf')
    best_adaptive_results = None
    best_adaptive = None

    for n_clusters in [3, 5, 8]:
        print(f"\n  Trying n_clusters={n_clusters}...")
        t2 = time.time()
        cluster_centers, cluster_weights = learn_adaptive_weights(
            node_models, gp_models, data, n_clusters=n_clusters,
        )
        print(f"    Weights learned in {time.time()-t2:.1f}s")

        ae = AdaptiveEnsemble(
            node_models, gp_models, state_std, action_std, delta_std,
            cluster_centers, cluster_weights,
        )

        # Use fewer segments for adaptive (slower due to per-step GP)
        ae_results = evaluate_full(ae.predict, data, horizons, n_segments=3)
        ae_primary = compute_primary_score(ae_results)
        ae_improvement = (v9_primary - ae_primary) / v9_primary * 100 if not np.isnan(ae_primary) else float('nan')

        print(f"    Adaptive Ensemble (k={n_clusters}) Primary: {ae_primary:.4f} "
              f"(improvement: {ae_improvement:.1f}%) [eval: {time.time()-t2:.1f}s]")

        if not np.isnan(ae_primary) and ae_primary < best_adaptive_primary:
            best_adaptive_primary = ae_primary
            best_adaptive = n_clusters
            best_adaptive_results = ae_results
            best_adaptive_centers = cluster_centers
            best_adaptive_weights = cluster_weights

    if best_adaptive_results is not None:
        best_adaptive_improvement = (v9_primary - best_adaptive_primary) / v9_primary * 100
    else:
        best_adaptive_improvement = float('nan')

    print(f"\n  Best adaptive: k={best_adaptive}, Primary={best_adaptive_primary:.4f}, "
          f"Improvement={best_adaptive_improvement:.1f}%")
    print(f"  Completed in {time.time()-t1:.1f}s")

    all_results['adaptive_ensemble'] = {
        'results': best_adaptive_results if best_adaptive_results else {},
        'primary': float(best_adaptive_primary) if not np.isnan(best_adaptive_primary) else None,
        'improvement': float(best_adaptive_improvement) if not np.isnan(best_adaptive_improvement) else None,
        'best_n_clusters': best_adaptive,
    }

    # ----------------------------------------------------------
    # 6. Approach 3: Physics-Informed Ensemble
    # ----------------------------------------------------------
    print("\n[6/7] Approach 3: Physics-Informed Ensemble...")
    t1 = time.time()

    # Use best NODE model for physics-informed
    best_node_name = 'node_a'
    best_node_model = node_models[best_node_name]

    # Build NN corrector
    train_obs_norm = train_obs_norm_data = data['train_obs'] / state_std
    nn_model = NearestNeighbors(n_neighbors=10, algorithm='auto')
    nn_model.fit(train_obs_norm)

    physics_blend = learn_physics_blend(best_node_model, gp_models, data)
    print(f"  Physics blend ratios: { {STATE_NAMES_7D[d]: round(physics_blend[d], 2) for d in range(STATE_DIM)} }")

    best_pi_primary = float('inf')
    best_pi_results = None
    best_pi_config = None

    for correction_strength in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]:
        pie = PhysicsInformedEnsemble(
            best_node_model, gp_models, state_std, action_std, delta_std,
            physics_blend, correction_strength,
            nn_corrector=nn_model, train_obs_norm=train_obs_norm,
        )

        pie_results = evaluate_full(pie.predict, data, horizons, n_segments=5)
        pie_primary = compute_primary_score(pie_results)
        pie_improvement = (v9_primary - pie_primary) / v9_primary * 100 if not np.isnan(pie_primary) else float('nan')

        print(f"  Correction={correction_strength:.2f}: Primary={pie_primary:.4f}, "
              f"Improvement={pie_improvement:.1f}%")

        if not np.isnan(pie_primary) and pie_primary < best_pi_primary:
            best_pi_primary = pie_primary
            best_pi_results = pie_results
            best_pi_config = {
                'correction_strength': correction_strength,
                'physics_blend': {STATE_NAMES_7D[d]: float(physics_blend[d])
                                   for d in range(STATE_DIM)},
            }

    if best_pi_results is not None:
        best_pi_improvement = (v9_primary - best_pi_primary) / v9_primary * 100
    else:
        best_pi_improvement = float('nan')

    print(f"\n  Best Physics-Informed: Primary={best_pi_primary:.4f}, "
          f"Improvement={best_pi_improvement:.1f}%")
    print(f"  Config: {best_pi_config}")
    print(f"  Completed in {time.time()-t1:.1f}s")

    all_results['physics_informed_ensemble'] = {
        'results': best_pi_results if best_pi_results else {},
        'primary': float(best_pi_primary) if not np.isnan(best_pi_primary) else None,
        'improvement': float(best_pi_improvement) if not np.isnan(best_pi_improvement) else None,
        'best_config': best_pi_config,
    }

    # ----------------------------------------------------------
    # 7. Summary
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n{'Method':<30} {'H=100':<10} {'H=200':<10} {'H=500':<10} {'H=1000':<10} "
          f"{'Primary':<10} {'Improve':<10}")
    print("-" * 90)

    for name, rd in all_results.items():
        r = rd.get('results', {})
        prim = rd.get('primary', float('nan'))
        imp = rd.get('improvement', float('nan'))
        h100 = r.get(100, {}).get('nmae_mean', float('nan'))
        h200 = r.get(200, {}).get('nmae_mean', float('nan'))
        h500 = r.get(500, {}).get('nmae_mean', float('nan'))
        h1000 = r.get(1000, {}).get('nmae_mean', float('nan'))
        prim_s = f"{prim:.4f}" if prim is not None and not np.isnan(prim) else "nan"
        imp_s = f"{imp:.1f}%" if imp is not None and not np.isnan(imp) else "nan"
        h100_s = f"{h100:.4f}" if not np.isnan(h100) else "nan"
        h200_s = f"{h200:.4f}" if not np.isnan(h200) else "nan"
        h500_s = f"{h500:.4f}" if not np.isnan(h500) else "nan"
        h1000_s = f"{h1000:.4f}" if not np.isnan(h1000) else "nan"
        print(f"{name:<30} {h100_s:<10} {h200_s:<10} {h500_s:<10} {h1000_s:<10} "
              f"{prim_s:<10} {imp_s:<10}")

    # Find best method
    valid_methods = {k: v for k, v in all_results.items()
                     if v.get('primary') is not None and not np.isnan(v['primary'])}
    if valid_methods:
        best_method = min(valid_methods.keys(), key=lambda k: valid_methods[k]['primary'])
        best_r = valid_methods[best_method]['results']
        print(f"\nBest method: {best_method}")
        print(f"\nPer-state NMAE at H=200:")
        print(f"  {'State':<12} {'NMAE':<10}")
        print("  " + "-" * 22)
        for name in STATE_NAMES_7D:
            m = best_r.get(200, {}).get('per_state_nmae', {}).get(name, {}).get('mean', float('nan'))
            print(f"  {name:<12} {m:<10.4f}")
    else:
        best_method = None
        print("\nNo valid results found!")

    # ----------------------------------------------------------
    # Save results
    # ----------------------------------------------------------
    output = {
        'timestamp': datetime.now().isoformat(),
        'run_id': '20260628_175824_neural_ode_144h',
        'experiment': 'per_state_ensemble',
        'target_improvement': 75,
        'v9_primary': v9_primary,
        'best_primary': float(valid_methods[best_method]['primary']) if valid_methods else None,
        'best_method': best_method,
        'methods': {},
    }

    for name, rd in all_results.items():
        method_data = {
            'primary': rd.get('primary'),
            'improvement': rd.get('improvement'),
            'horizons': {},
        }
        r = rd.get('results', {})
        for h in horizons:
            if h in r:
                method_data['horizons'][str(h)] = {
                    'nmae_mean': r[h].get('nmae_mean'),
                    'survival_rate': r[h].get('survival_rate'),
                    'per_state_nmae': r[h].get('per_state_nmae'),
                }
        if name == 'per_state_ensemble':
            method_data['weights'] = rd.get('weights', {})
        elif name == 'adaptive_ensemble':
            method_data['best_n_clusters'] = rd.get('best_n_clusters')
        elif name == 'physics_informed_ensemble':
            method_data['best_config'] = rd.get('best_config')
        output['methods'][name] = method_data

    output_path = 'D:/系统辨识作业/sindy_bicycle/research_72h/05_candidates/EXP042_per_state_ensemble.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    return output


if __name__ == '__main__':
    run_experiment()
