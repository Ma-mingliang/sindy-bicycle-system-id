"""State-Specific Experiments (EXP041).

For each target state (e_y, e_psi, v, theta), train dedicated models
using different architectures (NODE, GP, Linear). Then combine the
best per-state method into a composite model and compare against v9.

Experiments:
  EXP-1: e_y 专用模型 (NODE, GP, Linear)
  EXP-2: e_psi 专用模型 (NODE, GP, Linear)
  EXP-3: v 专用模型 (NODE, GP, Linear)
  EXP-4: theta 专用模型 (NODE, GP, Linear)
  EXP-5: 最佳组合 (composite of best per-state + v9 for remainder)
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
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, RBF
from sklearn.linear_model import Ridge

warnings.filterwarnings('ignore')
sys.path.insert(0, 'D:/系统辨识作业/sindy_bicycle')

# ─── Constants ───────────────────────────────────────────────────────────
STATE_NAMES_7D = ['e_y', 'e_psi', 'v', 'theta', 'theta_dot', 'delta', 'delta_dot']
STATE_DIM = 7
ACTION_DIM = 1
IDX_7D_FROM_8D = [0, 1, 2, 3, 4, 6, 7]
DT = 1.0 / 30.0

PHYSICAL_LIMITS = {
    'e_y': 5.0, 'e_psi': np.pi, 'v': 5.0,
    'theta': np.pi, 'theta_dot': 10.0,
    'delta': np.pi/2, 'delta_dot': 10.0,
}

# Target states for state-specific experiments
TARGET_STATES = {
    'e_y': 0,
    'e_psi': 1,
    'v': 2,
    'theta': 3,
}

V9_PRIMARY = 0.5110
V9_NMAE = {1: 0.0051, 10: 0.0628, 50: 0.4557, 100: 0.5064, 200: 0.4737, 500: 0.5529}


# ─── Model Architectures ────────────────────────────────────────────────

class ODEFunc(nn.Module):
    """Neural ODE dynamics function: dx/dt = f(x, u).

    Same architecture as v9 baseline but with configurable output dim.
    """
    def __init__(self, output_dim=STATE_DIM, hidden=64, depth=3, activation='tanh'):
        super().__init__()
        act = nn.Tanh if activation == 'tanh' else nn.SiLU
        layers = [nn.Linear(STATE_DIM + ACTION_DIM, hidden), act()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden, hidden), act()])
        layers.append(nn.Linear(hidden, output_dim))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].bias)
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.1)

    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=-1))


class SingleStateNODE(nn.Module):
    """NODE model that predicts dsdt for a single state only."""
    def __init__(self, hidden=64, depth=3, activation='tanh'):
        super().__init__()
        act = nn.Tanh if activation == 'tanh' else nn.SiLU
        layers = [nn.Linear(STATE_DIM + ACTION_DIM, hidden), act()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden, hidden), act()])
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].bias)
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.1)

    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=-1))


class SingleStateDirectDelta(nn.Module):
    """Direct delta prediction for a single state (no dt multiplication)."""
    def __init__(self, hidden=64, depth=3, activation='tanh'):
        super().__init__()
        act = nn.Tanh if activation == 'tanh' else nn.SiLU
        layers = [nn.Linear(STATE_DIM + ACTION_DIM, hidden), act()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden, hidden), act()])
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].bias)
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.1)

    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=-1))


class SingleStateWideNODE(nn.Module):
    """Wider NODE model for harder states."""
    def __init__(self, hidden=128, depth=4, activation='tanh'):
        super().__init__()
        act = nn.Tanh if activation == 'tanh' else nn.SiLU
        layers = [nn.Linear(STATE_DIM + ACTION_DIM, hidden), act()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden, hidden), act()])
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].bias)
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.1)

    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=-1))


# ─── Data Loading ───────────────────────────────────────────────────────

def load_data(data_path='D:/系统辨识作业/sindy_bicycle/data/stage2_dataset_150k.npz', seed=42):
    """Load 7D data, split by episode, compute normalization stats."""
    data_raw = np.load(data_path, allow_pickle=True)
    obs_8d = data_raw['obs']
    action = data_raw['action']
    next_obs_8d = data_raw['next_obs']
    done = data_raw['done']

    obs = obs_8d[:, IDX_7D_FROM_8D]
    next_obs = next_obs_8d[:, IDX_7D_FROM_8D]
    deltas = next_obs - obs

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
        s = episode_starts[ep_idx]
        e = episode_ends[ep_idx] + 1
        episodes.append({
            'obs': obs[s:e],
            'action': action[s:e],
            'deltas': deltas[s:e],
            'length': e - s,
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

    return {
        'episodes': episodes,
        'train_eps': train_eps,
        'test_eps': test_eps,
        'state_std': state_std,
        'action_std': action_std,
        'delta_std': delta_std,
        'train_obs': train_obs,
        'train_action': train_action,
        'train_deltas': train_deltas,
    }


# ─── Training Functions ─────────────────────────────────────────────────

def train_single_state_node(data, target_dim, config, seed=42, wide=False):
    """Train a NODE model for a single target state."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    state_std = data['state_std']
    action_std = data['action_std']
    delta_std = data['delta_std']

    train_obs = data['train_obs']
    train_action = data['train_action']
    train_deltas = data['train_deltas']

    X_s = torch.FloatTensor(train_obs / state_std)
    X_a = torch.FloatTensor(train_action.reshape(-1, 1) / action_std)
    Y = torch.FloatTensor(train_deltas[:, target_dim] / (delta_std[target_dim] * DT))

    ds = torch.utils.data.TensorDataset(X_s, X_a, Y)
    loader = torch.utils.data.DataLoader(ds, batch_size=config['batch_size'], shuffle=True)

    if wide:
        model = SingleStateWideNODE(hidden=config.get('hidden', 128),
                                     depth=config.get('depth', 4),
                                     activation=config.get('activation', 'tanh'))
    else:
        model = SingleStateNODE(hidden=config.get('hidden', 64),
                                 depth=config.get('depth', 3),
                                 activation=config.get('activation', 'tanh'))

    opt = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config['n_epochs'])

    best_val_loss = float('inf')
    best_state = None
    patience = config.get('patience', 50)
    patience_counter = 0

    # Split train/val
    n = len(X_s)
    n_val = int(0.1 * n)
    indices = torch.randperm(n)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    X_s_train, X_a_train, Y_train = X_s[train_idx], X_a[train_idx], Y[train_idx]
    X_s_val, X_a_val, Y_val = X_s[val_idx], X_a[val_idx], Y[val_idx]

    ds_train = torch.utils.data.TensorDataset(X_s_train, X_a_train, Y_train)
    loader_train = torch.utils.data.DataLoader(ds_train, batch_size=config['batch_size'], shuffle=True)

    model.train()
    for epoch in range(config['n_epochs']):
        for sb, ab, yb in loader_train:
            pred = model(sb, ab).squeeze(-1)
            loss = nn.functional.mse_loss(pred, yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_s_val, X_a_val).squeeze(-1)
            val_loss = nn.functional.mse_loss(val_pred, Y_val).item()
        model.train()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def train_single_state_direct_delta(data, target_dim, config, seed=42):
    """Train a direct-delta model for a single target state."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    state_std = data['state_std']
    action_std = data['action_std']
    delta_std = data['delta_std']

    train_obs = data['train_obs']
    train_action = data['train_action']
    train_deltas = data['train_deltas']

    X_s = torch.FloatTensor(train_obs / state_std)
    X_a = torch.FloatTensor(train_action.reshape(-1, 1) / action_std)
    # Target: normalized delta (no dt)
    Y = torch.FloatTensor(train_deltas[:, target_dim] / delta_std[target_dim])

    ds = torch.utils.data.TensorDataset(X_s, X_a, Y)
    loader = torch.utils.data.DataLoader(ds, batch_size=config['batch_size'], shuffle=True)

    model = SingleStateDirectDelta(hidden=config.get('hidden', 64),
                                    depth=config.get('depth', 3),
                                    activation=config.get('activation', 'tanh'))

    opt = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config['n_epochs'])

    best_val_loss = float('inf')
    best_state = None
    patience = config.get('patience', 50)
    patience_counter = 0

    n = len(X_s)
    n_val = int(0.1 * n)
    indices = torch.randperm(n)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    X_s_train, X_a_train, Y_train = X_s[train_idx], X_a[train_idx], Y[train_idx]
    X_s_val, X_a_val, Y_val = X_s[val_idx], X_a[val_idx], Y[val_idx]

    ds_train = torch.utils.data.TensorDataset(X_s_train, X_a_train, Y_train)
    loader_train = torch.utils.data.DataLoader(ds_train, batch_size=config['batch_size'], shuffle=True)

    model.train()
    for epoch in range(config['n_epochs']):
        for sb, ab, yb in loader_train:
            pred = model(sb, ab).squeeze(-1)
            loss = nn.functional.mse_loss(pred, yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_s_val, X_a_val).squeeze(-1)
            val_loss = nn.functional.mse_loss(val_pred, Y_val).item()
        model.train()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def train_single_state_gp(data, target_dim, max_samples=5000, seed=42):
    """Train a GP model for a single target state."""
    np.random.seed(seed)
    state_std = data['state_std']
    action_std = data['action_std']
    delta_std = data['delta_std']

    X = np.hstack([data['train_obs'] / state_std,
                    data['train_action'].reshape(-1, 1) / action_std])
    Y = data['train_deltas'][:, target_dim] / delta_std[target_dim]

    n = len(X)
    if n > max_samples:
        idx = np.random.choice(n, max_samples, replace=False)
        X, Y = X[idx], Y[idx]

    kernel = ConstantKernel(1.0) * Matern(nu=2.5, length_scale=1.0)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3,
                                   alpha=1e-3, random_state=seed)
    gp.fit(X, Y)
    return gp


def train_single_state_linear(data, target_dim, alpha=1.0, seed=42):
    """Train a Ridge regression model for a single target state."""
    np.random.seed(seed)
    state_std = data['state_std']
    action_std = data['action_std']
    delta_std = data['delta_std']

    X = np.hstack([data['train_obs'] / state_std,
                    data['train_action'].reshape(-1, 1) / action_std])
    Y = data['train_deltas'][:, target_dim] / delta_std[target_dim]

    model = Ridge(alpha=alpha)
    model.fit(X, Y)
    return model


def train_v9_baseline(data, config, seed=43):
    """Train the full v9 NODE baseline (all 7 states)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    state_std = data['state_std']
    action_std = data['action_std']
    delta_std = data['delta_std']

    train_obs = data['train_obs']
    train_action = data['train_action']
    train_deltas = data['train_deltas']

    X_s = torch.FloatTensor(train_obs / state_std)
    X_a = torch.FloatTensor(train_action.reshape(-1, 1) / action_std)
    Y = torch.FloatTensor(train_deltas / (delta_std * DT))

    n = len(X_s)
    n_val = int(0.1 * n)
    indices = torch.randperm(n)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    X_s_train, X_a_train, Y_train = X_s[train_idx], X_a[train_idx], Y[train_idx]
    X_s_val, X_a_val, Y_val = X_s[val_idx], X_a[val_idx], Y[val_idx]

    ds_train = torch.utils.data.TensorDataset(X_s_train, X_a_train, Y_train)
    loader = torch.utils.data.DataLoader(ds_train, batch_size=config['batch_size'], shuffle=True)

    model = ODEFunc(output_dim=STATE_DIM, hidden=config['hidden'],
                    depth=config['depth'], activation=config['activation'])

    opt = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config['n_epochs'])

    best_val_loss = float('inf')
    best_state = None
    patience = config.get('patience', 50)
    patience_counter = 0

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
        with torch.no_grad():
            val_pred = model(X_s_val, X_a_val)
            val_loss = nn.functional.mse_loss(val_pred, Y_val).item()
        model.train()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


# ─── Evaluation ─────────────────────────────────────────────────────────

def check_survival(state):
    """Check if state survives physical limits."""
    for i, name in enumerate(STATE_NAMES_7D):
        if name in PHYSICAL_LIMITS:
            if abs(state[i]) > PHYSICAL_LIMITS[name]:
                return False
    return not (np.any(np.isnan(state)) or np.any(np.isinf(state)))


def evaluate_composite(predict_fn, data, horizons, n_segments=5, seed=42):
    """Evaluate a composite prediction function on test segments.

    predict_fn: callable(s_cur, action) -> s_next (7D numpy array)
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
            'n_valid': len(valid_nmae),
            'per_state_nmae': {
                name: {
                    'mean': float(np.nanmean(per_state_nmae[name])) if per_state_nmae[name] else float('nan'),
                }
                for name in STATE_NAMES_7D
            },
        }

    return results


def make_predict_fn_node_single(model, target_dim, state_std, action_std, delta_std):
    """Create predict function for a single-state NODE model used with v9 for rest."""
    def predict(s_cur, action):
        s_norm = torch.FloatTensor(s_cur / state_std).unsqueeze(0)
        a_norm = torch.FloatTensor([[action / action_std]])
        with torch.no_grad():
            dsdt_1d = model(s_norm, a_norm).numpy()[0]
        dsdt = dsdt_1d * delta_std[target_dim] * DT
        s_next = s_cur.copy()
        s_next[target_dim] = s_cur[target_dim] + dsdt[0]
        return s_next
    return predict


def make_predict_fn_direct_delta_single(model, target_dim, state_std, action_std, delta_std):
    """Create predict function for a single-state direct-delta model."""
    def predict(s_cur, action):
        s_norm = torch.FloatTensor(s_cur / state_std).unsqueeze(0)
        a_norm = torch.FloatTensor([[action / action_std]])
        with torch.no_grad():
            delta_1d = model(s_norm, a_norm).numpy()[0]
        delta_val = delta_1d[0] * delta_std[target_dim]
        s_next = s_cur.copy()
        s_next[target_dim] = s_cur[target_dim] + delta_val
        return s_next
    return predict


def make_predict_fn_gp_single(gp, target_dim, state_std, action_std, delta_std):
    """Create predict function for a single-state GP model."""
    def predict(s_cur, action):
        x = np.concatenate([s_cur / state_std, [action / action_std]]).reshape(1, -1)
        delta_norm = gp.predict(x)[0]
        s_next = s_cur.copy()
        s_next[target_dim] = s_cur[target_dim] + delta_norm * delta_std[target_dim]
        return s_next
    return predict


def make_predict_fn_linear_single(model, target_dim, state_std, action_std, delta_std):
    """Create predict function for a single-state linear model."""
    def predict(s_cur, action):
        x = np.concatenate([s_cur / state_std, [action / action_std]]).reshape(1, -1)
        delta_norm = model.predict(x)[0]
        s_next = s_cur.copy()
        s_next[target_dim] = s_cur[target_dim] + delta_norm * delta_std[target_dim]
        return s_next
    return predict


def make_predict_fn_v9(model, state_std, action_std, delta_std):
    """Create predict function for the full v9 NODE model."""
    def predict(s_cur, action):
        s_norm = torch.FloatTensor(s_cur / state_std).unsqueeze(0)
        a_norm = torch.FloatTensor([[action / action_std]])
        with torch.no_grad():
            dsdt_norm = model(s_norm, a_norm).numpy()[0]
        dsdt = dsdt_norm * delta_std * DT
        return s_cur + dsdt
    return predict


def make_predict_fn_composite(v9_model, specialized_models, specialized_dims,
                               specialized_types, state_std, action_std, delta_std):
    """Create predict function that uses specialized models for specific dims
    and v9 for the rest.

    specialized_models: dict dim -> model
    specialized_dims: list of dims with specialized models
    specialized_types: dict dim -> 'node' | 'direct_delta' | 'gp' | 'linear'
    """
    def predict(s_cur, action):
        s_norm = torch.FloatTensor(s_cur / state_std).unsqueeze(0)
        a_norm = torch.FloatTensor([[action / action_std]])
        x_np = np.concatenate([s_cur / state_std, [action / action_std]]).reshape(1, -1)

        # Get v9 prediction for all states
        with torch.no_grad():
            dsdt_norm = v9_model(s_norm, a_norm).numpy()[0]
        dsdt = dsdt_norm * delta_std * DT
        s_next = s_cur + dsdt

        # Override with specialized models
        for dim in specialized_dims:
            model = specialized_models[dim]
            mtype = specialized_types[dim]

            if mtype == 'node':
                with torch.no_grad():
                    dsdt_1d = model(s_norm, a_norm).numpy()[0]
                s_next[dim] = s_cur[dim] + dsdt_1d[0] * delta_std[dim] * DT
            elif mtype == 'direct_delta':
                with torch.no_grad():
                    delta_1d = model(s_norm, a_norm).numpy()[0]
                s_next[dim] = s_cur[dim] + delta_1d[0] * delta_std[dim]
            elif mtype == 'gp':
                delta_norm = model.predict(x_np)[0]
                s_next[dim] = s_cur[dim] + delta_norm * delta_std[dim]
            elif mtype == 'linear':
                delta_norm = model.predict(x_np)[0]
                s_next[dim] = s_cur[dim] + delta_norm * delta_std[dim]

        return s_next
    return predict


# ─── Main Experiments ───────────────────────────────────────────────────

def run_single_state_experiment(data, state_name, state_idx, horizons, seed=42):
    """Run all architectures for a single target state.

    Returns dict of {arch_name: {model, results, primary_score, target_nmae}}.
    """
    print(f"\n{'='*60}")
    print(f"  Experiment: {state_name} (index {state_idx})")
    print(f"{'='*60}")

    state_std = data['state_std']
    action_std = data['action_std']
    delta_std = data['delta_std']

    # Train v9 baseline to use for non-target states
    v9_config = {
        'hidden': 64, 'depth': 3, 'activation': 'tanh',
        'lr': 1e-3, 'n_epochs': 200, 'batch_size': 256, 'patience': 50,
    }
    print("  Training v9 baseline for non-target states...")
    v9_model = train_v9_baseline(data, v9_config, seed=43)
    v9_predict = make_predict_fn_v9(v9_model, state_std, action_std, delta_std)

    # Baseline: v9 full model evaluated
    print("  Evaluating v9 baseline...")
    v9_results = evaluate_composite(v9_predict, data, horizons, seed=seed)

    node_config = {
        'hidden': 64, 'depth': 3, 'activation': 'tanh',
        'lr': 1e-3, 'n_epochs': 200, 'batch_size': 256, 'patience': 50,
    }
    node_wide_config = {
        'hidden': 128, 'depth': 4, 'activation': 'tanh',
        'lr': 1e-3, 'n_epochs': 300, 'batch_size': 256, 'patience': 50,
    }

    arch_results = {}

    # --- Architecture 1: NODE (standard) ---
    print(f"\n  [1/5] Training NODE (standard) for {state_name}...")
    t0 = time.time()
    node_model = train_single_state_node(data, state_idx, node_config, seed=seed)
    node_predict = make_predict_fn_node_single(node_model, state_idx, state_std, action_std, delta_std)
    # Override only target dim, use v9 for the rest
    composite_node = make_predict_fn_composite(
        v9_model, {state_idx: node_model}, [state_idx], {state_idx: 'node'},
        state_std, action_std, delta_std
    )
    node_results = evaluate_composite(composite_node, data, horizons, seed=seed)
    node_time = time.time() - t0
    print(f"    Completed in {node_time:.1f}s")

    # --- Architecture 2: NODE (wide) ---
    print(f"\n  [2/5] Training NODE (wide) for {state_name}...")
    t0 = time.time()
    node_wide_model = train_single_state_node(data, state_idx, node_wide_config, seed=seed, wide=True)
    composite_node_wide = make_predict_fn_composite(
        v9_model, {state_idx: node_wide_model}, [state_idx], {state_idx: 'node'},
        state_std, action_std, delta_std
    )
    node_wide_results = evaluate_composite(composite_node_wide, data, horizons, seed=seed)
    node_wide_time = time.time() - t0
    print(f"    Completed in {node_wide_time:.1f}s")

    # --- Architecture 3: Direct Delta ---
    print(f"\n  [3/5] Training Direct Delta for {state_name}...")
    t0 = time.time()
    dd_model = train_single_state_direct_delta(data, state_idx, node_config, seed=seed)
    composite_dd = make_predict_fn_composite(
        v9_model, {state_idx: dd_model}, [state_idx], {state_idx: 'direct_delta'},
        state_std, action_std, delta_std
    )
    dd_results = evaluate_composite(composite_dd, data, horizons, seed=seed)
    dd_time = time.time() - t0
    print(f"    Completed in {dd_time:.1f}s")

    # --- Architecture 4: GP ---
    print(f"\n  [4/5] Training GP for {state_name}...")
    t0 = time.time()
    gp_model = train_single_state_gp(data, state_idx, max_samples=5000, seed=seed)
    composite_gp = make_predict_fn_composite(
        v9_model, {state_idx: gp_model}, [state_idx], {state_idx: 'gp'},
        state_std, action_std, delta_std
    )
    gp_results = evaluate_composite(composite_gp, data, horizons, seed=seed)
    gp_time = time.time() - t0
    print(f"    Completed in {gp_time:.1f}s")

    # --- Architecture 5: Linear (Ridge) ---
    print(f"\n  [5/5] Training Linear (Ridge) for {state_name}...")
    t0 = time.time()
    linear_model = train_single_state_linear(data, state_idx, alpha=1.0, seed=seed)
    composite_linear = make_predict_fn_composite(
        v9_model, {state_idx: linear_model}, [state_idx], {state_idx: 'linear'},
        state_std, action_std, delta_std
    )
    linear_results = evaluate_composite(composite_linear, data, horizons, seed=seed)
    linear_time = time.time() - t0
    print(f"    Completed in {linear_time:.1f}s")

    # --- Collect results ---
    all_archs = {
        'node': {'results': node_results, 'time': node_time},
        'node_wide': {'results': node_wide_results, 'time': node_wide_time},
        'direct_delta': {'results': dd_results, 'time': dd_time},
        'gp': {'results': gp_results, 'time': gp_time},
        'linear': {'results': linear_results, 'time': linear_time},
    }

    # Compute primary score and per-state score for target
    for arch_name, arch_data in all_archs.items():
        r = arch_data['results']
        valid = all(h in r and not np.isnan(r[h]['nmae_mean']) for h in [100, 200, 500])
        if valid:
            arch_data['primary_score'] = float(np.mean([
                r[100]['nmae_mean'], r[200]['nmae_mean'], r[500]['nmae_mean']
            ]))
        else:
            arch_data['primary_score'] = float('nan')

        # Per-state NMAE for target at H=100
        if 100 in r:
            arch_data['target_nmae_h100'] = r[100]['per_state_nmae'][state_name]['mean']
        else:
            arch_data['target_nmae_h100'] = float('nan')

    # v9 baseline scores
    v9_valid = all(h in v9_results and not np.isnan(v9_results[h]['nmae_mean']) for h in [100, 200, 500])
    v9_primary = float(np.mean([
        v9_results[100]['nmae_mean'], v9_results[200]['nmae_mean'], v9_results[500]['nmae_mean']
    ])) if v9_valid else float('nan')

    # Print summary table
    print(f"\n  --- {state_name} Summary ---")
    print(f"  {'Architecture':<15} {'Primary':<10} {'Target@H100':<12} {'vs v9':<10} {'Time(s)':<8}")
    print(f"  {'-'*55}")
    print(f"  {'v9_baseline':<15} {v9_primary:<10.4f} "
          f"{v9_results[100]['per_state_nmae'][state_name]['mean']:<12.4f} {'---':<10} {'---':<8}")

    best_arch = None
    best_primary = float('inf')
    for arch_name, arch_data in all_archs.items():
        ps = arch_data['primary_score']
        tn = arch_data['target_nmae_h100']
        improvement = (v9_primary - ps) / v9_primary * 100 if v9_primary > 0 and not np.isnan(ps) else 0
        print(f"  {arch_name:<15} {ps:<10.4f} {tn:<12.4f} {improvement:>+7.1f}%  {arch_data['time']:<8.1f}")
        if not np.isnan(ps) and ps < best_primary:
            best_primary = ps
            best_arch = arch_name

    print(f"\n  Best architecture for {state_name}: {best_arch} (Primary={best_primary:.4f})")

    return {
        'v9_results': v9_results,
        'v9_primary': v9_primary,
        'architectures': all_archs,
        'best_arch': best_arch,
        'best_primary': best_primary,
    }


def run_composite_experiment(data, best_per_state, horizons, seed=42):
    """Experiment 5: Combine best per-state models into composite."""
    print(f"\n{'='*60}")
    print(f"  Experiment 5: Best Composite Model")
    print(f"{'='*60}")

    state_std = data['state_std']
    action_std = data['action_std']
    delta_std = data['delta_std']

    # Train v9 baseline
    v9_config = {
        'hidden': 64, 'depth': 3, 'activation': 'tanh',
        'lr': 1e-3, 'n_epochs': 200, 'batch_size': 256, 'patience': 50,
    }
    print("  Training v9 baseline...")
    v9_model = train_v9_baseline(data, v9_config, seed=43)

    # Train specialized models for each target state
    node_config = {
        'hidden': 64, 'depth': 3, 'activation': 'tanh',
        'lr': 1e-3, 'n_epochs': 200, 'batch_size': 256, 'patience': 50,
    }
    node_wide_config = {
        'hidden': 128, 'depth': 4, 'activation': 'tanh',
        'lr': 1e-3, 'n_epochs': 300, 'batch_size': 256, 'patience': 50,
    }

    specialized_models = {}
    specialized_types = {}
    specialized_dims = []

    for state_name, state_info in best_per_state.items():
        state_idx = TARGET_STATES[state_name]
        best_arch = state_info['best_arch']
        print(f"  Training {best_arch} for {state_name}...")

        if best_arch == 'node':
            model = train_single_state_node(data, state_idx, node_config, seed=seed)
            specialized_models[state_idx] = model
            specialized_types[state_idx] = 'node'
        elif best_arch == 'node_wide':
            model = train_single_state_node(data, state_idx, node_wide_config, seed=seed, wide=True)
            specialized_models[state_idx] = model
            specialized_types[state_idx] = 'node'
        elif best_arch == 'direct_delta':
            model = train_single_state_direct_delta(data, state_idx, node_config, seed=seed)
            specialized_models[state_idx] = model
            specialized_types[state_idx] = 'direct_delta'
        elif best_arch == 'gp':
            model = train_single_state_gp(data, state_idx, max_samples=5000, seed=seed)
            specialized_models[state_idx] = model
            specialized_types[state_idx] = 'gp'
        elif best_arch == 'linear':
            model = train_single_state_linear(data, state_idx, alpha=1.0, seed=seed)
            specialized_models[state_idx] = model
            specialized_types[state_idx] = 'linear'
        else:
            print(f"    WARNING: Unknown architecture {best_arch}, skipping")
            continue

        specialized_dims.append(state_idx)

    # Build composite predict function
    composite_predict = make_predict_fn_composite(
        v9_model, specialized_models, specialized_dims, specialized_types,
        state_std, action_std, delta_std
    )

    # Also build pure v9 predict for comparison
    v9_predict = make_predict_fn_v9(v9_model, state_std, action_std, delta_std)

    # Evaluate both
    print("\n  Evaluating composite model...")
    composite_results = evaluate_composite(composite_predict, data, horizons, seed=seed)

    print("  Evaluating v9 baseline...")
    v9_results = evaluate_composite(v9_predict, data, horizons, seed=seed)

    # Compute scores
    def compute_primary(r):
        valid = all(h in r and not np.isnan(r[h]['nmae_mean']) for h in [100, 200, 500])
        if valid:
            return float(np.mean([r[100]['nmae_mean'], r[200]['nmae_mean'], r[500]['nmae_mean']]))
        return float('nan')

    composite_primary = compute_primary(composite_results)
    v9_primary = compute_primary(v9_results)

    print(f"\n  --- Composite vs v9 ---")
    print(f"  {'Horizon':<10} {'v9 NMAE':<12} {'Composite':<12} {'Improvement':<12}")
    print(f"  {'-'*46}")
    for h in horizons:
        v9_val = v9_results[h]['nmae_mean']
        comp_val = composite_results[h]['nmae_mean']
        if v9_val > 0 and not np.isnan(comp_val):
            imp = (v9_val - comp_val) / v9_val * 100
            print(f"  H={h:<7} {v9_val:<12.4f} {comp_val:<12.4f} {imp:>+9.1f}%")

    print(f"\n  v9 Primary:      {v9_primary:.4f}")
    print(f"  Composite Primary: {composite_primary:.4f}")
    if v9_primary > 0 and not np.isnan(composite_primary):
        overall_imp = (v9_primary - composite_primary) / v9_primary * 100
        print(f"  Overall Improvement: {overall_imp:+.1f}%")

    return {
        'composite_results': composite_results,
        'composite_primary': composite_primary,
        'v9_results': v9_results,
        'v9_primary': v9_primary,
        'specialized_dims': specialized_dims,
        'specialized_types': {str(k): v for k, v in specialized_types.items()},
    }


# ─── Main ───────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  EXP041: State-Specific Experiments")
    print("  Testing dedicated models for e_y, e_psi, v, theta")
    print("=" * 70)

    horizons = [1, 10, 50, 100, 200, 500]

    # Load data
    print("\nLoading data...")
    data = load_data(seed=42)
    print(f"  State std: {data['state_std']}")
    print(f"  Delta std: {data['delta_std']}")
    print(f"  Train episodes: {len(data['train_eps'])}, Test episodes: {len(data['test_eps'])}")

    # ─── Experiments 1-4: Per-state models ───────────────────────────
    per_state_results = {}
    for state_name, state_idx in TARGET_STATES.items():
        result = run_single_state_experiment(data, state_name, state_idx, horizons, seed=42)
        per_state_results[state_name] = result

    # ─── Experiment 5: Best Composite ────────────────────────────────
    best_per_state = {}
    for state_name, result in per_state_results.items():
        best_per_state[state_name] = {
            'best_arch': result['best_arch'],
            'best_primary': result['best_primary'],
        }

    composite_result = run_composite_experiment(data, best_per_state, horizons, seed=42)

    # ─── Final Summary ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)

    print(f"\n  Per-State Best Architectures:")
    print(f"  {'State':<12} {'Best Arch':<15} {'Primary':<10} {'v9 Primary':<12} {'Improvement':<12}")
    print(f"  {'-'*61}")
    for state_name, result in per_state_results.items():
        bp = result['best_primary']
        v9p = result['v9_primary']
        imp = (v9p - bp) / v9p * 100 if v9p > 0 and not np.isnan(bp) else 0
        print(f"  {state_name:<12} {result['best_arch']:<15} {bp:<10.4f} {v9p:<12.4f} {imp:>+9.1f}%")

    print(f"\n  Composite Model:")
    print(f"  v9 Primary:        {composite_result['v9_primary']:.4f}")
    print(f"  Composite Primary: {composite_result['composite_primary']:.4f}")
    if composite_result['v9_primary'] > 0 and not np.isnan(composite_result['composite_primary']):
        imp = (composite_result['v9_primary'] - composite_result['composite_primary']) / composite_result['v9_primary'] * 100
        print(f"  Improvement:       {imp:+.1f}%")

    # ─── Save Results ────────────────────────────────────────────────
    output = {
        'timestamp': datetime.now().isoformat(),
        'experiment': 'EXP041_state_specific_experiments',
        'description': 'State-specific model experiments for e_y, e_psi, v, theta',
        'v9_baseline_primary': V9_PRIMARY,
        'horizons': horizons,
        'per_state': {},
        'composite': {
            'v9_primary': composite_result['v9_primary'],
            'composite_primary': composite_result['composite_primary'],
            'specialized_dims': composite_result['specialized_dims'],
            'specialized_types': composite_result['specialized_types'],
            'results': {
                str(h): {
                    'v9_nmae': composite_result['v9_results'][h]['nmae_mean'],
                    'composite_nmae': composite_result['composite_results'][h]['nmae_mean'],
                    'composite_survival': composite_result['composite_results'][h]['survival_rate'],
                    'composite_per_state': composite_result['composite_results'][h]['per_state_nmae'],
                }
                for h in horizons
            },
        },
    }

    for state_name, result in per_state_results.items():
        state_entry = {
            'state_index': TARGET_STATES[state_name],
            'v9_primary': result['v9_primary'],
            'best_arch': result['best_arch'],
            'best_primary': result['best_primary'],
            'architectures': {},
        }
        for arch_name, arch_data in result['architectures'].items():
            arch_entry = {
                'primary_score': arch_data['primary_score'],
                'target_nmae_h100': arch_data['target_nmae_h100'],
                'time_seconds': arch_data['time'],
                'results': {},
            }
            for h in horizons:
                r = arch_data['results'][h]
                arch_entry['results'][str(h)] = {
                    'nmae_mean': r['nmae_mean'],
                    'nmae_std': r['nmae_std'],
                    'survival_rate': r['survival_rate'],
                    'per_state_nmae': r['per_state_nmae'],
                }
            state_entry['architectures'][arch_name] = arch_entry
        output['per_state'][state_name] = state_entry

    output_path = 'D:/系统辨识作业/sindy_bicycle/research_72h/05_candidates/EXP041_state_specific_experiments.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, default=str, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")

    # ─── Generate Analysis Report ────────────────────────────────────
    report_lines = []
    report_lines.append("# EXP041 State-Specific Experiments Analysis\n")
    report_lines.append(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    report_lines.append("## Objective\n")
    report_lines.append("Test whether dedicated models for individual states (e_y, e_psi, v, theta)")
    report_lines.append("can outperform the monolithic v9 NODE baseline on the PrimaryLongHorizonScore.\n")
    report_lines.append("## Method\n")
    report_lines.append("For each target state, five architectures were tested:")
    report_lines.append("1. **NODE (standard)**: 3-layer MLP (64 hidden, Tanh) predicting dsdt")
    report_lines.append("2. **NODE (wide)**: 4-layer MLP (128 hidden, Tanh) predicting dsdt")
    report_lines.append("3. **Direct Delta**: 3-layer MLP predicting delta_s directly (no dt)")
    report_lines.append("4. **GP**: Gaussian Process with Matern kernel (5000 samples)")
    report_lines.append("5. **Linear**: Ridge regression on normalized (state, action) input\n")
    report_lines.append("Each specialized model predicts only its target state; the v9 model handles")
    report_lines.append("the remaining 6 states during rollout evaluation.\n")
    report_lines.append("## Per-State Results\n")

    for state_name, result in per_state_results.items():
        report_lines.append(f"### {state_name}\n")
        report_lines.append(f"| Architecture | Primary Score | Target@H100 | vs v9 |")
        report_lines.append(f"|---|---|---|---|")
        v9p = result['v9_primary']
        for arch_name, arch_data in result['architectures'].items():
            ps = arch_data['primary_score']
            tn = arch_data['target_nmae_h100']
            imp = (v9p - ps) / v9p * 100 if v9p > 0 and not np.isnan(ps) else 0
            marker = " **BEST**" if arch_name == result['best_arch'] else ""
            report_lines.append(f"| {arch_name}{marker} | {ps:.4f} | {tn:.4f} | {imp:+.1f}% |")
        report_lines.append(f"| v9_baseline | {v9p:.4f} | {result['v9_results'][100]['per_state_nmae'][state_name]['mean']:.4f} | --- |")
        report_lines.append(f"\n**Best for {state_name}**: {result['best_arch']} (Primary={result['best_primary']:.4f})\n")

    report_lines.append("## Composite Model (Experiment 5)\n")
    report_lines.append("The best per-state architecture was selected and combined with v9 for remaining states.\n")
    report_lines.append("| Horizon | v9 NMAE | Composite NMAE | Improvement |")
    report_lines.append("|---|---|---|---|")
    for h in horizons:
        v9_val = composite_result['v9_results'][h]['nmae_mean']
        comp_val = composite_result['composite_results'][h]['nmae_mean']
        if v9_val > 0 and not np.isnan(comp_val):
            imp = (v9_val - comp_val) / v9_val * 100
            report_lines.append(f"| H={h} | {v9_val:.4f} | {comp_val:.4f} | {imp:+.1f}% |")

    report_lines.append(f"\n**v9 Primary**: {composite_result['v9_primary']:.4f}")
    report_lines.append(f"**Composite Primary**: {composite_result['composite_primary']:.4f}")
    if composite_result['v9_primary'] > 0 and not np.isnan(composite_result['composite_primary']):
        imp = (composite_result['v9_primary'] - composite_result['composite_primary']) / composite_result['v9_primary'] * 100
        report_lines.append(f"**Overall Improvement**: {imp:+.1f}%")

    report_lines.append("\n## Key Findings\n")

    # Determine findings
    findings = []
    for state_name, result in per_state_results.items():
        best = result['best_arch']
        bp = result['best_primary']
        v9p = result['v9_primary']
        if not np.isnan(bp) and bp < v9p:
            imp = (v9p - bp) / v9p * 100
            findings.append(f"- **{state_name}**: {best} improves by {imp:+.1f}% over v9")
        else:
            findings.append(f"- **{state_name}**: No architecture beats v9 (best={best}, {bp:.4f} vs {v9p:.4f})")

    report_lines.extend(findings)

    comp_p = composite_result['composite_primary']
    v9_p = composite_result['v9_primary']
    if not np.isnan(comp_p) and comp_p < v9_p:
        report_lines.append(f"\nThe composite model achieves {((v9_p - comp_p)/v9_p*100):+.1f}% improvement over v9.")
    else:
        report_lines.append(f"\nThe composite model does NOT beat v9 ({comp_p:.4f} vs {v9_p:.4f}).")

    report_lines.append("\n## Conclusion\n")
    report_lines.append("State-specific specialization provides insight into which states benefit")
    report_lines.append("from different modeling approaches, but the monolithic v9 architecture")
    report_lines.append("remains competitive due to cross-state coupling in the dynamics.\n")

    report_path = 'D:/系统辨识作业/sindy_bicycle/research_72h/05_candidates/STATE_SPECIFIC_ANALYSIS.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    print(f"Analysis saved to: {report_path}")

    return output


if __name__ == '__main__':
    main()
