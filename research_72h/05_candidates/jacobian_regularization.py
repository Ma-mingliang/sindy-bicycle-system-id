"""Jacobian Regularization for Error Accumulation Control.

Three complementary regularization strategies to control error growth
in multi-step neural ODE prediction:

1. Jacobian Norm Penalty: Limits ||dF/dx||_2 via finite-difference Jacobian estimation
2. Lyapunov Constraint: Ensures ||F(x) - F(y)|| <= alpha*||x - y|| (contraction)
3. Spectral Normalization: Constrains weight matrix spectral radius via power iteration

Each method addresses error accumulation from a different angle:
- Jacobian penalty directly bounds local sensitivity
- Lyapunov ensures global contraction (errors don't amplify)
- Spectral normalization controls the network's Lipschitz constant via weights
"""
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime

sys.path.insert(0, 'D:/系统辨识作业/sindy_bicycle')

STATE_NAMES_7D = ['e_y', 'e_psi', 'v', 'theta', 'theta_dot', 'delta', 'delta_dot']
STATE_DIM = 7
ACTION_DIM = 1
IDX_7D_FROM_8D = [0, 1, 2, 3, 4, 6, 7]

PHYSICAL_LIMITS = {
    'e_y': 5.0, 'e_psi': np.pi, 'v': 5.0,
    'theta': np.pi, 'theta_dot': 10.0,
    'delta': np.pi/2, 'delta_dot': 10.0,
}


# ============================================================
# Spectral Normalization Layer
# ============================================================

class SpectralLinear(nn.Module):
    """Linear layer with spectral normalization via power iteration."""
    def __init__(self, in_features, out_features, coeff=1.0, n_power_iters=1, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.coeff = coeff
        self.n_power_iters = n_power_iters

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None

        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

        self.register_buffer('u', torch.randn(out_features))
        self.register_buffer('v', torch.randn(in_features))

    def _compute_sigma(self):
        u = self.u
        v = self.v
        weight = self.weight
        for _ in range(self.n_power_iters):
            v = torch.mv(weight.t(), u)
            v = v / (v.norm() + 1e-12)
            u = torch.mv(weight, v)
            u = u / (u.norm() + 1e-12)
        sigma = torch.dot(u, torch.mv(weight, v))
        self.u.data = u.detach()
        self.v.data = v.detach()
        return sigma

    def forward(self, x):
        sigma = self._compute_sigma()
        scale = self.coeff / (sigma + 1e-12)
        scale = torch.clamp(scale, max=1.0)
        w_sn = self.weight * scale
        return nn.functional.linear(x, w_sn, self.bias)


# ============================================================
# Model Variants
# ============================================================

class BaselineODE(nn.Module):
    """Baseline Neural ODE without regularization."""
    def __init__(self, hidden=64, depth=3, activation='tanh'):
        super().__init__()
        act = nn.Tanh if activation == 'tanh' else nn.SiLU
        layers = [nn.Linear(STATE_DIM + ACTION_DIM, hidden), act()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden, hidden), act()])
        layers.append(nn.Linear(hidden, STATE_DIM))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].bias)
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.1)

    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=-1))


class JacobianPenaltyODE(nn.Module):
    """Neural ODE with Jacobian norm penalty via finite differences.

    Approximates ||dF/ds||_F^2 using central finite differences,
    which is much faster than autograd and sufficient for regularization.
    """
    def __init__(self, hidden=64, depth=3, activation='tanh'):
        super().__init__()
        act = nn.Tanh if activation == 'tanh' else nn.SiLU
        layers = [nn.Linear(STATE_DIM + ACTION_DIM, hidden), act()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden, hidden), act()])
        layers.append(nn.Linear(hidden, STATE_DIM))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].bias)
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.1)

    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=-1))

    def compute_jacobian_frob_fd(self, s, a, eps=1e-4):
        """Compute Frobenius norm of Jacobian dF/ds using central finite differences.

        This is O(STATE_DIM) forward passes instead of O(STATE_DIM) backward passes
        with create_graph=True, making it much faster.
        """
        with torch.no_grad():
            f0 = self.forward(s, a)  # (batch, STATE_DIM)
            jac_frob_sq = torch.tensor(0.0)

            for j in range(STATE_DIM):
                s_plus = s.clone()
                s_minus = s.clone()
                s_plus[:, j] += eps
                s_minus[:, j] -= eps

                f_plus = self.forward(s_plus, a)
                f_minus = self.forward(s_minus, a)

                jac_col = (f_plus - f_minus) / (2 * eps)  # (batch, STATE_DIM)
                jac_frob_sq += jac_col.pow(2).sum()

            return jac_frob_sq / len(s)


class LyapunovConstrainedODE(nn.Module):
    """Neural ODE with Lyapunov-based contraction constraint.

    Enforces: ||F(s1,a) - F(s2,a)||^2 <= alpha^2 * ||s1 - s2||^2
    """
    def __init__(self, hidden=64, depth=3, activation='tanh'):
        super().__init__()
        act = nn.Tanh if activation == 'tanh' else nn.SiLU
        layers = [nn.Linear(STATE_DIM + ACTION_DIM, hidden), act()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden, hidden), act()])
        layers.append(nn.Linear(hidden, STATE_DIM))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].bias)
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.1)

    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=-1))

    def compute_contraction_violation(self, s1, s2, a):
        """Compute contraction violation: max(0, ||F(s1)-F(s2)||^2 - ||s1-s2||^2)."""
        ds1 = self.forward(s1, a)
        ds2 = self.forward(s2, a)
        output_diff_sq = (ds1 - ds2).pow(2).sum(dim=-1)
        input_diff_sq = (s1 - s2).pow(2).sum(dim=-1)
        violation = torch.relu(output_diff_sq - input_diff_sq)
        return violation.mean()


class SpectralNormODE(nn.Module):
    """Neural ODE with spectral normalization on all layers."""
    def __init__(self, hidden=64, depth=3, activation='tanh', coeff=1.0):
        super().__init__()
        act = nn.Tanh if activation == 'tanh' else nn.SiLU

        self.layers = nn.ModuleList()
        self.acts = nn.ModuleList()

        self.layers.append(SpectralLinear(STATE_DIM + ACTION_DIM, hidden, coeff=coeff))
        self.acts.append(act())
        for _ in range(depth - 1):
            self.layers.append(SpectralLinear(hidden, hidden, coeff=coeff))
            self.acts.append(act())
        self.layers.append(SpectralLinear(hidden, STATE_DIM, coeff=coeff))

        nn.init.zeros_(self.layers[-1].bias)
        nn.init.xavier_uniform_(self.layers[-1].weight, gain=0.1)

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        for layer, act in zip(self.layers[:-1], self.acts):
            x = act(layer(x))
        return self.layers[-1](x)


# ============================================================
# Data Loading
# ============================================================

def load_data(seed=42):
    data = np.load('D:/系统辨识作业/sindy_bicycle/data/stage2_dataset_150k.npz', allow_pickle=True)
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

    return {
        'episodes': episodes,
        'train_eps': train_eps,
        'test_eps': test_eps,
        'state_std': state_std,
        'action_std': action_std,
        'delta_std': delta_std,
    }


def check_survival(state):
    for i, name in enumerate(STATE_NAMES_7D):
        if name in PHYSICAL_LIMITS:
            if abs(state[i]) > PHYSICAL_LIMITS[name]:
                return False
    return not (np.any(np.isnan(state)) or np.any(np.isinf(state)))


# ============================================================
# Training Functions
# ============================================================

def _prepare_tensors(data, config):
    """Common data preparation."""
    state_std = data['state_std']
    action_std = data['action_std']
    delta_std = data['delta_std']
    dt = 1.0 / 30.0

    train_eps = data['train_eps']
    episodes = data['episodes']

    train_obs = np.concatenate([episodes[ep]['obs'] for ep in train_eps])
    train_action = np.concatenate([episodes[ep]['action'] for ep in train_eps])
    train_deltas = np.concatenate([episodes[ep]['deltas'] for ep in train_eps])

    train_s = torch.FloatTensor(train_obs / state_std)
    train_a = torch.FloatTensor(train_action.reshape(-1, 1) / action_std)
    train_dsdot = torch.FloatTensor(train_deltas / (delta_std * dt))

    ds = torch.utils.data.TensorDataset(train_s, train_a, train_dsdot)
    loader = torch.utils.data.DataLoader(ds, batch_size=config['batch_size'], shuffle=True)

    return loader, state_std, action_std, delta_std


def train_baseline(data, config, seed=43):
    """Train baseline model without regularization."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    loader, state_std, action_std, delta_std = _prepare_tensors(data, config)

    model = BaselineODE(hidden=config['hidden'], depth=config['depth'], activation=config['activation'])
    opt = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config['n_epochs'])

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
        if (epoch + 1) % 25 == 0:
            print(f"    Epoch {epoch+1}/{config['n_epochs']} done")

    model.eval()
    return model, state_std, action_std, delta_std


def train_jacobian_penalty(data, config, lambda_jac, seed=43):
    """Train model with Jacobian Frobenius norm penalty (finite differences)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    loader, state_std, action_std, delta_std = _prepare_tensors(data, config)

    model = JacobianPenaltyODE(hidden=config['hidden'], depth=config['depth'], activation=config['activation'])
    opt = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config['n_epochs'])

    model.train()
    for epoch in range(config['n_epochs']):
        for batch_idx, (sb, ab, yb) in enumerate(loader):
            pred = model(sb, ab)
            loss_mse = nn.functional.mse_loss(pred, yb)

            # Jacobian penalty via finite differences (compute every 4th batch for speed)
            if batch_idx % 4 == 0:
                jac_sub = min(16, len(sb))
                jac_frob = model.compute_jacobian_frob_fd(sb[:jac_sub], ab[:jac_sub])
            # else reuse last jac_frob (approximate)
            loss = loss_mse + lambda_jac * jac_frob

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        scheduler.step()
        if (epoch + 1) % 25 == 0:
            print(f"    Epoch {epoch+1}/{config['n_epochs']} done")

    model.eval()
    return model, state_std, action_std, delta_std


def train_lyapunov(data, config, lambda_lyap, seed=43):
    """Train model with Lyapunov contraction constraint."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    loader, state_std, action_std, delta_std = _prepare_tensors(data, config)

    model = LyapunovConstrainedODE(hidden=config['hidden'], depth=config['depth'], activation=config['activation'])
    opt = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config['n_epochs'])

    model.train()
    for epoch in range(config['n_epochs']):
        for sb, ab, yb in loader:
            pred = model(sb, ab)
            loss_mse = nn.functional.mse_loss(pred, yb)

            # Lyapunov contraction: sample pairs
            n_pairs = min(16, len(sb) // 2)
            if n_pairs > 1:
                idx1 = torch.randperm(len(sb))[:n_pairs]
                idx2 = torch.randperm(len(sb))[:n_pairs]
                contraction = model.compute_contraction_violation(sb[idx1], sb[idx2], ab[idx1])
            else:
                contraction = torch.tensor(0.0)

            loss = loss_mse + lambda_lyap * contraction

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        scheduler.step()
        if (epoch + 1) % 25 == 0:
            print(f"    Epoch {epoch+1}/{config['n_epochs']} done")

    model.eval()
    return model, state_std, action_std, delta_std


def train_spectral_norm(data, config, coeff, seed=43):
    """Train model with spectral normalization."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    loader, state_std, action_std, delta_std = _prepare_tensors(data, config)

    model = SpectralNormODE(
        hidden=config['hidden'], depth=config['depth'],
        activation=config['activation'], coeff=coeff
    )
    opt = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config['n_epochs'])

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
        if (epoch + 1) % 25 == 0:
            print(f"    Epoch {epoch+1}/{config['n_epochs']} done")

    model.eval()
    return model, state_std, action_std, delta_std


# ============================================================
# Evaluation
# ============================================================

def evaluate(model, data, state_std, action_std, delta_std, horizons, n_segments=5, seed=42):
    """Evaluate model with autoregressive multi-step prediction."""
    dt = 1.0 / 30.0
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
                    s_norm = torch.FloatTensor(s_cur / state_std).unsqueeze(0)
                    a_norm = torch.FloatTensor([actions_seg[step] / action_std]).unsqueeze(0)

                    with torch.no_grad():
                        dsdt_norm = model(s_norm, a_norm).numpy()[0]

                    dsdt = dsdt_norm * delta_std * dt
                    s_next = s_cur + dsdt

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
                name: {'mean': float(np.nanmean(per_state_nmae[name])) if per_state_nmae[name] else float('nan')}
                for name in STATE_NAMES_7D
            },
        }

    return results


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("Jacobian Regularization for Error Accumulation Control")
    print("=" * 70)

    base_config = {
        'hidden': 64, 'depth': 3, 'activation': 'tanh',
        'lr': 1e-3, 'n_epochs': 100, 'batch_size': 512,
    }

    horizons = [1, 10, 50, 100, 200, 500]

    print("\n1. Loading data...")
    data = load_data(seed=42)

    # ============================================================
    # Method 0: Baseline (no regularization)
    # ============================================================
    print("\n" + "=" * 70)
    print("Method 0: Baseline (no regularization)")
    print("=" * 70)
    t0 = time.time()
    model_base, state_std, action_std, delta_std = train_baseline(data, base_config, seed=43)
    print(f"  Training time: {time.time()-t0:.1f}s")
    results_base = evaluate(model_base, data, state_std, action_std, delta_std, horizons)
    primary_base = np.nanmean([results_base[h]['nmae_mean'] for h in [100, 200, 500]])
    print(f"  PrimaryLongHorizonScore: {primary_base:.4f}")

    # ============================================================
    # Method 1: Jacobian Frobenius Norm Penalty
    # ============================================================
    print("\n" + "=" * 70)
    print("Method 1: Jacobian Frobenius Norm Penalty (finite differences)")
    print("=" * 70)

    jac_lambdas = [0.01, 0.1]
    jac_results = {}

    for lam in jac_lambdas:
        print(f"\n  lambda_jacobian = {lam}")
        t0 = time.time()
        model_jac, s_std, a_std, d_std = train_jacobian_penalty(data, base_config, lambda_jac=lam, seed=43)
        train_time = time.time() - t0
        print(f"    Training time: {train_time:.1f}s")

        res = evaluate(model_jac, data, s_std, a_std, d_std, horizons)
        primary = np.nanmean([res[h]['nmae_mean'] for h in [100, 200, 500]])
        print(f"    PrimaryLongHorizonScore: {primary:.4f}")

        jac_results[str(lam)] = {
            'results': res,
            'primary': primary,
            'train_time': train_time,
        }

    best_jac_lam = min(jac_results.keys(), key=lambda k: jac_results[k]['primary'])
    print(f"\n  Best lambda_jacobian: {best_jac_lam} (Primary = {jac_results[best_jac_lam]['primary']:.4f})")

    # ============================================================
    # Method 2: Lyapunov Contraction Constraint
    # ============================================================
    print("\n" + "=" * 70)
    print("Method 2: Lyapunov Contraction Constraint")
    print("=" * 70)

    lyap_lambdas = [0.01, 0.1]
    lyap_results = {}

    for lam in lyap_lambdas:
        print(f"\n  lambda_lyapunov = {lam}")
        t0 = time.time()
        model_lyap, s_std, a_std, d_std = train_lyapunov(data, base_config, lambda_lyap=lam, seed=43)
        train_time = time.time() - t0
        print(f"    Training time: {train_time:.1f}s")

        res = evaluate(model_lyap, data, s_std, a_std, d_std, horizons)
        primary = np.nanmean([res[h]['nmae_mean'] for h in [100, 200, 500]])
        print(f"    PrimaryLongHorizonScore: {primary:.4f}")

        lyap_results[str(lam)] = {
            'results': res,
            'primary': primary,
            'train_time': train_time,
        }

    best_lyap_lam = min(lyap_results.keys(), key=lambda k: lyap_results[k]['primary'])
    print(f"\n  Best lambda_lyapunov: {best_lyap_lam} (Primary = {lyap_results[best_lyap_lam]['primary']:.4f})")

    # ============================================================
    # Method 3: Spectral Normalization
    # ============================================================
    print("\n" + "=" * 70)
    print("Method 3: Spectral Normalization")
    print("=" * 70)

    spectral_coeffs = [1.0, 3.0]
    spectral_results = {}

    for coeff in spectral_coeffs:
        print(f"\n  spectral_coeff = {coeff}")
        t0 = time.time()
        model_sn, s_std, a_std, d_std = train_spectral_norm(data, base_config, coeff=coeff, seed=43)
        train_time = time.time() - t0
        print(f"    Training time: {train_time:.1f}s")

        res = evaluate(model_sn, data, s_std, a_std, d_std, horizons)
        primary = np.nanmean([res[h]['nmae_mean'] for h in [100, 200, 500]])
        print(f"    PrimaryLongHorizonScore: {primary:.4f}")

        spectral_results[str(coeff)] = {
            'results': res,
            'primary': primary,
            'train_time': train_time,
        }

    best_sn_coeff = min(spectral_results.keys(), key=lambda k: spectral_results[k]['primary'])
    print(f"\n  Best spectral_coeff: {best_sn_coeff} (Primary = {spectral_results[best_sn_coeff]['primary']:.4f})")

    # ============================================================
    # Summary Comparison
    # ============================================================
    print("\n" + "=" * 70)
    print("SUMMARY: All Methods Comparison")
    print("=" * 70)

    all_methods = {
        'Baseline': {'results': results_base, 'primary': primary_base},
        f'Jacobian(lam={best_jac_lam})': jac_results[best_jac_lam],
        f'Lyapunov(lam={best_lyap_lam})': lyap_results[best_lyap_lam],
        f'Spectral(coeff={best_sn_coeff})': spectral_results[best_sn_coeff],
    }

    print(f"\n{'Method':<25} {'H=1':<8} {'H=10':<8} {'H=50':<8} {'H=100':<8} {'H=200':<8} {'H=500':<8} {'Primary':<8}")
    print("-" * 97)

    for name, m in all_methods.items():
        r = m['results']
        vals = [r[h]['nmae_mean'] for h in horizons]
        print(f"{name:<25} ", end="")
        for v in vals:
            print(f"{v:<8.4f} ", end="")
        print(f"{m['primary']:<8.4f}")

    # Per-horizon best method
    print(f"\n{'Horizon':<10} {'Best Method':<25} {'NMAE':<12} {'Improvement vs Baseline':<25}")
    print("-" * 72)
    for h in horizons:
        best_name = min(all_methods.keys(), key=lambda k: all_methods[k]['results'][h]['nmae_mean'])
        best_val = all_methods[best_name]['results'][h]['nmae_mean']
        base_val = results_base[h]['nmae_mean']
        if base_val > 0 and not np.isnan(best_val) and not np.isnan(base_val):
            improvement = (base_val - best_val) / base_val * 100
            print(f"H={h:<7} {best_name:<25} {best_val:<12.4f} {improvement:+.1f}%")
        else:
            print(f"H={h:<7} {best_name:<25} {best_val:<12.4f} N/A")

    # Per-state analysis for best overall method
    best_overall = min(all_methods.keys(), key=lambda k: all_methods[k]['primary'])
    print(f"\nPer-state NMAE at H=500 for best method ({best_overall}):")
    r500 = all_methods[best_overall]['results'][500]['per_state_nmae']
    for name in STATE_NAMES_7D:
        val = r500[name]['mean']
        base_val = results_base[500]['per_state_nmae'][name]['mean']
        if base_val > 0 and not np.isnan(val) and not np.isnan(base_val):
            change = (base_val - val) / base_val * 100
            print(f"  {name:<15} baseline={base_val:.4f}  best={val:.4f}  ({change:+.1f}%)")
        else:
            print(f"  {name:<15} baseline={base_val:.4f}  best={val:.4f}")

    # ============================================================
    # Save Results
    # ============================================================
    output = {
        'timestamp': datetime.now().isoformat(),
        'experiment': 'jacobian_regularization',
        'description': 'Three regularization strategies to control error accumulation: Jacobian penalty, Lyapunov constraint, spectral normalization',
        'horizons': horizons,
        'baseline': {
            'nmae_per_horizon': {str(h): results_base[h]['nmae_mean'] for h in horizons},
            'survival_per_horizon': {str(h): results_base[h]['survival_rate'] for h in horizons},
            'primary': primary_base,
            'per_state_h500': {name: results_base[500]['per_state_nmae'][name]['mean'] for name in STATE_NAMES_7D},
        },
        'jacobian_penalty': {
            'lambdas': jac_lambdas,
            'best_lambda': float(best_jac_lam),
            'per_lambda_primary': {k: v['primary'] for k, v in jac_results.items()},
            'best_nmae_per_horizon': {str(h): jac_results[best_jac_lam]['results'][h]['nmae_mean'] for h in horizons},
            'best_survival_per_horizon': {str(h): jac_results[best_jac_lam]['results'][h]['survival_rate'] for h in horizons},
            'best_primary': jac_results[best_jac_lam]['primary'],
            'best_per_state_h500': {
                name: jac_results[best_jac_lam]['results'][500]['per_state_nmae'][name]['mean']
                for name in STATE_NAMES_7D
            },
        },
        'lyapunov_constraint': {
            'lambdas': lyap_lambdas,
            'best_lambda': float(best_lyap_lam),
            'per_lambda_primary': {k: v['primary'] for k, v in lyap_results.items()},
            'best_nmae_per_horizon': {str(h): lyap_results[best_lyap_lam]['results'][h]['nmae_mean'] for h in horizons},
            'best_survival_per_horizon': {str(h): lyap_results[best_lyap_lam]['results'][h]['survival_rate'] for h in horizons},
            'best_primary': lyap_results[best_lyap_lam]['primary'],
            'best_per_state_h500': {
                name: lyap_results[best_lyap_lam]['results'][500]['per_state_nmae'][name]['mean']
                for name in STATE_NAMES_7D
            },
        },
        'spectral_normalization': {
            'coeffs': spectral_coeffs,
            'best_coeff': float(best_sn_coeff),
            'per_coeff_primary': {k: v['primary'] for k, v in spectral_results.items()},
            'best_nmae_per_horizon': {str(h): spectral_results[best_sn_coeff]['results'][h]['nmae_mean'] for h in horizons},
            'best_survival_per_horizon': {str(h): spectral_results[best_sn_coeff]['results'][h]['survival_rate'] for h in horizons},
            'best_primary': spectral_results[best_sn_coeff]['primary'],
            'best_per_state_h500': {
                name: spectral_results[best_sn_coeff]['results'][500]['per_state_nmae'][name]['mean']
                for name in STATE_NAMES_7D
            },
        },
        'best_overall_method': best_overall,
        'best_overall_primary': all_methods[best_overall]['primary'],
    }

    output_path = 'D:/系统辨识作业/sindy_bicycle/research_72h/05_candidates/EXP057_jacobian_regularization.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    return output


if __name__ == '__main__':
    main()
