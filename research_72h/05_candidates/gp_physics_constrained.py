"""GP + Physics Constraints for Bicycle System Identification.

Three approaches to integrate known physics into GP prediction:

1. Post-prediction physics correction (Alpha blending):
   - GP predicts all 7 state deltas
   - Physics model computes e_y_dot and e_psi_dot from known kinematics
   - Blend GP and physics using alpha parameter

2. Physics-augmented features:
   - Add physics-derived features to GP input (v*sin(e_psi), -v*delta/L)
   - GP learns residuals on top of physics priors

3. Physics residual GP:
   - GP predicts (delta - physics_delta) instead of raw delta
   - At prediction time: output = physics_delta + GP_residual

Known physics relationships:
  e_y_dot  = v * sin(e_psi)           [exact kinematics]
  e_psi_dot = -v * delta / L          [exact kinematics, L=wheelbase]
  theta_dot = d(theta)/dt             [definition]
  delta_dot = d(delta)/dt             [definition]
"""

import sys
import json
import time
import numpy as np
from datetime import datetime
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel

sys.path.insert(0, 'D:/系统辨识作业/sindy_bicycle')

# ============================================================
# Constants
# ============================================================

STATE_NAMES_7D = ['e_y', 'e_psi', 'v', 'theta', 'theta_dot', 'delta', 'delta_dot']
STATE_DIM = 7
ACTION_DIM = 1
IDX_7D_FROM_8D = [0, 1, 2, 3, 4, 6, 7]

# Physical parameters
WHEELBASE = 1.0   # m (matches path_tracking_env.py)
DT = 1.0 / 30.0   # simulation timestep

PHYSICAL_LIMITS = {
    'e_y': 5.0, 'e_psi': np.pi, 'v': 5.0,
    'theta': np.pi, 'theta_dot': 10.0,
    'delta': np.pi/2, 'delta_dot': 10.0,
}

# Index mapping for readability
IDX_EY = 0
IDX_EPSI = 1
IDX_V = 2
IDX_THETA = 3
IDX_THETA_DOT = 4
IDX_DELTA = 5
IDX_DELTA_DOT = 6


# ============================================================
# Physics Model
# ============================================================

def compute_physics_deltas(state, action):
    """Compute physics-based state deltas from known kinematics.

    Args:
        state: (7,) array [e_y, e_psi, v, theta, theta_dot, delta, delta_dot]
        action: scalar steering torque

    Returns:
        physics_delta: (7,) array of predicted state changes per timestep
    """
    e_y = state[IDX_EY]
    e_psi = state[IDX_EPSI]
    v = state[IDX_V]
    theta = state[IDX_THETA]
    theta_dot = state[IDX_THETA_DOT]
    delta = state[IDX_DELTA]
    delta_dot = state[IDX_DELTA_DOT]

    physics_delta = np.zeros(7)

    # e_y_dot = v * sin(e_psi) [exact kinematics]
    physics_delta[IDX_EY] = v * np.sin(e_psi) * DT

    # e_psi_dot = -v * delta / L [exact kinematics]
    physics_delta[IDX_EPSI] = -v * delta / WHEELBASE * DT

    # v: no exact physics model (involves drag, acceleration)
    # Leave as 0 -- GP will learn this
    physics_delta[IDX_V] = 0.0

    # theta_dot = d(theta)/dt [definition]
    # Use current theta_dot as the derivative
    physics_delta[IDX_THETA] = theta_dot * DT

    # theta_ddot: complex dynamics (gravity, steering coupling)
    # Leave as 0 -- GP will learn this
    physics_delta[IDX_THETA_DOT] = 0.0

    # delta_dot = d(delta)/dt [definition]
    physics_delta[IDX_DELTA] = delta_dot * DT

    # delta_ddot: complex dynamics (spring, damping, control)
    # Leave as 0 -- GP will learn this
    physics_delta[IDX_DELTA_DOT] = 0.0

    return physics_delta


def compute_physics_deltas_batch(states, actions):
    """Batch version of physics delta computation."""
    n = len(states)
    physics_deltas = np.zeros((n, 7))
    for i in range(n):
        physics_deltas[i] = compute_physics_deltas(states[i], actions[i])
    return physics_deltas


# ============================================================
# Data Loading
# ============================================================

def load_data(seed=42):
    """Load and split episode data."""
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
        'train_obs': train_obs,
        'train_action': train_action,
        'train_deltas': train_deltas,
    }


# ============================================================
# Survival Check
# ============================================================

def check_survival(state):
    """Check if state is within physical limits."""
    for i, name in enumerate(STATE_NAMES_7D):
        if name in PHYSICAL_LIMITS:
            if abs(state[i]) > PHYSICAL_LIMITS[name]:
                return False
    return not (np.any(np.isnan(state)) or np.any(np.isinf(state)))


# ============================================================
# GP Training
# ============================================================

def train_standard_gps(data, kernel, n_samples=5000, seed=42):
    """Train standard GP models (one per state dimension)."""
    np.random.seed(seed)
    state_std = data['state_std']
    action_std = data['action_std']
    delta_std = data['delta_std']

    n_total = len(data['train_obs'])
    indices = np.random.choice(n_total, min(n_samples, n_total), replace=False)

    X = np.hstack([
        data['train_obs'][indices] / state_std,
        data['train_action'][indices].reshape(-1, 1) / action_std
    ])
    Y = data['train_deltas'][indices] / delta_std

    gps = []
    for i in range(STATE_DIM):
        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=3,
            random_state=seed,
            alpha=1e-6
        )
        gp.fit(X, Y[:, i])
        gps.append(gp)

    return gps


def train_physics_augmented_gps(data, kernel, n_samples=5000, seed=42):
    """Train GP with physics-augmented features.

    Adds 2 physics-derived features to the input:
      - v * sin(e_psi)  (physics prediction for e_y_dot)
      - -v * delta / L  (physics prediction for e_psi_dot)

    This gives the GP explicit knowledge of the physics relationships.
    """
    np.random.seed(seed)
    state_std = data['state_std']
    action_std = data['action_std']
    delta_std = data['delta_std']

    n_total = len(data['train_obs'])
    indices = np.random.choice(n_total, min(n_samples, n_total), replace=False)

    obs = data['train_obs'][indices]
    acts = data['train_action'][indices]

    # Compute physics features
    phys_feat1 = (obs[:, IDX_V] * np.sin(obs[:, IDX_EPSI])).reshape(-1, 1)
    phys_feat2 = (-obs[:, IDX_V] * obs[:, IDX_DELTA] / WHEELBASE).reshape(-1, 1)

    # Normalize physics features
    phys_std1 = np.std(phys_feat1)
    phys_std1 = phys_std1 if phys_std1 > 1e-10 else 1.0
    phys_std2 = np.std(phys_feat2)
    phys_std2 = phys_std2 if phys_std2 > 1e-10 else 1.0

    X = np.hstack([
        obs / state_std,
        acts.reshape(-1, 1) / action_std,
        phys_feat1 / phys_std1,
        phys_feat2 / phys_std2,
    ])
    Y = data['train_deltas'][indices] / delta_std

    gps = []
    for i in range(STATE_DIM):
        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=3,
            random_state=seed,
            alpha=1e-6
        )
        gp.fit(X, Y[:, i])
        gps.append(gp)

    phys_params = {'phys_std1': phys_std1, 'phys_std2': phys_std2}
    return gps, phys_params


def train_residual_gps(data, kernel, n_samples=5000, seed=42):
    """Train GP on physics residuals.

    GP learns: residual = actual_delta - physics_delta
    At prediction: output = physics_delta + GP_residual

    For dimensions with no physics model (v, theta_dot, delta_dot),
    GP learns the full delta directly.
    """
    np.random.seed(seed)
    state_std = data['state_std']
    action_std = data['action_std']
    delta_std = data['delta_std']

    n_total = len(data['train_obs'])
    indices = np.random.choice(n_total, min(n_samples, n_total), replace=False)

    obs = data['train_obs'][indices]
    acts = data['train_action'][indices]
    actual_deltas = data['train_deltas'][indices]

    # Compute physics deltas for all training samples
    physics_deltas = compute_physics_deltas_batch(obs, acts)

    # Residual = actual - physics
    residual_deltas = actual_deltas - physics_deltas

    X = np.hstack([obs / state_std, acts.reshape(-1, 1) / action_std])
    Y = residual_deltas / delta_std

    gps = []
    for i in range(STATE_DIM):
        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=3,
            random_state=seed,
            alpha=1e-6
        )
        gp.fit(X, Y[:, i])
        gps.append(gp)

    return gps


# ============================================================
# GP Prediction with Physics Constraints
# ============================================================

def predict_standard(gps, state, action, state_std, action_std, delta_std):
    """Standard GP prediction."""
    x = np.hstack([state / state_std, [action / action_std]]).reshape(1, -1)
    delta_pred = np.array([gp.predict(x)[0] for gp in gps])
    return delta_pred * delta_std * DT


def predict_physics_blended(gps, state, action, state_std, action_std, delta_std,
                            alpha=0.5):
    """Alpha-blended GP + Physics prediction.

    For constrained dimensions (e_y, e_psi, theta, delta):
      final = alpha * physics + (1-alpha) * gp

    For unconstrained dimensions (v, theta_dot, delta_dot):
      final = gp (no physics model available)

    Args:
        alpha: blending weight for physics (0=GP only, 1=physics only)
    """
    x = np.hstack([state / state_std, [action / action_std]]).reshape(1, -1)
    gp_delta = np.array([gp.predict(x)[0] for gp in gps]) * delta_std * DT

    physics_delta = compute_physics_deltas(state, action)

    final_delta = gp_delta.copy()

    # Blend constrained dimensions
    for dim in [IDX_EY, IDX_EPSI, IDX_THETA, IDX_DELTA]:
        final_delta[dim] = alpha * physics_delta[dim] + (1 - alpha) * gp_delta[dim]

    return final_delta


def predict_physics_augmented(gps, state, action, state_std, action_std, delta_std,
                               phys_params):
    """GP prediction with physics-augmented features."""
    phys_feat1 = state[IDX_V] * np.sin(state[IDX_EPSI])
    phys_feat2 = -state[IDX_V] * state[IDX_DELTA] / WHEELBASE

    x = np.hstack([
        state / state_std,
        [action / action_std],
        [phys_feat1 / phys_params['phys_std1']],
        [phys_feat2 / phys_params['phys_std2']],
    ]).reshape(1, -1)

    delta_pred = np.array([gp.predict(x)[0] for gp in gps])
    return delta_pred * delta_std * DT


def predict_residual(gps, state, action, state_std, action_std, delta_std):
    """Residual GP prediction: physics + GP_residual."""
    x = np.hstack([state / state_std, [action / action_std]]).reshape(1, -1)
    residual_delta = np.array([gp.predict(x)[0] for gp in gps]) * delta_std * DT

    physics_delta = compute_physics_deltas(state, action)

    return physics_delta + residual_delta


# ============================================================
# Evaluation
# ============================================================

def evaluate_model(predict_fn, data, state_std, action_std, delta_std,
                   horizons, n_segments=5, seed=42, label="Model"):
    """Evaluate a model on multi-step prediction.

    Args:
        predict_fn: callable(state, action) -> delta (7,)
    """
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
                    dsdt = predict_fn(s_cur, actions_seg[step])
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
                name: {'mean': float(np.nanmean(per_state_nmae[name]))
                       if per_state_nmae[name] else float('nan')}
                for name in STATE_NAMES_7D
            },
        }

    return results


def format_results_table(results, horizons, label):
    """Format results as a printable table."""
    lines = [f"\n{label}:", f"{'Horizon':<10} {'NMAE':<12} {'Survival':<12}"]
    lines.append("-" * 34)
    for h in horizons:
        r = results[h]
        lines.append(f"H={h:<7} {r['nmae_mean']:<12.4f} {r['survival_rate']:<12.2%}")
    primary = np.mean([results[100]['nmae_mean'], results[200]['nmae_mean'],
                       results[500]['nmae_mean']])
    lines.append(f"PrimaryLongHorizonScore: {primary:.4f}")
    return "\n".join(lines), primary


def format_per_state_table(results, horizons, label):
    """Format per-state NMAE table."""
    lines = [f"\n{label} - Per-State NMAE:"]
    header = f"{'Horizon':<8}" + "".join(f"{name:<12}" for name in STATE_NAMES_7D)
    lines.append(header)
    lines.append("-" * (8 + 12 * STATE_DIM))
    for h in horizons:
        r = results[h]
        row = f"H={h:<5}"
        for name in STATE_NAMES_7D:
            val = r['per_state_nmae'][name]['mean']
            row += f"{val:<12.4f}"
        lines.append(row)
    return "\n".join(lines)


# ============================================================
# Alpha Sweep
# ============================================================

def sweep_alpha(gps, data, state_std, action_std, delta_std,
                horizons, alphas, n_segments=5, seed=42):
    """Sweep alpha values to find optimal blending weight."""
    best_alpha = 0.0
    best_primary = float('inf')
    alpha_results = {}

    for alpha in alphas:
        predict_fn = lambda s, a, alpha=alpha: predict_physics_blended(
            gps, s, a, state_std, action_std, delta_std, alpha=alpha
        )
        results = evaluate_model(
            predict_fn, data, state_std, action_std, delta_std,
            horizons, n_segments=n_segments, seed=seed,
            label=f"Alpha={alpha:.2f}"
        )
        primary = np.mean([results[100]['nmae_mean'],
                           results[200]['nmae_mean'],
                           results[500]['nmae_mean']])
        alpha_results[alpha] = {'results': results, 'primary': primary}

        if primary < best_primary:
            best_primary = primary
            best_alpha = alpha

    return best_alpha, best_primary, alpha_results


# ============================================================
# Main Experiment
# ============================================================

def main():
    print("=" * 70)
    print("GP + Physics Constraints for Bicycle System Identification")
    print("=" * 70)

    horizons = [1, 10, 50, 100, 200, 500]
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
    n_samples = 5000
    n_segments = 5
    seed = 42

    # ----------------------------------------------------------
    # 1. Load Data
    # ----------------------------------------------------------
    print("\n[1/6] Loading data...")
    data = load_data(seed=seed)
    state_std = data['state_std']
    action_std = data['action_std']
    delta_std = data['delta_std']
    print(f"  Train samples: {len(data['train_obs'])}")
    print(f"  Test episodes: {len(data['test_eps'])}")
    print(f"  State std: {state_std}")
    print(f"  Delta std: {delta_std}")

    # ----------------------------------------------------------
    # 2. Analyze Physics Accuracy on Training Data
    # ----------------------------------------------------------
    print("\n[2/6] Analyzing physics model accuracy on training data...")
    train_obs = data['train_obs'][:5000]
    train_acts = data['train_action'][:5000].flatten()
    train_deltas = data['train_deltas'][:5000]

    physics_deltas = compute_physics_deltas_batch(train_obs, train_acts)

    # Compute physics error per dimension
    physics_errors = np.abs(physics_deltas - train_deltas)
    physics_nmae = np.mean(physics_errors, axis=0) / (np.std(train_deltas, axis=0) + 1e-10)

    print(f"\n  Physics Model NMAE per dimension (on training data):")
    print(f"  {'Dimension':<15} {'Physics NMAE':<15} {'GP-Relevant?':<15}")
    print(f"  {'-'*45}")
    for i, name in enumerate(STATE_NAMES_7D):
        relevant = "YES" if i in [IDX_EY, IDX_EPSI, IDX_THETA, IDX_DELTA] else "NO"
        print(f"  {name:<15} {physics_nmae[i]:<15.4f} {relevant:<15}")

    # Correlation between physics and actual deltas
    print(f"\n  Correlation between physics and actual deltas:")
    for i, name in enumerate(STATE_NAMES_7D):
        corr = np.corrcoef(physics_deltas[:, i], train_deltas[:, i])[0, 1]
        print(f"  {name:<15} r = {corr:.4f}")

    # ----------------------------------------------------------
    # 3. Train Models
    # ----------------------------------------------------------
    print("\n[3/6] Training models...")

    # 3a. Standard GP (baseline)
    print("\n  Training standard GP...")
    t0 = time.time()
    gps_standard = train_standard_gps(data, kernel, n_samples=n_samples, seed=seed)
    t_standard = time.time() - t0
    print(f"  Done in {t_standard:.1f}s")

    # 3b. Physics-augmented GP
    print("\n  Training physics-augmented GP...")
    t0 = time.time()
    gps_augmented, phys_params = train_physics_augmented_gps(
        data, kernel, n_samples=n_samples, seed=seed
    )
    t_augmented = time.time() - t0
    print(f"  Done in {t_augmented:.1f}s")

    # 3c. Residual GP
    print("\n  Training residual GP...")
    t0 = time.time()
    gps_residual = train_residual_gps(data, kernel, n_samples=n_samples, seed=seed)
    t_residual = time.time() - t0
    print(f"  Done in {t_residual:.1f}s")

    # ----------------------------------------------------------
    # 4. Evaluate Models
    # ----------------------------------------------------------
    print("\n[4/6] Evaluating models...")

    # Standard GP
    predict_std_fn = lambda s, a: predict_standard(
        gps_standard, s, a, state_std, action_std, delta_std
    )
    results_standard = evaluate_model(
        predict_std_fn, data, state_std, action_std, delta_std,
        horizons, n_segments=n_segments, seed=seed, label="Standard GP"
    )

    # Physics-augmented GP
    predict_aug_fn = lambda s, a: predict_physics_augmented(
        gps_augmented, s, a, state_std, action_std, delta_std, phys_params
    )
    results_augmented = evaluate_model(
        predict_aug_fn, data, state_std, action_std, delta_std,
        horizons, n_segments=n_segments, seed=seed, label="Physics-Augmented GP"
    )

    # Residual GP
    predict_res_fn = lambda s, a: predict_residual(
        gps_residual, s, a, state_std, action_std, delta_std
    )
    results_residual = evaluate_model(
        predict_res_fn, data, state_std, action_std, delta_std,
        horizons, n_segments=n_segments, seed=seed, label="Residual GP"
    )

    # Alpha-blended GP (sweep alpha)
    print("\n  Sweeping alpha for blended GP...")
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    best_alpha, best_alpha_primary, alpha_results = sweep_alpha(
        gps_standard, data, state_std, action_std, delta_std,
        horizons, alphas, n_segments=n_segments, seed=seed
    )
    results_blended = alpha_results[best_alpha]['results']
    print(f"  Best alpha = {best_alpha:.2f} (Primary = {best_alpha_primary:.4f})")

    # ----------------------------------------------------------
    # 5. Summary
    # ----------------------------------------------------------
    print("\n[5/6] Generating summary...")

    all_models = {
        'Standard GP': {'results': results_standard, 'time': t_standard},
        'Physics-Augmented GP': {'results': results_augmented, 'time': t_augmented},
        'Residual GP': {'results': results_residual, 'time': t_residual},
        f'Blended GP (alpha={best_alpha:.1f})': {'results': results_blended, 'time': t_standard},
    }

    # Compute primary scores
    for name, info in all_models.items():
        r = info['results']
        info['primary'] = np.mean([r[100]['nmae_mean'], r[200]['nmae_mean'],
                                   r[500]['nmae_mean']])

    # Print comparison table
    print("\n" + "=" * 90)
    print("COMPARISON: All Models")
    print("=" * 90)

    header = f"{'Model':<28} {'H=1':<8} {'H=10':<8} {'H=50':<8} {'H=100':<8} {'H=200':<8} {'H=500':<8} {'Primary':<8}"
    print(header)
    print("-" * 90)

    for name, info in all_models.items():
        r = info['results']
        primary = info['primary']
        print(f"{name:<28} {r[1]['nmae_mean']:<8.4f} {r[10]['nmae_mean']:<8.4f} "
              f"{r[50]['nmae_mean']:<8.4f} {r[100]['nmae_mean']:<8.4f} "
              f"{r[200]['nmae_mean']:<8.4f} {r[500]['nmae_mean']:<8.4f} {primary:<8.4f}")

    # Per-state analysis for best model
    best_model_name = min(all_models.keys(), key=lambda k: all_models[k]['primary'])
    print(f"\nBest model: {best_model_name} (Primary = {all_models[best_model_name]['primary']:.4f})")

    print(format_per_state_table(results_standard, [100, 200, 500], "Standard GP"))
    print(format_per_state_table(all_models[best_model_name]['results'], [100, 200, 500],
                                  best_model_name))

    # Alpha sweep results
    print(f"\nAlpha Sweep Results:")
    print(f"{'Alpha':<8} {'H=100':<10} {'H=200':<10} {'H=500':<10} {'Primary':<10}")
    print("-" * 48)
    for alpha in alphas:
        r = alpha_results[alpha]['results']
        primary = alpha_results[alpha]['primary']
        marker = " <-- best" if alpha == best_alpha else ""
        print(f"{alpha:<8.1f} {r[100]['nmae_mean']:<10.4f} {r[200]['nmae_mean']:<10.4f} "
              f"{r[500]['nmae_mean']:<10.4f} {primary:<10.4f}{marker}")

    # Comparison with v9 baseline
    print(f"\nComparison with v9 Neural ODE baseline:")
    v9_nmae = {1: 0.0051, 10: 0.0628, 50: 0.4557, 100: 0.5064, 200: 0.4737, 500: 0.5529}
    v9_primary = np.mean([v9_nmae[100], v9_nmae[200], v9_nmae[500]])
    print(f"  v9 PrimaryLongHorizonScore: {v9_primary:.4f}")
    for name, info in all_models.items():
        improvement = (v9_primary - info['primary']) / v9_primary * 100
        direction = "better" if improvement > 0 else "worse"
        print(f"  {name:<28} Primary={info['primary']:.4f}  ({improvement:+.1f}% {direction})")

    # ----------------------------------------------------------
    # 6. Save Results
    # ----------------------------------------------------------
    print("\n[6/6] Saving results...")

    output = {
        'timestamp': datetime.now().isoformat(),
        'run_id': '20260628_175824_neural_ode_72h',
        'experiment': 'gp_physics_constrained',
        'config': {
            'kernel': 'ConstantKernel(1.0) * RBF(length_scale=1.0)',
            'n_samples': n_samples,
            'n_segments': n_segments,
            'seed': seed,
            'wheelbase': WHEELBASE,
            'dt': DT,
        },
        'physics_accuracy': {
            name: float(physics_nmae[i]) for i, name in enumerate(STATE_NAMES_7D)
        },
        'models': {},
        'alpha_sweep': {
            str(alpha): {'primary': alpha_results[alpha]['primary']}
            for alpha in alphas
        },
        'best_alpha': best_alpha,
        'best_model': best_model_name,
    }

    for name, info in all_models.items():
        output['models'][name] = {
            'primary': info['primary'],
            'train_time': info['time'],
            'per_horizon': {
                str(h): {
                    'nmae_mean': info['results'][h]['nmae_mean'],
                    'survival_rate': info['results'][h]['survival_rate'],
                }
                for h in horizons
            },
        }

    output_path = 'D:/系统辨识作业/sindy_bicycle/research_72h/05_candidates/EXP013_gp_physics_constrained.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Results saved to: {output_path}")

    return output, all_models, alpha_results


if __name__ == '__main__':
    output, all_models, alpha_results = main()
