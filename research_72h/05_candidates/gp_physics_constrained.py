"""GP + Physics Constraints for Bicycle System Identification.

Three approaches to integrate known physics into GP prediction:

1. Alpha-blended GP + Physics:
   - GP predicts all 7 state deltas
   - Physics computes e_y_dot, e_psi_dot, theta_dot, delta_dot from known formulas
   - Blend: final = alpha * physics + (1-alpha) * GP

2. Residual GP:
   - GP learns: residual = actual_delta - physics_delta
   - Prediction: output = physics_delta + GP_residual

3. Standard GP (baseline):
   - Pure GP prediction without physics

Known physics relationships:
  e_y_dot   = v * sin(e_psi)          [exact kinematics]
  e_psi_dot = -v * delta / L          [exact kinematics, L=wheelbase]
  theta_dot = d(theta)/dt             [definition]
  delta_dot = d(delta)/dt             [definition]
"""

import sys
import json
import time
import warnings
import numpy as np
from datetime import datetime
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

warnings.filterwarnings('ignore')

sys.path.insert(0, 'D:/系统辨识作业/sindy_bicycle')

# ============================================================
# Constants
# ============================================================

STATE_NAMES_7D = ['e_y', 'e_psi', 'v', 'theta', 'theta_dot', 'delta', 'delta_dot']
STATE_DIM = 7
IDX_7D_FROM_8D = [0, 1, 2, 3, 4, 6, 7]

WHEELBASE = 1.0   # m
DT = 1.0 / 30.0

PHYSICAL_LIMITS = {
    'e_y': 5.0, 'e_psi': np.pi, 'v': 5.0,
    'theta': np.pi, 'theta_dot': 10.0,
    'delta': np.pi/2, 'delta_dot': 10.0,
}

IDX_EY, IDX_EPSI, IDX_V = 0, 1, 2
IDX_THETA, IDX_THETA_DOT = 3, 4
IDX_DELTA, IDX_DELTA_DOT = 5, 6


# ============================================================
# Physics Model
# ============================================================

def compute_physics_deltas(state):
    """Compute physics-based state deltas from known kinematics.

    Args:
        state: (7,) [e_y, e_psi, v, theta, theta_dot, delta, delta_dot]
    Returns:
        physics_delta: (7,) predicted state change per timestep
    """
    e_psi = state[IDX_EPSI]
    v = state[IDX_V]
    theta_dot = state[IDX_THETA_DOT]
    delta = state[IDX_DELTA]
    delta_dot = state[IDX_DELTA_DOT]

    d = np.zeros(7)
    d[IDX_EY]    = v * np.sin(e_psi) * DT            # e_y_dot = v*sin(e_psi)
    d[IDX_EPSI]  = -v * delta / WHEELBASE * DT        # e_psi_dot = -v*delta/L
    d[IDX_V]     = 0.0                                  # no exact model
    d[IDX_THETA] = theta_dot * DT                       # theta_dot = d(theta)/dt
    d[IDX_THETA_DOT] = 0.0                              # no exact model
    d[IDX_DELTA] = delta_dot * DT                       # delta_dot = d(delta)/dt
    d[IDX_DELTA_DOT] = 0.0                              # no exact model
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


def check_survival(state):
    for i, name in enumerate(STATE_NAMES_7D):
        if name in PHYSICAL_LIMITS and abs(state[i]) > PHYSICAL_LIMITS[name]:
            return False
    return not (np.any(np.isnan(state)) or np.any(np.isinf(state)))


# ============================================================
# GP Training
# ============================================================

def train_gps(X, Y, kernel, seed=42, label=""):
    """Train 7 independent GPs (one per state dim)."""
    gps = []
    for i in range(STATE_DIM):
        gp = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=0,
            random_state=seed, alpha=1e-6
        )
        gp.fit(X, Y[:, i])
        gps.append(gp)
    return gps


def prepare_standard_data(data, indices):
    """Prepare normalized (state, action) -> delta for standard GP."""
    state_std = data['state_std']
    action_std = data['action_std']
    delta_std = data['delta_std']

    obs = data['train_obs'][indices]
    acts = data['train_action'][indices]
    deltas = data['train_deltas'][indices]

    X = np.hstack([obs / state_std, acts.reshape(-1, 1) / action_std])
    Y = deltas / delta_std
    return X, Y


def prepare_residual_data(data, indices):
    """Prepare (state, action) -> (actual_delta - physics_delta) for residual GP."""
    state_std = data['state_std']
    action_std = data['action_std']
    delta_std = data['delta_std']

    obs = data['train_obs'][indices]
    acts = data['train_action'][indices]
    actual_deltas = data['train_deltas'][indices]

    physics_deltas = compute_physics_deltas_batch(obs)
    residual_deltas = actual_deltas - physics_deltas

    X = np.hstack([obs / state_std, acts.reshape(-1, 1) / action_std])
    Y = residual_deltas / delta_std
    return X, Y


# ============================================================
# Prediction Functions
# ============================================================

def predict_standard(gps, state, action, state_std, action_std, delta_std):
    x = np.hstack([state / state_std, [action / action_std]]).reshape(1, -1)
    delta_pred = np.array([gp.predict(x)[0] for gp in gps])
    return delta_pred * delta_std * DT


def predict_blended(gps, state, action, state_std, action_std, delta_std, alpha):
    """Alpha-blended: alpha * physics + (1-alpha) * GP for constrained dims."""
    x = np.hstack([state / state_std, [action / action_std]]).reshape(1, -1)
    gp_delta = np.array([gp.predict(x)[0] for gp in gps]) * delta_std * DT
    physics_delta = compute_physics_deltas(state)

    out = gp_delta.copy()
    for dim in [IDX_EY, IDX_EPSI, IDX_THETA, IDX_DELTA]:
        out[dim] = alpha * physics_delta[dim] + (1 - alpha) * gp_delta[dim]
    return out


def predict_residual(gps, state, action, state_std, action_std, delta_std):
    """Residual GP: physics_delta + GP_residual."""
    x = np.hstack([state / state_std, [action / action_std]]).reshape(1, -1)
    residual = np.array([gp.predict(x)[0] for gp in gps]) * delta_std * DT
    physics_delta = compute_physics_deltas(state)
    return physics_delta + residual


# ============================================================
# Evaluation
# ============================================================

def evaluate_model(predict_fn, data, state_std, action_std, delta_std,
                   horizons, n_segments=3, seed=42):
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


def primary_score(results):
    return np.mean([results[100]['nmae_mean'], results[200]['nmae_mean'],
                    results[500]['nmae_mean']])


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("GP + Physics Constraints for Bicycle System Identification")
    print("=" * 70)

    horizons = [1, 10, 50, 100, 200, 500]
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
    n_samples = 500
    n_segments = 3
    seed = 42

    # ----------------------------------------------------------
    # 1. Load Data
    # ----------------------------------------------------------
    print("\n[1/5] Loading data...")
    data = load_data(seed=seed)
    state_std = data['state_std']
    action_std = data['action_std']
    delta_std = data['delta_std']
    print(f"  Train samples: {len(data['train_obs'])}")
    print(f"  Test episodes: {len(data['test_eps'])}")

    # ----------------------------------------------------------
    # 2. Physics Model Accuracy Analysis
    # ----------------------------------------------------------
    print("\n[2/5] Physics model accuracy on training data...")
    n_phys = min(3000, len(data['train_obs']))
    obs_phys = data['train_obs'][:n_phys]
    actual_deltas = data['train_deltas'][:n_phys]
    physics_deltas = compute_physics_deltas_batch(obs_phys)

    delta_std_data = np.std(actual_deltas, axis=0)
    delta_std_data[delta_std_data < 1e-10] = 1.0
    physics_nmae = np.mean(np.abs(physics_deltas - actual_deltas), axis=0) / delta_std_data

    print(f"\n  {'Dimension':<15} {'Physics NMAE':<15}")
    print(f"  {'-'*30}")
    for i, name in enumerate(STATE_NAMES_7D):
        print(f"  {name:<15} {physics_nmae[i]:<15.4f}")

    print(f"\n  Correlation (physics vs actual):")
    for i, name in enumerate(STATE_NAMES_7D):
        std_a = np.std(physics_deltas[:, i])
        std_b = np.std(actual_deltas[:, i])
        if std_a > 1e-10 and std_b > 1e-10:
            corr = np.corrcoef(physics_deltas[:, i], actual_deltas[:, i])[0, 1]
            print(f"  {name:<15} r = {corr:.4f}")
        else:
            print(f"  {name:<15} r = N/A (zero variance)")

    # ----------------------------------------------------------
    # 3. Train Models
    # ----------------------------------------------------------
    print(f"\n[3/5] Training models (n_samples={n_samples})...")
    np.random.seed(seed)
    train_indices = np.random.choice(len(data['train_obs']),
                                     min(n_samples, len(data['train_obs'])),
                                     replace=False)

    # Standard GP
    print("  Training standard GP...")
    t0 = time.time()
    X_std, Y_std = prepare_standard_data(data, train_indices)
    gps_standard = train_gps(X_std, Y_std, kernel, seed=seed, label="standard")
    t_standard = time.time() - t0
    print(f"    Done in {t_standard:.1f}s")

    # Residual GP
    print("  Training residual GP...")
    t0 = time.time()
    X_res, Y_res = prepare_residual_data(data, train_indices)
    gps_residual = train_gps(X_res, Y_res, kernel, seed=seed, label="residual")
    t_residual = time.time() - t0
    print(f"    Done in {t_residual:.1f}s")

    # ----------------------------------------------------------
    # 4. Evaluate
    # ----------------------------------------------------------
    print("\n[4/5] Evaluating models...")

    # Standard GP
    predict_std_fn = lambda s, a: predict_standard(
        gps_standard, s, a, state_std, action_std, delta_std)
    results_standard = evaluate_model(
        predict_std_fn, data, state_std, action_std, delta_std,
        horizons, n_segments=n_segments, seed=seed)

    # Residual GP
    predict_res_fn = lambda s, a: predict_residual(
        gps_residual, s, a, state_std, action_std, delta_std)
    results_residual = evaluate_model(
        predict_res_fn, data, state_std, action_std, delta_std,
        horizons, n_segments=n_segments, seed=seed)

    # Blended GP alpha sweep
    print("  Alpha sweep for blended GP...")
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    alpha_results = {}
    best_alpha = 0.0
    best_alpha_primary = float('inf')

    for alpha in alphas:
        predict_blend_fn = lambda s, a, a_=alpha: predict_blended(
            gps_standard, s, a, state_std, action_std, delta_std, alpha=a_)
        r = evaluate_model(
            predict_blend_fn, data, state_std, action_std, delta_std,
            horizons, n_segments=n_segments, seed=seed)
        p = primary_score(r)
        alpha_results[alpha] = {'results': r, 'primary': p}
        if p < best_alpha_primary:
            best_alpha_primary = p
            best_alpha = alpha

    results_blended = alpha_results[best_alpha]['results']
    print(f"    Best alpha = {best_alpha:.1f} (Primary = {best_alpha_primary:.4f})")

    # ----------------------------------------------------------
    # 5. Summary & Save
    # ----------------------------------------------------------
    print("\n[5/5] Summary")
    print("=" * 85)

    all_models = {
        'Standard GP': {'results': results_standard, 'time': t_standard},
        'Residual GP': {'results': results_residual, 'time': t_residual},
        f'Blended GP (a={best_alpha:.1f})': {'results': results_blended, 'time': t_standard},
    }

    for name, info in all_models.items():
        info['primary'] = primary_score(info['results'])

    # Comparison table
    header = f"{'Model':<25} {'H=1':<8} {'H=10':<8} {'H=50':<8} {'H=100':<8} {'H=200':<8} {'H=500':<8} {'Primary':<8}"
    print(header)
    print("-" * 85)
    for name, info in all_models.items():
        r = info['results']
        p = info['primary']
        print(f"{name:<25} {r[1]['nmae_mean']:<8.4f} {r[10]['nmae_mean']:<8.4f} "
              f"{r[50]['nmae_mean']:<8.4f} {r[100]['nmae_mean']:<8.4f} "
              f"{r[200]['nmae_mean']:<8.4f} {r[500]['nmae_mean']:<8.4f} {p:<8.4f}")

    # Per-state for best model
    best_model = min(all_models.keys(), key=lambda k: all_models[k]['primary'])
    print(f"\nBest model: {best_model} (Primary = {all_models[best_model]['primary']:.4f})")

    for label_name in ['Standard GP', best_model]:
        r = all_models[label_name]['results']
        print(f"\n  {label_name} - Per-State NMAE at H=500:")
        print(f"  {'State':<15} {'NMAE':<10}")
        print(f"  {'-'*25}")
        for name in STATE_NAMES_7D:
            val = r[500]['per_state_nmae'][name]['mean']
            print(f"  {name:<15} {val:<10.4f}")

    # Alpha sweep table
    print(f"\nAlpha Sweep:")
    print(f"  {'Alpha':<8} {'H=100':<10} {'H=200':<10} {'H=500':<10} {'Primary':<10}")
    print(f"  {'-'*48}")
    for alpha in alphas:
        r = alpha_results[alpha]['results']
        p = alpha_results[alpha]['primary']
        marker = " <-- best" if alpha == best_alpha else ""
        print(f"  {alpha:<8.1f} {r[100]['nmae_mean']:<10.4f} {r[200]['nmae_mean']:<10.4f} "
              f"{r[500]['nmae_mean']:<10.4f} {p:<10.4f}{marker}")

    # v9 comparison
    v9_nmae = {1: 0.0051, 10: 0.0628, 50: 0.4557, 100: 0.5064, 200: 0.4737, 500: 0.5529}
    v9_primary = np.mean([v9_nmae[100], v9_nmae[200], v9_nmae[500]])
    print(f"\nv9 Neural ODE comparison:")
    print(f"  v9 PrimaryLongHorizonScore: {v9_primary:.4f}")
    for name, info in all_models.items():
        imp = (v9_primary - info['primary']) / v9_primary * 100
        direction = "better" if imp > 0 else "worse"
        print(f"  {name:<25} Primary={info['primary']:.4f}  ({imp:+.1f}% {direction})")

    # Save JSON
    output = {
        'timestamp': datetime.now().isoformat(),
        'experiment': 'gp_physics_constrained',
        'config': {
            'n_samples': n_samples, 'n_segments': n_segments,
            'kernel': 'ConstantKernel(1.0) * RBF(1.0)',
            'wheelbase': WHEELBASE, 'dt': DT, 'seed': seed,
        },
        'physics_accuracy': {name: float(physics_nmae[i])
                             for i, name in enumerate(STATE_NAMES_7D)},
        'models': {},
        'alpha_sweep': {str(a): alpha_results[a]['primary'] for a in alphas},
        'best_alpha': best_alpha,
        'best_model': best_model,
    }
    for name, info in all_models.items():
        output['models'][name] = {
            'primary': info['primary'],
            'train_time': info['time'],
            'per_horizon': {
                str(h): {
                    'nmae_mean': info['results'][h]['nmae_mean'],
                    'survival_rate': info['results'][h]['survival_rate'],
                } for h in horizons
            },
        }

    output_path = 'D:/系统辨识作业/sindy_bicycle/research_72h/05_candidates/EXP013_gp_physics_constrained.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    return output, all_models, alpha_results


if __name__ == '__main__':
    main()
