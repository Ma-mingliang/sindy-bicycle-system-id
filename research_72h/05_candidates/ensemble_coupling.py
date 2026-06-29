"""Ensemble of Improved Coupling Networks (256-dim).

Trains 5 Improved Coupling models with different seeds and evaluates:
1. Single model baseline (each seed)
2. Simple average ensemble
3. Weighted average ensemble (weights from validation NMAE)

Best single model primary: 0.4712 (EXP051 xlarge_coupling)
Goal: Beat 0.4712 via ensemble averaging.
"""
import sys
import json
import time
import os
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime

# GPU setup
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Log file setup
LOG_PATH = 'D:/系统辨识作业/sindy_bicycle/research_72h/05_candidates/ensemble_coupling_log.txt'
_log = open(LOG_PATH, 'w', buffering=1)

def log(msg):
    """Write to both stdout and log file with immediate flush."""
    print(msg, flush=True)
    _log.write(msg + '\n')
    _log.flush()

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


# ===================== Model Architecture =====================

class ImprovedCouplingEncoder(nn.Module):
    """Shared encoder with residual connections."""
    def __init__(self, input_dim, coupling_dim=256, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, coupling_dim),
        )
        self.residual = nn.Linear(input_dim, coupling_dim) if input_dim != coupling_dim else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.residual(x)


class ImprovedStateDecoder(nn.Module):
    """State decoder with access to all states."""
    def __init__(self, coupling_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(coupling_dim + STATE_DIM, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.net[-1].bias)
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.01)

    def forward(self, z, all_states):
        x = torch.cat([z, all_states], dim=-1)
        return self.net(x).squeeze(-1)


class ImprovedCouplingNetwork(nn.Module):
    """Improved Coupling Network (256-dim)."""
    def __init__(self, state_dim=7, action_dim=1, coupling_dim=256, hidden=256):
        super().__init__()
        self.encoder = ImprovedCouplingEncoder(
            input_dim=state_dim + action_dim,
            coupling_dim=coupling_dim,
            hidden=hidden
        )
        self.decoders = nn.ModuleList([
            ImprovedStateDecoder(coupling_dim=coupling_dim, hidden=64)
            for _ in range(state_dim)
        ])

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        z = self.encoder(x)
        deltas = []
        for decoder in self.decoders:
            delta_i = decoder(z, s)
            deltas.append(delta_i)
        return torch.stack(deltas, dim=-1)


# ===================== Utilities =====================

def check_survival(state):
    for i, name in enumerate(STATE_NAMES_7D):
        if name in PHYSICAL_LIMITS:
            if abs(state[i]) > PHYSICAL_LIMITS[name]:
                return False
    return not (np.any(np.isnan(state)) or np.any(np.isinf(state)))


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


# ===================== Training =====================

CONFIG = {
    'coupling_dim': 256,
    'hidden': 256,
    'lr': 1e-3,
    'n_epochs': 200,
    'batch_size': 1024,
}


def train_model(data, config, seed):
    """Train a single Improved Coupling Network on GPU."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    state_std = data['state_std']
    action_std = data['action_std']
    delta_std = data['delta_std']
    dt = 1.0 / 30.0

    train_obs = data['train_obs']
    train_action = data['train_action']
    train_deltas = data['train_deltas']

    X = np.hstack([train_obs / state_std, train_action.reshape(-1, 1) / action_std])
    Y = train_deltas / (delta_std * dt)

    X_t = torch.FloatTensor(X).to(DEVICE)
    Y_t = torch.FloatTensor(Y).to(DEVICE)

    ds = torch.utils.data.TensorDataset(X_t, Y_t)
    loader = torch.utils.data.DataLoader(ds, batch_size=config['batch_size'], shuffle=True, pin_memory=False)

    model = ImprovedCouplingNetwork(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        coupling_dim=config['coupling_dim'],
        hidden=config['hidden']
    ).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config['n_epochs'])

    start_time = time.time()
    model.train()

    for epoch in range(config['n_epochs']):
        epoch_loss = 0.0
        n_batches = 0

        for xb, yb in loader:
            s = xb[:, :STATE_DIM]
            a = xb[:, STATE_DIM:]
            pred = model(s, a)
            loss = nn.functional.mse_loss(pred, yb)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        if (epoch + 1) % 50 == 0:
            avg_loss = epoch_loss / max(n_batches, 1)
            elapsed = time.time() - start_time
            log(f"    Epoch {epoch+1}/{config['n_epochs']}: loss={avg_loss:.6f}, time={elapsed:.1f}s")

    model.eval()
    total_time = time.time() - start_time
    log(f"    Training completed in {total_time:.1f}s")
    return model


# ===================== Evaluation =====================

def _rollout_predict(model, s_cur, action_val, state_std, action_std, delta_std, dt):
    """Run one-step prediction, returns next state (numpy)."""
    s_norm = torch.FloatTensor(s_cur / state_std).unsqueeze(0).to(DEVICE)
    a_norm = torch.FloatTensor([[action_val / action_std]]).to(DEVICE)
    with torch.no_grad():
        dsdt_norm = model(s_norm, a_norm).cpu().numpy()[0]
    dsdt = dsdt_norm * delta_std * dt
    return s_cur + dsdt


def evaluate_single_model(model, data, horizons, n_segments=5, seed=42):
    """Evaluate a single model on trajectory rollout."""
    dt = 1.0 / 30.0
    state_std = data['state_std']
    action_std = data['action_std']
    delta_std = data['delta_std']
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
                    s_next = _rollout_predict(model, s_cur, actions_seg[step], state_std, action_std, delta_std, dt)

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
                except:
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


def evaluate_ensemble(models, data, horizons, weights=None, n_segments=5, seed=42):
    """Evaluate ensemble of models with optional weighted averaging."""
    dt = 1.0 / 30.0
    state_std = data['state_std']
    action_std = data['action_std']
    delta_std = data['delta_std']
    episodes = data['episodes']
    test_eps = data['test_eps']

    if weights is None:
        weights = np.ones(len(models)) / len(models)
    else:
        weights = np.array(weights)
        weights = weights / weights.sum()

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
                    # Collect predictions from all models
                    predictions = []
                    for model in models:
                        s_next_i = _rollout_predict(model, s_cur, actions_seg[step], state_std, action_std, delta_std, dt)
                        predictions.append(s_next_i)

                    # Weighted average of next-state predictions
                    predictions = np.array(predictions)
                    s_next = np.average(predictions, axis=0, weights=weights)

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
                except:
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


# ===================== Validation Weight Computation =====================

def compute_validation_scores(models, data, seeds):
    """Compute per-model validation NMAE for weighting."""
    dt = 1.0 / 30.0
    state_std = data['state_std']
    action_std = data['action_std']
    delta_std = data['delta_std']
    episodes = data['episodes']
    test_eps = data['test_eps']

    np.random.seed(42)
    segments = []
    for ep_idx in test_eps:
        ep = episodes[ep_idx]
        if ep['length'] >= 1100:
            segments.append(ep)
    val_segments = segments[:3]
    h_val = 100

    scores = []
    for model_idx, model in enumerate(models):
        nmae_list = []
        for seg in val_segments:
            s0 = seg['obs'][0].copy()
            actions_seg = seg['action'].flatten()
            real_states = seg['obs']
            n = min(h_val, len(actions_seg))
            s_cur = s0.copy()
            survived = True
            step_errors = []

            for step in range(n):
                try:
                    s_next = _rollout_predict(model, s_cur, actions_seg[step], state_std, action_std, delta_std, dt)

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
                except:
                    survived = False
                    break

            if step_errors:
                errors = np.array(step_errors)
                nmae = float(np.mean(np.mean(errors, axis=0)))
                nmae_list.append(nmae)

        avg_score = np.mean(nmae_list) if nmae_list else float('inf')
        scores.append(avg_score)
        log(f"    Model {model_idx} (seed={seeds[model_idx]}): val_NMAE={avg_score:.4f}")

    return np.array(scores)


# ===================== Main =====================

SEEDS = [42, 43, 44, 45, 46]
HORIZONS = [1, 10, 50, 100, 200, 500, 1000]


def main():
    log("=" * 70)
    log("Ensemble of Improved Coupling Networks (256-dim)")
    log("=" * 70)
    log(f"Device: {DEVICE}")
    log(f"Seeds: {SEEDS}")
    log(f"Config: coupling_dim={CONFIG['coupling_dim']}, hidden={CONFIG['hidden']}")
    log(f"Epochs: {CONFIG['n_epochs']}, Batch: {CONFIG['batch_size']}")

    # 1. Load data
    log("\n1. Loading data...")
    data = load_data(seed=42)
    log(f"   Loaded: {len(data['train_obs'])} train samples, {len(data['test_eps'])} test episodes")

    # 2. Train 5 models
    log("\n2. Training 5 models with different seeds...")
    models = []
    train_times = []

    for i, seed in enumerate(SEEDS):
        log(f"\n  --- Model {i+1}/5 (seed={seed}) ---")
        t0 = time.time()
        model = train_model(data, CONFIG, seed)
        elapsed = time.time() - t0
        models.append(model)
        train_times.append(elapsed)
        log(f"  Model {i+1} done in {elapsed:.1f}s")

    total_train_time = sum(train_times)
    log(f"\n  Total training time: {total_train_time:.1f}s ({total_train_time/60:.1f} min)")

    # 3. Evaluate each single model
    log("\n3. Evaluating individual models...")
    single_results = []
    for i, (model, seed) in enumerate(zip(models, SEEDS)):
        log(f"\n  --- Single Model {i+1} (seed={seed}) ---")
        result = evaluate_single_model(model, data, HORIZONS)
        single_results.append(result)
        primary = np.mean([result[100]['nmae_mean'], result[200]['nmae_mean'], result[500]['nmae_mean']])
        log(f"    PrimaryLongHorizonScore: {primary:.4f}")

    # 4. Compute validation weights
    log("\n4. Computing validation-based weights...")
    val_scores = compute_validation_scores(models, data, SEEDS)
    inv_scores = 1.0 / (val_scores + 1e-8)
    weighted_weights = inv_scores / inv_scores.sum()
    log(f"\n    Validation NMAE scores: {val_scores}")
    log(f"    Weighted ensemble weights: {weighted_weights}")

    # 5. Evaluate simple average ensemble
    log("\n5. Evaluating Simple Average Ensemble...")
    simple_avg_results = evaluate_ensemble(models, data, HORIZONS, weights=None)
    simple_primary = np.mean([
        simple_avg_results[100]['nmae_mean'],
        simple_avg_results[200]['nmae_mean'],
        simple_avg_results[500]['nmae_mean'],
    ])
    log(f"    PrimaryLongHorizonScore: {simple_primary:.4f}")

    # 6. Evaluate weighted average ensemble
    log("\n6. Evaluating Weighted Average Ensemble...")
    weighted_avg_results = evaluate_ensemble(models, data, HORIZONS, weights=weighted_weights)
    weighted_primary = np.mean([
        weighted_avg_results[100]['nmae_mean'],
        weighted_avg_results[200]['nmae_mean'],
        weighted_avg_results[500]['nmae_mean'],
    ])
    log(f"    PrimaryLongHorizonScore: {weighted_primary:.4f}")

    # 7. Summary comparison
    log("\n" + "=" * 70)
    log("SUMMARY: Ensemble vs Single Model")
    log("=" * 70)

    single_primaries = []
    for i, result in enumerate(single_results):
        p = np.mean([result[100]['nmae_mean'], result[200]['nmae_mean'], result[500]['nmae_mean']])
        single_primaries.append(p)

    best_single_idx = np.argmin(single_primaries)
    best_single_primary = single_primaries[best_single_idx]
    avg_single_primary = np.mean(single_primaries)

    log(f"\n{'Method':<30} {'Primary':>10} {'vs Best Single':>15}")
    log("-" * 55)
    for i, p in enumerate(single_primaries):
        marker = " <-- best single" if i == best_single_idx else ""
        log(f"  Single model seed={SEEDS[i]:<10}  {p:>10.4f}{marker}")
    log(f"  Average of singles            {avg_single_primary:>10.4f}  {(avg_single_primary - best_single_primary)/best_single_primary*100:>+12.1f}%")
    log(f"  Simple Average Ensemble       {simple_primary:>10.4f}  {(simple_primary - best_single_primary)/best_single_primary*100:>+12.1f}%")
    log(f"  Weighted Average Ensemble     {weighted_primary:>10.4f}  {(weighted_primary - best_single_primary)/best_single_primary*100:>+12.1f}%")
    log(f"\n  Baseline (EXP051 xlarge):     0.4712")

    best_ensemble_primary = min(simple_primary, weighted_primary)
    best_ensemble_name = "simple_average" if simple_primary <= weighted_primary else "weighted_average"
    improvement_vs_baseline = (0.4712 - best_ensemble_primary) / 0.4712 * 100

    log(f"\n  Best ensemble: {best_ensemble_name}")
    log(f"  Improvement vs EXP051 baseline: {improvement_vs_baseline:+.1f}%")

    # Per-horizon comparison table
    header = f"{'Horizon':<10}"
    for i in range(len(SEEDS)):
        header += f"{'S'+str(i):>8}"
    header += f"{'SimpleAvg':>10} {'WeightedAvg':>12}"
    log(f"\n{header}")
    log("-" * len(header))

    for h in HORIZONS:
        line = f"H={h:<7}"
        for result in single_results:
            line += f"{result[h]['nmae_mean']:>8.4f}"
        line += f"{simple_avg_results[h]['nmae_mean']:>10.4f} {weighted_avg_results[h]['nmae_mean']:>12.4f}"
        log(line)

    # 8. Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'run_id': '20260628_175824_neural_ode_144h',
        'experiment': 'ensemble_coupling',
        'device': str(DEVICE),
        'config': CONFIG,
        'seeds': SEEDS,
        'train_times': train_times,
        'total_train_time': total_train_time,
        'validation_scores': val_scores.tolist(),
        'ensemble_weights': weighted_weights.tolist(),
        'single_model_results': {
            f'seed_{seed}': {
                'primary': float(p),
                'per_horizon': {
                    str(h): {'nmae': float(result[h]['nmae_mean']), 'survival': float(result[h]['survival_rate'])}
                    for h in HORIZONS
                },
            }
            for seed, p, result in zip(SEEDS, single_primaries, single_results)
        },
        'simple_average_ensemble': {
            'primary': float(simple_primary),
            'per_horizon': {
                str(h): {
                    'nmae': float(simple_avg_results[h]['nmae_mean']),
                    'survival': float(simple_avg_results[h]['survival_rate']),
                    'per_state': simple_avg_results[h]['per_state_nmae'],
                }
                for h in HORIZONS
            },
        },
        'weighted_average_ensemble': {
            'primary': float(weighted_primary),
            'weights': weighted_weights.tolist(),
            'per_horizon': {
                str(h): {
                    'nmae': float(weighted_avg_results[h]['nmae_mean']),
                    'survival': float(weighted_avg_results[h]['survival_rate']),
                    'per_state': weighted_avg_results[h]['per_state_nmae'],
                }
                for h in HORIZONS
            },
        },
        'summary': {
            'best_single_primary': float(best_single_primary),
            'best_single_seed': int(SEEDS[best_single_idx]),
            'avg_single_primary': float(avg_single_primary),
            'simple_avg_primary': float(simple_primary),
            'weighted_avg_primary': float(weighted_primary),
            'best_ensemble_method': best_ensemble_name,
            'best_ensemble_primary': float(best_ensemble_primary),
            'improvement_vs_exp051_baseline': float(improvement_vs_baseline),
            'improvement_vs_best_single': float((best_single_primary - best_ensemble_primary) / best_single_primary * 100),
        },
    }

    output_path = 'D:/系统辨识作业/sindy_bicycle/research_72h/05_candidates/EXP058_ensemble_coupling.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    log(f"\nResults saved to: {output_path}")

    _log.close()
    return output


if __name__ == '__main__':
    main()
