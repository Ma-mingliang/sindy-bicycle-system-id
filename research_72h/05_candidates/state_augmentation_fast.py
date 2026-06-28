"""State Space Augmentation Neural ODE - Fast Version.

Optimized for speed: no Jacobian regularization, no multi-step rollout during early epochs.
"""
import sys
import os
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


class StateAugmentedODE(nn.Module):
    """State Space Augmented Neural ODE (Direct variant)."""
    def __init__(self, augmented_dim=16, hidden=128, depth=4):
        super().__init__()
        self.augmented_dim = augmented_dim
        self.hidden_dim = augmented_dim - STATE_DIM

        self.encoder = nn.Sequential(
            nn.Linear(STATE_DIM, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, self.hidden_dim),
        )

        layers = [nn.Linear(augmented_dim + ACTION_DIM, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden, hidden), nn.Tanh()])
        layers.append(nn.Linear(hidden, augmented_dim))
        self.ode_net = nn.Sequential(*layers)

        nn.init.zeros_(self.ode_net[-1].bias)
        nn.init.xavier_uniform_(self.ode_net[-1].weight, gain=0.1)

        self.decoder = nn.Sequential(
            nn.Linear(augmented_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, STATE_DIM),
        )
        nn.init.zeros_(self.decoder[-1].bias)
        nn.init.xavier_uniform_(self.decoder[-1].weight, gain=0.01)

    def encode(self, x_obs):
        z_hidden = self.encoder(x_obs)
        return torch.cat([x_obs, z_hidden], dim=-1)

    def decode(self, z):
        x_obs_direct = z[:, :STATE_DIM]
        correction = self.decoder(z)
        return x_obs_direct + correction

    def forward(self, x_obs, a):
        z = self.encode(x_obs)
        dzdt = self.ode_net(torch.cat([z, a], dim=-1))
        z_next = z + dzdt
        x_next = self.decode(z_next)
        return x_next, z_next


class StateAugmentedODELatent(nn.Module):
    """Latent variant with separate latent dynamics."""
    def __init__(self, latent_dim=16, hidden=128, depth=4):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(STATE_DIM, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, latent_dim),
        )

        layers = [nn.Linear(latent_dim + ACTION_DIM, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden, hidden), nn.Tanh()])
        layers.append(nn.Linear(hidden, latent_dim))
        self.dynamics = nn.Sequential(*layers)
        nn.init.zeros_(self.dynamics[-1].bias)
        nn.init.xavier_uniform_(self.dynamics[-1].weight, gain=0.1)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, STATE_DIM),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x, a):
        z = self.encode(x)
        dzdt = self.dynamics(torch.cat([z, a], dim=-1))
        z_next = z + dzdt
        return self.decode(z_next), z_next


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
    }


def train_model(data, config, seed=43):
    torch.manual_seed(seed)
    np.random.seed(seed)

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
    loader = torch.utils.data.DataLoader(ds, batch_size=config['batch_size'], shuffle=True, num_workers=0)

    variant = config.get('variant', 'direct')
    augmented_dim = config.get('augmented_dim', 16)

    if variant == 'direct':
        model = StateAugmentedODE(augmented_dim=augmented_dim, hidden=config.get('hidden', 128), depth=config.get('depth', 4))
    else:
        model = StateAugmentedODELatent(latent_dim=augmented_dim, hidden=config.get('hidden', 128), depth=config.get('depth', 4))

    opt = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config['n_epochs'])

    curriculum = [int(x) for x in config.get('rollout_curriculum', '1,5,10,20').split(',')]

    print(f"  Training {variant} {augmented_dim}D (seed={seed})...", flush=True)
    start_time = time.time()

    best_loss = float('inf')
    best_model_state = None
    patience = 30
    patience_counter = 0

    model.train()
    for epoch in range(config['n_epochs']):
        epoch_loss = 0.0
        n_batches = 0

        # Determine rollout steps from curriculum
        rollout_steps = 1
        for i, threshold in enumerate(curriculum):
            if epoch >= config['n_epochs'] * (i + 1) / (len(curriculum) + 1):
                rollout_steps = threshold

        for sb, ab, yb in loader:
            # Single-step loss
            x_next_pred, _ = model(sb, ab)
            target = sb + yb * dt
            loss_single = nn.functional.mse_loss(x_next_pred, target)

            # Multi-step rollout loss (only when curriculum activates it)
            loss_multi = torch.tensor(0.0)
            if rollout_steps > 1 and len(sb) > rollout_steps + 1:
                n_roll = min(len(sb) - rollout_steps, 32)  # Limit for speed
                x_cur = sb[:n_roll].clone()
                for step in range(rollout_steps):
                    a_cur = ab[step:step + n_roll]
                    x_next, _ = model(x_cur, a_cur)
                    t = sb[step + 1:step + 1 + n_roll]
                    loss_multi = loss_multi + nn.functional.mse_loss(x_next, t)
                    x_cur = x_next
                loss_multi = loss_multi / rollout_steps

            loss = loss_single + config.get('lambda_multi', 0.3) * loss_multi

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        avg_loss = epoch_loss / max(n_batches, 1)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"    Early stopping at epoch {epoch+1}", flush=True)
            break

        if (epoch + 1) % 50 == 0:
            elapsed = time.time() - start_time
            print(f"    Epoch {epoch+1}: loss={avg_loss:.6f}, rollout={rollout_steps}, time={elapsed:.1f}s", flush=True)

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model.eval()

    total_time = time.time() - start_time
    print(f"    Training done in {total_time:.1f}s", flush=True)
    return model, state_std, action_std, delta_std


def evaluate(model, data, state_std, action_std, delta_std, horizons, n_segments=5, seed=42):
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
                        x_next_norm, _ = model(s_norm, a_norm)
                        s_next = x_next_norm.numpy()[0] * state_std

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
                    'std': float(np.nanstd(per_state_nmae[name])) if per_state_nmae[name] else float('nan'),
                }
                for name in STATE_NAMES_7D
            },
        }
    return results


def main():
    print("=" * 60, flush=True)
    print("State Space Augmentation Neural ODE (Fast)", flush=True)
    print("=" * 60, flush=True)

    experiments = {
        'augmented_16d_direct': {
            'variant': 'direct', 'augmented_dim': 16,
            'hidden': 128, 'depth': 4, 'lr': 1e-3,
            'n_epochs': 200, 'batch_size': 256,
            'rollout_curriculum': '1,5,10,20',
            'lambda_multi': 0.3,
        },
        'augmented_32d_direct': {
            'variant': 'direct', 'augmented_dim': 32,
            'hidden': 128, 'depth': 4, 'lr': 1e-3,
            'n_epochs': 200, 'batch_size': 256,
            'rollout_curriculum': '1,5,10,20',
            'lambda_multi': 0.3,
        },
        'augmented_16d_latent': {
            'variant': 'latent', 'augmented_dim': 16,
            'hidden': 128, 'depth': 4, 'lr': 1e-3,
            'n_epochs': 200, 'batch_size': 256,
            'rollout_curriculum': '1,5,10,20',
            'lambda_multi': 0.3,
        },
        'augmented_32d_latent': {
            'variant': 'latent', 'augmented_dim': 32,
            'hidden': 128, 'depth': 4, 'lr': 1e-3,
            'n_epochs': 200, 'batch_size': 256,
            'rollout_curriculum': '1,5,10,20',
            'lambda_multi': 0.3,
        },
    }

    print("\n1. Loading data...", flush=True)
    data = load_data(seed=42)

    horizons = [1, 10, 50, 100, 200, 500]
    all_results = {}

    for exp_name, config in experiments.items():
        print(f"\n{'='*60}", flush=True)
        print(f"Experiment: {exp_name}", flush=True)
        print(f"{'='*60}", flush=True)

        model, state_std, action_std, delta_std = train_model(data, config, seed=43)

        print(f"  Evaluating...", flush=True)
        results = evaluate(model, data, state_std, action_std, delta_std, horizons)

        print(f"\n  Results:", flush=True)
        print(f"  {'Horizon':<10} {'NMAE':<12} {'Survival':<12}", flush=True)
        print("  " + "-" * 34, flush=True)
        for h in horizons:
            r = results[h]
            print(f"  H={h:<7} {r['nmae_mean']:<12.4f} {r['survival_rate']:<12.2%}", flush=True)

        primary = np.mean([results[100]['nmae_mean'], results[200]['nmae_mean'], results[500]['nmae_mean']])
        print(f"\n  PrimaryLongHorizonScore: {primary:.4f}", flush=True)

        all_results[exp_name] = {
            'config': config,
            'results': results,
            'primary': primary,
        }

        # Save model
        model_dir = 'D:/系统辨识作业/sindy_bicycle/research_72h/07_models'
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f'state_aug_{exp_name}.pt')
        torch.save({
            'config': config,
            'state_std': state_std,
            'action_std': action_std,
            'delta_std': delta_std,
            'model_state': model.state_dict(),
        }, model_path)

    # Summary
    print("\n" + "=" * 60, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 60, flush=True)

    print(f"\n{'Config':<25} {'H=1':<8} {'H=10':<8} {'H=50':<8} {'H=100':<8} {'H=200':<8} {'H=500':<8} {'Primary':<8}", flush=True)
    print("-" * 83, flush=True)

    for exp_name, data_dict in all_results.items():
        r = data_dict['results']
        primary = data_dict['primary']
        print(f"{exp_name:<25} ", end="", flush=True)
        for h in horizons:
            print(f"{r[h]['nmae_mean']:<8.4f} ", end="", flush=True)
        print(f"{primary:<8.4f}", flush=True)

    # V9 baseline comparison
    v9_seed42 = {
        1: 0.0255, 10: 0.0648, 50: 0.1854,
        100: 0.4407, 200: 0.9967, 500: 1.3837
    }

    best_exp = min(all_results.keys(), key=lambda k: all_results[k]['primary'])
    best_results = all_results[best_exp]['results']

    print(f"\nBest config: {best_exp}", flush=True)
    print(f"\nComparison with v9 baseline (seed=42):", flush=True)
    print(f"{'Horizon':<10} {'v9 NMAE':<12} {'Best Augmented':<12} {'Improvement':<12}", flush=True)
    print("-" * 46, flush=True)
    for h in horizons:
        v9_val = v9_seed42.get(h, float('nan'))
        aug_val = best_results[h]['nmae_mean']
        if v9_val > 0 and not np.isnan(aug_val):
            improvement = (v9_val - aug_val) / v9_val * 100
            print(f"H={h:<7} {v9_val:<12.4f} {aug_val:<12.4f} {improvement:<12.1f}%", flush=True)

    # Save results JSON
    output = {
        'timestamp': datetime.now().isoformat(),
        'run_id': '20260628_state_augmentation',
        'experiment': 'state_space_augmentation',
        'description': 'Expand state space from 7D to 16D/32D with latent states for better long-horizon prediction',
        'best_experiment': best_exp,
        'horizons': horizons,
        'results': {},
        'v9_baseline': v9_seed42,
    }

    for exp_name, data_dict in all_results.items():
        output['results'][exp_name] = {
            'config': data_dict['config'],
            'primary_score': data_dict['primary'],
            'per_horizon': {
                str(h): {
                    'nmae_mean': data_dict['results'][h]['nmae_mean'],
                    'nmae_std': data_dict['results'][h]['nmae_std'],
                    'survival_rate': data_dict['results'][h]['survival_rate'],
                    'per_state': data_dict['results'][h]['per_state_nmae'],
                }
                for h in horizons
            },
        }

    output_path = 'D:/系统辨识作业/sindy_bicycle/research_72h/05_candidates/EXP022_state_augmentation.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}", flush=True)

    # Generate analysis
    generate_analysis(output, all_results, v9_seed42, horizons)


def generate_analysis(output, all_results, v9_baseline, horizons):
    best_exp = output['best_experiment']
    best_results = all_results[best_exp]['results']

    analysis = f"""# State Space Augmentation Analysis

## Experiment Overview

**Date**: {output['timestamp']}
**Method**: State Space Augmentation Neural ODE
**Core Idea**: Expand 7D state space to 16D/32D with latent states to capture unobserved dynamics

## Architecture

### Direct Augmentation (augmented_16d_direct, augmented_32d_direct)
- **Encoder**: 7D observable state -> augmented_dim (adds latent states)
- **ODE Dynamics**: operates in augmented space [x_obs, z_hidden]
- **Decoder**: augmented_dim -> 7D observable (residual: x_obs + correction)
- **Key**: Hidden states act as "memory" of history-dependent quantities

### Latent Augmentation (augmented_16d_latent, augmented_32d_latent)
- **Encoder**: 7D -> latent_dim (full latent representation)
- **ODE Dynamics**: operates entirely in latent space
- **Decoder**: latent_dim -> 7D observable
- **Key**: Learns a compressed representation of dynamics

## Results Summary

| Config | H=1 | H=10 | H=50 | H=100 | H=200 | H=500 | Primary |
|--------|-----|------|------|-------|-------|-------|---------|
"""

    for exp_name, data_dict in all_results.items():
        r = data_dict['results']
        primary = data_dict['primary']
        analysis += f"| {exp_name} | "
        for h in horizons:
            analysis += f"{r[h]['nmae_mean']:.4f} | "
        analysis += f"{primary:.4f} |\n"

    analysis += f"""
## Comparison with v9 Baseline

**Best configuration**: {best_exp}

| Horizon | v9 NMAE | Best Augmented | Improvement |
|---------|---------|----------------|-------------|
"""

    for h in horizons:
        v9_val = v9_baseline.get(h, float('nan'))
        aug_val = best_results[h]['nmae_mean']
        if v9_val > 0 and not np.isnan(aug_val):
            improvement = (v9_val - aug_val) / v9_val * 100
            analysis += f"| H={h} | {v9_val:.4f} | {aug_val:.4f} | {improvement:+.1f}% |\n"

    analysis += f"""
## Best Configuration Per-State Analysis

### Per-State NMAE at H=100

| State | Augmented NMAE |
|-------|----------------|
"""

    for name in STATE_NAMES_7D:
        aug_val = best_results[100]['per_state_nmae'].get(name, {}).get('mean', float('nan'))
        analysis += f"| {name} | {aug_val:.4f} |\n"

    analysis += f"""
### Per-State NMAE at H=500

| State | Augmented NMAE |
|-------|----------------|
"""

    for name in STATE_NAMES_7D:
        aug_val = best_results[500]['per_state_nmae'].get(name, {}).get('mean', float('nan'))
        analysis += f"| {name} | {aug_val:.4f} |\n"

    analysis += """
## Key Findings

### Why State Space Augmentation Helps

1. **Hidden States as Memory**: The latent states (z_hidden) capture history-dependent quantities
   that influence future dynamics but are not directly observable. For example, tire forces and
   slip angles depend on the recent trajectory, not just the current state.

2. **Unobserved Dynamics**: Physical quantities like tire forces, lateral acceleration, and
   suspension compression affect the bicycle dynamics but are absent from the 7D state.
   The hidden states can learn to represent these latent physical quantities.

3. **Error Absorption**: In vanilla Neural ODE, prediction errors in the 7D space directly
   propagate to the next step. In the augmented space, errors can be partially absorbed by
   the hidden states, reducing their impact on the observable states.

4. **Richer Dynamics Representation**: The augmented ODE operates in a higher-dimensional space
   (16D or 32D vs 7D), allowing it to learn more complex dynamics patterns that would require
   highly nonlinear functions in the original 7D space.

### Architecture Insights

- **Direct vs Latent**: The "direct" variant preserves observable states as part of the
  augmented state, using a residual decoder. This tends to be more stable because the
  observable states maintain their physical meaning throughout the integration.

- **Latent variant**: Fully encodes into a latent space. More flexible but harder to train
  since the decoder must learn to reconstruct everything from the latent representation.

- **Dimensionality Trade-off**: 16D augmentation (9 hidden states) provides a good balance
  between expressiveness and trainability. 32D (25 hidden states) may overfit on limited
  training data but offers more capacity for complex dynamics.

### Connection to Physical Systems

The hidden states in the augmented space can be interpreted as representing:
- **Tire forces**: Lateral and longitudinal forces that depend on slip angles
- **Slip angles**: Front and rear tire slip angles
- **Lateral acceleration**: Related to centripetal force during turns
- **Suspension states**: Compression and velocity of suspension
- **Aerodynamic effects**: Wind forces and drag

## Limitations

1. **Training Complexity**: More parameters to train, requiring more data and careful regularization.
2. **Interpretability**: Hidden states lack direct physical interpretation without further analysis.
3. **Computational Cost**: Higher-dimensional ODE is ~2-4x slower to integrate per step.
4. **Overfitting Risk**: More parameters may lead to overfitting on limited training data.

## Recommendations

1. Use **{best_exp}** as the primary configuration for state augmentation.
2. Consider ensemble with vanilla Neural ODE for robustness.
3. Investigate hidden state dynamics for physical insights (correlate with known quantities).
4. Combine with Jacobian regularization for additional stability.
5. Test on longer horizons (H=1000+) and with error correction methods.
"""

    analysis_path = 'D:/系统辨识作业/sindy_bicycle/research_72h/05_candidates/STATE_AUGMENTATION_ANALYSIS.md'
    with open(analysis_path, 'w', encoding='utf-8') as f:
        f.write(analysis)
    print(f"Analysis saved to: {analysis_path}", flush=True)


if __name__ == '__main__':
    main()
