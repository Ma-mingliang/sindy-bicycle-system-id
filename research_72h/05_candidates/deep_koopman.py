"""Deep Koopman for Bicycle Dynamics.

Implements Deep Koopman operator learning for long-horizon prediction.
Key idea: Learn lifting function phi(x) such that phi(x_{t+1}) = K * phi(x_t) + B * u_t
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
LIFTED_DIM = 20  # Dimension of lifted space
IDX_7D_FROM_8D = [0, 1, 2, 3, 4, 6, 7]

PHYSICAL_LIMITS = {
    'e_y': 5.0, 'e_psi': np.pi, 'v': 5.0,
    'theta': np.pi, 'theta_dot': 10.0,
    'delta': np.pi/2, 'delta_dot': 10.0,
}


class DeepKoopman(nn.Module):
    """Deep Koopman model with autoencoder structure.

    Architecture:
    - Encoder: phi: R^7 -> R^lifted_dim (lifting)
    - Decoder: psi: R^lifted_dim -> R^7 (inverse lifting)
    - Linear dynamics: z_{t+1} = K * z_t + B * u_t
    """
    def __init__(self, state_dim=7, action_dim=1, lifted_dim=20, hidden=64):
        super().__init__()

        self.lifted_dim = lifted_dim

        # Encoder (lifting function)
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, lifted_dim),
        )

        # Decoder (inverse lifting)
        self.decoder = nn.Sequential(
            nn.Linear(lifted_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, state_dim),
        )

        # Linear dynamics in lifted space
        self.K = nn.Linear(lifted_dim, lifted_dim, bias=False)  # State transition
        self.B = nn.Linear(action_dim, lifted_dim, bias=False)  # Control input

        # Initialize K close to identity for stability
        nn.init.eye_(self.K.weight)
        self.K.weight.data *= 0.95  # Slight contraction

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def predict_next_z(self, z, u):
        return self.K(z) + self.B(u)

    def forward(self, x, u):
        z = self.encode(x)
        z_next = self.predict_next_z(z, u)
        x_next = self.decode(z_next)
        return x_next, z_next

    def rollout(self, x0, actions, steps):
        """Rollout for multi-step prediction."""
        z = self.encode(x0)
        predictions = [self.decode(z)]

        for step in range(steps):
            u = actions[:, step:step+1] if actions.dim() == 2 else actions[step:step+1]
            z = self.predict_next_z(z, u)
            predictions.append(self.decode(z))

        return torch.stack(predictions, dim=1)


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

    state_std = np.std(train_obs, axis=0)
    state_std[state_std < 1e-10] = 1.0
    action_std = np.std(train_action)
    if action_std < 1e-10:
        action_std = 1.0

    return {
        'episodes': episodes,
        'train_eps': train_eps,
        'test_eps': test_eps,
        'state_std': state_std,
        'action_std': action_std,
    }


def train_deep_koopman(data, config, seed=43):
    torch.manual_seed(seed)
    np.random.seed(seed)

    state_std = data['state_std']
    action_std = data['action_std']

    train_eps = data['train_eps']
    episodes = data['episodes']

    n_train = len(train_eps)
    n_val = max(1, n_train // 10)
    val_eps = train_eps[:n_val]
    actual_train_eps = train_eps[n_val:]

    model = DeepKoopman(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        lifted_dim=config.get('lifted_dim', LIFTED_DIM),
        hidden=config.get('hidden', 64)
    )

    opt = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config['n_epochs'])

    print(f"Training Deep Koopman (seed={seed})...")
    print(f"  Lifted dim: {config.get('lifted_dim', LIFTED_DIM)}")
    start_time = time.time()

    best_val_loss = float('inf')
    best_model_state = None
    patience = 50
    patience_counter = 0

    model.train()
    for epoch in range(config['n_epochs']):
        epoch_loss = 0.0
        n_batches = 0

        for _ in range(100):
            batch_size = config['batch_size']
            states_list = []
            next_states_list = []
            actions_list = []

            for _ in range(batch_size):
                ep_idx = np.random.choice(actual_train_eps)
                ep = episodes[ep_idx]
                idx = np.random.randint(0, ep['length'] - 1)
                states_list.append(ep['obs'][idx])
                next_states_list.append(ep['obs'][idx + 1])
                actions_list.append(ep['action'][idx])

            sb = torch.FloatTensor(np.array(states_list) / state_std)
            sb_next = torch.FloatTensor(np.array(next_states_list) / state_std)
            ab = torch.FloatTensor(np.array(actions_list).reshape(-1, 1) / action_std)

            # Single-step prediction
            x_next_pred, z_next = model(sb, ab)
            loss_single = nn.functional.mse_loss(x_next_pred, sb_next)

            # Autoencoder reconstruction loss
            z = model.encode(sb)
            x_recon = model.decode(z)
            loss_recon = nn.functional.mse_loss(x_recon, sb)

            # Linear dynamics loss (in lifted space)
            z_target = model.encode(sb_next)
            loss_linear = nn.functional.mse_loss(z_next, z_target)

            loss = loss_single + 0.1 * loss_recon + 0.1 * loss_linear

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            val_loss = 0.0
            n_val_batches = 0

            for _ in range(20):
                states_list = []
                next_states_list = []
                actions_list = []

                for _ in range(batch_size):
                    ep_idx = np.random.choice(val_eps)
                    ep = episodes[ep_idx]
                    idx = np.random.randint(0, ep['length'] - 1)
                    states_list.append(ep['obs'][idx])
                    next_states_list.append(ep['obs'][idx + 1])
                    actions_list.append(ep['action'][idx])

                sb = torch.FloatTensor(np.array(states_list) / state_std)
                sb_next = torch.FloatTensor(np.array(next_states_list) / state_std)
                ab = torch.FloatTensor(np.array(actions_list).reshape(-1, 1) / action_std)

                with torch.no_grad():
                    x_next_pred, _ = model(sb, ab)
                    val_loss += nn.functional.mse_loss(x_next_pred, sb_next).item()
                n_val_batches += 1

            val_loss = val_loss / max(n_val_batches, 1)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

            model.train()

        if (epoch + 1) % 50 == 0:
            avg_loss = epoch_loss / max(n_batches, 1)
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch+1}/{config['n_epochs']}: loss={avg_loss:.6f}, "
                  f"val_loss={val_loss:.6f}, time={elapsed:.1f}s")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model.eval()

    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.1f}s")

    return model, state_std, action_std


def evaluate(model, data, state_std, action_std, horizons, n_segments=5, seed=42):
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
                        s_next_norm, _ = model(s_norm, a_norm)
                        s_next = s_next_norm.numpy()[0] * state_std

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


def main():
    print("=" * 60)
    print("Deep Koopman for Bicycle Dynamics")
    print("=" * 60)

    config = {
        'lifted_dim': 20,
        'hidden': 64,
        'lr': 1e-3,
        'n_epochs': 200,
        'batch_size': 256,
    }

    print("\n1. Loading data...")
    data = load_data(seed=42)

    print("\n2. Training Deep Koopman...")
    model, state_std, action_std = train_deep_koopman(data, config, seed=43)

    print("\n3. Evaluating...")
    horizons = [1, 10, 50, 100, 200, 500]
    results = evaluate(model, data, state_std, action_std, horizons)

    print("\n" + "=" * 60)
    print("RESULTS: Deep Koopman")
    print("=" * 60)
    print(f"\n{'Horizon':<10} {'NMAE':<12} {'Survival':<12}")
    print("-" * 34)
    for h in horizons:
        r = results[h]
        print(f"H={h:<7} {r['nmae_mean']:<12.4f} {r['survival_rate']:<12.2%}")

    print(f"\nComparison with v9 baseline:")
    print(f"{'Horizon':<10} {'v9 NMAE':<12} {'Koopman':<12} {'Improvement':<12}")
    print("-" * 46)
    v9_nmae = {1: 0.0051, 10: 0.0628, 50: 0.4557, 100: 0.5064, 200: 0.4737, 500: 0.5529}
    for h in horizons:
        v9_val = v9_nmae.get(h, float('nan'))
        k_val = results[h]['nmae_mean']
        if v9_val > 0 and not np.isnan(k_val):
            improvement = (v9_val - k_val) / v9_val * 100
            print(f"H={h:<7} {v9_val:<12.4f} {k_val:<12.4f} {improvement:<12.1f}%")

    output = {
        'timestamp': datetime.now().isoformat(),
        'run_id': '20260628_175824_neural_ode_72h',
        'experiment': 'deep_koopman',
        'config': config,
        'results': results,
    }

    output_path = 'D:/系统辨识作业/sindy_bicycle/research_72h/05_candidates/EXP008_deep_koopman.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    model_path = 'D:/系统辨识作业/sindy_bicycle/research_72h/07_models/deep_koopman_seed43.pt'
    torch.save({
        'config': config,
        'state_std': state_std,
        'action_std': action_std,
        'model_state': model.state_dict(),
    }, model_path)
    print(f"Model saved to: {model_path}")


if __name__ == '__main__':
    main()
