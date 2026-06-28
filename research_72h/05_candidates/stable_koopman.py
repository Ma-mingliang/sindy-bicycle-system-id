"""Stable Koopman for Bicycle Dynamics.

Core idea: Koopman operator linearizes nonlinear systems.
Linear systems do NOT accumulate errors the way nonlinear ones do.
Key improvement over Deep Koopman (EXP008):
1. Eigenvalue clamping on K to enforce |eigenvalues| < 1 (stability)
2. Multi-step rollout loss to prevent error accumulation
3. Larger lifted space (32 vs 20)
4. Eigenvalue monitoring during training

Stability mechanism: After each gradient update, we clamp the eigenvalues of K
to lie within a circle of radius spectral_bound < 1, ensuring contractive dynamics
that prevent long-horizon error accumulation.
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


class StableKoopman(nn.Module):
    """Stable Koopman model with autoencoder structure.

    Architecture:
    - Encoder: phi: R^7 -> R^lifted_dim (lifting)
    - Decoder: psi: R^lifted_dim -> R^7 (inverse lifting)
    - Linear dynamics: z_{t+1} = K * z_t + B * u_t
    - K eigenvalues clamped to |lambda| <= spectral_bound after each step

    Stability is enforced by eigenvalue clamping: after each gradient update,
    we decompose K = V * diag(lambda) * V^{-1} and clamp |lambda| <= spectral_bound.
    This guarantees the dynamics are contractive.
    """
    def __init__(self, state_dim=7, action_dim=1, lifted_dim=32, hidden=128,
                 spectral_bound=0.98):
        super().__init__()

        self.lifted_dim = lifted_dim
        self.spectral_bound = spectral_bound

        # Encoder (lifting function)
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden // 2),
            nn.Tanh(),
            nn.Linear(hidden // 2, lifted_dim),
        )

        # Decoder (inverse lifting)
        self.decoder = nn.Sequential(
            nn.Linear(lifted_dim, hidden // 2),
            nn.Tanh(),
            nn.Linear(hidden // 2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, state_dim),
        )

        # Linear dynamics in lifted space
        self.K = nn.Linear(lifted_dim, lifted_dim, bias=False)
        self.B = nn.Linear(action_dim, lifted_dim, bias=False)

        # Initialize K as scaled identity (well within unit circle)
        nn.init.eye_(self.K.weight)
        self.K.weight.data *= 0.5
        nn.init.xavier_uniform_(self.B.weight, gain=0.1)

    def clamp_K_eigenvalues(self):
        """Clamp eigenvalues of K to lie within unit circle of given radius.

        Decomposes K = V * diag(lambda) * V^{-1}, clamps |lambda| <= spectral_bound,
        then reconstructs K = V * diag(clamped_lambda) * V^{-1}.
        This is done in-place after each gradient update step.
        """
        with torch.no_grad():
            W = self.K.weight.data
            try:
                # Eigendecomposition (may produce complex eigenvalues)
                eigenvalues, V = torch.linalg.eig(W)
                # Clamp magnitude
                magnitudes = torch.abs(eigenvalues)
                clamped_magnitudes = torch.clamp(magnitudes, max=self.spectral_bound)
                # Apply clamping: lambda_clamped = lambda * (clamped_mag / mag)
                scale = clamped_magnitudes / (magnitudes + 1e-10)
                eigenvalues_clamped = eigenvalues * scale.to(eigenvalues.dtype)
                # Reconstruct: W = V * diag(lambda) * V^{-1}
                V_inv = torch.linalg.inv(V)
                W_new = (V @ torch.diag(eigenvalues_clamped) @ V_inv).real
                self.K.weight.data.copy_(W_new)
            except Exception:
                # Fallback: just rescale by spectral norm
                sigma = torch.linalg.svdvals(W)[0]
                if sigma > self.spectral_bound:
                    self.K.weight.data *= self.spectral_bound / sigma

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

    def get_max_eigenvalue_magnitude(self):
        """Get the maximum eigenvalue magnitude of K for monitoring."""
        W = self.K.weight.detach()
        try:
            eigenvalues = torch.linalg.eigvals(W)
            return torch.abs(eigenvalues).max().item()
        except:
            return float('nan')


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


def train_stable_koopman(data, config, seed=43):
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

    model = StableKoopman(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        lifted_dim=config.get('lifted_dim', 32),
        hidden=config.get('hidden', 128),
        spectral_bound=config.get('spectral_bound', 0.98),
    )

    opt = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config['n_epochs'])

    batch_size = config['batch_size']

    print(f"Training Stable Koopman (seed={seed})...")
    print(f"  Lifted dim: {config.get('lifted_dim', 32)}, Hidden: {config.get('hidden', 128)}")
    print(f"  Spectral bound: {config.get('spectral_bound', 0.98)}")
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
            # Sample batch
            states_list, next_states_list, actions_list = [], [], []
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

            # Multi-step rollout loss (key for stability, enabled after warmup)
            loss_multi = torch.tensor(0.0, device=sb.device)
            if epoch >= 20 and np.random.random() < 0.3:
                rollout_len = min(np.random.choice([3, 5, 8]), 8)
                seq_s, seq_a, seq_sn = [], [], []

                for _ in range(batch_size):
                    ep_idx = np.random.choice(actual_train_eps)
                    ep = episodes[ep_idx]
                    max_start = ep['length'] - rollout_len - 1
                    if max_start <= 0:
                        continue
                    si = np.random.randint(0, max_start)
                    seq_s.append(ep['obs'][si])
                    seq_a.append(ep['action'][si:si + rollout_len].flatten())
                    seq_sn.append(ep['obs'][si + 1:si + rollout_len + 1])

                if len(seq_s) >= 16:
                    s0_t = torch.FloatTensor(np.array(seq_s) / state_std)
                    acts_t = torch.FloatTensor(np.array(seq_a) / action_std)  # (N, rollout_len)
                    targets_t = torch.FloatTensor(np.array(seq_sn) / state_std)  # (N, rollout_len, 7)

                    z_cur = model.encode(s0_t)  # (N, lifted_dim)
                    for t in range(rollout_len):
                        u_t = acts_t[:, t:t+1]  # (N, 1)
                        z_cur = model.predict_next_z(z_cur, u_t)
                        x_pred_t = model.decode(z_cur)  # (N, 7)
                        weight = 1.0 + 0.1 * t
                        loss_multi = loss_multi + weight * nn.functional.mse_loss(
                            x_pred_t, targets_t[:, t, :])

                    loss_multi = loss_multi / rollout_len

            loss = loss_single + 0.1 * loss_recon + 0.1 * loss_linear
            if loss_multi.item() > 0:
                loss = loss + 0.5 * loss_multi

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            # Clamp eigenvalues after each gradient step
            model.clamp_K_eigenvalues()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            val_loss = 0.0
            n_val_batches = 0

            for _ in range(20):
                states_list, next_states_list, actions_list = [], [], []
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
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
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
            max_eig = model.get_max_eigenvalue_magnitude()
            print(f"  Epoch {epoch+1}/{config['n_epochs']}: loss={avg_loss:.6f}, "
                  f"val_loss={val_loss:.6f}, max|eig(K)|={max_eig:.4f}, "
                  f"time={elapsed:.1f}s")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model.eval()

    total_time = time.time() - start_time
    max_eig = model.get_max_eigenvalue_magnitude()
    print(f"Training completed in {total_time:.1f}s")
    print(f"Final max|eigenvalue(K)|: {max_eig:.4f} "
          f"(target <= {config.get('spectral_bound', 0.98)})")

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
                    a_norm = torch.FloatTensor(
                        [actions_seg[step] / action_std]).unsqueeze(0)

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
                name: {
                    'mean': float(np.nanmean(per_state_nmae[name]))
                    if per_state_nmae[name] else float('nan')
                }
                for name in STATE_NAMES_7D
            },
        }

    return results


def main():
    print("=" * 60)
    print("Stable Koopman for Bicycle Dynamics")
    print("=" * 60)
    print("Key improvements over Deep Koopman (EXP008):")
    print("  1. Eigenvalue clamping on K (|lambda| <= 0.98)")
    print("  2. Multi-step rollout loss (prevents error accumulation)")
    print("  3. Larger lifted space (32 vs 20)")
    print("  4. Deeper encoder/decoder (128 hidden)")
    print()

    config = {
        'lifted_dim': 32,
        'hidden': 128,
        'lr': 1e-3,
        'n_epochs': 200,
        'batch_size': 256,
        'spectral_bound': 0.98,
    }

    print("\n1. Loading data...")
    data = load_data(seed=42)
    print(f"   {len(data['train_eps'])} train episodes, "
          f"{len(data['test_eps'])} test episodes")

    print("\n2. Training Stable Koopman...")
    model, state_std, action_std = train_stable_koopman(data, config, seed=43)

    print("\n3. Evaluating...")
    horizons = [1, 10, 50, 100, 200, 500]
    results = evaluate(model, data, state_std, action_std, horizons)

    print("\n" + "=" * 60)
    print("RESULTS: Stable Koopman")
    print("=" * 60)
    print(f"\n{'Horizon':<10} {'NMAE':<12} {'Survival':<12}")
    print("-" * 34)
    for h in horizons:
        r = results[h]
        print(f"H={h:<7} {r['nmae_mean']:<12.4f} {r['survival_rate']:<12.2%}")

    # Per-state breakdown
    print(f"\n{'Horizon':<8}", end="")
    for name in STATE_NAMES_7D:
        print(f"{name:<12}", end="")
    print()
    print("-" * (8 + 12 * len(STATE_NAMES_7D)))
    for h in horizons:
        print(f"H={h:<6}", end="")
        for name in STATE_NAMES_7D:
            val = results[h]['per_state_nmae'][name]['mean']
            print(f"{val:<12.4f}", end="")
        print()

    # v9 baseline comparison
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

    # Primary score
    primary = np.mean([results[h]['nmae_mean'] for h in [100, 200, 500]])
    v9_primary = np.mean([v9_nmae[h] for h in [100, 200, 500]])
    print(f"\nPrimary Score (avg H=100,200,500):")
    print(f"  v9 baseline: {v9_primary:.4f}")
    print(f"  Stable Koopman: {primary:.4f}")
    if v9_primary > 0:
        print(f"  Improvement: {(v9_primary - primary) / v9_primary * 100:.1f}%")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'run_id': '20260628_175824_neural_ode_72h',
        'experiment': 'stable_koopman',
        'config': config,
        'results': results,
        'primary_score': float(primary),
        'v9_primary_score': float(v9_primary),
    }

    output_path = ('D:/系统辨识作业/sindy_bicycle/research_72h/05_candidates/'
                   'EXP023_stable_koopman.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    model_path = ('D:/系统辨识作业/sindy_bicycle/research_72h/07_models/'
                  'stable_koopman_seed43.pt')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save({
        'config': config,
        'state_std': state_std,
        'action_std': action_std,
        'model_state': model.state_dict(),
    }, model_path)
    print(f"Model saved to: {model_path}")


if __name__ == '__main__':
    main()
