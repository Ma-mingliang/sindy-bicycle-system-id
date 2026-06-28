"""Multiple Shooting Neural ODE for Long-Horizon Error Reduction.

Core idea: Instead of single-step loss (which allows error to silently accumulate),
train with segment-level loss on short multi-step rollouts (10-20 steps).

This forces the model to minimize accumulated error within each segment, directly
addressing the root cause of long-horizon prediction degradation.

Multiple Shooting formulation:
  - Split trajectories into short segments of length L (shooting interval)
  - Each segment starts from the REAL observed state (shooting node)
  - Roll out the ODE for L steps from each shooting node
  - Loss = accumulated prediction error over the entire segment
  - At test time, chain segments: predicted endpoint becomes next segment's initial state

This is fundamentally different from single-step training because:
  - Single-step loss: model only learns one-step accuracy
  - Segment loss: model learns to minimize error COMPOSITION over L steps
  - The model is penalized for errors that compound, not just local errors
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


class ShootingODEFunc(nn.Module):
    """Neural ODE dynamics function for multiple shooting."""
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


class MultipleShootingNeuralODE:
    """Multiple Shooting Neural ODE trainer and evaluator."""

    def __init__(self, config):
        self.config = config
        self.dt = 1.0 / 30.0
        self.model = None
        self.state_std = None
        self.action_std = None
        self.delta_std = None

    def load_data(self, seed=42):
        """Load and prepare data with train/test split."""
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

        # Compute normalization from training episodes
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

        self.state_std = state_std
        self.action_std = action_std
        self.delta_std = delta_std

        return {
            'episodes': episodes,
            'train_eps': train_eps,
            'test_eps': test_eps,
            'state_std': state_std,
            'action_std': action_std,
            'delta_std': delta_std,
        }

    def build_shooting_segments(self, episodes, train_eps, segment_len):
        """Extract shooting segments from training episodes.

        Each segment is a contiguous sub-trajectory of length segment_len+1
        (segment_len transitions). Segments start at random positions within
        episodes, with real observed states as shooting nodes.

        Returns:
            List of dicts with 'obs', 'action', 'deltas' arrays of shape (segment_len+1, ...)
        """
        segments = []
        for ep_idx in train_eps:
            ep = episodes[ep_idx]
            ep_len = ep['length']
            if ep_len < segment_len + 1:
                continue
            # Sample multiple segments per episode with stride
            stride = max(1, segment_len // 2)
            for start in range(0, ep_len - segment_len, stride):
                end = start + segment_len + 1
                segments.append({
                    'obs': ep['obs'][start:end],
                    'action': ep['action'][start:end],
                    'deltas': ep['deltas'][start:end],
                })
        return segments

    def train(self, data, seed=42):
        """Train Multiple Shooting Neural ODE.

        Key difference from standard training:
        - Instead of single-step loss on shuffled (s, a, delta) tuples,
          we use segment-level loss on short rollouts.
        - Each segment starts from a real observed state (shooting node).
        - The model rolls out segment_len steps, and loss is computed over
          all predicted states in the segment.
        """
        torch.manual_seed(seed)
        np.random.seed(seed)

        config = self.config
        state_std = data['state_std']
        action_std = data['action_std']
        delta_std = data['delta_std']
        dt = self.dt
        train_eps = data['train_eps']
        episodes = data['episodes']

        segment_len = config['segment_len']
        lambda_single = config.get('lambda_single', 0.3)
        lambda_segment = config.get('lambda_segment', 1.0)
        lambda_final = config.get('lambda_final', 0.5)

        # Build shooting segments
        print(f"Building shooting segments (len={segment_len})...")
        segments = self.build_shooting_segments(episodes, train_eps, segment_len)
        print(f"  Total segments: {len(segments)}")

        # Pre-normalize segments
        norm_segments = []
        for seg in segments:
            norm_segments.append({
                'obs': torch.FloatTensor(seg['obs'] / state_std),
                'action': torch.FloatTensor(seg['action'].reshape(-1, 1) / action_std),
                'deltas': torch.FloatTensor(seg['deltas'] / (delta_std * dt)),
            })

        model = ShootingODEFunc(
            hidden=config['hidden'],
            depth=config['depth'],
            activation=config['activation'],
        )

        opt = torch.optim.Adam(model.parameters(), lr=config['lr'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config['n_epochs'])

        batch_size = config['batch_size']
        n_segments = len(norm_segments)

        print(f"\nTraining Multiple Shooting Neural ODE (seed={seed})...")
        print(f"  segment_len={segment_len}, lambda_single={lambda_single}, "
              f"lambda_segment={lambda_segment}, lambda_final={lambda_final}")
        print(f"  hidden={config['hidden']}, depth={config['depth']}, "
              f"activation={config['activation']}")
        start_time = time.time()

        # Curriculum: gradually increase segment length
        curriculum = config.get('curriculum', None)
        if curriculum:
            curriculum_stages = [(int(s.split(':')[0]), int(s.split(':')[1]))
                                 for s in curriculum.split(',')]
        else:
            curriculum_stages = [(0, segment_len)]

        model.train()
        for epoch in range(config['n_epochs']):
            epoch_loss = 0.0
            n_batches = 0

            # Determine current segment length from curriculum
            cur_seg_len = segment_len
            if curriculum:
                for threshold, length in curriculum_stages:
                    if epoch >= threshold:
                        cur_seg_len = length

            # Filter segments that are long enough
            valid_indices = [i for i, seg in enumerate(norm_segments)
                            if len(seg['obs']) >= cur_seg_len + 1]

            if not valid_indices:
                valid_indices = list(range(len(norm_segments)))
                cur_seg_len = min(cur_seg_len,
                                  min(len(norm_segments[i]['obs']) - 1 for i in valid_indices))

            np.random.shuffle(valid_indices)

            for batch_start in range(0, len(valid_indices), batch_size):
                batch_indices = valid_indices[batch_start:batch_start + batch_size]

                # --- Single-step loss (from all transitions in the batch) ---
                all_obs = []
                all_actions = []
                all_deltas = []
                for idx in batch_indices:
                    seg = norm_segments[idx]
                    n = min(cur_seg_len + 1, len(seg['obs']))
                    all_obs.append(seg['obs'][:n])
                    all_actions.append(seg['action'][:n])
                    all_deltas.append(seg['deltas'][:n])

                obs_cat = torch.cat(all_obs, dim=0)
                act_cat = torch.cat(all_actions, dim=0)
                delta_cat = torch.cat(all_deltas, dim=0)

                pred_single = model(obs_cat, act_cat)
                loss_single = nn.functional.mse_loss(pred_single, delta_cat)

                # --- Segment-level rollout loss ---
                loss_segment = torch.tensor(0.0)
                loss_final = torch.tensor(0.0)
                n_valid_segments = 0

                for idx in batch_indices:
                    seg = norm_segments[idx]
                    n = min(cur_seg_len + 1, len(seg['obs']))
                    if n < 3:
                        continue

                    s_cur = seg['obs'][0:1].clone()  # (1, 7)
                    segment_errors = []

                    for step in range(n - 1):
                        a_step = seg['action'][step:step + 1]
                        dsdt_norm = model(s_cur, a_step)
                        s_cur = s_cur + dsdt_norm * dt
                        target = seg['obs'][step + 1:step + 2]
                        step_err = nn.functional.mse_loss(s_cur, target)
                        segment_errors.append(step_err)

                    # Accumulated error over segment (this is the key!)
                    segment_loss = sum(segment_errors) / len(segment_errors)
                    loss_segment = loss_segment + segment_loss

                    # Extra penalty on final state (end of segment)
                    final_err = nn.functional.mse_loss(s_cur, seg['obs'][n - 1:n])
                    loss_final = loss_final + final_err

                    n_valid_segments += 1

                if n_valid_segments > 0:
                    loss_segment = loss_segment / n_valid_segments
                    loss_final = loss_final / n_valid_segments

                # Total loss
                loss = (lambda_single * loss_single +
                        lambda_segment * loss_segment +
                        lambda_final * loss_final)

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
                print(f"  Epoch {epoch+1}/{config['n_epochs']}: loss={avg_loss:.6f}, "
                      f"seg_len={cur_seg_len}, time={elapsed:.1f}s")

        model.eval()
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.1f}s")

        self.model = model
        return model

    def predict_step(self, s_cur, a_cur):
        """Single-step prediction using the trained model."""
        s_norm = torch.FloatTensor(s_cur / self.state_std).unsqueeze(0)
        a_norm = torch.FloatTensor([[a_cur / self.action_std]])

        with torch.no_grad():
            dsdt_norm = self.model(s_norm, a_norm).numpy()[0]

        dsdt = dsdt_norm * self.delta_std * self.dt
        return s_cur + dsdt

    def evaluate(self, data, horizons, n_segments=5, seed=42):
        """Evaluate model with full rollout (chained predictions).

        At test time, we do NOT use shooting nodes (real states) as anchors.
        Instead, the model predicts the full trajectory from the initial state.
        This tests whether the segment-level training reduced error accumulation.
        """
        episodes = data['episodes']
        test_eps = data['test_eps']

        np.random.seed(seed)
        segments = []
        for ep_idx in test_eps:
            ep = episodes[ep_idx]
            if ep['length'] >= max(horizons) + 10:
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
                        s_next = self.predict_step(s_cur, actions_seg[step])

                        if np.any(np.isnan(s_next)) or np.any(np.isinf(s_next)):
                            survived = False
                            break
                        if not self.check_survival(s_next):
                            survived = False
                            break
                        if step + 1 < len(real_states):
                            step_err = np.abs(s_next - real_states[step + 1]) / self.state_std
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
                        if per_state_nmae[name] else float('nan')
                    }
                    for name in STATE_NAMES_7D
                },
            }

        return results

    def evaluate_with_shooting(self, data, horizons, shooting_interval, n_segments=5, seed=42):
        """Evaluate with shooting-anchored predictions.

        At every `shooting_interval` steps, reset the state to the REAL observed state.
        This is the "cheating" version that shows the theoretical benefit of shooting.
        Useful for diagnostic purposes.
        """
        episodes = data['episodes']
        test_eps = data['test_eps']

        np.random.seed(seed)
        segments = []
        for ep_idx in test_eps:
            ep = episodes[ep_idx]
            if ep['length'] >= max(horizons) + 10:
                segments.append(ep)
        segments = segments[:n_segments]

        results = {}
        for h in horizons:
            nmae_list = []
            survival_list = []

            for seg in segments:
                actions_seg = seg['action'].flatten()
                real_states = seg['obs']
                n = min(h, len(actions_seg))
                s_cur = seg['obs'][0].copy()
                survived = True
                step_errors = []

                for step in range(n):
                    try:
                        # Shooting: reset to real state at interval boundaries
                        if step > 0 and step % shooting_interval == 0:
                            if step < len(real_states):
                                s_cur = real_states[step].copy()

                        s_next = self.predict_step(s_cur, actions_seg[step])

                        if np.any(np.isnan(s_next)) or np.any(np.isinf(s_next)):
                            survived = False
                            break
                        if not self.check_survival(s_next):
                            survived = False
                            break
                        if step + 1 < len(real_states):
                            step_err = np.abs(s_next - real_states[step + 1]) / self.state_std
                            step_errors.append(step_err)
                        s_cur = s_next
                    except Exception:
                        survived = False
                        break

                if step_errors:
                    errors = np.array(step_errors)
                    nmae_overall = np.mean(np.mean(errors, axis=1))
                    nmae_list.append(nmae_overall)
                else:
                    nmae_list.append(float('nan'))
                survival_list.append(survived and len(step_errors) >= n - 1)

            valid_nmae = [x for x in nmae_list if not np.isnan(x)]
            results[h] = {
                'nmae_mean': float(np.nanmean(valid_nmae)) if valid_nmae else float('nan'),
                'survival_rate': float(np.mean(survival_list)),
            }

        return results

    @staticmethod
    def check_survival(state):
        for i, name in enumerate(STATE_NAMES_7D):
            if name in PHYSICAL_LIMITS:
                if abs(state[i]) > PHYSICAL_LIMITS[name]:
                    return False
        return not (np.any(np.isnan(state)) or np.any(np.isinf(state)))


def train_baseline_single_step(data, config, seed=42):
    """Train baseline with single-step loss only (for fair comparison)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    state_std = data['state_std']
    action_std = data['action_std']
    delta_std = data['delta_std']
    dt = 1.0 / 30.0
    train_eps = data['train_eps']
    episodes = data['episodes']

    # Flat single-step data
    train_obs = np.concatenate([episodes[ep]['obs'] for ep in train_eps])
    train_action = np.concatenate([episodes[ep]['action'] for ep in train_eps])
    train_deltas = np.concatenate([episodes[ep]['deltas'] for ep in train_eps])

    train_s = torch.FloatTensor(train_obs / state_std)
    train_a = torch.FloatTensor(train_action.reshape(-1, 1) / action_std)
    train_dsdot = torch.FloatTensor(train_deltas / (delta_std * dt))

    ds = torch.utils.data.TensorDataset(train_s, train_a, train_dsdot)
    loader = torch.utils.data.DataLoader(ds, batch_size=config['batch_size'], shuffle=True)

    model = ShootingODEFunc(
        hidden=config['hidden'],
        depth=config['depth'],
        activation=config['activation'],
    )

    opt = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config['n_epochs'])

    print(f"\nTraining single-step baseline (seed={seed})...")
    start_time = time.time()

    model.train()
    for epoch in range(config['n_epochs']):
        epoch_loss = 0.0
        n_batches = 0

        for sb, ab, yb in loader:
            pred = model(sb, ab)
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
            print(f"  Epoch {epoch+1}/{config['n_epochs']}: loss={avg_loss:.6f}, "
                  f"time={elapsed:.1f}s")

    model.eval()
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.1f}s")

    return model, state_std, action_std, delta_std


def evaluate_baseline(model, data, state_std, action_std, delta_std, horizons,
                      n_segments=5, seed=42):
    """Evaluate baseline model."""
    dt = 1.0 / 30.0
    episodes = data['episodes']
    test_eps = data['test_eps']

    np.random.seed(seed)
    segments = []
    for ep_idx in test_eps:
        ep = episodes[ep_idx]
        if ep['length'] >= max(horizons) + 10:
            segments.append(ep)
    segments = segments[:n_segments]

    results = {}
    for h in horizons:
        nmae_list = []
        survival_list = []

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
                    a_norm = torch.FloatTensor([[actions_seg[step] / action_std]])

                    with torch.no_grad():
                        dsdt_norm = model(s_norm, a_norm).numpy()[0]

                    dsdt = dsdt_norm * delta_std * dt
                    s_next = s_cur + dsdt

                    if np.any(np.isnan(s_next)) or np.any(np.isinf(s_next)):
                        survived = False
                        break
                    for i, name in enumerate(STATE_NAMES_7D):
                        if name in PHYSICAL_LIMITS and abs(s_next[i]) > PHYSICAL_LIMITS[name]:
                            survived = False
                            break
                    if not survived:
                        break
                    if step + 1 < len(real_states):
                        step_err = np.abs(s_next - real_states[step + 1]) / state_std
                        step_errors.append(step_err)
                    s_cur = s_next
                except Exception:
                    survived = False
                    break

            if step_errors:
                errors = np.array(step_errors)
                nmae_overall = np.mean(np.mean(errors, axis=1))
                nmae_list.append(nmae_overall)
            else:
                nmae_list.append(float('nan'))
            survival_list.append(survived and len(step_errors) >= n - 1)

        valid_nmae = [x for x in nmae_list if not np.isnan(x)]
        results[h] = {
            'nmae_mean': float(np.nanmean(valid_nmae)) if valid_nmae else float('nan'),
            'nmae_std': float(np.nanstd(valid_nmae)) if valid_nmae else float('nan'),
            'survival_rate': float(np.mean(survival_list)),
        }

    return results


def main():
    print("=" * 70)
    print("Multiple Shooting Neural ODE for Long-Horizon Error Reduction")
    print("=" * 70)

    # === Configuration ===
    shooting_config = {
        'hidden': 64,
        'depth': 3,
        'activation': 'tanh',
        'lr': 1e-3,
        'n_epochs': 200,
        'batch_size': 64,       # Number of segments per batch
        'segment_len': 15,      # Shooting interval: 15 steps
        'lambda_single': 0.2,   # Weight for single-step loss
        'lambda_segment': 1.0,  # Weight for segment rollout loss
        'lambda_final': 0.5,    # Weight for final-state penalty
        'curriculum': '0:5,50:10,100:15,150:20',  # Gradually increase segment length
    }

    baseline_config = {
        'hidden': 64,
        'depth': 3,
        'activation': 'tanh',
        'lr': 1e-3,
        'n_epochs': 200,
        'batch_size': 256,
    }

    # === Load Data ===
    print("\n1. Loading data...")
    shooter = MultipleShootingNeuralODE(shooting_config)
    data = shooter.load_data(seed=42)
    print(f"  Episodes: {len(data['episodes'])}")
    print(f"  Train: {len(data['train_eps'])}, Test: {len(data['test_eps'])}")

    horizons = [1, 10, 50, 100, 200, 500]

    # === Train Multiple Shooting Model ===
    print(f"\n{'='*70}")
    print("2. Training Multiple Shooting Neural ODE")
    print(f"{'='*70}")
    shooter.train(data, seed=42)

    # === Evaluate Multiple Shooting Model ===
    print(f"\n{'='*70}")
    print("3. Evaluating Multiple Shooting Model (full rollout)")
    print(f"{'='*70}")
    shooting_results = shooter.evaluate(data, horizons, n_segments=5, seed=42)

    print(f"\n{'Horizon':<10} {'NMAE':<12} {'Std':<12} {'Survival':<12}")
    print("-" * 46)
    for h in horizons:
        r = shooting_results[h]
        print(f"H={h:<7} {r['nmae_mean']:<12.4f} {r['nmae_std']:<12.4f} "
              f"{r['survival_rate']:<12.2%}")

    # === Evaluate with Shooting Anchors (diagnostic) ===
    print(f"\n{'='*70}")
    print("4. Evaluating with Shooting Anchors (every 15 steps)")
    print(f"{'='*70}")
    anchored_results = shooter.evaluate_with_shooting(
        data, horizons, shooting_interval=15, n_segments=5, seed=42
    )

    print(f"\n{'Horizon':<10} {'NMAE':<12} {'Survival':<12}")
    print("-" * 34)
    for h in horizons:
        r = anchored_results[h]
        print(f"H={h:<7} {r['nmae_mean']:<12.4f} {r['survival_rate']:<12.2%}")

    # === Train Single-Step Baseline ===
    print(f"\n{'='*70}")
    print("5. Training Single-Step Baseline (same architecture)")
    print(f"{'='*70}")
    baseline_model, bs_std, ba_std, bd_std = train_baseline_single_step(
        data, baseline_config, seed=42
    )

    print(f"\n{'='*70}")
    print("6. Evaluating Single-Step Baseline")
    print(f"{'='*70}")
    baseline_results = evaluate_baseline(
        baseline_model, data, bs_std, ba_std, bd_std, horizons, n_segments=5, seed=42
    )

    print(f"\n{'Horizon':<10} {'NMAE':<12} {'Std':<12} {'Survival':<12}")
    print("-" * 46)
    for h in horizons:
        r = baseline_results[h]
        print(f"H={h:<7} {r['nmae_mean']:<12.4f} {r['nmae_std']:<12.4f} "
              f"{r['survival_rate']:<12.2%}")

    # === Comparison ===
    print(f"\n{'='*70}")
    print("7. Comparison: Multiple Shooting vs Single-Step Baseline")
    print(f"{'='*70}")

    v9_nmae = {1: 0.0051, 10: 0.0628, 50: 0.4557, 100: 0.5064, 200: 0.4737, 500: 0.5529}

    print(f"\n{'Horizon':<10} {'v9 Baseline':<14} {'Single-Step':<14} {'Shooting':<14} "
          f"{'vs v9':<12} {'vs Single':<12}")
    print("-" * 76)
    for h in horizons:
        v9_val = v9_nmae.get(h, float('nan'))
        ss_val = baseline_results[h]['nmae_mean']
        ms_val = shooting_results[h]['nmae_mean']

        vs_v9 = ((v9_val - ms_val) / v9_val * 100) if v9_val > 0 and not np.isnan(ms_val) else float('nan')
        vs_ss = ((ss_val - ms_val) / ss_val * 100) if ss_val > 0 and not np.isnan(ms_val) else float('nan')

        print(f"H={h:<7} {v9_val:<14.4f} {ss_val:<14.4f} {ms_val:<14.4f} "
              f"{vs_v9:<12.1f}% {vs_ss:<12.1f}%")

    # Per-state analysis for shooting model at H=500
    print(f"\nPer-state NMAE at H=500 (Multiple Shooting):")
    print(f"{'State':<15} {'NMAE':<12}")
    print("-" * 27)
    if 500 in shooting_results and 'per_state_nmae' in shooting_results[500]:
        for name in STATE_NAMES_7D:
            val = shooting_results[500]['per_state_nmae'].get(name, {}).get('mean', float('nan'))
            print(f"{name:<15} {val:<12.4f}")

    # === Compute primary scores ===
    ms_primary = np.mean([shooting_results[h]['nmae_mean'] for h in [100, 200, 500]
                          if not np.isnan(shooting_results[h]['nmae_mean'])])
    ss_primary = np.mean([baseline_results[h]['nmae_mean'] for h in [100, 200, 500]
                          if not np.isnan(baseline_results[h]['nmae_mean'])])
    v9_primary = np.mean([v9_nmae[h] for h in [100, 200, 500]])

    print(f"\nPrimary Long-Horizon Score (avg H=100,200,500):")
    print(f"  v9 Baseline:       {v9_primary:.4f}")
    print(f"  Single-Step:       {ss_primary:.4f}")
    print(f"  Multiple Shooting: {ms_primary:.4f}")
    print(f"  Shooting vs v9:    {(v9_primary - ms_primary) / v9_primary * 100:.1f}% improvement")
    print(f"  Shooting vs Single: {(ss_primary - ms_primary) / ss_primary * 100:.1f}% improvement")

    # === Save Results ===
    output = {
        'timestamp': datetime.now().isoformat(),
        'run_id': '20260628_multiple_shooting',
        'experiment': 'multiple_shooting',
        'description': 'Multiple Shooting Neural ODE - segment-level loss training',
        'shooting_config': shooting_config,
        'baseline_config': baseline_config,
        'results': {
            'multiple_shooting': shooting_results,
            'single_step_baseline': baseline_results,
            'anchored_shooting': anchored_results,
        },
        'primary_scores': {
            'v9_baseline': float(v9_primary),
            'single_step': float(ss_primary),
            'multiple_shooting': float(ms_primary),
            'improvement_vs_v9_pct': float((v9_primary - ms_primary) / v9_primary * 100),
            'improvement_vs_single_pct': float((ss_primary - ms_primary) / ss_primary * 100),
        },
        'v9_reference': v9_nmae,
    }

    output_path = 'D:/系统辨识作业/sindy_bicycle/research_72h/05_candidates/EXP020_multiple_shooting.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
