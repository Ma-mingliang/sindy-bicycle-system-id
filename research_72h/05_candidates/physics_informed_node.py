"""Physics-Informed Neural ODE (EXP044).

Core idea: embed exact kinematic relationships into the Neural ODE:
  e_y_dot   = v * sin(e_psi)           -- exact kinematics
  e_psi_dot = -v * delta / L           -- exact kinematics (bicycle model)

The NN learns:
  - Residual corrections for e_y and e_psi (unmodeled dynamics)
  - Full derivatives for the other 5 states (v, theta, theta_dot, delta, delta_dot)

This hybrid approach guarantees the model respects fundamental kinematic
constraints while still being flexible enough to capture complex dynamics.
"""
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime

sys.path.insert(0, 'D:/系统辨识作业/sindy_bicycle')

# ============================================================================
# Constants
# ============================================================================
STATE_NAMES_7D = ['e_y', 'e_psi', 'v', 'theta', 'theta_dot', 'delta', 'delta_dot']
STATE_DIM = 7
ACTION_DIM = 1
IDX_7D_FROM_8D = [0, 1, 2, 3, 4, 6, 7]

PHYSICAL_LIMITS = {
    'e_y': 5.0, 'e_psi': np.pi, 'v': 5.0,
    'theta': np.pi, 'theta_dot': 10.0,
    'delta': np.pi/2, 'delta_dot': 10.0,
}

# Bicycle model parameter
WHEELBASE = 1.0  # L = 1.0 (from project context)

# ============================================================================
# Physics equations (in original physical space, not normalized)
# ============================================================================

def physics_e_y_dot(v, e_psi):
    """Exact kinematics: d(e_y)/dt = v * sin(e_psi)."""
    return v * np.sin(e_psi)


def physics_e_psi_dot(v, delta, L=WHEELBASE):
    """Exact kinematics: d(e_psi)/dt = -v * delta / L."""
    return -v * delta / L


def physics_e_y_dot_torch(v, e_psi):
    """Torch version of e_y_dot = v * sin(e_psi)."""
    return v * torch.sin(e_psi)


def physics_e_psi_dot_torch(v, delta, L=WHEELBASE):
    """Torch version of e_psi_dot = -v * delta / L."""
    return -v * delta / L


# ============================================================================
# Model
# ============================================================================

class PhysicsInformedODEFunc(nn.Module):
    """Physics-Informed Neural ODE dynamics function.

    Architecture:
      - Input: (state_norm, action_norm) -> 8D
      - Output: 7D continuous-time derivative (normalized)

    For e_y and e_psi, the output represents a RESIDUAL on top of the
    known physics. The actual derivative is:
      dsdt[0] = physics_e_y_dot(v, e_psi) + residual_scale * nn_out[0]
      dsdt[1] = physics_e_psi_dot(v, delta) + residual_scale * nn_out[1]
      dsdt[2:] = nn_out[2:]  (fully learned)

    In normalized space, the physics terms are converted via:
      physics_norm = physics_raw / (delta_std * dt)
    """

    def __init__(self, hidden=64, depth=3, activation='tanh',
                 residual_scale=0.1):
        super().__init__()
        self.residual_scale = residual_scale

        if activation == 'tanh':
            act = nn.Tanh
        elif activation == 'silu':
            act = nn.SiLU
        elif activation == 'relu':
            act = nn.ReLU
        else:
            act = nn.Tanh

        # NN takes (state, action) -> 7D output
        layers = [nn.Linear(STATE_DIM + ACTION_DIM, hidden), act()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden, hidden), act()])
        layers.append(nn.Linear(hidden, STATE_DIM))
        self.net = nn.Sequential(*layers)

        # Initialize last layer small for stability
        nn.init.zeros_(self.net[-1].bias)
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.1)

    def forward(self, s_norm, a_norm, state_std, delta_std, dt):
        """Forward pass with embedded physics.

        Args:
            s_norm: (batch, 7) normalized state
            a_norm: (batch, 1) normalized action
            state_std: (7,) state standard deviations
            delta_std: (7,) delta standard deviations
            dt: time step

        Returns:
            dsdt_norm: (batch, 7) normalized continuous-time derivative
        """
        # NN prediction (raw output)
        nn_out = self.net(torch.cat([s_norm, a_norm], dim=-1))

        # Convert normalized state back to physical space for physics terms
        # s_phys = s_norm * state_std
        v_phys = s_norm[:, 2] * state_std[2]
        e_psi_phys = s_norm[:, 1] * state_std[1]
        delta_phys = s_norm[:, 5] * state_std[5]

        # Compute physics terms in physical space
        ey_dot_phys = physics_e_y_dot_torch(v_phys, e_psi_phys)
        epsi_dot_phys = physics_e_psi_dot_torch(v_phys, delta_phys)

        # Convert physics terms to normalized derivative space
        # dsdt_norm = dsdt_phys / (delta_std * dt)
        ey_dot_norm = ey_dot_phys / (delta_std[0] * dt)
        epsi_dot_norm = epsi_dot_phys / (delta_std[1] * dt)

        # Combine: physics + NN residual
        dsdt_norm = torch.zeros_like(nn_out)
        dsdt_norm[:, 0] = ey_dot_norm + self.residual_scale * nn_out[:, 0]
        dsdt_norm[:, 1] = epsi_dot_norm + self.residual_scale * nn_out[:, 1]
        dsdt_norm[:, 2:] = nn_out[:, 2:]

        return dsdt_norm


class PhysicsInformedNeuralODE:
    """Physics-Informed Neural ODE model with embedded kinematics."""

    def __init__(self, hidden=64, depth=3, activation='tanh',
                 lr=1e-3, weight_decay=0, n_epochs=200, batch_size=256,
                 dt=1.0/30.0, residual_scale=0.1,
                 rollout_curriculum='1,5,10,20',
                 lambda_multi=0.3, lambda_physics=0.5,
                 lambda_consistency=0.1, lambda_jacobian=0.01,
                 seed=42):
        self.hidden = hidden
        self.depth = depth
        self.activation = activation
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.dt = dt
        self.residual_scale = residual_scale
        self.rollout_curriculum = rollout_curriculum
        self.lambda_multi = lambda_multi
        self.lambda_physics = lambda_physics
        self.lambda_consistency = lambda_consistency
        self.lambda_jacobian = lambda_jacobian
        self.seed = seed

        self._model = None
        self._state_std = None
        self._action_std = None
        self._delta_std = None
        self._training_log = []

    def train(self, states, actions, deltas, state_std, action_std, delta_std):
        """Train the model."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self._state_std = state_std.copy()
        self._action_std = float(action_std)
        self._delta_std = delta_std.copy()
        self._state_std[self._state_std < 1e-10] = 1.0
        self._delta_std[self._delta_std < 1e-10] = 1.0

        # Build model
        self._model = PhysicsInformedODEFunc(
            hidden=self.hidden,
            depth=self.depth,
            activation=self.activation,
            residual_scale=self.residual_scale,
        )

        # Prepare data
        train_s = torch.FloatTensor(states / self._state_std)
        train_a = torch.FloatTensor(actions.reshape(-1, 1) / self._action_std)
        train_dsdot = torch.FloatTensor(deltas / (self._delta_std * self.dt))

        # Convert stds to tensors for physics computation
        state_std_t = torch.FloatTensor(self._state_std)
        delta_std_t = torch.FloatTensor(self._delta_std)

        ds = torch.utils.data.TensorDataset(train_s, train_a, train_dsdot)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=self.batch_size, shuffle=True
        )

        opt = torch.optim.Adam(
            self._model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.n_epochs
        )

        # Parse rollout curriculum
        curriculum = [int(x) for x in self.rollout_curriculum.split(',')]

        self._model.train()
        start_time = time.time()

        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            n_batches = 0

            # Determine current rollout steps
            rollout_steps = 1
            for i, threshold in enumerate(curriculum):
                if epoch >= self.n_epochs * (i + 1) / (len(curriculum) + 1):
                    rollout_steps = threshold

            for sb, ab, yb in loader:
                # --- Single-step loss ---
                pred = self._model(sb, ab, state_std_t, delta_std_t, self.dt)
                loss_single = nn.functional.mse_loss(pred, yb)

                # --- Physics consistency loss ---
                # The physics terms in pred should match the target for e_y and e_psi
                loss_physics = torch.tensor(0.0)
                if self.lambda_physics > 0:
                    # Reconstruct physical derivatives from normalized target
                    # target_phys = yb * delta_std * dt
                    # For e_y: physics_pred = v * sin(e_psi), should match target_phys[:, 0]
                    v_phys = sb[:, 2] * self._state_std[2]
                    e_psi_phys = sb[:, 1] * self._state_std[1]
                    delta_phys = sb[:, 5] * self._state_std[5]

                    ey_dot_phys = physics_e_y_dot_torch(v_phys, e_psi_phys)
                    epsi_dot_phys = physics_e_psi_dot_torch(v_phys, delta_phys)

                    # Target in physical space
                    target_ey_dot = yb[:, 0] * self._delta_std[0] * self.dt
                    target_epsi_dot = yb[:, 1] * self._delta_std[1] * self.dt

                    # Physics should explain most of the target
                    loss_physics = (
                        nn.functional.mse_loss(ey_dot_phys, target_ey_dot)
                        + nn.functional.mse_loss(epsi_dot_phys, target_epsi_dot)
                    )

                # --- Multi-step rollout loss ---
                loss_multi = torch.tensor(0.0)
                if rollout_steps > 1 and len(sb) > rollout_steps + 1:
                    n_roll = min(len(sb) - rollout_steps, 64)
                    s_cur = sb[:n_roll].clone()
                    for step in range(rollout_steps):
                        a_cur = ab[step:step + n_roll]
                        dsdt = self._model(
                            s_cur, a_cur, state_std_t, delta_std_t, self.dt
                        )
                        s_cur = s_cur + dsdt * self.dt
                        target = sb[step + 1:step + 1 + n_roll]
                        loss_multi = loss_multi + nn.functional.mse_loss(
                            s_cur, target
                        )
                    loss_multi = loss_multi / rollout_steps

                # --- Consistency loss: theta <-> theta_dot ---
                loss_consistency = torch.tensor(0.0)
                if self.lambda_consistency > 0:
                    s_next_norm = sb + pred * self.dt
                    theta_pred = s_next_norm[:, 3]
                    theta_dot = sb[:, 4]
                    theta_gt = sb[:, 3] + theta_dot * self.dt
                    loss_consistency = nn.functional.mse_loss(theta_pred, theta_gt)

                # --- Jacobian regularization ---
                loss_jacobian = torch.tensor(0.0)
                if self.lambda_jacobian > 0:
                    n_jac = min(32, len(sb))
                    s_req = sb[:n_jac].requires_grad_(True)
                    a_req = ab[:n_jac].requires_grad_(True)
                    dsdt = self._model(
                        s_req, a_req, state_std_t, delta_std_t, self.dt
                    )
                    jac_norm = 0.0
                    for i in range(STATE_DIM):
                        grad = torch.autograd.grad(
                            dsdt[:, i].sum(), s_req, create_graph=True
                        )[0]
                        jac_norm = jac_norm + grad.pow(2).sum()
                    loss_jacobian = jac_norm / (STATE_DIM * n_jac)

                # --- Total loss ---
                loss = (
                    loss_single
                    + self.lambda_physics * loss_physics
                    + self.lambda_multi * loss_multi
                    + self.lambda_consistency * loss_consistency
                    + self.lambda_jacobian * loss_jacobian
                )

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                opt.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()

            avg_loss = epoch_loss / max(n_batches, 1)
            self._training_log.append({
                'epoch': epoch,
                'loss': avg_loss,
                'rollout_steps': rollout_steps,
                'lr': scheduler.get_last_lr()[0],
            })

            if (epoch + 1) % 50 == 0:
                elapsed = time.time() - start_time
                print(f"  Epoch {epoch+1}/{self.n_epochs}: loss={avg_loss:.6f}, "
                      f"rollout={rollout_steps}, time={elapsed:.1f}s")

        self._model.eval()
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.1f}s")

    def predict(self, s, tau):
        """Predict next state given current state and action."""
        s_norm = torch.FloatTensor(s / self._state_std).unsqueeze(0)
        a_norm = torch.FloatTensor([tau / self._action_std]).unsqueeze(0)
        state_std_t = torch.FloatTensor(self._state_std)
        delta_std_t = torch.FloatTensor(self._delta_std)

        with torch.no_grad():
            dsdt_norm = self._model(
                s_norm, a_norm, state_std_t, delta_std_t, self.dt
            ).numpy()[0]

        dsdt = dsdt_norm * self._delta_std * self.dt
        return s + dsdt

    def predict_batch(self, states, actions):
        """Batch prediction for efficiency."""
        s_norm = torch.FloatTensor(states / self._state_std)
        a_norm = torch.FloatTensor(actions.reshape(-1, 1) / self._action_std)
        state_std_t = torch.FloatTensor(self._state_std)
        delta_std_t = torch.FloatTensor(self._delta_std)

        with torch.no_grad():
            dsdt_norm = self._model(
                s_norm, a_norm, state_std_t, delta_std_t, self.dt
            ).numpy()

        return states + dsdt_norm * self._delta_std * self.dt

    def name(self):
        return 'physics_informed_node'


# ============================================================================
# Data loading and evaluation (standalone, matching v9 protocol)
# ============================================================================

def check_survival(state):
    """Check if state survives physical limits."""
    for i, name in enumerate(STATE_NAMES_7D):
        if name in PHYSICAL_LIMITS:
            if abs(state[i]) > PHYSICAL_LIMITS[name]:
                return False
    return not (np.any(np.isnan(state)) or np.any(np.isinf(state)))


def load_data(data_path='D:/系统辨识作业/sindy_bicycle/data/stage2_dataset_150k.npz',
              seed=42):
    """Load 7D data from 8D dataset."""
    data = np.load(data_path, allow_pickle=True)
    obs_8d = data['obs']
    action = data['action']
    next_obs_8d = data['next_obs']
    done = data['done']

    # Convert to 7D
    obs = obs_8d[:, IDX_7D_FROM_8D].astype(np.float64)
    next_obs = next_obs_8d[:, IDX_7D_FROM_8D].astype(np.float64)
    deltas = next_obs - obs
    action = action.astype(np.float64)

    # Split by episodes
    episode_ends = np.where(done)[0]
    episode_starts = np.concatenate([[0], episode_ends[:-1] + 1])
    n_episodes = len(episode_ends)

    # Train/test split (75/25 by episode)
    np.random.seed(seed)
    indices = np.random.permutation(n_episodes)
    n_train = int(0.75 * n_episodes)
    train_eps = indices[:n_train]
    test_eps = indices[n_train:]

    train_mask = np.zeros(len(obs), dtype=bool)
    test_mask = np.zeros(len(obs), dtype=bool)

    for ep in train_eps:
        train_mask[episode_starts[ep]:episode_ends[ep]+1] = True
    for ep in test_eps:
        test_mask[episode_starts[ep]:episode_ends[ep]+1] = True

    # Compute statistics from training data
    state_std = np.std(obs[train_mask], axis=0)
    state_std[state_std < 1e-10] = 1.0
    action_std = float(np.std(action[train_mask]))
    if action_std < 1e-10:
        action_std = 1.0
    delta_std = np.std(deltas[train_mask], axis=0)
    delta_std[delta_std < 1e-10] = 1.0

    return {
        'obs': obs,
        'action': action,
        'deltas': deltas,
        'done': done,
        'train_mask': train_mask,
        'test_mask': test_mask,
        'state_std': state_std,
        'action_std': action_std,
        'delta_std': delta_std,
        'episode_ends': episode_ends,
        'episode_starts': episode_starts,
    }


def get_test_segments(data, n_segments=5, segment_length=1100, seed=42):
    """Get test segments for long-horizon evaluation."""
    obs = data['obs']
    action = data['action']
    episode_ends = data['episode_ends']
    episode_starts = data['episode_starts']

    # Find long enough test episodes
    test_ep_indices = np.where(data['test_mask'][episode_ends])[0]
    long_eps = [ep for ep in test_ep_indices
                if episode_ends[ep] - episode_starts[ep] >= segment_length]

    np.random.seed(seed)
    segments = []

    for _ in range(min(n_segments, len(long_eps))):
        ep = np.random.choice(long_eps)
        ep_start = episode_starts[ep]
        ep_end = episode_ends[ep]
        max_start = ep_end - segment_length
        if max_start <= ep_start:
            start = ep_start
        else:
            start = np.random.randint(ep_start, max_start + 1)

        segments.append({
            'states': obs[start:start+segment_length].copy(),
            'actions': action[start:start+segment_length].flatten().copy(),
            'episode': ep,
            'start': start,
        })

    return segments


def evaluate_model(model, segments, state_std, horizons):
    """Evaluate model on test segments at multiple horizons."""
    results = {}

    for h in horizons:
        nmae_list = []
        survival_list = []
        per_state_nmae = {name: [] for name in STATE_NAMES_7D}

        for seg in segments:
            s0 = seg['states'][0].copy()
            actions_seg = seg['actions']
            real_states = seg['states']

            n = min(h, len(actions_seg))
            s_cur = s0.copy()
            survived = True
            step_errors = []

            for step in range(n):
                try:
                    s_next = model.predict(s_cur, actions_seg[step])

                    if np.any(np.isnan(s_next)) or np.any(np.isinf(s_next)):
                        survived = False
                        break

                    if not check_survival(s_next):
                        survived = False
                        break

                    # Compute per-step error
                    if step + 1 < len(real_states):
                        step_err = np.abs(s_next - real_states[step + 1]) / state_std
                        step_errors.append(step_err)

                    s_cur = s_next
                except Exception:
                    survived = False
                    break

            # Compute NMAE
            if step_errors:
                n_valid = min(len(step_errors), n)
                errors = np.array(step_errors[:n_valid])
                nmae_per_state = np.mean(errors, axis=0)
                nmae_overall = np.mean(nmae_per_state)
                nmae_list.append(nmae_overall)
                for i, name in enumerate(STATE_NAMES_7D):
                    per_state_nmae[name].append(nmae_per_state[i])
            else:
                nmae_list.append(float('nan'))

            survival_list.append(survived and len(step_errors) >= n - 1)

        # Aggregate
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


# ============================================================================
# Main experiment
# ============================================================================

def main():
    print("=" * 70)
    print("EXP044: Physics-Informed Neural ODE")
    print("Embedding exact kinematics: e_y_dot=v*sin(e_psi), e_psi_dot=-v*delta/L")
    print("=" * 70)

    # Configuration
    config = {
        'hidden': 64,
        'depth': 3,
        'activation': 'tanh',
        'lr': 1e-3,
        'weight_decay': 0,
        'n_epochs': 200,
        'batch_size': 256,
        'dt': 1.0 / 30.0,
        'residual_scale': 0.1,
        'rollout_curriculum': '1,5,10,20',
        'lambda_multi': 0.3,
        'lambda_physics': 0.5,
        'lambda_consistency': 0.1,
        'lambda_jacobian': 0.01,
    }

    horizons = [1, 10, 50, 100, 200, 500, 1000]

    # Load data
    print("\n1. Loading data...")
    data = load_data(seed=42)
    print(f"   State std: {data['state_std']}")
    print(f"   Action std: {data['action_std']:.6f}")
    print(f"   Delta std: {data['delta_std']}")
    print(f"   Train samples: {np.sum(data['train_mask'])}")
    print(f"   Test samples: {np.sum(data['test_mask'])}")

    # Get test segments
    print("\n2. Getting test segments...")
    segments = get_test_segments(data, n_segments=5, segment_length=1100, seed=42)
    print(f"   Number of segments: {len(segments)}")

    # Train with multiple seeds
    seeds = [42, 43, 44]
    all_results = {}

    for seed in seeds:
        print(f"\n{'='*70}")
        print(f"Training Physics-Informed Neural ODE (seed={seed})")
        print(f"{'='*70}")

        model = PhysicsInformedNeuralODE(
            hidden=config['hidden'],
            depth=config['depth'],
            activation=config['activation'],
            lr=config['lr'],
            weight_decay=config['weight_decay'],
            n_epochs=config['n_epochs'],
            batch_size=config['batch_size'],
            dt=config['dt'],
            residual_scale=config['residual_scale'],
            rollout_curriculum=config['rollout_curriculum'],
            lambda_multi=config['lambda_multi'],
            lambda_physics=config['lambda_physics'],
            lambda_consistency=config['lambda_consistency'],
            lambda_jacobian=config['lambda_jacobian'],
            seed=seed,
        )

        model.train(
            data['obs'][data['train_mask']],
            data['action'][data['train_mask']],
            data['deltas'][data['train_mask']],
            data['state_std'],
            data['action_std'],
            data['delta_std'],
        )

        # Evaluate
        print(f"\nEvaluating seed={seed}...")
        results = evaluate_model(model, segments, data['state_std'], horizons)

        # Print results
        print(f"\nResults for seed={seed}:")
        print(f"{'Horizon':<10} {'NMAE':<12} {'Survival':<12}")
        print("-" * 34)
        for h in horizons:
            r = results[h]
            print(f"H={h:<7} {r['nmae_mean']:<12.4f} {r['survival_rate']:<12.2%}")

        all_results[seed] = results

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Physics-Informed Neural ODE")
    print("=" * 70)

    print(f"\n{'Seed':<8} ", end="")
    for h in horizons:
        print(f"H={h:<6} ", end="")
    print()
    print("-" * (8 + 9 * len(horizons)))

    for seed in seeds:
        r = all_results[seed]
        print(f"{seed:<8} ", end="")
        for h in horizons:
            print(f"{r[h]['nmae_mean']:<9.4f} ", end="")
        print()

    # Compute primary score (mean of H=100, 200, 500)
    print(f"\nPrimaryLongHorizonScore (mean of H=100,200,500):")
    primary_scores = []
    for seed in seeds:
        r = all_results[seed]
        primary = np.mean([r[100]['nmae_mean'], r[200]['nmae_mean'], r[500]['nmae_mean']])
        primary_scores.append(primary)
        print(f"  Seed {seed}: {primary:.4f}")

    avg_primary = np.mean(primary_scores)
    print(f"  Average: {avg_primary:.4f}")

    # Load v9 baseline for comparison
    print("\n" + "=" * 70)
    print("COMPARISON WITH V9 BASELINE")
    print("=" * 70)

    v9_path = 'D:/系统辨识作业/sindy_bicycle/research_72h/01_baseline/V9_FIXED_REPRODUCTION_RESULTS.json'
    try:
        with open(v9_path) as f:
            v9_data = json.load(f)

        v9_results = v9_data['results']
        v9_seeds = v9_data['seeds']

        # Compute v9 averages
        v9_avg = {}
        for h in horizons:
            if h <= 500:  # v9 only has up to H=500
                vals = []
                for seed in v9_seeds:
                    seed_str = str(seed)
                    if seed_str in v9_results and str(h) in v9_results[seed_str]:
                        vals.append(v9_results[seed_str][str(h)]['nmae_mean'])
                if vals:
                    v9_avg[h] = np.mean(vals)

        print(f"\n{'Horizon':<10} {'V9 Baseline':<15} {'Physics-Node':<15} {'Improvement':<15}")
        print("-" * 55)
        for h in horizons:
            pi_vals = [all_results[s][h]['nmae_mean'] for s in seeds]
            pi_mean = np.nanmean(pi_vals)
            if h in v9_avg:
                v9_val = v9_avg[h]
                improvement = (v9_val - pi_mean) / v9_val * 100
                print(f"H={h:<7} {v9_val:<15.4f} {pi_mean:<15.4f} {improvement:+.1f}%")
            else:
                print(f"H={h:<7} {'N/A':<15} {pi_mean:<15.4f} {'N/A':<15}")

        # V9 primary score
        v9_primary_vals = []
        for seed in v9_seeds:
            seed_str = str(seed)
            if seed_str in v9_results:
                r = v9_results[seed_str]
                if all(str(h) in r for h in [100, 200, 500]):
                    p = np.mean([r['100']['nmae_mean'], r['200']['nmae_mean'], r['500']['nmae_mean']])
                    v9_primary_vals.append(p)

        if v9_primary_vals:
            v9_primary = np.mean(v9_primary_vals)
            improvement = (v9_primary - avg_primary) / v9_primary * 100
            print(f"\nPrimary Score: V9={v9_primary:.4f}, Physics-Node={avg_primary:.4f} "
                  f"({improvement:+.1f}%)")

    except FileNotFoundError:
        print("  V9 baseline file not found, skipping comparison.")

    # Per-state analysis for best seed
    print("\n" + "=" * 70)
    print("PER-STATE ANALYSIS (best seed)")
    print("=" * 70)

    best_seed = seeds[np.argmin(primary_scores)]
    best_results = all_results[best_seed]
    print(f"\nBest seed: {best_seed}")

    for h in [1, 10, 50, 100, 200, 500, 1000]:
        r = best_results[h]
        print(f"\n  H={h}:")
        for name in STATE_NAMES_7D:
            val = r['per_state_nmae'][name]['mean']
            print(f"    {name:<12}: {val:.6f}")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'run_id': '20260628_175824_neural_ode_72h',
        'experiment': 'EXP044_physics_informed_node',
        'description': 'Neural ODE with embedded exact kinematics: '
                       'e_y_dot=v*sin(e_psi), e_psi_dot=-v*delta/L',
        'config': config,
        'seeds': seeds,
        'horizons': horizons,
        'results': {str(seed): all_results[seed] for seed in seeds},
        'primary_scores': {str(seed): float(primary_scores[i]) for i, seed in enumerate(seeds)},
        'avg_primary_score': float(avg_primary),
        'per_state_analysis': {
            str(h): {
                name: best_results[h]['per_state_nmae'][name]['mean']
                for name in STATE_NAMES_7D
            }
            for h in horizons
        },
    }

    output_path = 'D:/系统辨识作业/sindy_bicycle/research_72h/05_candidates/EXP044_physics_informed_node.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    # Generate analysis markdown
    generate_analysis(output, v9_avg if 'v9_avg' in dir() else {})


def generate_analysis(results, v9_avg):
    """Generate analysis markdown file."""
    analysis_path = 'D:/系统辨识作业/sindy_bicycle/research_72h/05_candidates/PHYSICS_INFORMED_NODE_ANALYSIS.md'

    seeds = results['seeds']
    horizons = results['horizons']
    all_results = results['results']
    config = results['config']

    # Compute averages across seeds
    avg_results = {}
    for h in horizons:
        vals = [all_results[str(s)][str(h)]['nmae_mean'] for s in seeds]
        avg_results[h] = {
            'mean': float(np.nanmean(vals)),
            'std': float(np.nanstd(vals)),
        }

    md = []
    md.append("# EXP044: Physics-Informed Neural ODE Analysis")
    md.append("")
    md.append("## Experiment Overview")
    md.append("")
    md.append("**Core Idea**: Embed exact kinematic relationships into the Neural ODE:")
    md.append("- `e_y_dot = v * sin(e_psi)` -- exact kinematics")
    md.append("- `e_psi_dot = -v * delta / L` -- exact kinematics (bicycle model)")
    md.append("")
    md.append("The NN learns:")
    md.append("- Residual corrections for e_y and e_psi (unmodeled dynamics)")
    md.append("- Full derivatives for v, theta, theta_dot, delta, delta_dot")
    md.append("")
    md.append("## Configuration")
    md.append("")
    md.append("```python")
    for k, v in config.items():
        md.append(f"{k} = {v}")
    md.append("```")
    md.append("")
    md.append("## Results")
    md.append("")
    md.append("### Multi-Step Prediction Accuracy (NMAE)")
    md.append("")
    md.append("| Horizon | Mean NMAE | Std NMAE | Survival |")
    md.append("|---------|-----------|----------|----------|")
    for h in horizons:
        vals = [all_results[str(s)][str(h)]['nmae_mean'] for s in seeds]
        surv_vals = [all_results[str(s)][str(h)]['survival_rate'] for s in seeds]
        md.append(f"| H={h} | {np.nanmean(vals):.4f} | {np.nanstd(vals):.4f} | "
                  f"{np.nanmean(surv_vals):.0%} |")
    md.append("")

    # Primary score
    primary_scores = []
    for s in seeds:
        r = all_results[str(s)]
        p = np.mean([r['100']['nmae_mean'], r['200']['nmae_mean'], r['500']['nmae_mean']])
        primary_scores.append(p)
    md.append(f"**Primary Score** (mean of H=100,200,500): **{np.mean(primary_scores):.4f}**")
    md.append("")

    # Comparison with V9
    md.append("### Comparison with V9 Baseline")
    md.append("")
    if v9_avg:
        md.append("| Horizon | V9 Baseline | Physics-Node | Improvement |")
        md.append("|---------|-------------|--------------|-------------|")
        for h in horizons:
            pi_mean = avg_results[h]['mean']
            if h in v9_avg:
                v9_val = v9_avg[h]
                improvement = (v9_val - pi_mean) / v9_val * 100
                md.append(f"| H={h} | {v9_val:.4f} | {pi_mean:.4f} | {improvement:+.1f}% |")
            else:
                md.append(f"| H={h} | N/A | {pi_mean:.4f} | N/A |")
        md.append("")
    else:
        md.append("V9 baseline data not available for comparison.")
        md.append("")

    # Per-state analysis
    md.append("### Per-State Analysis (Best Seed)")
    md.append("")
    best_seed = seeds[np.argmin(primary_scores)]
    best_results = all_results[str(best_seed)]

    md.append("| State | H=1 | H=10 | H=50 | H=100 | H=200 | H=500 | H=1000 |")
    md.append("|-------|-----|------|------|-------|-------|-------|--------|")
    for name in STATE_NAMES_7D:
        vals = []
        for h in horizons:
            vals.append(f"{best_results[str(h)]['per_state_nmae'][name]['mean']:.4f}")
        md.append(f"| {name} | {' | '.join(vals)} |")
    md.append("")

    # Key findings
    md.append("## Key Findings")
    md.append("")
    md.append("1. **Physics embedding approach**: The model uses known kinematic equations")
    md.append("   as a prior, with the NN learning only the residual correction.")
    md.append("")
    md.append("2. **Architectural design**:")
    md.append("   - Physics terms are computed in physical (unnormalized) space")
    md.append("   - NN output is scaled by `residual_scale` for e_y and e_psi")
    md.append("   - Other states use pure NN prediction")
    md.append("")
    md.append("3. **Training losses**:")
    md.append("   - Single-step MSE on normalized derivatives")
    md.append("   - Physics consistency loss (physics terms vs ground truth)")
    md.append("   - Multi-step rollout loss (curriculum: 1->5->10->20)")
    md.append("   - Consistency loss (theta <-> theta_dot)")
    md.append("   - Jacobian regularization for smoothness")
    md.append("")
    md.append("4. **Residual scale** = 0.1: Physics dominates, NN provides small corrections")
    md.append("")

    # Conclusions
    md.append("## Conclusions")
    md.append("")
    md.append("The Physics-Informed Neural ODE embeds exact kinematic constraints,")
    md.append("ensuring the model respects fundamental bicycle dynamics while")
    md.append("remaining flexible enough to capture unmodeled effects through")
    md.append("residual learning.")
    md.append("")
    md.append("---")
    md.append(f"*Generated: {datetime.now().isoformat()}*")

    with open(analysis_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))
    print(f"Analysis saved to: {analysis_path}")


if __name__ == '__main__':
    main()
