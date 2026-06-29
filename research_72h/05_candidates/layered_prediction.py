"""Layered Prediction Architecture.

Based on deep root cause analysis:
- e_y: Physics integration (exact kinematics)
- e_psi: Physics integration (exact kinematics)
- v: Constant model (dv/dt ≈ 0)
- theta: Neural ODE (medium dynamics)
- delta: Neural ODE (medium dynamics)
- theta_dot: Derive from theta
- delta_dot: Derive from delta
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

WHEELBASE = 1.0
DT = 1.0 / 30.0


class ODEFunc(nn.Module):
    """Neural ODE for specific states."""
    def __init__(self, input_dim, output_dim, hidden=128, depth=4, activation='tanh'):
        super().__init__()
        act = nn.Tanh if activation == 'tanh' else nn.SiLU
        layers = [nn.Linear(input_dim, hidden), act()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden, hidden), act()])
        layers.append(nn.Linear(hidden, output_dim))
        self.net = nn.Sequential(*layers)

        nn.init.zeros_(self.net[-1].bias)
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.1)

    def forward(self, x):
        return self.net(x)


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


class LayeredPredictor:
    """Layered prediction model.

    Layers:
    1. Physics: e_y, e_psi (exact kinematics)
    2. Constant: v (dv/dt ≈ 0)
    3. Neural ODE: theta, delta (medium dynamics)
    4. Derivative: theta_dot, delta_dot (from theta, delta)
    """
    def __init__(self, state_std, action_std, delta_std, node_model=None):
        self._state_std = state_std
        self._action_std = action_std
        self._delta_std = delta_std
        self._node_model = node_model
        self._prev_theta = None
        self._prev_delta = None

    def predict(self, s, tau):
        """Predict next state using layered approach."""
        s_next = np.zeros(STATE_DIM)

        # Layer 1: Physics for e_y, e_psi
        # e_y_dot = v * sin(e_psi)
        v = s[2]
        e_psi = s[1]
        e_y_dot = v * np.sin(e_psi)
        s_next[0] = s[0] + e_y_dot * DT

        # e_psi_dot = -v * delta / L
        delta = s[6]
        e_psi_dot = -v * delta / WHEELBASE
        s_next[1] = s[1] + e_psi_dot * DT

        # Layer 2: Constant for v
        s_next[2] = s[2]  # dv/dt ≈ 0

        # Layer 3: Neural ODE for theta, delta
        if self._node_model is not None:
            s_norm = s / self._state_std
            a_norm = tau / self._action_std
            x = np.concatenate([s_norm, [a_norm]])
            x_torch = torch.FloatTensor(x).unsqueeze(0)

            with torch.no_grad():
                pred = self._node_model(x_torch).numpy()[0]

            # pred is normalized delta for [theta, delta]
            theta_delta = pred * self._delta_std[[3, 5]] * DT
            s_next[3] = s[3] + theta_delta[0]  # theta
            s_next[5] = s[5] + theta_delta[1]  # delta
        else:
            # Fallback: use simple Euler
            s_next[3] = s[3] + s[4] * DT  # theta += theta_dot * dt
            s_next[5] = s[5] + s[6] * DT  # delta += delta_dot * dt

        # Layer 4: Derive theta_dot, delta_dot
        if self._prev_theta is not None:
            s_next[4] = (s_next[3] - self._prev_theta) / DT  # theta_dot
            s_next[6] = (s_next[5] - self._prev_delta) / DT  # delta_dot
        else:
            s_next[4] = s[4]  # First step: keep current
            s_next[6] = s[6]

        # Update previous states
        self._prev_theta = s_next[3]
        self._prev_delta = s_next[5]

        return s_next


def train_node_for_theta_delta(data, config, seed=42):
    """Train Neural ODE for theta and delta."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    state_std = data['state_std']
    action_std = data['action_std']
    delta_std = data['delta_std']

    train_obs = data['train_obs']
    train_action = data['train_action']
    train_deltas = data['train_deltas']

    # Prepare input: all states + action
    X = np.hstack([train_obs / state_std, train_action.reshape(-1, 1) / action_std])
    # Prepare target: theta and delta only
    Y = train_deltas[:, [3, 5]] / (delta_std[[3, 5]] * DT)

    X_t = torch.FloatTensor(X)
    Y_t = torch.FloatTensor(Y)

    ds = torch.utils.data.TensorDataset(X_t, Y_t)
    loader = torch.utils.data.DataLoader(ds, batch_size=config['batch_size'], shuffle=True)

    model = ODEFunc(
        input_dim=STATE_DIM + ACTION_DIM,
        output_dim=2,  # theta and delta only
        hidden=config['hidden'],
        depth=config['depth'],
        activation=config['activation']
    )

    opt = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config['n_epochs'])

    print(f"Training NODE for theta, delta (seed={seed})...")
    start_time = time.time()

    model.train()
    for epoch in range(config['n_epochs']):
        epoch_loss = 0.0
        n_batches = 0

        for xb, yb in loader:
            pred = model(xb)
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
            print(f"  Epoch {epoch+1}/{config['n_epochs']}: loss={avg_loss:.6f}, time={elapsed:.1f}s")

    model.eval()
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.1f}s")

    return model


def evaluate_layered(predictor, data, horizons, n_segments=5, seed=42):
    """Evaluate layered predictor."""
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

            # Reset predictor state
            predictor._prev_theta = None
            predictor._prev_delta = None

            for step in range(n):
                try:
                    s_next = predictor.predict(s_cur, actions_seg[step])

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
    print("Layered Prediction Architecture")
    print("=" * 60)

    node_config = {
        'hidden': 128, 'depth': 4, 'activation': 'tanh',
        'lr': 1e-3, 'n_epochs': 200, 'batch_size': 256,
    }

    print("\n1. Loading data...")
    data = load_data(seed=42)

    print("\n2. Training NODE for theta, delta...")
    node_model = train_node_for_theta_delta(data, node_config, seed=42)

    print("\n3. Creating layered predictor...")
    predictor = LayeredPredictor(
        state_std=data['state_std'],
        action_std=data['action_std'],
        delta_std=data['delta_std'],
        node_model=node_model
    )

    print("\n4. Evaluating...")
    horizons = [1, 10, 50, 100, 200, 500, 1000]
    results = evaluate_layered(predictor, data, horizons)

    print("\n" + "=" * 60)
    print("RESULTS: Layered Prediction")
    print("=" * 60)
    print(f"\n{'Horizon':<10} {'NMAE':<12} {'Survival':<12}")
    print("-" * 34)
    for h in horizons:
        r = results[h]
        print(f"H={h:<7} {r['nmae_mean']:<12.4f} {r['survival_rate']:<12.2%}")

    # Compare with v9 baseline
    print(f"\nComparison with v9 baseline:")
    print(f"{'Horizon':<10} {'v9 NMAE':<12} {'Layered':<12} {'Improvement':<12}")
    print("-" * 46)
    v9_nmae = {1: 0.0051, 10: 0.0628, 50: 0.4557, 100: 0.5064, 200: 0.4737, 500: 0.5529, 1000: 0.6443}
    for h in horizons:
        v9_val = v9_nmae.get(h, float('nan'))
        l_val = results[h]['nmae_mean']
        if v9_val > 0 and not np.isnan(l_val):
            improvement = (v9_val - l_val) / v9_val * 100
            print(f"H={h:<7} {v9_val:<12.4f} {l_val:<12.4f} {improvement:<12.1f}%")

    # Per-state results for H=500
    if 500 in results:
        print(f"\nPer-state NMAE (H=500):")
        print("-" * 40)
        for name in STATE_NAMES_7D:
            mean = results[500]['per_state_nmae'][name]['mean']
            print(f"  {name:<12}: {mean:.4f}")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'run_id': '20260628_175824_neural_ode_144h',
        'experiment': 'layered_prediction',
        'architecture': {
            'e_y': 'physics_integration',
            'e_psi': 'physics_integration',
            'v': 'constant',
            'theta': 'neural_ode',
            'delta': 'neural_ode',
            'theta_dot': 'derived_from_theta',
            'delta_dot': 'derived_from_delta',
        },
        'results': {str(h): {'nmae_mean': r['nmae_mean'], 'survival_rate': r['survival_rate']}
                   for h, r in results.items()},
    }

    output_path = 'D:/系统辨识作业/sindy_bicycle/research_72h/05_candidates/EXP043_layered_prediction.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
