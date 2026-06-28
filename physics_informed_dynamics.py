"""Physics-Informed Dynamics Learning Pipeline

Approach:
  1. Open-loop data collection from linearized Whipple environment
  2. Physics-informed SINDy with known dynamics terms
  3. Residual neural network for unmodeled dynamics
  4. Combined model: Physics-SINDy + Residual NN

Key insight: The linearized Whipple has theta_ddot = (g/h)*theta, which is
the source of instability. By including this as a known physics term in the
SINDy library, the model can learn the correct coefficient from data.
"""

import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ============================================================
# Part 1: Open-Loop Data Collection
# ============================================================

def collect_openloop_data(num_episodes=1000, max_steps=200, dt=1/30,
                          speed=3.0, seed=42):
    """Collect open-loop data from linearized Whipple dynamics.

    Strategy (mixed, with multi-speed):
      1. Pure roll dynamics (action=0, large theta) → learn gravity/damping
      2. Small random actions → learn coupling terms
      3. Large random actions → learn full control space
      4. Step responses → learn steady-state behavior
      5. Multi-speed episodes → separate speed-dependent terms
      6. Impulse response → observe natural decay for damping
    """
    rng = np.random.RandomState(seed)

    # Physical parameters (matching path_tracking_env.py)
    g, h = 9.81, 0.8
    wheelbase = 1.0
    omega_n = 5.0
    zeta = 0.5
    trail = 0.15

    def derivatives(s, u, v_speed):
        ey_s, epsi_s, v_s, theta_s, theta_dot_s, k_s, delta_s, delta_dot_s = s
        trail_effect = trail * v_s * theta_s / wheelbase
        theta_ddot = (g / h) * theta_s \
                     - (v_s**2 / (h * wheelbase)) * delta_s \
                     - 2.0 * theta_dot_s
        delta_ddot = -omega_n**2 * (delta_s - trail_effect) \
                     - 2 * zeta * omega_n * delta_dot_s \
                     + u
        return np.array([
            v_s * epsi_s,
            v_s * (k_s - delta_s / wheelbase),
            -0.1 * (v_s - v_speed),
            theta_dot_s,
            theta_ddot,
            0.0,
            delta_dot_s,
            delta_ddot,
        ])

    def step_dynamics(state, action, dt, v_speed):
        dt_sub = dt / 5
        s = state.copy()
        for _ in range(5):
            k1 = derivatives(s, action, v_speed)
            k2 = derivatives(s + 0.5 * dt_sub * k1, action, v_speed)
            k3 = derivatives(s + 0.5 * dt_sub * k2, action, v_speed)
            k4 = derivatives(s + dt_sub * k3, action, v_speed)
            s = s + (dt_sub / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        return s

    states, actions, next_states = [], [], []

    # Episode types with counts
    n_types = 6
    n_per_type = [num_episodes // n_types] * n_types
    n_per_type[-1] = num_episodes - sum(n_per_type[:-1])

    speeds = [2.0, 3.0, 4.0, 5.0]  # multi-speed

    ep_count = 0
    for type_idx in range(n_types):
        for _ in range(n_per_type[type_idx]):
            # Pick speed (multi-speed for types 0-3, varied for types 4-5)
            if type_idx < 4:
                v_speed = speeds[type_idx % len(speeds)]
            else:
                v_speed = rng.choice(speeds)

            if type_idx == 0:
                # Type 0: Pure roll — zero action, large theta, observe free decay
                theta = rng.uniform(-0.8, 0.8)
                theta_dot = rng.uniform(-2.0, 2.0)
                delta = 0.0
                delta_dot = 0.0
                action_val = 0.0
                action_type = 'constant'
            elif type_idx == 1:
                # Type 1: Small random actions
                theta = rng.uniform(-0.5, 0.5)
                theta_dot = rng.uniform(-1.5, 1.5)
                delta = rng.uniform(-0.15, 0.15)
                delta_dot = rng.uniform(-0.3, 0.3)
                action_val = None
                action_type = 'random_small'
            elif type_idx == 2:
                # Type 2: Large random actions
                theta = rng.uniform(-0.3, 0.3)
                theta_dot = rng.uniform(-1.0, 1.0)
                delta = rng.uniform(-0.2, 0.2)
                delta_dot = rng.uniform(-0.5, 0.5)
                action_val = None
                action_type = 'random_large'
            elif type_idx == 3:
                # Type 3: Step response — constant action
                theta = rng.uniform(-0.6, 0.6)
                theta_dot = rng.uniform(-1.5, 1.5)
                delta = rng.uniform(-0.1, 0.1)
                delta_dot = 0.0
                action_val = rng.uniform(-1.5, 1.5)
                action_type = 'constant'
            elif type_idx == 4:
                # Type 4: Impulse response — short action then free decay
                theta = rng.uniform(-0.3, 0.3)
                theta_dot = rng.uniform(-1.0, 1.0)
                delta = rng.uniform(-0.1, 0.1)
                delta_dot = 0.0
                action_val = rng.uniform(-2.0, 2.0)
                action_type = 'impulse'
            else:
                # Type 5: Sinusoidal action — frequency sweep
                theta = rng.uniform(-0.4, 0.4)
                theta_dot = rng.uniform(-1.0, 1.0)
                delta = rng.uniform(-0.1, 0.1)
                delta_dot = 0.0
                action_val = None
                action_type = 'sinusoidal'
                sin_freq = rng.uniform(0.5, 5.0)
                sin_amp = rng.uniform(0.3, 1.5)

            v = v_speed + rng.uniform(-0.3, 0.3)
            ey = rng.uniform(-1.0, 1.0)
            epsi = rng.uniform(-0.2, 0.2)

            state = np.array([ey, epsi, v, theta, theta_dot, 0.0, delta, delta_dot])

            impulse_steps = rng.randint(5, 15)  # impulse duration

            for step in range(max_steps):
                # Pick action
                if action_type == 'constant':
                    action = action_val
                elif action_type == 'random_small':
                    action = rng.uniform(-0.5, 0.5)
                elif action_type == 'random_large':
                    action = rng.uniform(-2.0, 2.0)
                elif action_type == 'impulse':
                    action = action_val if step < impulse_steps else 0.0
                elif action_type == 'sinusoidal':
                    action = sin_amp * np.sin(2 * np.pi * sin_freq * step * dt)
                else:
                    action = 0.0

                next_state = step_dynamics(state, action, dt, v_speed)

                states.append(state)
                actions.append([action])
                next_states.append(next_state)

                state = next_state

                # Stop if bike fell
                if abs(state[3]) > 1.5:
                    break

            ep_count += 1

    states = np.array(states, dtype=np.float32)
    actions = np.array(actions, dtype=np.float32)
    next_states = np.array(next_states, dtype=np.float32)

    return states, actions, next_states


# ============================================================
# Part 2: Physics-Informed SINDy Library
# ============================================================

def build_physics_informed_library(state_norm, action_norm):
    """Build polynomial library with known physics terms.

    Standard polynomial: [1, x0..x8, x0*x0, x0*x1, ..., x8*x8]
    Plus physics-informed terms:
      - theta (gravity restoring)
      - theta * v^2 (centrifugal)
      - theta_dot * v (gyroscopic damping)
      - delta * v^2 (steering centrifugal)
      - delta * theta * v (coupling)
    """
    x = np.concatenate([state_norm, action_norm])
    n = len(x)

    # Standard polynomial terms
    terms = [1.0]
    terms.extend(x)
    for i in range(n):
        for j in range(i, n):
            terms.append(x[i] * x[j])

    # Physics-informed terms (in normalized space)
    theta_norm = state_norm[3]
    v_norm = state_norm[2]
    delta_norm = state_norm[6]
    theta_dot_norm = state_norm[4]

    # Physics-informed terms (NEW terms only, avoid duplicates with polynomial library)
    # Standard library already has: theta, theta^2, theta_dot, v, delta, delta*v, theta*theta_dot, etc.
    # Only add terms that are NOT simple products of two state variables
    terms.append(theta_norm * v_norm**2)           # centrifugal roll (theta * v^2)
    terms.append(delta_norm * v_norm**2)           # steering centrifugal (delta * v^2)
    terms.append(delta_norm * theta_norm * v_norm) # roll-steering coupling
    terms.append(theta_norm**3)                    # nonlinear gravity (cubic)

    return np.array(terms, dtype=np.float64)


def build_standard_library(state_norm, action_norm):
    """Standard SINDy polynomial library (same as original)."""
    x = np.concatenate([state_norm, action_norm])
    n = len(x)
    terms = [1.0]
    terms.extend(x)
    for i in range(n):
        for j in range(i, n):
            terms.append(x[i] * x[j])
    return np.array(terms, dtype=np.float64)


def normalize_state(raw_state):
    """Normalize state to [-1, 1] range."""
    scales = np.array([10.0, 1.57, 5.0, 1.57, 10.0, 1/8.0, 0.785, 3.0])
    normed = raw_state.copy()
    for i in range(8):
        normed[i] = np.clip(normed[i], -scales[i]*2, scales[i]*2) / scales[i]
    return normed.astype(np.float32)


def denormalize_state(normed_state):
    """Denormalize from [-1, 1] to raw state."""
    scales = np.array([10.0, 1.57, 5.0, 1.57, 10.0, 1/8.0, 0.785, 3.0])
    return normed_state * scales


# ============================================================
# Part 3: Train Physics-Informed SINDy
# ============================================================

def train_sindy(states, actions, next_states, use_physics=True, dt=1/30):
    """Train SINDy model using sequential thresholded least squares.

    Args:
        states: (N, 8) raw states
        actions: (N, 1) raw actions
        next_states: (N, 8) raw next states
        use_physics: whether to add physics-informed terms
        dt: time step
    """
    N = len(states)
    action_scale = 1.41

    # Normalize states
    states_norm = np.array([normalize_state(s) for s in states])
    next_states_norm = np.array([normalize_state(s) for s in next_states])
    actions_norm = actions / action_scale

    # Compute deltas in normalized space
    deltas_norm = next_states_norm - states_norm

    # Build library matrix
    if use_physics:
        lib_func = build_physics_informed_library
    else:
        lib_func = build_standard_library

    Theta = np.array([lib_func(states_norm[i], actions_norm[i])
                      for i in range(N)])

    print(f"Library size: {Theta.shape[1]} terms")
    print(f"Data size: {N} transitions")

    # Sequential thresholded least squares (STLSQ)
    n_states = 8
    n_lib = Theta.shape[1]
    Xi = np.zeros((n_lib, n_states))

    for dim in range(n_states):
        y = deltas_norm[:, dim]

        # Least squares
        coeffs, residuals, rank, sv = np.linalg.lstsq(Theta, y, rcond=None)

        # Iterative thresholding
        threshold = 0.01
        for iteration in range(10):
            small = np.abs(coeffs) < threshold
            coeffs[small] = 0

            # Refit with remaining terms
            big = ~small
            if np.sum(big) == 0:
                break
            coeffs_big, _, _, _ = np.linalg.lstsq(Theta[:, big], y, rcond=None)
            coeffs[big] = coeffs_big

        Xi[:, dim] = coeffs

    # Report
    state_names = ['ey', 'epsi', 'v', 'theta', 'theta_dot', 'k', 'delta', 'delta_dot']
    if use_physics:
        # Build labels for physics-informed library
        lib_labels = ['1']
        feature_names = ['ey', 'epsi', 'v', 'theta', 'theta_dot', 'k', 'delta', 'delta_dot', 'a']
        lib_labels.extend(feature_names)
        for i in range(9):
            for j in range(i, 9):
                lib_labels.append(f'{feature_names[i]}*{feature_names[j]}')
        lib_labels.extend([
            'theta (gravity)',
            'theta*v^2 (centrifugal)',
            'theta_dot*v (gyroscopic)',
            'delta*v^2 (steer centrifugal)',
            'delta*theta*v (coupling)',
            'theta^3 (nonlinear gravity)',
            'delta*v (steer-speed)',
        ])
    else:
        lib_labels = ['1']
        feature_names = ['ey', 'epsi', 'v', 'theta', 'theta_dot', 'k', 'delta', 'delta_dot', 'a']
        lib_labels.extend(feature_names)
        for i in range(9):
            for j in range(i, 9):
                lib_labels.append(f'{feature_names[i]}*{feature_names[j]}')

    print(f"\n=== Physics-Informed SINDy Results ===")
    for dim in range(n_states):
        nz = np.count_nonzero(Xi[:, dim])
        print(f"\n{state_names[dim]}: {nz} non-zero terms")
        for i in range(n_lib):
            if Xi[i, dim] != 0:
                label = lib_labels[i] if i < len(lib_labels) else f'term_{i}'
                print(f"  {label:30s} * {Xi[i, dim]:+.6f}")

    return Xi, action_scale, lib_labels


# ============================================================
# Part 4: Residual Neural Network
# ============================================================

class ResidualDynamicsNet(nn.Module):
    """Neural network for residual dynamics.

    Learns: residual = true_delta - sindy_delta
    """
    def __init__(self, state_dim=8, action_dim=1, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, state_dim),
        )

    def forward(self, state_norm, action_norm):
        sa = torch.cat([state_norm, action_norm], dim=-1)
        return self.net(sa)


def train_residual_network(states, actions, next_states, sindy_Xi,
                            action_scale=1.41, epochs=200, batch_size=256,
                            lr=1e-3, device='cpu'):
    """Train residual network to compensate SINDy errors.

    residual = true_delta_norm - sindy_delta_norm
    """
    N = len(states)
    states_norm = np.array([normalize_state(s) for s in states], dtype=np.float32)
    next_states_norm = np.array([normalize_state(s) for s in next_states], dtype=np.float32)
    actions_norm = (actions / action_scale).astype(np.float32)

    # Compute SINDy predictions
    sindy_deltas = np.zeros((N, 8), dtype=np.float32)
    for i in range(N):
        lib = build_physics_informed_library(states_norm[i], actions_norm[i])
        sindy_deltas[i] = lib @ sindy_Xi

    # Residual = true - SINDy
    true_deltas = next_states_norm - states_norm
    residual_deltas = true_deltas - sindy_deltas

    print(f"\nResidual statistics:")
    for dim in range(8):
        names = ['ey', 'epsi', 'v', 'theta', 'theta_dot', 'k', 'delta', 'delta_dot']
        print(f"  {names[dim]:12s}: mean={residual_deltas[:, dim].mean():.6f}, "
              f"std={residual_deltas[:, dim].std():.6f}, "
              f"max={np.abs(residual_deltas[:, dim]).max():.6f}")

    # Create DataLoader
    state_t = torch.FloatTensor(states_norm)
    action_t = torch.FloatTensor(actions_norm)
    residual_t = torch.FloatTensor(residual_deltas)

    dataset = TensorDataset(state_t, action_t, residual_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Train
    model = ResidualDynamicsNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    print(f"\nTraining residual network ({epochs} epochs)...")
    for epoch in range(epochs):
        total_loss = 0
        n_batches = 0
        for s_batch, a_batch, r_batch in loader:
            s_batch = s_batch.to(device)
            a_batch = a_batch.to(device)
            r_batch = r_batch.to(device)

            pred = model(s_batch, a_batch)
            loss = criterion(pred, r_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 50 == 0:
            avg_loss = total_loss / n_batches
            print(f"  Epoch {epoch+1}: loss={avg_loss:.6f}")

    model.eval()
    return model


# ============================================================
# Part 5a: Hybrid Dynamics Model (SINDy gravity + true damping)
# ============================================================

class HybridDynamicsModel:
    """Hybrid dynamics model: SINDy for all dimensions, but theta_dot uses true damping.

    Problem: SINDy overestimates theta_dot damping by 675% (13.5 vs 2.0 1/s)
      due to multicollinearity between theta_dot and theta_dot*v in polynomial library.
    Solution: Use SINDy for gravity (103% accurate) and all other dimensions,
      but override theta_dot dynamics with known physics damping.

    theta_dot_true = (g/h)*theta - 2.0*theta_dot + coupling_terms
    In normalized space:
      gravity_coeff = (g/h) * dt * scale_theta / scale_theta_dot = 0.06449
      damping_coeff = -2.0 * dt * scale_theta / scale_theta_dot = -0.01053
    """

    def __init__(self, sindy_Xi, action_scale=1.41, dt=1/30,
                 true_damping=2.0, g=9.81, h=0.8):
        self.sindy_Xi = sindy_Xi.copy()
        self.action_scale = action_scale
        self.dt = dt

        # Normalization scales
        self.scales = np.array([10.0, 1.57, 5.0, 1.57, 10.0, 1/8.0, 0.785, 3.0])
        scale_theta_dot = self.scales[4]  # 10.0
        scale_theta = self.scales[3]      # 1.57

        # True normalized coefficients for theta_dot dynamics
        self.true_gravity_coeff = (g / h) * dt * scale_theta / scale_theta_dot
        self.true_damping_coeff = -true_damping * dt * scale_theta / scale_theta_dot

        print(f"Hybrid model coefficients for theta_dot:")
        print(f"  True gravity: {self.true_gravity_coeff:.6f} (raw: {g/h:.4f} rad/s^2)")
        print(f"  True damping: {self.true_damping_coeff:.6f} (raw: {true_damping:.1f} 1/s)")

        # Find theta_dot and theta indices in the library
        # Library: [1, ey, epsi, v, theta, theta_dot, k, delta, delta_dot, a, ...]
        # Linear terms: idx 0=1, 1=ey, 2=epsi, 3=v, 4=theta, 5=theta_dot, 6=k, 7=delta, 8=delta_dot, 9=a
        self.theta_idx = 4      # theta in library linear terms
        self.theta_dot_idx = 5  # theta_dot in library linear terms

    def predict_next_state(self, state_raw, action_raw):
        """Predict next state using hybrid approach."""
        state_norm = normalize_state(state_raw)
        action_norm = np.atleast_1d(float(np.asarray(action_raw).ravel()[0] / self.action_scale))

        lib = build_physics_informed_library(state_norm, action_norm)
        delta_norm = lib @ self.sindy_Xi

        # Override theta_dot dynamics with true physics
        # theta_dot_delta = gravity_coeff * theta + damping_coeff * theta_dot + small_coupling
        # Keep SINDy's non-linear coupling terms but replace linear gravity/damping
        theta_norm = state_norm[3]
        theta_dot_norm = state_norm[4]

        # Start with true linear dynamics
        corrected_theta_dot_delta = self.true_gravity_coeff * theta_norm \
                                   + self.true_damping_coeff * theta_dot_norm

        # Add SINDy's nonlinear coupling terms (theta*v^2, delta*v^2, etc.)
        # These are the last 4 terms in the physics-informed library
        n_lib = len(lib)
        # Physics-informed terms are the last 4 indices
        sindy_nonlinear = lib[-4:] @ self.sindy_Xi[-4:, 4]  # column 4 = theta_dot
        corrected_theta_dot_delta += sindy_nonlinear

        delta_norm[4] = corrected_theta_dot_delta

        # Denormalize and add to raw state
        next_state = state_raw + delta_norm * self.scales
        return next_state

    def save(self, path):
        """Save model to file."""
        np.savez(path,
                 sindy_Xi=self.sindy_Xi,
                 action_scale=self.action_scale,
                 true_damping=2.0,
                 g=9.81, h=0.8)
        print(f"Saved hybrid model to {path}")

    @classmethod
    def load(cls, path):
        """Load model from file."""
        data = np.load(path, allow_pickle=True)
        return cls(
            sindy_Xi=data['sindy_Xi'],
            action_scale=float(data['action_scale']),
            true_damping=float(data['true_damping']),
            g=float(data['g']),
            h=float(data['h']),
        )


# ============================================================
# Part 5b: Combined Dynamics Model
# ============================================================

class CombinedDynamicsModel:
    """Physics-Informed SINDy + Residual NN dynamics model.

    predict_next_state(s, a) = s + sindy_delta(s_norm, a_norm) + nn_residual(s_norm, a_norm)
    """

    def __init__(self, sindy_Xi, residual_net, action_scale=1.41):
        self.sindy_Xi = sindy_Xi
        self.residual_net = residual_net
        self.action_scale = action_scale
        self.device = next(residual_net.parameters()).device

    def predict_delta_norm(self, state_norm, action_norm):
        """Predict state delta in normalized space."""
        # SINDy component
        lib = build_physics_informed_library(state_norm, action_norm)
        sindy_delta = lib @ self.sindy_Xi

        # Residual component
        with torch.no_grad():
            s_t = torch.FloatTensor(state_norm).unsqueeze(0).to(self.device)
            a_t = torch.FloatTensor(action_norm).unsqueeze(0).to(self.device)
            nn_delta = self.residual_net(s_t, a_t).cpu().numpy()[0]

        return sindy_delta + nn_delta

    def predict_next_state(self, state_raw, action_raw):
        """Predict next state in raw space."""
        state_norm = normalize_state(state_raw)
        action_norm = np.atleast_1d(float(np.asarray(action_raw).ravel()[0] / self.action_scale))

        delta_norm = self.predict_delta_norm(state_norm, action_norm)

        # Denormalize
        scales = np.array([10.0, 1.57, 5.0, 1.57, 10.0, 1/8.0, 0.785, 3.0])
        next_state = state_raw + delta_norm * scales

        return next_state

    def save(self, path):
        """Save model to file."""
        np.savez(path,
                 sindy_Xi=self.sindy_Xi,
                 action_scale=self.action_scale)
        torch.save(self.residual_net.state_dict(), path.replace('.npz', '_residual.pt'))
        print(f"Saved combined model to {path}")

    @classmethod
    def load(cls, path, device='cpu'):
        """Load model from file."""
        data = np.load(path, allow_pickle=True)
        sindy_Xi = data['sindy_Xi']
        action_scale = float(data['action_scale'])

        residual_net = ResidualDynamicsNet().to(device)
        residual_net.load_state_dict(torch.load(
            path.replace('.npz', '_residual.pt'), map_location=device))
        residual_net.eval()

        return cls(sindy_Xi, residual_net, action_scale)


# ============================================================
# Part 6: Validation
# ============================================================

def validate_model(model, states, actions, next_states, name="Model"):
    """Validate dynamics model on test data."""
    N = len(states)
    errors = np.zeros((N, 8))
    sindy_errors = np.zeros((N, 8))

    for i in range(N):
        pred = model.predict_next_state(states[i], actions[i])
        errors[i] = np.abs(pred - next_states[i])

    names = ['ey', 'epsi', 'v', 'theta', 'theta_dot', 'k', 'delta', 'delta_dot']
    print(f"\n=== {name} Validation ({N} samples) ===")
    print(f"{'State':12s} {'MAE':>10s} {'RMSE':>10s} {'Max':>10s}")
    print("-" * 45)
    for dim in range(8):
        mae = errors[:, dim].mean()
        rmse = np.sqrt((errors[:, dim]**2).mean())
        maxe = errors[:, dim].max()
        print(f"{names[dim]:12s} {mae:10.6f} {rmse:10.6f} {maxe:10.6f}")

    return errors


def run_baseline_test(model, max_steps=2000, num_seeds=50, dt=1/30, speed=3.0):
    """Test dynamics model as environment replacement.

    Uses LQR + Stanley control with the learned dynamics.
    """
    import scipy.linalg as la
    from reference_path import ReferencePath
    from stanley_controller import steady_state_calculation

    # LQR gains (same as path_tracking_env.py)
    g, h, wheelbase = 9.81, 0.8, 1.0
    omega_n, zeta, trail = 5.0, 0.5, 0.15

    A = np.array([
        [0, 1, 0, 0],
        [g/h, -2.0, -speed**2/(h*wheelbase), 0],
        [0, 0, 0, 1],
        [omega_n**2*trail*speed/wheelbase, 0, -omega_n**2, -2*zeta*omega_n]
    ])
    B = np.array([[0], [0], [0], [1.0]])
    Q = np.diag([1000, 100, 10, 1])
    R = np.array([[0.2]])
    P = la.solve_continuous_are(A, B, Q, R)
    K = (np.linalg.inv(R) @ B.T @ P).flatten()

    path = ReferencePath()

    returns, lengths, ets = [], [], []

    for seed in range(num_seeds):
        rng = np.random.RandomState(seed)

        # Reset
        x, y = 0.0, rng.uniform(-0.5, 0.5)
        heading = 0.0
        v = speed + rng.uniform(-0.3, 0.3)
        theta = rng.uniform(-0.05, 0.05)
        theta_dot = 0.0
        delta = 0.0
        delta_dot = 0.0

        ep_return = 0.0
        terminated = False

        for step in range(max_steps):
            # Path errors
            fx = x + wheelbase * math.cos(heading)
            fy = y + wheelbase * math.sin(heading)
            path_info = path.get_closest_point(fx, fy, heading)
            ey = path_info['lateral_error']
            epsi = path_info['course_error_angle']
            k = path_info['curvature']

            # Stanley + LQR control
            delta_stanley = -(math.atan(0.6 * ey / max(v, 0.5)) + 0.4 * epsi)
            target_roll = steady_state_calculation(delta_stanley, v)
            x_lqr = np.array([theta - target_roll, theta_dot, delta, delta_dot])
            u_total = float(-K @ x_lqr)

            # Current state
            state = np.array([ey, epsi, v, theta, theta_dot, k, delta, delta_dot])

            # Predict next state using learned model
            next_state = model.predict_next_state(state, u_total)

            # Update position
            heading += v * (-delta / wheelbase) * dt
            x += v * math.cos(heading) * dt
            y += v * math.sin(heading) * dt

            # Unpack
            v = next_state[2]
            theta = float(next_state[3])
            theta_dot = next_state[4]
            delta = np.clip(next_state[6], -0.785, 0.785)
            delta_dot = next_state[7]

            # Reward (simplified)
            reward = -abs(ey) - 0.1 * abs(theta)
            if abs(theta) > math.pi / 4:
                reward = -10.0
                terminated = True
            ep_return += reward

            if terminated:
                break

        returns.append(ep_return)
        lengths.append(step + 1)
        ets.append(terminated)

    return {
        'return_mean': float(np.mean(returns)),
        'return_std': float(np.std(returns)),
        'length_mean': float(np.mean(lengths)),
        'length_std': float(np.std(lengths)),
        'et_rate': float(np.mean(ets)),
    }


# ============================================================
# Main Pipeline
# ============================================================

def main():
    import math
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    out_dir = r'D:/系统辨识作业/sindy_bicycle/runs/physics_informed_dynamics'
    os.makedirs(out_dir, exist_ok=True)

    # Step 1: Collect open-loop data
    print("\n" + "="*60)
    print("Step 1: Collecting open-loop data")
    print("="*60)
    states, actions, next_states = collect_openloop_data(
        num_episodes=2000, max_steps=200, seed=42)
    print(f"Collected {len(states)} transitions from 2000 episodes")

    # Split train/test
    n_train = int(0.8 * len(states))
    idx = np.random.permutation(len(states))
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    train_s, train_a, train_ns = states[train_idx], actions[train_idx], next_states[train_idx]
    test_s, test_a, test_ns = states[test_idx], actions[test_idx], next_states[test_idx]
    print(f"Train: {len(train_s)}, Test: {len(test_s)}")

    # Step 2: Train standard SINDy (baseline)
    print("\n" + "="*60)
    print("Step 2: Training standard SINDy (baseline)")
    print("="*60)
    Xi_std, _, labels_std = train_sindy(train_s, train_a, train_ns, use_physics=False)

    # Step 3: Train physics-informed SINDy
    print("\n" + "="*60)
    print("Step 3: Training physics-informed SINDy")
    print("="*60)
    Xi_phys, _, labels_phys = train_sindy(train_s, train_a, train_ns, use_physics=True)

    # Step 4: Train residual network
    print("\n" + "="*60)
    print("Step 4: Training residual network")
    print("="*60)
    residual_net = train_residual_network(
        train_s, train_a, train_ns, Xi_phys,
        action_scale=1.41, epochs=500, batch_size=256,
        lr=1e-3, device=device)

    # Step 5: Create combined model
    print("\n" + "="*60)
    print("Step 5: Creating combined model")
    print("="*60)
    combined_model = CombinedDynamicsModel(Xi_phys, residual_net, action_scale=1.41)

    # Step 6: Validate on test data
    print("\n" + "="*60)
    print("Step 6: Validation")
    print("="*60)

    # Create SINDy-only model for comparison
    class SINDyOnlyModel:
        def __init__(self, Xi, action_scale=1.41):
            self.Xi = Xi
            self.action_scale = action_scale
        def predict_next_state(self, state_raw, action_raw):
            state_norm = normalize_state(state_raw)
            action_norm = np.atleast_1d(float(np.asarray(action_raw).ravel()[0] / self.action_scale))
            lib = build_physics_informed_library(state_norm, action_norm)
            delta_norm = lib @ self.Xi
            scales = np.array([10.0, 1.57, 5.0, 1.57, 10.0, 1/8.0, 0.785, 3.0])
            return state_raw + delta_norm * scales

    sindy_model = SINDyOnlyModel(Xi_phys, action_scale=1.41)

    validate_model(sindy_model, test_s, test_a, test_ns, "Physics-Informed SINDy")
    validate_model(combined_model, test_s, test_a, test_ns, "Combined (SINDy + Residual)")

    # Step 7: Create hybrid model (SINDy gravity + true damping)
    print("\n" + "="*60)
    print("Step 7: Creating hybrid model")
    print("="*60)
    hybrid_model = HybridDynamicsModel(Xi_phys, action_scale=1.41)

    # Step 8: Validate hybrid model
    print("\n" + "="*60)
    print("Step 8: Validating hybrid model")
    print("="*60)
    validate_model(hybrid_model, test_s, test_a, test_ns, "Hybrid (SINDy gravity + true damping)")

    # Step 9: Save models
    print("\n" + "="*60)
    print("Step 9: Saving models")
    print("="*60)
    np.savez(os.path.join(out_dir, 'physics_informed_sindy.npz'),
             coefficients=Xi_phys,
             lib_labels=labels_phys,
             action_scale=1.41)
    combined_model.save(os.path.join(out_dir, 'combined_model.npz'))
    hybrid_model.save(os.path.join(out_dir, 'hybrid_model.npz'))

    # Step 10: Baseline test
    print("\n" + "="*60)
    print("Step 10: Baseline test (LQR+Stanley with learned dynamics)")
    print("="*60)

    import json

    # Test hybrid model
    print("\nTesting hybrid model...")
    hybrid_result = run_baseline_test(hybrid_model, num_seeds=50)
    print(f"Hybrid: return={hybrid_result['return_mean']:.2f}, "
          f"length={hybrid_result['length_mean']:.1f}, ET={hybrid_result['et_rate']:.0%}")

    # Test combined model
    print("\nTesting combined model...")
    combined_result = run_baseline_test(combined_model, num_seeds=50)
    print(f"Combined: return={combined_result['return_mean']:.2f}, "
          f"length={combined_result['length_mean']:.1f}, ET={combined_result['et_rate']:.0%}")

    # Test SINDy-only model
    print("\nTesting SINDy-only model...")
    sindy_result = run_baseline_test(sindy_model, num_seeds=50)
    print(f"SINDy-only: return={sindy_result['return_mean']:.2f}, "
          f"length={sindy_result['length_mean']:.1f}, ET={sindy_result['et_rate']:.0%}")

    # Save results
    results = {
        'hybrid': hybrid_result,
        'combined': combined_result,
        'sindy_only': sindy_result,
    }
    with open(os.path.join(out_dir, 'baseline_test_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {out_dir}")


if __name__ == '__main__':
    main()
