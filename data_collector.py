"""
Data collection for SINDy bicycle system identification.

Implements stratified sampling across scenarios (straight, curved, sharp turns)
with randomized initial states and mixed action strategies, as described in the paper.

Collects (state, action, next_state, reward) tuples at 30 Hz.
"""

import numpy as np
from bicycle_dynamics import (
    BicycleParams, simulate_bicycle, full_state_to_path_tracking
)


# Normalization constants from the paper
NORMALIZATION = {
    'ey': 10.0,
    'epsi': 1.57,  # ~pi/2
    'v': 5.0,
    'theta': 1.57,
    'theta_dot': 10.0,
    'k': 8.0,      # Note: paper says k*8, so we divide by 1/8 = multiply by 8
    'delta': 0.785, # ~pi/4
    'delta_dot': 3.0
}

STATE_NAMES = ['ey', 'epsi', 'v', 'theta', 'theta_dot', 'k', 'delta', 'delta_dot']


def normalize_state(s):
    """Normalize state vector per paper specification."""
    s_norm = s.copy()
    s_norm[:, 0] = s[:, 0] / NORMALIZATION['ey']
    s_norm[:, 1] = s[:, 1] / NORMALIZATION['epsi']
    s_norm[:, 2] = s[:, 2] / NORMALIZATION['v']
    s_norm[:, 3] = s[:, 3] / NORMALIZATION['theta']
    s_norm[:, 4] = s[:, 4] / NORMALIZATION['theta_dot']
    s_norm[:, 5] = s[:, 5] * NORMALIZATION['k']  # k * 8
    s_norm[:, 6] = s[:, 6] / NORMALIZATION['delta']
    s_norm[:, 7] = s[:, 7] / NORMALIZATION['delta_dot']
    return s_norm


def denormalize_state(s_norm):
    """Denormalize state vector."""
    s = s_norm.copy()
    s[:, 0] = s[:, 0] * NORMALIZATION['ey']
    s[:, 1] = s[:, 1] * NORMALIZATION['epsi']
    s[:, 2] = s[:, 2] * NORMALIZATION['v']
    s[:, 3] = s[:, 3] * NORMALIZATION['theta']
    s[:, 4] = s[:, 4] * NORMALIZATION['theta_dot']
    s[:, 5] = s[:, 5] / NORMALIZATION['k']
    s[:, 6] = s[:, 6] * NORMALIZATION['delta']
    s[:, 7] = s[:, 7] * NORMALIZATION['delta_dot']
    return s


def compute_reward(s, a, s_next):
    """
    Compute reward as in the paper:
    R = w1*e_y^2 + w2*e_psi^2 + w3*theta^2 + w4*delta^2 + w5*(delta_t)^2
    """
    w1, w2, w3, w4, w5 = 1.0, 0.5, 0.3, 0.1, 0.05
    reward = -(
        w1 * s_next[0]**2 +
        w2 * s_next[1]**2 +
        w3 * s_next[3]**2 +
        w4 * s_next[6]**2 +
        w5 * (s_next[6] - s[6])**2
    )
    return reward


class BicycleDataCollector:
    """
    Collects bicycle dynamics data for SINDy identification.

    Implements stratified sampling across three scenario types:
    1. Straight segments (k ≈ 0)
    2. Continuous curvature segments (moderate k)
    3. Sharp turns (large k)
    """

    def __init__(self, dt=1/30, seed=42):
        """
        Parameters
        ----------
        dt : float
            Sampling period (default 1/30 s as in paper)
        seed : int
            Random seed for reproducibility
        """
        self.dt = dt
        self.params = BicycleParams()
        self.rng = np.random.RandomState(seed)

    def _random_initial_state(self, scenario='straight'):
        """
        Generate randomized initial state for a given scenario.

        Parameters
        ----------
        scenario : str
            One of 'straight', 'curved', 'sharp'

        Returns
        -------
        x0 : ndarray, shape (10,)
            Full bicycle state
        k_ref : float
            Reference path curvature
        """
        p = self.params

        # Speed range: 2-6 m/s (typical bicycle)
        v = self.rng.uniform(2.0, 6.0)

        # Small initial perturbations
        y0 = self.rng.uniform(-1.0, 1.0)
        psi0 = self.rng.uniform(-0.2, 0.2)
        theta0 = self.rng.uniform(-0.15, 0.15)
        theta_dot0 = self.rng.uniform(-0.5, 0.5)
        delta0 = self.rng.uniform(-0.2, 0.2)
        delta_dot0 = self.rng.uniform(-0.5, 0.5)

        # Reference curvature depends on scenario
        if scenario == 'straight':
            k_ref = self.rng.uniform(-0.02, 0.02)
        elif scenario == 'curved':
            k_ref = self.rng.uniform(-0.1, 0.1)
        elif scenario == 'sharp':
            k_ref = self.rng.uniform(-0.3, 0.3)
        else:
            k_ref = 0.0

        # Full state: [x, y, psi, v, theta, theta_dot, delta, delta_dot, phi_f, omega_f]
        x0 = np.array([0.0, y0, psi0, v, theta0, theta_dot0, delta0, delta_dot0, 0.0, v/p.r_front])
        return x0, k_ref

    def _make_control_func(self, scenario, t_episode):
        """
        Create a control function for data collection.

        Uses a mix of:
        1. A simple balance controller (PD on roll angle)
        2. Random exploration perturbations

        This ensures the action distribution is broad enough for SINDy.
        """
        # Pre-generate random perturbations
        n_perturb = int(t_episode / self.dt) + 100
        perturb_times = np.cumsum(self.rng.exponential(0.5, n_perturb))
        perturb_values = self.rng.uniform(-2.0, 2.0, n_perturb)

        def control_func(t, state):
            x, y, psi, v, theta, theta_dot, delta, delta_dot, _, _ = state

            # PD controller for roll stabilization
            # Steering into the fall to recover
            Kp_roll = 8.0   # Proportional gain
            Kd_roll = 2.0   # Derivative gain
            Kp_delta = -1.5  # Steering damping

            u_balance = -Kp_roll * theta - Kd_roll * theta_dot + Kp_delta * delta

            # Random exploration perturbation (mixed strategy from paper)
            idx = np.searchsorted(perturb_times, t)
            u_explore = perturb_values[min(idx, n_perturb-1)]

            # Scenario-dependent weighting
            if scenario == 'straight':
                u = u_balance + 0.3 * u_explore
            elif scenario == 'curved':
                u = u_balance + 0.5 * u_explore
            else:  # sharp
                u = u_balance + 0.8 * u_explore

            # Clip to reasonable range
            u = np.clip(u, -10.0, 10.0)
            return u

        return control_func

    def collect_episode(self, scenario='straight', t_episode=10.0):
        """
        Collect one episode of data.

        Parameters
        ----------
        scenario : str
            Scenario type
        t_episode : float
            Episode duration in seconds

        Returns
        -------
        data : dict
            Contains 'states_8d', 'actions', 'next_states_8d', 'rewards', 'scenario'
        """
        x0, k_ref = self._random_initial_state(scenario)
        control_func = self._make_control_func(scenario, t_episode)

        t, x_full = simulate_bicycle(
            self.params, x0, (0, t_episode), self.dt, control_func
        )

        # Convert to 8D path-tracking states
        states_8d = np.array([full_state_to_path_tracking(xi, k_ref) for xi in x_full])

        # Extract actions (control inputs applied)
        actions = np.array([control_func(t[i], x_full[i]) for i in range(len(t))])

        # Build (s, a, s') transitions
        n = len(states_8d) - 1
        s = states_8d[:n]
        a = actions[:n].reshape(-1, 1)
        s_next = states_8d[1:n+1]

        # Compute rewards
        rewards = np.array([compute_reward(s[i], a[i], s_next[i]) for i in range(n)])

        # Filter out fallen states (|theta| > 1.0 rad indicates fall)
        valid = np.abs(s[:, 3]) < 1.0
        # Also filter extreme states
        valid &= np.all(np.abs(s) < 50, axis=1)
        valid &= np.all(np.abs(s_next) < 50, axis=1)

        return {
            'states': s[valid],
            'actions': a[valid],
            'next_states': s_next[valid],
            'rewards': rewards[valid],
            'scenario': scenario,
            'n_total': len(s),
            'n_valid': np.sum(valid)
        }

    def collect_dataset(self, n_episodes_per_scenario=50, t_episode=10.0):
        """
        Collect a full stratified dataset.

        Parameters
        ----------
        n_episodes_per_scenario : int
            Number of episodes per scenario type
        t_episode : float
            Duration of each episode

        Returns
        -------
        dataset : dict
            Combined dataset with metadata
        """
        all_data = []
        scenario_counts = {}

        for scenario in ['straight', 'curved', 'sharp']:
            scenario_data = []
            n_valid_total = 0

            for ep in range(n_episodes_per_scenario):
                data = self.collect_episode(scenario, t_episode)
                scenario_data.append(data)
                n_valid_total += data['n_valid']

                if (ep + 1) % 10 == 0:
                    print(f"  [{scenario}] Episode {ep+1}/{n_episodes_per_scenario}, "
                          f"valid samples: {n_valid_total}")

            # Concatenate scenario data
            s = np.vstack([d['states'] for d in scenario_data])
            a = np.vstack([d['actions'] for d in scenario_data])
            s_next = np.vstack([d['next_states'] for d in scenario_data])
            r = np.concatenate([d['rewards'] for d in scenario_data])

            all_data.append({
                'states': s,
                'actions': a,
                'next_states': s_next,
                'rewards': r,
                'scenario': scenario
            })
            scenario_counts[scenario] = len(s)

        # Combine all scenarios
        dataset = {
            'states': np.vstack([d['states'] for d in all_data]),
            'actions': np.vstack([d['actions'] for d in all_data]),
            'next_states': np.vstack([d['next_states'] for d in all_data]),
            'rewards': np.concatenate([d['rewards'] for d in all_data]),
            'scenario_labels': np.concatenate([
                np.full(len(d['states']), d['scenario']) for d in all_data
            ]),
            'scenario_counts': scenario_counts,
            'dt': self.dt
        }

        print(f"\nTotal dataset: {len(dataset['states'])} samples")
        for sc, cnt in scenario_counts.items():
            print(f"  {sc}: {cnt} samples")

        return dataset


def collect_from_gym_environment(env_name='BicycleBalance-v0', n_steps=10000):
    """
    Alternative: collect data from an OpenAI Gym bicycle environment.

    Several open-source bicycle environments exist:
    - BicycleBalance-v0 (from gym-envs)
    - BicycleEnv (from bicycle-rl)

    This function provides a template for such collection.
    """
    try:
        import gymnasium as gym
        # Try to use a bicycle environment if available
        env = gym.make(env_name)
        states, actions, next_states, rewards = [], [], [], []

        obs, _ = env.reset()
        for i in range(n_steps):
            action = env.action_space.sample()  # Random policy for exploration
            next_obs, reward, terminated, truncated, _ = env.step(action)

            states.append(obs)
            actions.append(action)
            next_states.append(next_obs)
            rewards.append(reward)

            if terminated or truncated:
                obs, _ = env.reset()
            else:
                obs = next_obs

        env.close()
        return {
            'states': np.array(states),
            'actions': np.array(actions).reshape(-1, 1),
            'next_states': np.array(next_states),
            'rewards': np.array(rewards)
        }
    except Exception as e:
        print(f"Gym environment not available: {e}")
        print("Using built-in simulator instead.")
        return None


if __name__ == '__main__':
    print("=" * 60)
    print("Bicycle Data Collection for SINDy")
    print("=" * 60)

    collector = BicycleDataCollector(dt=1/30, seed=42)

    print("\nCollecting stratified dataset...")
    dataset = collector.collect_dataset(n_episodes_per_scenario=30, t_episode=8.0)

    # Save dataset
    np.savez(
        'bicycle_data.npz',
        states=dataset['states'],
        actions=dataset['actions'],
        next_states=dataset['next_states'],
        rewards=dataset['rewards'],
        dt=dataset['dt']
    )
    print(f"\nDataset saved to bicycle_data.npz")
    print(f"States shape: {dataset['states'].shape}")
    print(f"Actions shape: {dataset['actions'].shape}")
