"""
SINDy-based World Model for bicycle dynamics.

This module implements the identified SINDy model as a Markov transition
function suitable for integration with TD-MPC2 or other model-based RL.

The world model provides:
1. State transition: s_{t+1} = f(s_t, a_t)
2. Uncertainty estimation: Σ_ξ (covariance of residuals)
3. Rollout capability for planning
"""

import numpy as np
from data_collector import normalize_state, denormalize_state, NORMALIZATION, STATE_NAMES


class SINDyWorldModel:
    """
    SINDy-based discrete world model for bicycle dynamics.

    Implements the paper's equation:
        s_{t+1} = s_t + f_SINDy(s_t, a_t) + ξ_t

    where f_SINDy is the sparse polynomial identified by SINDy,
    and ξ_t is the residual noise with covariance Σ_ξ.
    """

    def __init__(self, model_path='sindy_model.npz'):
        """Load identified model."""
        data = np.load(model_path, allow_pickle=True)
        self.Xi = data['coefficients']
        self.lib_labels = list(data['lib_labels'])
        self.threshold = float(data['threshold'])
        self.state_names = list(data['state_names'])
        self.n_states = len(self.state_names)

        # Compute residual statistics from training data (loaded separately)
        self.residual_mean = None
        self.residual_cov = None

        # Normalization parameters
        self.norm_factors = np.array([
            NORMALIZATION['ey'], NORMALIZATION['epsi'], NORMALIZATION['v'],
            NORMALIZATION['theta'], NORMALIZATION['theta_dot'],
            1.0 / NORMALIZATION['k'],  # k is multiplied, not divided
            NORMALIZATION['delta'], NORMALIZATION['delta_dot']
        ])

    def fit_residual_statistics(self, states, actions, next_states):
        """
        Compute residual statistics for uncertainty estimation.

        Parameters
        ----------
        states : ndarray, shape (n, 8)
        actions : ndarray, shape (n, 1)
        next_states : ndarray, shape (n, 8)
        """
        delta_pred = self._predict_delta_normalized(states, actions)
        states_norm = normalize_state(states)
        next_states_norm = normalize_state(next_states)
        delta_true = next_states_norm - states_norm

        residuals = delta_true - delta_pred
        self.residual_mean = np.mean(residuals, axis=0)
        self.residual_cov = np.cov(residuals.T)

    def _build_library(self, X_norm, action_norm):
        """Build polynomial library from normalized features."""
        n = X_norm.shape[0]
        features = [np.ones((n, 1))]  # constant

        # Linear
        for i in range(X_norm.shape[1]):
            features.append(X_norm[:, i:i+1])
        features.append(action_norm.reshape(-1, 1))

        # Quadratic
        X_aug = np.hstack([X_norm, action_norm.reshape(-1, 1)])
        n_feat = X_aug.shape[1]
        for i in range(n_feat):
            for j in range(i, n_feat):
                features.append(X_aug[:, i:i+1] * X_aug[:, j:j+1])

        return np.hstack(features)

    def _predict_delta_normalized(self, states, actions):
        """
        Predict delta state in normalized space.

        Parameters
        ----------
        states : ndarray, shape (n, 8)
        actions : ndarray, shape (n, 1)

        Returns
        -------
        delta_norm : ndarray, shape (n, 8)
        """
        states_norm = normalize_state(states)
        action_scale = max(np.std(actions), 1.0)
        actions_norm = actions / action_scale

        Theta = self._build_library(states_norm, actions_norm)

        # Ensure library matches model dimensions
        min_cols = min(Theta.shape[1], self.Xi.shape[0])
        delta_norm = Theta[:, :min_cols] @ self.Xi[:min_cols, :]
        return delta_norm

    def predict_next_state(self, state, action):
        """
        Predict next state given current state and action.

        Parameters
        ----------
        state : ndarray, shape (8,) or (n, 8)
        action : float or ndarray, shape (1,) or (n, 1)

        Returns
        -------
        next_state : ndarray, same shape as state
        """
        state = np.atleast_2d(state)
        action = np.atleast_2d(action).reshape(-1, 1)

        delta_norm = self._predict_delta_normalized(state, action)
        states_norm = normalize_state(state)
        next_states_norm = states_norm + delta_norm
        next_state = denormalize_state(next_states_norm)

        if next_state.shape[0] == 1:
            return next_state[0]
        return next_state

    def rollout(self, state_init, actions, add_noise=False):
        """
        Perform multi-step rollout.

        Parameters
        ----------
        state_init : ndarray, shape (8,)
        actions : ndarray, shape (n_steps,) or (n_steps, 1)
        add_noise : bool
            Whether to add residual noise

        Returns
        -------
        states : ndarray, shape (n_steps+1, 8)
        """
        actions = np.atleast_2d(actions).reshape(-1, 1)
        n_steps = len(actions)
        states = np.zeros((n_steps + 1, 8))
        states[0] = state_init

        for i in range(n_steps):
            states[i+1] = self.predict_next_state(states[i:i+1], actions[i:i+1])
            if add_noise and self.residual_cov is not None:
                noise = np.random.multivariate_normal(self.residual_mean, self.residual_cov)
                states[i+1] += denormalize_state(noise.reshape(1, -1))[0]

        return states

    def get_transition_uncertainty(self, state, action):
        """
        Get uncertainty estimate for a single transition.

        Returns the residual covariance Σ_ξ as described in the paper.

        Parameters
        ----------
        state : ndarray, shape (8,)
        action : float

        Returns
        -------
        cov : ndarray, shape (8, 8)
            State-dependent uncertainty (approximate)
        """
        if self.residual_cov is not None:
            return self.residual_cov
        return np.eye(8) * 0.01  # Default small uncertainty

    def summary(self):
        """Print model summary."""
        print("=" * 60)
        print("SINDy World Model Summary")
        print("=" * 60)
        print(f"Threshold: {self.threshold}")
        print(f"Library terms: {self.Xi.shape[0]}")
        print(f"State dimensions: {self.Xi.shape[1]}")

        print(f"\nActive terms per state:")
        for i in range(self.Xi.shape[1]):
            n_active = np.sum(np.abs(self.Xi[:, i]) > 1e-6)
            print(f"  {self.state_names[i]:>12}: {n_active} terms")

        print(f"\nIdentified equations (top terms):")
        for i in range(self.Xi.shape[1]):
            coefs = self.Xi[:, i]
            active_idx = np.argsort(np.abs(coefs))[::-1]
            terms = []
            for j in active_idx[:3]:
                if abs(coefs[j]) > 1e-6:
                    label = self.lib_labels[j] if j < len(self.lib_labels) else f"f{j}"
                    terms.append(f"{coefs[j]:+.3f}*{label}")
            if terms:
                print(f"  Δ{self.state_names[i]} ≈ {' '.join(terms)}")
            else:
                print(f"  Δ{self.state_names[i]} ≈ 0")
        print("=" * 60)


class SimpleBicycleWorldModel:
    """
    Simple physics-based world model for comparison.

    Uses the kinematic bicycle model with identified parameters.
    This serves as a baseline to compare against the SINDy model.
    """

    def __init__(self, wheelbase=1.02):
        self.wheelbase = wheelbase
        self.dt = 1/30

    def predict_next_state(self, state, action):
        """
        Kinematic bicycle model prediction.

        Parameters
        ----------
        state : ndarray, shape (8,)
            [ey, epsi, v, theta, theta_dot, k, delta, delta_dot]
        action : float
            Steering torque
        """
        ey, epsi, v, theta, theta_dot, k, delta, delta_dot = state

        # Steering dynamics (simplified)
        delta_ddot = action - 2.0 * delta_dot - 5.0 * delta
        delta_next = delta + delta_dot * self.dt
        delta_dot_next = delta_dot + delta_ddot * self.dt

        # Roll dynamics (simplified coupling)
        theta_ddot = 9.81 * theta / 1.0 - 0.5 * delta
        theta_next = theta + theta_dot * self.dt
        theta_dot_next = theta_dot + theta_ddot * self.dt

        # Kinematics
        psi_dot = v * np.tan(delta) / self.wheelbase
        epsi_next = epsi + psi_dot * self.dt
        ey_next = ey + v * np.sin(epsi) * self.dt

        # Speed (constant)
        v_next = v
        k_next = k

        return np.array([ey_next, epsi_next, v_next, theta_next,
                         theta_dot_next, k_next, delta_next, delta_dot_next])


def compare_models(sindy_model, states, actions, next_states, n_rollout=50):
    """
    Compare SINDy model with simple physics model.

    Parameters
    ----------
    sindy_model : SINDyWorldModel
    states, actions, next_states : ndarray
    n_rollout : int
    """
    simple_model = SimpleBicycleWorldModel()

    print("\n" + "=" * 60)
    print("Model Comparison: SINDy vs Simple Physics")
    print("=" * 60)

    # Single-step prediction errors
    sindy_errors = []
    simple_errors = []

    for i in range(min(n_rollout, len(states))):
        s = states[i]
        a = actions[i].item() if actions.ndim > 1 else actions[i]
        s_next_true = next_states[i]

        s_next_sindy = sindy_model.predict_next_state(s, a)
        s_next_simple = simple_model.predict_next_state(s, a)

        sindy_errors.append(np.sqrt(np.mean((s_next_sindy - s_next_true)**2)))
        simple_errors.append(np.sqrt(np.mean((s_next_simple - s_next_true)**2)))

    print(f"\nSingle-step RMSE:")
    print(f"  SINDy model:   {np.mean(sindy_errors):.6f}")
    print(f"  Simple physics: {np.mean(simple_errors):.6f}")

    # Multi-step rollout
    s0 = states[0]
    a_seq = actions[:n_rollout]

    sindy_rollout = sindy_model.rollout(s0, a_seq)
    simple_rollout = np.zeros((n_rollout + 1, 8))
    simple_rollout[0] = s0
    for i in range(n_rollout):
        a = a_seq[i].item() if a_seq.ndim > 1 else a_seq[i]
        simple_rollout[i+1] = simple_model.predict_next_state(simple_rollout[i], a)

    true_states = np.vstack([s0.reshape(1, -1), next_states[:n_rollout]])

    sindy_rollout_rmse = np.sqrt(np.mean((sindy_rollout - true_states)**2))
    simple_rollout_rmse = np.sqrt(np.mean((simple_rollout - true_states)**2))

    print(f"\n{n_rollout}-step rollout RMSE:")
    print(f"  SINDy model:   {sindy_rollout_rmse:.6f}")
    print(f"  Simple physics: {simple_rollout_rmse:.6f}")

    return {
        'sindy_single_rmse': np.mean(sindy_errors),
        'simple_single_rmse': np.mean(simple_errors),
        'sindy_rollout_rmse': sindy_rollout_rmse,
        'simple_rollout_rmse': simple_rollout_rmse
    }


if __name__ == '__main__':
    # Load model and data
    model = SINDyWorldModel('sindy_model.npz')
    model.summary()

    data = np.load('bicycle_data.npz')
    states = data['states']
    actions = data['actions']
    next_states = data['next_states']

    # Fit residual statistics
    model.fit_residual_statistics(states, actions, next_states)

    if model.residual_cov is not None:
        print(f"\nResidual covariance (diagonal):")
        for i, name in enumerate(STATE_NAMES):
            print(f"  {name}: σ² = {model.residual_cov[i, i]:.6f}")

    # Compare models
    compare_models(model, states, actions, next_states)
