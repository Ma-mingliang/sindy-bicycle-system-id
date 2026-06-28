"""Canonical 4D model interfaces - GP, SINDy, base class."""
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Tuple
from .config import TrainingConfig


class BaseModel(ABC):
    """Unified model interface."""

    @abstractmethod
    def predict(self, s: np.ndarray, tau: float) -> np.ndarray:
        pass

    @abstractmethod
    def name(self) -> str:
        pass

    def predict_with_uncertainty(self, s: np.ndarray, tau: float) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        return self.predict(s, tau), None

    @abstractmethod
    def train(self, states, actions, deltas, state_std, action_std, delta_std):
        pass


class GPModel(BaseModel):
    """Standard GP model - one GP per state dimension."""

    def __init__(self, max_samples: int = 2000, n_restarts: int = 2):
        self.max_samples = max_samples
        self.n_restarts = n_restarts
        self._gps = None
        self._state_std = None
        self._action_std = None
        self._delta_std = None

    def train(self, states, actions, deltas, state_std, action_std, delta_std):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel
        self._state_std = state_std.copy()
        self._action_std = action_std
        self._delta_std = delta_std.copy()
        # Avoid division by zero for constant features
        self._state_std[self._state_std < 1e-10] = 1.0
        self._delta_std[self._delta_std < 1e-10] = 1.0

        n = len(states)
        if n > self.max_samples:
            rng = np.random.RandomState(42)
            idx = rng.choice(n, self.max_samples, replace=False)
        else:
            idx = np.arange(n)

        X = np.column_stack([states[idx] / self._state_std, actions[idx].reshape(-1, 1) / action_std])
        Y = deltas[idx] / self._delta_std

        self._gps = []
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        for col in range(4):
            gp = GaussianProcessRegressor(
                kernel=kernel, n_restarts_optimizer=self.n_restarts, alpha=1e-6
            )
            gp.fit(X, Y[:, col])
            self._gps.append(gp)

    def predict(self, s, tau):
        s_norm = s / self._state_std
        a_norm = tau / self._action_std
        x = np.concatenate([s_norm, [a_norm]]).reshape(1, -1)
        delta_norm = np.array([gp.predict(x)[0] for gp in self._gps])
        return s + delta_norm * self._delta_std

    def name(self):
        return 'gp'


class SINDyModel(BaseModel):
    """SINDy polynomial regression model."""

    def __init__(self):
        self._coefficients = None
        self._state_std = None
        self._delta_std = None

    def _build_library(self, X):
        n = X.shape[0]
        lib = [np.ones(n)]
        for i in range(X.shape[1]):
            lib.append(X[:, i])
        for i in range(X.shape[1]):
            for j in range(i, X.shape[1]):
                lib.append(X[:, i] * X[:, j])
        return np.column_stack(lib)

    def train(self, states, actions, deltas, state_std, action_std, delta_std):
        self._state_std = state_std
        self._action_std = action_std
        self._delta_std = delta_std
        X = np.column_stack([states, actions.reshape(-1, 1)])
        Theta = self._build_library(X)
        self._coefficients = np.linalg.lstsq(Theta, deltas, rcond=None)[0]

    def predict(self, s, tau):
        x = np.concatenate([s, [tau]])
        Theta = self._build_library(x.reshape(1, -1))
        delta = (Theta @ self._coefficients).flatten()
        return s + delta

    def name(self):
        return 'sindy'
