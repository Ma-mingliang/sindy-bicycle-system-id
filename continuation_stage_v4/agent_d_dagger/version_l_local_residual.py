"""
Version L: Local Dynamics Residual DAgger Implementation

Residual formula:
    r_k = (F_real(s_model_k, u_k) - GP(s_model_k, u_k)) / delta_std

Both real dynamics and GP are evaluated at the SAME state (model state).
This is a pure model-error identification objective.

The key difference from Version T:
- Version L: real dynamics at s_model[k], GP at s_model[k] -> same state
- Version T: s_real[k+1] vs GP(s_model[k], tau) -> different trajectory states
"""

import numpy as np
import math
import torch
from typing import Optional, Tuple, List

# Delayed imports
_methods_common = None
_methods_nn = None


def _get_methods_common():
    global _methods_common
    if _methods_common is None:
        import methods_common
        _methods_common = methods_common
    return _methods_common


def _get_methods_nn():
    global _methods_nn
    if _methods_nn is None:
        import methods_nn
        _methods_nn = methods_nn
    return _methods_nn


class GPStandard:
    """Standard GP model (copied from eval_models.py for independence)."""

    def __init__(self, max_samples: int = 5000, n_restarts: int = 2):
        self.max_samples = max_samples
        self.n_restarts = n_restarts
        self._gps = None
        self._state_std = None
        self._action_std = None
        self._delta_std = None

    def train(self, states, actions, deltas, state_std, action_std, delta_std):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel

        self._state_std = state_std
        self._action_std = action_std
        self._delta_std = delta_std

        n = len(states)
        if n > self.max_samples:
            rng = np.random.RandomState(42)
            idx = rng.choice(n, self.max_samples, replace=False)
        else:
            idx = np.arange(n)

        X = np.column_stack([states[idx] / state_std, actions[idx].reshape(-1, 1) / action_std])
        Y = deltas[idx] / delta_std

        self._gps = []
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        for col in range(4):
            gp = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=self.n_restarts,
                alpha=1e-6
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
        return 'gp_standard'


class VersionLLocalResidual:
    """
    GP + NN ensemble with Version L (Local Dynamics Residual) DAgger.

    Residual at each DAgger step:
        r_k = (F_real(s_model_k, u_k) - GP(s_model_k, u_k)) / delta_std

    Both real dynamics and GP evaluated at the model state s_model[k].
    This is a pure local error identification objective.
    """

    def __init__(self, n_models: int = 5, dagger_rounds: int = 3,
                 residual_scale: float = 0.3, n_epochs: int = 100,
                 use_scheduler: bool = True):
        self.n_models = n_models
        self.dagger_rounds = dagger_rounds
        self.residual_scale = residual_scale
        self.n_epochs = n_epochs
        self.use_scheduler = use_scheduler
        self._baseline = None
        self._ensemble_models = []
        self._state_std = None
        self._action_std = None
        self._delta_std = None

    def train(self, states: np.ndarray, actions: np.ndarray, deltas: np.ndarray,
              state_std: np.ndarray, action_std: float, delta_std: np.ndarray):
        """Train GP baseline + NN ensemble with Version L DAgger."""
        mc = _get_methods_common()
        me = __import__('methods_evaluate')
        nn_mod = _get_methods_nn()
        ResidualNet = nn_mod.ResidualNet
        DEVICE = nn_mod.DEVICE

        self._state_std = state_std
        self._action_std = action_std
        self._delta_std = delta_std

        # 1. Train GP baseline
        self._baseline = GPStandard(max_samples=5000)
        self._baseline.train(states, actions, deltas, state_std, action_std, delta_std)

        # 2. Round 0: Pure local dynamics residual at expert states
        residuals = np.empty_like(deltas)
        for i in range(len(states)):
            s_next_base = self._baseline.predict(states[i], actions[i])
            # Version L: both at expert state
            residuals[i] = (deltas[i] - (s_next_base - states[i])) / delta_std

        # 3. DAgger rounds with Version L residual
        train_inputs_base = np.column_stack([states / state_std, actions.reshape(-1, 1) / action_std])
        all_train_inputs = [train_inputs_base.copy()]
        all_train_residuals = [residuals.copy()]

        self._ensemble_models = []

        for round_i in range(self.dagger_rounds):
            train_inputs = np.vstack(all_train_inputs)
            train_residuals = np.vstack(all_train_residuals)

            # Train ensemble
            models = []
            for i in range(self.n_models):
                torch.manual_seed(i * 42)
                np.random.seed(i * 42)

                model = ResidualNet().to(DEVICE)
                train_x = torch.FloatTensor(train_inputs).to(DEVICE)
                train_y = torch.FloatTensor(train_residuals).to(DEVICE)
                ds = torch.utils.data.TensorDataset(train_x, train_y)
                loader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True)
                opt = torch.optim.Adam(model.parameters(), lr=1e-3)

                if self.use_scheduler:
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.n_epochs)
                else:
                    scheduler = None

                crit = torch.nn.MSELoss()
                model.train()
                for epoch in range(self.n_epochs):
                    for xb, yb in loader:
                        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                        loss = crit(model(xb), yb)
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                    if scheduler:
                        scheduler.step()
                model.eval()
                models.append(model)

            self._ensemble_models = models

            # DAgger: collect new data with Version L residual
            if round_i < self.dagger_rounds - 1:
                new_inputs, new_residuals = self._collect_dagger_data_vL(
                    models, state_std, action_std, delta_std, me.make_tau_func, mc.real_step
                )
                if new_inputs:
                    all_train_inputs.append(np.array(new_inputs))
                    all_train_residuals.append(np.array(new_residuals))

    def _collect_dagger_data_vL(self, models, state_std, action_std, delta_std,
                                 make_tau_func, real_step):
        """
        Collect DAgger data with Version L residual.

        At each step k:
            1. Compute tau from real state (standard DAgger)
            2. Compute GP prediction at MODEL state s_model[k]
            3. Compute real dynamics at MODEL state s_model[k] (NOT s_real[k])
            4. Residual = (F_real(s_model[k], u_k) - GP(s_model[k], u_k)) / delta_std
            5. Update model state: s_model[k+1] = GP(s_model[k], u_k) + scaled_NN(s_model[k], u_k)

        Key insight: The real dynamics are evaluated at the MODEL state,
        not the real state. This gives a clean local error identification.
        """
        new_inputs = []
        new_residuals = []
        for seg_i in range(3):
            tau_func = make_tau_func(100 + seg_i, 500)
            phi_init = np.random.uniform(-0.25, 0.25)
            s0 = np.array([phi_init, 0.0, 0.0, 0.0])
            s_model = s0.copy()
            s_real = s0.copy()

            for step in range(500):
                # Control from REAL state (standard DAgger behavior)
                tau = tau_func(step, s_real)

                # Advance REAL trajectory (for tau computation reference)
                s_real = real_step(s_real, tau)

                if abs(s_real[0]) > math.pi / 3:
                    break

                # ============================================================
                # VERSION L: Local Dynamics Residual
                # ============================================================
                # Both real dynamics and GP evaluated at MODEL state s_model[k]
                s_model_k = s_model.copy()  # save pre-update state

                # GP prediction at model state
                s_next_gp = self._baseline.predict(s_model_k, tau)

                # Real dynamics at MODEL state (NOT s_real)
                s_next_real = real_step(s_model_k, tau)

                # Residual: local dynamics error at model state
                gp_delta = s_next_gp - s_model_k
                real_delta = s_next_real - s_model_k
                residual = (real_delta - gp_delta) / delta_std

                # NN input: model state before update
                s_norm = s_model_k / state_std
                a_norm = tau / action_std

                # NN prediction
                delta_nn = self._predict_ensemble_mean(models, s_norm, a_norm)

                # Update model state
                s_model = s_next_gp + delta_nn * delta_std * self.residual_scale

                # Store training data
                new_inputs.append(np.concatenate([s_norm, [a_norm]]))
                new_residuals.append(residual)

        return new_inputs, new_residuals

    def _predict_ensemble_mean(self, models, s_norm, a_norm):
        """Ensemble mean prediction."""
        nn_mod = _get_methods_nn()
        DEVICE = nn_mod.DEVICE
        s_t = torch.FloatTensor(s_norm).unsqueeze(0).to(DEVICE)
        a_t = torch.FloatTensor([a_norm]).unsqueeze(0).to(DEVICE)
        preds = []
        for model in models:
            with torch.no_grad():
                pred = model(s_t, a_t).cpu().numpy()[0]
            preds.append(pred)
        return np.mean(preds, axis=0)

    def _predict_ensemble_with_std(self, models, s_norm, a_norm):
        """Ensemble mean + std prediction."""
        nn_mod = _get_methods_nn()
        DEVICE = nn_mod.DEVICE
        s_t = torch.FloatTensor(s_norm).unsqueeze(0).to(DEVICE)
        a_t = torch.FloatTensor([a_norm]).unsqueeze(0).to(DEVICE)
        preds = []
        for model in models:
            with torch.no_grad():
                pred = model(s_t, a_t).cpu().numpy()[0]
            preds.append(pred)
        preds = np.array(preds)
        return np.mean(preds, axis=0), np.std(preds, axis=0)

    def predict(self, s: np.ndarray, tau: float) -> np.ndarray:
        """Single-step prediction."""
        s_next_base = self._baseline.predict(s, tau)
        s_norm = s / self._state_std
        a_norm = tau / self._action_std
        delta_nn = self._predict_ensemble_mean(self._ensemble_models, s_norm, a_norm)
        return s_next_base + delta_nn * self._delta_std * self.residual_scale

    def predict_with_uncertainty(self, s: np.ndarray, tau: float) -> Tuple[np.ndarray, np.ndarray]:
        """Prediction with uncertainty."""
        s_next_base = self._baseline.predict(s, tau)
        s_norm = s / self._state_std
        a_norm = tau / self._action_std
        mean_nn, std_nn = self._predict_ensemble_with_std(
            self._ensemble_models, s_norm, a_norm
        )
        s_next = s_next_base + mean_nn * self._delta_std * self.residual_scale
        uncertainty = std_nn * self._delta_std * self.residual_scale
        return s_next, uncertainty

    def name(self) -> str:
        return f'version_L_{self.n_models}m_{self.dagger_rounds}r'
