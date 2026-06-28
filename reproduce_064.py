"""Reproduce the 0.064 rad result: GP + Ensemble(5) on 4D Meijaard model.

This script exactly follows the code path in test_ensemble.py:
1. generate_training_data(30000) from methods_common
2. GPMethod trains 4 independent GPs
3. EnsembleResidualNet trains 5 NNs on residuals
4. DAgger with 3 rounds
5. Evaluation: 5 segments, 500 steps, LQR from real state
"""
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['PYTHONWARNINGS'] = 'ignore'

import sys
sys.path.insert(0, 'D:/系统辨识作业/sindy_bicycle')

import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

# Import from project
from methods_common import generate_training_data, real_step, K_lqr, dt
from methods_evaluate import make_tau_func

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")


# ============================================================
# ResidualNet (from methods_nn.py)
# ============================================================
class ResidualNet(nn.Module):
    def __init__(self, state_dim=4, action_dim=1, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, state_dim),
        )

    def forward(self, s, a=None):
        if a is not None:
            return self.net(torch.cat([s, a], dim=-1))
        return self.net(s)


# ============================================================
# GP Baseline (from methods_classic.py GPMethod)
# ============================================================
class GPBaseline:
    def __init__(self, max_samples=5000):
        self.max_samples = max_samples
        self._gps = None
        self._state_std = None
        self._action_std = None
        self._delta_std = None

    def train(self, states, actions, deltas, state_std, action_std, delta_std):
        self._state_std = state_std
        self._action_std = action_std
        self._delta_std = delta_std

        n = len(states)
        if n > self.max_samples:
            idx = np.random.choice(n, self.max_samples, replace=False)
        else:
            idx = np.arange(n)
        X = np.column_stack([states[idx] / state_std, actions[idx].reshape(-1, 1) / action_std])
        Y = deltas[idx] / delta_std

        self._gps = []
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        for col in range(4):
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=1e-6)
            gp.fit(X, Y[:, col])
            self._gps.append(gp)
            print(f"    GP output {col} trained")

    def predict(self, s, tau):
        s_norm = s / self._state_std
        a_norm = tau / self._action_std
        x = np.concatenate([s_norm, [a_norm]]).reshape(1, -1)
        delta_norm = np.array([gp.predict(x)[0] for gp in self._gps])
        return s + delta_norm * self._delta_std


# ============================================================
# EnsembleResidualNet (from test_ensemble.py)
# ============================================================
class EnsembleResidualNet:
    def __init__(self, n_models=5):
        self.n_models = n_models
        self.models = []

    def train(self, train_inputs, train_residuals, n_epochs=100):
        self.models = []
        for i in range(self.n_models):
            print(f"    Training model {i+1}/{self.n_models}")
            torch.manual_seed(i * 42)
            np.random.seed(i * 42)

            model = ResidualNet().to(DEVICE)
            train_x = torch.FloatTensor(train_inputs).to(DEVICE)
            train_y = torch.FloatTensor(train_residuals).to(DEVICE)
            ds = TensorDataset(train_x, train_y)
            loader = DataLoader(ds, batch_size=256, shuffle=True)
            opt = optim.Adam(model.parameters(), lr=1e-3)
            crit = nn.MSELoss()

            model.train()
            for _ in range(n_epochs):
                for xb, yb in loader:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    loss = crit(model(xb), yb)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            model.eval()
            self.models.append(model)

    def predict(self, s_norm, a_norm):
        s_t = torch.FloatTensor(s_norm).unsqueeze(0).to(DEVICE)
        a_t = torch.FloatTensor([a_norm]).unsqueeze(0).to(DEVICE)
        predictions = []
        for model in self.models:
            with torch.no_grad():
                pred = model(s_t, a_t).cpu().numpy()[0]
            predictions.append(pred)
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        return mean_pred, std_pred

    def predict_mean(self, s_norm, a_norm):
        mean_pred, _ = self.predict(s_norm, a_norm)
        return mean_pred


# ============================================================
# train_ensemble (from test_ensemble.py)
# ============================================================
def train_ensemble(n_models=5, n_epochs=100, dagger_rounds=3, residual_scale=0.3):
    states, actions, deltas, state_std, action_std, delta_std = generate_training_data(30000)
    baseline = GPBaseline(max_samples=5000)
    baseline.train(states, actions, deltas, state_std, action_std, delta_std)

    residuals = np.empty_like(deltas)
    for i in range(len(states)):
        s_next_base = baseline.predict(states[i], actions[i])
        residuals[i] = (deltas[i] - (s_next_base - states[i])) / delta_std

    train_inputs_base = np.column_stack([states / state_std, actions.reshape(-1, 1) / action_std])
    all_train_inputs = [train_inputs_base.copy()]
    all_train_residuals = [residuals.copy()]

    for round_i in range(dagger_rounds):
        train_inputs = np.vstack(all_train_inputs)
        train_residuals = np.vstack(all_train_residuals)

        ensemble = EnsembleResidualNet(n_models=n_models)
        ensemble.train(train_inputs, train_residuals, n_epochs=n_epochs)

        if round_i < dagger_rounds - 1:
            new_inputs = []
            new_residuals = []
            for seg_i in range(3):
                tau_func = make_tau_func(100 + seg_i, 500)
                phi_init = np.random.uniform(-0.25, 0.25)
                s0 = np.array([phi_init, 0.0, 0.0, 0.0])
                s_base = s0.copy()
                s_real = s0.copy()
                for step in range(500):
                    tau = tau_func(step, s_real)
                    s_real = real_step(s_real, tau)
                    s_norm = s_base / state_std
                    a_norm = tau / action_std
                    delta_nn = ensemble.predict_mean(s_norm, a_norm)
                    s_base = baseline.predict(s_base, tau) + delta_nn * delta_std * residual_scale
                    if abs(s_real[0]) > math.pi / 3:
                        break
                    new_inputs.append(np.concatenate([s_norm, [a_norm]]))
                    s_next_base = baseline.predict(s_base - delta_nn * delta_std * residual_scale, tau)
                    base_delta = s_next_base - (s_base - delta_nn * delta_std * residual_scale)
                    real_delta = s_real - (s_base - delta_nn * delta_std * residual_scale)
                    new_residuals.append((real_delta - base_delta) / delta_std)

            if new_inputs:
                all_train_inputs.append(np.array(new_inputs))
                all_train_residuals.append(np.array(new_residuals))
                print(f"    DAgger round {round_i+1}: collected {len(new_inputs)} new samples")

    return baseline, ensemble, (state_std, action_std, delta_std)


# ============================================================
# Evaluation (from test_ensemble.py test_ensemble)
# ============================================================
def evaluate_ensemble(baseline, ensemble, state_std, action_std, delta_std, residual_scale=0.3):
    eval_steps = [1, 5, 10, 20, 50, 100, 200, 500]
    n_segments = 5
    all_errors = {step: [] for step in eval_steps}
    baseline_errors = {step: [] for step in eval_steps}

    for seg_i in range(n_segments):
        tau_func = make_tau_func(seg_i, 500)
        phi_init = np.random.uniform(-0.25, 0.25)
        s0 = np.array([phi_init, 0.0, 0.0, 0.0])
        s_real = s0.copy()
        s_base = s0.copy()
        s_ensemble = s0.copy()

        for step in range(500):
            tau = tau_func(step, s_real)
            s_real = real_step(s_real, tau)
            s_base = baseline.predict(s_base, tau)

            # ensemble predict
            s_norm = s_ensemble / state_std
            a_norm = tau / action_std
            delta_nn, _ = ensemble.predict(s_norm, a_norm)
            s_next_base = baseline.predict(s_ensemble, tau)
            s_ensemble = s_next_base + delta_nn * delta_std * residual_scale

            if abs(s_real[0]) > math.pi / 3:
                break
            if step + 1 in eval_steps:
                all_errors[step + 1].append(abs(s_ensemble[0] - s_real[0]))
                baseline_errors[step + 1].append(abs(s_base[0] - s_real[0]))

    print(f"\n  {'Step':>6} | {'Baseline':>10} | {'Ensemble':>10}")
    print(f"  {'-'*35}")
    for step in eval_steps:
        b_err = np.mean(baseline_errors[step]) if baseline_errors[step] else float('nan')
        h_err = np.mean(all_errors[step]) if all_errors[step] else float('nan')
        print(f"  {step:6d} | {b_err:10.5f} | {h_err:10.5f}")

    return baseline_errors, all_errors


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("Reproducing GP + Ensemble(5) result")
    print("=" * 60)

    # Train
    print("\n--- Training ---")
    baseline, ensemble, (state_std, action_std, delta_std) = train_ensemble(
        n_models=5, n_epochs=100, dagger_rounds=3, residual_scale=0.3
    )

    # Evaluate
    print("\n--- Evaluation ---")
    baseline_errors, all_errors = evaluate_ensemble(
        baseline, ensemble, state_std, action_std, delta_std, residual_scale=0.3
    )

    # Final result
    mae_500 = np.mean(all_errors[500]) if all_errors[500] else float('nan')
    print(f"\n{'='*60}")
    print(f"RESULT: GP + Ensemble(5) 500-step MAE = {mae_500:.5f} rad")
    print(f"{'='*60}")
