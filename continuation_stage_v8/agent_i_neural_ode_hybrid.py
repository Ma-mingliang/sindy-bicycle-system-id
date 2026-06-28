"""Agent I: Neural ODE hybrid methods + long-horizon (200/500) evaluation.

Tests:
1. Neural ODE (baseline)
2. Neural ODE + NN residual (like E1 but with Neural ODE instead of GP)
3. Neural ODE + GP residual (Neural ODE as base, GP corrects errors)
4. GP Ensemble 5 (baseline comparison)
5. GP (baseline comparison)

All evaluated at H=1, 5, 10, 20, 50, 100, 200, 500
"""
import sys
sys.path.insert(0, 'D:/系统辨识作业/sindy_bicycle/continuation_stage_v8')

import numpy as np
import json
import time
import traceback
from canonical_7d import (
    load_7d_data, DataConfig, get_test_segments,
    GPModel, GPEnsemble, NeuralODEModel,
    compute_metrics, STATE_NAMES_7D, STATE_DIM,
)


# ============================================================
# Hybrid Model: Neural ODE + NN Residual
# ============================================================
class NeuralODENNResidual:
    """Neural ODE base + NN ensemble residual correction.

    Like E1 but uses Neural ODE instead of GP as the base model.
    The NN learns to predict the residual: delta_true - delta_ode.
    """

    def __init__(self, ode_epochs=100, n_models=5, nn_epochs=50, lr=1e-3, hidden=128):
        self.ode_epochs = ode_epochs
        self.n_models = n_models
        self.nn_epochs = nn_epochs
        self.lr = lr
        self.hidden = hidden
        self._ode = None
        self._nn_ensemble = None
        self._state_std = None
        self._action_std = None
        self._delta_std = None

    def train(self, states, actions, deltas, state_std, action_std, delta_std):
        import torch
        import torch.nn as nn

        self._state_std = state_std.copy()
        self._action_std = action_std
        self._delta_std = delta_std.copy()
        self._state_std[self._state_std < 1e-10] = 1.0
        self._delta_std[self._delta_std < 1e-10] = 1.0

        # Step 1: Train Neural ODE base
        print("    [1/2] Training Neural ODE base...", end=' ', flush=True)
        self._ode = NeuralODEModel(n_epochs=self.ode_epochs, lr=self.lr, hidden=self.hidden)
        self._ode.train(states, actions, deltas, state_std, action_std, delta_std)
        print("DONE")

        # Step 2: Compute residuals and train NN ensemble
        print("    [2/2] Training NN residual ensemble...", end=' ', flush=True)
        residuals = np.empty_like(deltas)
        for i in range(len(states)):
            s_next_ode = self._ode.predict(states[i], actions[i])
            delta_ode = s_next_ode - states[i]
            residuals[i] = (deltas[i] - delta_ode) / delta_std

        # Train NN ensemble on normalized inputs
        train_s = states / self._state_std
        train_a = actions.reshape(-1, 1) / action_std
        train_x = np.column_stack([train_s, train_a])

        self._nn_ensemble = _NNEnsemble(self.n_models, self.nn_epochs, self.hidden)
        self._nn_ensemble.train(train_x, residuals)
        print("DONE")

    def predict(self, s, tau):
        # Base prediction from Neural ODE
        s_next_ode = self._ode.predict(s, tau)

        # NN residual correction
        s_norm = s / self._state_std
        a_norm = tau / self._action_std
        x = np.concatenate([s_norm, [a_norm]]).reshape(1, -1)
        delta_nn = self._nn_ensemble.predict_mean(x)
        correction = delta_nn * self._delta_std

        return s_next_ode + correction

    def predict_with_uncertainty(self, s, tau):
        s_next_ode = self._ode.predict(s, tau)
        s_norm = s / self._state_std
        a_norm = tau / self._action_std
        x = np.concatenate([s_norm, [a_norm]]).reshape(1, -1)
        mean_nn, std_nn = self._nn_ensemble.predict_mean_std(x)
        s_next = s_next_ode + mean_nn * self._delta_std
        uncertainty = std_nn * self._delta_std
        return s_next, uncertainty

    def name(self):
        return 'neural_ode_nn_residual'


class NeuralODEGPResidual:
    """Neural ODE base + GP residual correction.

    Uses GP to correct Neural ODE errors. GP is trained on
    (s, a) -> residual_delta.
    """

    def __init__(self, ode_epochs=100, gp_max_samples=2000):
        self.ode_epochs = ode_epochs
        self.gp_max_samples = gp_max_samples
        self._ode = None
        self._gp_residuals = None  # 7 independent GPs for residuals
        self._state_std = None
        self._action_std = None
        self._delta_std = None

    def train(self, states, actions, deltas, state_std, action_std, delta_std):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel

        self._state_std = state_std.copy()
        self._action_std = action_std
        self._delta_std = delta_std.copy()
        self._state_std[self._state_std < 1e-10] = 1.0
        self._delta_std[self._delta_std < 1e-10] = 1.0

        # Step 1: Train Neural ODE base
        print("    [1/2] Training Neural ODE base...", end=' ', flush=True)
        self._ode = NeuralODEModel(n_epochs=self.ode_epochs, lr=1e-3, hidden=128)
        self._ode.train(states, actions, deltas, state_std, action_std, delta_std)
        print("DONE")

        # Step 2: Compute residuals and train GP
        print("    [2/2] Training GP residual...", end=' ', flush=True)
        residuals = np.empty_like(deltas)
        for i in range(len(states)):
            s_next_ode = self._ode.predict(states[i], actions[i])
            delta_ode = s_next_ode - states[i]
            residuals[i] = (deltas[i] - delta_ode) / delta_std

        # Subsample for GP
        n = len(states)
        if n > self.gp_max_samples:
            idx = np.random.RandomState(42).choice(n, self.gp_max_samples, replace=False)
        else:
            idx = np.arange(n)

        X = np.column_stack([states[idx] / self._state_std,
                             actions[idx].reshape(-1, 1) / action_std])
        Y = residuals[idx]

        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        self._gp_residuals = []
        for col in range(STATE_DIM):
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=1e-6)
            gp.fit(X, Y[:, col])
            self._gp_residuals.append(gp)
        print("DONE")

    def predict(self, s, tau):
        s_next_ode = self._ode.predict(s, tau)
        s_norm = s / self._state_std
        a_norm = tau / self._action_std
        x = np.concatenate([s_norm, [a_norm]]).reshape(1, -1)
        delta_gp = np.array([gp.predict(x)[0] for gp in self._gp_residuals])
        correction = delta_gp * self._delta_std
        return s_next_ode + correction

    def predict_with_uncertainty(self, s, tau):
        s_next_ode = self._ode.predict(s, tau)
        s_norm = s / self._state_std
        a_norm = tau / self._action_std
        x = np.concatenate([s_norm, [a_norm]]).reshape(1, -1)
        delta_means = []
        delta_stds = []
        for gp in self._gp_residuals:
            mean, std = gp.predict(x, return_std=True)
            delta_means.append(mean[0])
            delta_stds.append(std[0])
        s_next = s_next_ode + np.array(delta_means) * self._delta_std
        uncertainty = np.array(delta_stds) * self._delta_std
        return s_next, uncertainty

    def name(self):
        return 'neural_ode_gp_residual'


class _NNEnsemble:
    """Simple NN ensemble for residual prediction."""

    def __init__(self, n_models, n_epochs, hidden=128):
        self.n_models = n_models
        self.n_epochs = n_epochs
        self.hidden = hidden
        self._models = []

    def train(self, X, Y):
        import torch
        import torch.nn as nn

        input_dim = X.shape[1]
        output_dim = Y.shape[1]

        class ResNet(nn.Module):
            def __init__(self, in_dim, out_dim, h):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(in_dim, h), nn.ReLU(),
                    nn.Linear(h, h), nn.ReLU(),
                    nn.Linear(h, out_dim),
                )
            def forward(self, x):
                return self.net(x)

        for i in range(self.n_models):
            model = ResNet(input_dim, output_dim, self.hidden)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            crit = nn.MSELoss()

            Xt = torch.FloatTensor(X)
            Yt = torch.FloatTensor(Y)
            ds = torch.utils.data.TensorDataset(Xt, Yt)
            loader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True)

            model.train()
            for epoch in range(self.n_epochs):
                for xb, yb in loader:
                    pred = model(xb)
                    loss = crit(pred, yb)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            model.eval()
            self._models.append(model)

    def predict_mean(self, X):
        import torch
        Xt = torch.FloatTensor(X)
        preds = []
        with torch.no_grad():
            for model in self._models:
                preds.append(model(Xt).numpy())
        return np.mean(preds, axis=0).flatten()

    def predict_mean_std(self, X):
        import torch
        Xt = torch.FloatTensor(X)
        preds = []
        with torch.no_grad():
            for model in self._models:
                preds.append(model(Xt).numpy())
        preds = np.array(preds)
        return preds.mean(axis=0).flatten(), preds.std(axis=0).flatten()


# ============================================================
# Main evaluation
# ============================================================
ROLLOUT_HORIZONS = [1, 5, 10, 20, 50, 100, 200, 500]
N_ROLLOUT_SEGMENTS = 5
ROLLOUT_SEGMENT_LENGTH = 600  # Need 500+ steps
OUTPUT_PATH = 'D:/系统辨识作业/sindy_bicycle/continuation_stage_v8/raw_results/AGENT_I_NEURAL_ODE_HYBRID.json'


def evaluate_multistep(model, segments, state_std, horizons):
    results = {}
    for h in horizons:
        metrics_list = []
        for seg in segments:
            s0 = seg['states'][0].copy()
            actions_seg = seg['actions']
            real_states = seg['states']

            n = min(h, len(actions_seg))
            sm = [s0.copy()]
            s_cur = s0.copy()
            survived = True

            for step in range(n):
                try:
                    s_next = model.predict(s_cur, actions_seg[step])
                    if np.any(np.isnan(s_next)) or np.any(np.isinf(s_next)):
                        survived = False
                        break
                    if np.any(np.abs(s_next) > 100.0):
                        survived = False
                        break
                    sm.append(s_next.copy())
                    s_cur = s_next
                except Exception:
                    survived = False
                    break

            sm = np.array(sm)
            sr = real_states[:len(sm)]
            n_valid = min(len(sm), len(sr)) - 1

            if n_valid > 0 and survived:
                m = compute_metrics(sm[:n_valid+1], sr[:n_valid+1], state_std, n,
                                    survival_steps=n_valid,
                                    failure_before_horizon=(n_valid < h))
            else:
                m = compute_metrics(sm[:n_valid+1] if n_valid > 0 else sm[:1],
                                    sr[:n_valid+1] if n_valid > 0 else sr[:1],
                                    state_std, n,
                                    survival_steps=max(n_valid, 0),
                                    failure_before_horizon=True)
            metrics_list.append(m)

        agg = {}
        for key in ['overall'] + STATE_NAMES_7D:
            vals = [m.get(key, {}).get('nmae', float('nan')) for m in metrics_list
                    if key in m]
            if vals:
                agg[key] = {
                    'nmae_mean': float(np.nanmean(vals)),
                    'nmae_std': float(np.nanstd(vals)),
                }
        results[h] = agg

    return results


def main():
    print("=" * 70)
    print("Agent I: Neural ODE Hybrids + Long-Horizon (200/500) Evaluation")
    print("=" * 70)

    # Load data
    print("\n[1/4] Loading 7D data...")
    t0 = time.time()
    cfg = DataConfig(data_path='D:/系统辨识作业/sindy_bicycle/data/stage2_dataset_150k.npz')
    data = load_7d_data(cfg)
    print(f"  Loaded in {time.time()-t0:.1f}s")
    print(f"  Train: {len(data['train_states'])}, Test: {len(data['test_states'])}")

    # Get rollout segments (need 500+ steps)
    rollout_segments = get_test_segments(
        data, n_segments=N_ROLLOUT_SEGMENTS,
        segment_length=ROLLOUT_SEGMENT_LENGTH
    )
    print(f"  Rollout segments: {len(rollout_segments)}, length={ROLLOUT_SEGMENT_LENGTH}")

    # Define models
    models = {
        'gp': GPModel(max_samples=2000, n_restarts=2),
        'gp_ensemble_5': GPEnsemble(max_samples=2000, n_models=5),
        'neural_ode': NeuralODEModel(n_epochs=100),
        'neural_ode_nn_res': NeuralODENNResidual(ode_epochs=100, n_models=5, nn_epochs=50),
        'neural_ode_gp_res': NeuralODEGPResidual(ode_epochs=100, gp_max_samples=2000),
    }

    # Train
    print("\n[2/4] Training models...")
    trained = {}
    times = {}
    for name, model in models.items():
        print(f"\n  Training {name}...")
        t0 = time.time()
        try:
            model.train(
                data['train_states'], data['train_actions'], data['train_deltas'],
                data['state_std'], data['action_std'], data['delta_std']
            )
            elapsed = time.time() - t0
            times[name] = round(elapsed, 2)
            trained[name] = model
            print(f"  DONE ({elapsed:.1f}s)")
        except Exception as e:
            elapsed = time.time() - t0
            times[name] = round(elapsed, 2)
            print(f"  FAILED ({elapsed:.1f}s): {e}")
            traceback.print_exc()

    print(f"\n  Trained: {len(trained)}/{len(models)}")

    # Multi-step evaluation
    print("\n[3/4] Multi-step rollout evaluation...")
    multistep = {}
    for name, model in trained.items():
        print(f"  Rollout {name}...", end=' ', flush=True)
        t0 = time.time()
        try:
            result = evaluate_multistep(model, rollout_segments, data['state_std'], ROLLOUT_HORIZONS)
            multistep[name] = result
            summaries = []
            for h in [1, 10, 50, 100, 200, 500]:
                if h in result and 'overall' in result[h]:
                    summaries.append(f"h={h}:{result[h]['overall']['nmae_mean']:.4f}")
            print(f"OK ({time.time()-t0:.1f}s) [{', '.join(summaries)}]")
        except Exception as e:
            print(f"FAILED: {e}")
            traceback.print_exc()

    # Save
    output = {
        'agent': 'I',
        'description': 'Neural ODE hybrids + long-horizon evaluation',
        'config': {
            'horizons': ROLLOUT_HORIZONS,
            'n_segments': N_ROLLOUT_SEGMENTS,
            'segment_length': ROLLOUT_SEGMENT_LENGTH,
        },
        'training_times': times,
        'multi_step': multistep,
    }

    import os
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    # Summary table
    print("\n" + "=" * 90)
    print("MULTI-STEP ROLLOUT NMAE SUMMARY")
    print("=" * 90)
    header = f"{'Method':<28}" + "".join(f" H={h:>5}" for h in ROLLOUT_HORIZONS)
    print(header)
    print("-" * (28 + 8 * len(ROLLOUT_HORIZONS)))
    for name in multistep:
        row = f"{name:<28}"
        for h in ROLLOUT_HORIZONS:
            if h in multistep[name] and 'overall' in multistep[name][h]:
                v = multistep[name][h]['overall']['nmae_mean']
                row += f" {v:>7.4f}"
            else:
                row += f"    N/A"
        print(row)

    print(f"\nResults saved to: {OUTPUT_PATH}")
    print("Done.")


if __name__ == '__main__':
    main()
