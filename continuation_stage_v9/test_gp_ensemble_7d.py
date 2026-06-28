"""GP + Ensemble(5) + OOD + DAgger + 0.3 — 7D 基线测试。

将 4D 项目的最佳方法移植到 7D，使用 V9 评估框架对比 Neural ODE。

用法:
    python test_gp_ensemble_7d.py
"""
import sys
sys.path.insert(0, 'D:/系统辨识作业/sindy_bicycle')
sys.path.insert(0, 'D:/系统辨识作业/sindy_bicycle/continuation_stage_v8')
sys.stdout.reconfigure(line_buffering=True)

import json
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

from canonical_7d.data_loader import load_7d_data, get_test_segments
from canonical_7d.config import DataConfig
from canonical_node.evaluation_v9 import multi_step_evaluate
from canonical_node.config_v9 import (
    STATE_NAMES_7D, STATE_DIM, ACTION_DIM,
    ROLLOUT_HORIZONS, N_EVAL_SEGMENTS, SEGMENT_LENGTH,
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42

# ============================================================
# GP 基线 (7D)
# ============================================================
class GP7D:
    """7 个独立高斯过程，每个预测一个状态维度的归一化 delta。"""

    def __init__(self, max_samples: int = 2000):
        self.max_samples = max_samples
        self._gps = None
        self._state_std = None
        self._action_std = None
        self._delta_std = None

    def train(self, states, actions, deltas, state_std, action_std, delta_std):
        import warnings
        from sklearn.exceptions import ConvergenceWarning
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern

        self._state_std = state_std
        self._action_std = action_std
        self._delta_std = delta_std

        n = len(states)
        if n > self.max_samples:
            rng = np.random.RandomState(SEED)
            idx = rng.choice(n, self.max_samples, replace=False)
        else:
            idx = np.arange(n)

        X = np.column_stack([
            states[idx] / state_std,
            actions[idx].reshape(-1, 1) / action_std,
        ])  # (n, 8)
        Y = deltas[idx] / delta_std  # (n, 7)

        # 标准化 X 以加速 GP 收敛
        self._x_mean = X.mean(axis=0)
        self._x_std = X.std(axis=0) + 1e-8
        X_scaled = (X - self._x_mean) / self._x_std

        self._gps = []
        kernel = Matern(nu=2.5, length_scale=1.0)
        for col in range(STATE_DIM):
            t0 = time.time()
            gp = GaussianProcessRegressor(
                kernel=kernel, n_restarts_optimizer=1, alpha=1e-3,
            )
            gp.fit(X_scaled, Y[:, col])
            self._gps.append(gp)
            print(f"    GP dim {col} ({STATE_NAMES_7D[col]}) done ({time.time()-t0:.1f}s)")

    def _prepare_input(self, s, tau):
        s_norm = s / self._state_std
        a_norm = tau / self._action_std
        x = np.concatenate([s_norm, [a_norm]]).reshape(1, -1)
        return (x - self._x_mean) / self._x_std

    def predict(self, s, tau):
        x_scaled = self._prepare_input(s, tau)
        delta_norm = np.array([gp.predict(x_scaled)[0] for gp in self._gps])
        return s + delta_norm * self._delta_std

    def predict_delta_norm(self, s, tau):
        """返回归一化 delta（供残差计算用）。"""
        x_scaled = self._prepare_input(s, tau)
        return np.array([gp.predict(x_scaled)[0] for gp in self._gps])


# ============================================================
# 集成残差网络 (7D)
# ============================================================
class ResidualNet7D(nn.Module):
    """输入: [s_norm(7), a_norm(1)] = 8D  输出: 归一化残差 (7,)"""

    def __init__(self, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM + ACTION_DIM, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, STATE_DIM),
        )

    def forward(self, s, a=None):
        if a is not None:
            return self.net(torch.cat([s, a], dim=-1))
        return self.net(s)


class EnsembleResidualNet7D:
    """5 个 ResidualNet7D 的集成。"""

    def __init__(self, n_models: int = 5):
        self.n_models = n_models
        self.models = []

    def train(self, train_inputs, train_residuals, n_epochs: int = 100):
        self.models = []
        for i in range(self.n_models):
            torch.manual_seed(i * 42)
            np.random.seed(i * 42)

            model = ResidualNet7D().to(DEVICE)
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
        print(f"    Ensemble ({self.n_models} models) trained")

    def predict(self, s_norm, a_norm):
        s_t = torch.FloatTensor(s_norm).unsqueeze(0).to(DEVICE)
        a_t = torch.FloatTensor([a_norm]).unsqueeze(0).to(DEVICE)
        preds = []
        for model in self.models:
            with torch.no_grad():
                preds.append(model(s_t, a_t).cpu().numpy()[0])
        preds = np.array(preds)
        return np.mean(preds, axis=0), np.std(preds, axis=0)

    def predict_mean(self, s_norm, a_norm):
        mean, _ = self.predict(s_norm, a_norm)
        return mean


# ============================================================
# OOD 检测器
# ============================================================
class OODDetector:
    """基于 Mahalanobis 距离的自适应 OOD 检测。"""

    def __init__(self, train_inputs, threshold: float = 3.0, max_scale: float = 0.5):
        self.mean = np.mean(train_inputs, axis=0)
        self.std = np.std(train_inputs, axis=0) + 1e-8
        self.threshold = threshold
        self.max_scale = max_scale

    def compute_scale(self, x) -> float:
        z = np.abs(x - self.mean) / self.std
        mahal_dist = np.max(z)
        k = 3.0
        normalized_dist = (mahal_dist - 1.0) / (self.threshold - 1.0)
        clipped = max(-10.0, min(10.0, k * normalized_dist * 4))
        sigmoid = 1.0 / (1.0 + math.exp(clipped))
        return sigmoid * self.max_scale


# ============================================================
# 训练管道
# ============================================================
def train_gp_ensemble_7d(data, n_models=5, n_epochs=100, dagger_rounds=3,
                         residual_scale=0.3, gp_max_samples=2000):
    """完整训练: GP 基线 + Ensemble NN 残差 + DAgger。"""
    train_s = data['train_states']
    train_a = data['train_actions']
    train_d = data['train_deltas']
    state_std = data['state_std']
    action_std = data['action_std']
    delta_std = data['delta_std']

    # Step 1: 训练 GP 基线
    print("[1/4] Training GP baseline (7D)...")
    gp = GP7D(max_samples=gp_max_samples)
    gp.train(train_s, train_a, train_d, state_std, action_std, delta_std)

    # Step 2: 计算初始残差
    print("[2/4] Computing initial residuals...")
    n_train = len(train_s)
    residuals = np.empty((n_train, STATE_DIM))
    for i in range(n_train):
        s_next_gp = gp.predict(train_s[i], train_a[i])
        gp_delta = s_next_gp - train_s[i]
        residuals[i] = (train_d[i] - gp_delta) / delta_std

    train_inputs_base = np.column_stack([
        train_s / state_std,
        train_a.reshape(-1, 1) / action_std,
    ])
    all_train_inputs = [train_inputs_base.copy()]
    all_train_residuals = [residuals.copy()]

    # Step 3: DAgger
    print("[3/4] DAgger rounds...")
    ensemble = None
    for round_i in range(dagger_rounds):
        train_inputs = np.vstack(all_train_inputs)
        train_residuals = np.vstack(all_train_residuals)

        ensemble = EnsembleResidualNet7D(n_models=n_models)
        ensemble.train(train_inputs, train_residuals, n_epochs=n_epochs)

        if round_i < dagger_rounds - 1:
            new_inputs, new_residuals = _dagger_collect_7d(
                data, gp, ensemble, state_std, action_std, delta_std, residual_scale,
            )
            if new_inputs:
                all_train_inputs.append(np.array(new_inputs))
                all_train_residuals.append(np.array(new_residuals))
                print(f"    DAgger round {round_i+1}: {len(new_inputs)} new samples")

    # Step 4: OOD 检测器
    print("[4/4] Building OOD detector...")
    final_inputs = np.vstack(all_train_inputs)
    ood = OODDetector(final_inputs, threshold=3.0, max_scale=0.5)

    return gp, ensemble, ood, (state_std, action_std, delta_std)


def _dagger_collect_7d(data, gp, ensemble, state_std, action_std, delta_std,
                       residual_scale, n_segments=3, seg_len=500):
    """DAgger 数据收集：用真实轨迹段作为 ground truth。"""
    all_states = data['all_states']
    all_actions = data['all_actions']
    all_deltas = data['all_deltas']
    train_mask = data['train_mask']
    train_indices = np.where(train_mask)[0]

    # 找连续段
    runs = []
    run_start = train_indices[0]
    for i in range(1, len(train_indices)):
        if train_indices[i] != train_indices[i - 1] + 1:
            runs.append((run_start, train_indices[i - 1] + 1))
            run_start = train_indices[i]
    runs.append((run_start, train_indices[-1] + 1))

    rng = np.random.RandomState(SEED + 100)
    new_inputs = []
    new_residuals = []

    for _ in range(n_segments):
        # 随机选一个足够长的段
        valid_runs = [(s, e) for s, e in runs if e - s >= seg_len + 1]
        if not valid_runs:
            continue
        r_start, r_end = valid_runs[rng.randint(len(valid_runs))]
        offset = rng.randint(r_start, r_end - seg_len)

        s_cur = all_states[offset].copy()
        for step in range(seg_len):
            tau = all_actions[offset + step]
            real_next = all_states[offset + step + 1]
            real_delta = all_deltas[offset + step]

            s_norm = s_cur / state_std
            a_norm = tau / action_std
            delta_nn = ensemble.predict_mean(s_norm, a_norm)

            # GP + ensemble 预测
            s_next_pred = gp.predict(s_cur, tau) + delta_nn * delta_std * residual_scale

            # 计算残差 (相对于 GP 基线)
            gp_delta = gp.predict(s_cur, tau) - s_cur
            res = (real_delta - gp_delta) / delta_std
            new_inputs.append(np.concatenate([s_norm, [a_norm]]))
            new_residuals.append(res)

            s_cur = s_next_pred  # 用模型预测继续 rollout

    return new_inputs, new_residuals


# ============================================================
# 评估模型包装
# ============================================================
class GPEnsemble7DModel:
    """包装 GP + Ensemble + OOD，暴露 predict(s, tau) 接口。"""

    def __init__(self, gp, ensemble, ood, state_std, action_std, delta_std,
                 residual_scale=0.3):
        self.gp = gp
        self.ensemble = ensemble
        self.ood = ood
        self.state_std = state_std
        self.action_std = action_std
        self.delta_std = delta_std
        self.residual_scale = residual_scale

    def predict(self, s, tau):
        s_next_gp = self.gp.predict(s, tau)
        s_norm = s / self.state_std
        a_norm = tau / self.action_std
        inp = np.concatenate([s_norm, [a_norm]])

        scale = self.ood.compute_scale(inp)
        if scale < 0.01:
            return s_next_gp

        delta_nn = self.ensemble.predict_mean(s_norm, a_norm)
        return s_next_gp + delta_nn * self.delta_std * scale


class GPBaseline7DModel:
    """纯 GP 基线，暴露 predict(s, tau) 接口。"""

    def __init__(self, gp, state_std, action_std, delta_std):
        self.gp = gp
        self.state_std = state_std
        self.action_std = action_std
        self.delta_std = delta_std

    def predict(self, s, tau):
        return self.gp.predict(s, tau)


# ============================================================
# 主函数
# ============================================================
def main():
    print("=" * 70)
    print("GP + Ensemble(5) + OOD + DAgger + 0.3 — 7D Baseline")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"State dim: {STATE_DIM}, Action dim: {ACTION_DIM}")
    print()

    # 加载数据
    print("Loading 7D data...")
    t0 = time.time()
    cfg = DataConfig(
        data_path='D:/系统辨识作业/sindy_bicycle/data/stage2_dataset_150k.npz',
        seed=SEED,
    )
    data = load_7d_data(cfg)
    print(f"  Train: {len(data['train_states'])} samples")
    print(f"  Test:  {len(data['test_states'])} samples")
    print(f"  state_std: {data['state_std']}")
    print(f"  delta_std: {data['delta_std']}")
    print(f"  Loaded in {time.time()-t0:.1f}s")
    print()

    # 训练
    print("Training GP + Ensemble + DAgger...")
    t0 = time.time()
    gp, ensemble, ood, (state_std, action_std, delta_std) = train_gp_ensemble_7d(
        data, n_models=5, n_epochs=100, dagger_rounds=3, residual_scale=0.3,
        gp_max_samples=2000,
    )
    print(f"Training done in {time.time()-t0:.1f}s")
    print()

    # 构建评估模型
    gp_ensemble_model = GPEnsemble7DModel(
        gp, ensemble, ood, state_std, action_std, delta_std, residual_scale=0.3,
    )
    gp_baseline_model = GPBaseline7DModel(gp, state_std, action_std, delta_std)

    # 获取测试段
    print("Getting test segments...")
    segments = get_test_segments(data, n_segments=N_EVAL_SEGMENTS,
                                 segment_length=SEGMENT_LENGTH, seed=SEED)
    print(f"  {len(segments)} segments, length {SEGMENT_LENGTH}")
    print()

    # 评估 GP 基线
    print("Evaluating GP baseline...")
    results_gp = multi_step_evaluate(
        gp_baseline_model, segments, state_std, ROLLOUT_HORIZONS,
    )

    # 评估 GP + Ensemble
    print("Evaluating GP + Ensemble + OOD...")
    results_ensemble = multi_step_evaluate(
        gp_ensemble_model, segments, state_std, ROLLOUT_HORIZONS,
    )

    # 打印结果表
    print()
    print("=" * 90)
    print("RESULTS: GP + Ensemble(5) + OOD + DAgger + 0.3 (7D)")
    print("=" * 90)
    print(f"{'Horizon':>8} | {'GP NMAE':>10} {'Surv':>6} | {'GP+Ens NMAE':>12} {'Surv':>6} | {'V9 NeuralODE':>13}")
    print("-" * 90)

    v9_baseline = {
        1: 0.0051, 5: 0.0170, 10: 0.0628, 20: 0.1817,
        50: 0.4557, 100: 0.5064, 200: 0.4737, 500: 0.5529, 1000: 0.6443,
    }

    all_results = {}
    for h in ROLLOUT_HORIZONS:
        gp_r = results_gp[h]
        ens_r = results_ensemble[h]
        v9_nmae = v9_baseline.get(h, float('nan'))
        print(f"  H={h:<5} | {gp_r['nmae_mean']:10.4f} {gp_r['survival_rate']:5.0%} "
              f"| {ens_r['nmae_mean']:12.4f} {ens_r['survival_rate']:5.0%} "
              f"| {v9_nmae:13.4f}")
        all_results[h] = {
            'gp_nmae': gp_r['nmae_mean'],
            'gp_survival': gp_r['survival_rate'],
            'gp_ensemble_nmae': ens_r['nmae_mean'],
            'gp_ensemble_survival': ens_r['survival_rate'],
            'v9_neural_ode_nmae': v9_nmae,
        }

    # 每状态详细结果 (H=100)
    print()
    print("Per-state NMAE at H=100:")
    print(f"  {'State':>12} | {'GP':>8} | {'GP+Ens':>8} | {'V9 NODE':>8}")
    print("  " + "-" * 45)
    for name in STATE_NAMES_7D:
        gp_val = results_gp[100]['per_state_nmae'][name]['mean']
        ens_val = results_ensemble[100]['per_state_nmae'][name]['mean']
        print(f"  {name:>12} | {gp_val:8.4f} | {ens_val:8.4f} | {'---':>8}")

    # 保存结果
    log_dir = Path('D:/系统辨识作业/sindy_bicycle/continuation_stage_v9/logs')
    log_dir.mkdir(exist_ok=True)
    out_path = log_dir / 'gp_ensemble_7d_results.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            'method': 'GP+Ensemble(5)+OOD+DAgger+0.3',
            'state_dim': 7,
            'n_models': 5,
            'dagger_rounds': 3,
            'residual_scale': 0.3,
            'gp_max_samples': 2000,
            'n_eval_segments': N_EVAL_SEGMENTS,
            'segment_length': SEGMENT_LENGTH,
            'results': {str(k): v for k, v in all_results.items()},
        }, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")
    print("Done.")


if __name__ == '__main__':
    main()
