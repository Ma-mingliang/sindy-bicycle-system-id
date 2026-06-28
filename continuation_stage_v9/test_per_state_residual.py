"""方案 5: 分状态残差学习。

不同状态用不同策略：
- e_y, e_psi (动态大): 用 Neural ODE
- v, theta, delta, delta_dot (变化小): 用 GP
- theta_dot: 看情况

用法:
    python test_per_state_residual.py
"""
import sys
sys.path.insert(0, 'D:/系统辨识作业/sindy_bicycle')
sys.path.insert(0, 'D:/系统辨识作业/sindy_bicycle/continuation_stage_v8')
sys.stdout.reconfigure(line_buffering=True)

import json
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
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
DT = 1.0 / 30.0


# ============================================================
# Neural ODE (用于动态状态)
# ============================================================
class ODEFunc(nn.Module):
    def __init__(self, hidden=64, depth=3, activation='tanh'):
        super().__init__()
        act = nn.Tanh if activation == 'tanh' else nn.SiLU
        layers = [nn.Linear(STATE_DIM + ACTION_DIM, hidden), act()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), act()]
        layers.append(nn.Linear(hidden, STATE_DIM))
        self.net = nn.Sequential(*layers)
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.1)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=-1))


class NeuralODE7D:
    def __init__(self, state_std, action_std, delta_std):
        self._state_std = state_std
        self._action_std = action_std
        self._delta_std = delta_std
        self._model = None

    def load_checkpoint(self, path):
        self._model = ODEFunc(hidden=64, depth=3, activation='tanh').to(DEVICE)
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        if isinstance(ckpt, dict) and 'model_state' in ckpt:
            state_dict = ckpt['model_state']
        elif isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
        else:
            state_dict = ckpt
        self._model.load_state_dict(state_dict)
        self._model.eval()

    def predict(self, s, tau):
        s_norm = torch.FloatTensor(s / self._state_std).unsqueeze(0).to(DEVICE)
        a_norm = torch.FloatTensor([tau / self._action_std]).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            dsdt_norm = self._model(s_norm, a_norm).cpu().numpy()[0]
        return s + dsdt_norm * self._delta_std * DT

    def predict_delta(self, s, tau):
        s_norm = torch.FloatTensor(s / self._state_std).unsqueeze(0).to(DEVICE)
        a_norm = torch.FloatTensor([tau / self._action_std]).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            dsdt_norm = self._model(s_norm, a_norm).cpu().numpy()[0]
        return dsdt_norm * self._delta_std * DT


# ============================================================
# GP (用于稳定状态)
# ============================================================
class GPSingleDim:
    """单维度 GP。"""

    def __init__(self, max_samples=2000):
        self._max_samples = max_samples
        self._gp = None
        self._x_mean = None
        self._x_std = None

    def train(self, X, y):
        from sklearn.exceptions import ConvergenceWarning
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern

        n = len(X)
        if n > self._max_samples:
            rng = np.random.RandomState(SEED)
            idx = rng.choice(n, self._max_samples, replace=False)
            X, y = X[idx], y[idx]

        self._x_mean = X.mean(axis=0)
        self._x_std = X.std(axis=0) + 1e-8
        X_scaled = (X - self._x_mean) / self._x_std

        kernel = Matern(nu=2.5, length_scale=1.0)
        self._gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1, alpha=1e-3)
        self._gp.fit(X_scaled, y)

    def predict(self, x):
        x_scaled = (x - self._x_mean) / self._x_std
        return self._gp.predict(x_scaled.reshape(1, -1))[0]


# ============================================================
# 分状态混合模型
# ============================================================
class PerStateHybridModel:
    """分状态混合：动态状态用 NODE，稳定状态用 GP。"""

    def __init__(self, node, state_std, action_std, delta_std,
                 node_dims, gp_dims):
        """
        node_dims: 使用 Neural ODE 的状态维度索引
        gp_dims: 使用 GP 的状态维度索引
        """
        self._node = node
        self._state_std = state_std
        self._action_std = action_std
        self._delta_std = delta_std
        self._node_dims = node_dims
        self._gp_dims = gp_dims
        self._gp_models = {}

    def train_gp(self, train_s, train_a, train_d):
        """为 GP 维度训练 GP 模型。"""
        X = np.column_stack([
            train_s / self._state_std,
            train_a.reshape(-1, 1) / self._action_std,
        ])  # (n, 8)

        for dim in self._gp_dims:
            Y = train_d[:, dim] / self._delta_std[dim]
            gp = GPSingleDim(max_samples=2000)
            gp.train(X, Y)
            self._gp_models[dim] = gp
            print(f"    GP for {STATE_NAMES_7D[dim]} trained")

    def predict(self, s, tau):
        # Neural ODE 预测所有维度
        delta_node = self._node.predict_delta(s, tau)

        # GP 输入
        s_norm = s / self._state_std
        a_norm = tau / self._action_std
        x = np.concatenate([s_norm, [a_norm]])

        # 组合
        delta = np.zeros(STATE_DIM)
        for dim in range(STATE_DIM):
            if dim in self._gp_dims and dim in self._gp_models:
                delta[dim] = self._gp_models[dim].predict(x) * self._delta_std[dim]
            else:
                delta[dim] = delta_node[dim]

        return s + delta


# ============================================================
# 主函数
# ============================================================
def main():
    print("=" * 80)
    print("方案 5: 分状态残差学习")
    print("=" * 80)
    print(f"Device: {DEVICE}\n")

    # 加载数据
    print("Loading data...")
    cfg = DataConfig(
        data_path='D:/系统辨识作业/sindy_bicycle/data/stage2_dataset_150k.npz',
        seed=SEED,
    )
    data = load_7d_data(cfg)
    train_s = data['train_states']
    train_a = data['train_actions']
    train_d = data['train_deltas']
    state_std = data['state_std']
    action_std = data['action_std']
    delta_std = data['delta_std']
    segments = get_test_segments(data, n_segments=N_EVAL_SEGMENTS,
                                 segment_length=SEGMENT_LENGTH, seed=SEED)
    print(f"  Train: {len(train_s)}, Segments: {len(segments)}\n")

    # 加载 V9 Neural ODE
    print("[1] Loading V9 Neural ODE checkpoint...")
    node = NeuralODE7D(state_std, action_std, delta_std)
    ckpt_path = Path('D:/系统辨识作业/sindy_bicycle/continuation_stage_v9/checkpoints/BEST_NEURAL_ODE_V9.pt')
    node.load_checkpoint(ckpt_path)

    # 评估 V9 NODE
    print("[2] Evaluating V9 NODE baseline...")
    results_v9 = multi_step_evaluate(node, segments, state_std, ROLLOUT_HORIZONS)

    # ============================================================
    # 分析每个状态的 delta 大小，决定分组
    # ============================================================
    print("\n[3] Analyzing per-state delta magnitudes...")
    test_s = data['test_states']
    test_a = data['test_actions']
    test_d = data['test_deltas']

    rng = np.random.RandomState(42)
    sample_idx = rng.choice(len(test_s), 500, replace=False)

    node_deltas = np.array([node.predict_delta(test_s[i], test_a[i]) for i in sample_idx])
    real_deltas = test_d[sample_idx]

    print(f"\n{'状态':>12} | {'真实|delta|':>12} | {'NODE|delta|':>12} | {'NODE残差std':>12} | {'分组':>8}")
    print("-" * 65)

    groups = {}
    for j, name in enumerate(STATE_NAMES_7D):
        real_mean = np.mean(np.abs(real_deltas[:, j]))
        node_mean = np.mean(np.abs(node_deltas[:, j]))
        residual_std = np.std(real_deltas[:, j] - node_deltas[:, j])

        # 如果 NODE 残差 std < 真实 delta std 的 50%，用 NODE
        orig_std = np.std(real_deltas[:, j])
        ratio = residual_std / orig_std if orig_std > 1e-10 else 1.0

        if ratio < 0.5:
            group = "NODE"
        else:
            group = "GP"

        groups[j] = group
        print(f"  {name:>12} | {real_mean:12.6f} | {node_mean:12.6f} | {residual_std:12.6f} | {group:>8}")

    node_dims = [j for j, g in groups.items() if g == "NODE"]
    gp_dims = [j for j, g in groups.items() if g == "GP"]
    print(f"\n  NODE dims: {[STATE_NAMES_7D[j] for j in node_dims]}")
    print(f"  GP dims:   {[STATE_NAMES_7D[j] for j in gp_dims]}")

    # ============================================================
    # 训练分状态混合模型
    # ============================================================
    print("\n[4] Training per-state hybrid model...")
    t0 = time.time()
    hybrid = PerStateHybridModel(node, state_std, action_std, delta_std,
                                  node_dims=node_dims, gp_dims=gp_dims)
    hybrid.train_gp(train_s, train_a, train_d)
    print(f"  Done in {time.time()-t0:.1f}s")

    # ============================================================
    # 测试不同分组策略
    # ============================================================
    print("\n[5] Testing different grouping strategies...")

    # 策略 A: 自动分组 (基于残差分析)
    print("\n  Strategy A: Auto (based on residual analysis)")
    results_a = multi_step_evaluate(hybrid, segments, state_std, ROLLOUT_HORIZONS)

    # 策略 B: NODE 只用于 e_y, e_psi
    print("  Strategy B: NODE for e_y,e_psi only")
    hybrid_b = PerStateHybridModel(node, state_std, action_std, delta_std,
                                    node_dims=[0, 1], gp_dims=[2, 3, 4, 5, 6])
    hybrid_b.train_gp(train_s, train_a, train_d)
    results_b = multi_step_evaluate(hybrid_b, segments, state_std, ROLLOUT_HORIZONS)

    # 策略 C: NODE 用于 e_y, e_psi, theta_dot
    print("  Strategy C: NODE for e_y,e_psi,theta_dot")
    hybrid_c = PerStateHybridModel(node, state_std, action_std, delta_std,
                                    node_dims=[0, 1, 4], gp_dims=[2, 3, 5, 6])
    hybrid_c.train_gp(train_s, train_a, train_d)
    results_c = multi_step_evaluate(hybrid_c, segments, state_std, ROLLOUT_HORIZONS)

    # 策略 D: 纯 GP 基线
    print("  Strategy D: Pure GP (all dims)")
    from test_improved_gp_v2 import PureGP
    pure_gp = PureGP(max_samples=2000)
    pure_gp.train(train_s, train_a, train_d, state_std, action_std, delta_std)
    results_d = multi_step_evaluate(pure_gp, segments, state_std, ROLLOUT_HORIZONS)

    # ============================================================
    # 输出对比表
    # ============================================================
    print("\n" + "=" * 100)
    print("综合对比: 分状态残差学习")
    print("=" * 100)

    v9_ref = {
        1: 0.0051, 5: 0.0170, 10: 0.0628, 20: 0.1817,
        50: 0.4557, 100: 0.5064, 200: 0.4737, 500: 0.5529, 1000: 0.6443,
    }

    methods = ['V9(ref)', 'PureNODE', 'Auto', 'NODE(e_y,e_psi)', 'NODE(e_y,e_psi,thd)', 'PureGP']
    results_list = [v9_ref, results_v9, results_a, results_b, results_c, results_d]

    header = f"{'H':>5}"
    for m in methods:
        header += f" | {m:>16}"
    print(header)
    print("-" * (5 + 19 * len(methods)))

    for h in ROLLOUT_HORIZONS:
        row = f"{h:>5}"
        for i, (m, r) in enumerate(zip(methods, results_list)):
            if isinstance(r, dict) and h in r:
                if isinstance(r[h], dict):
                    row += f" | {r[h]['nmae_mean']:16.4f}"
                else:
                    row += f" | {r[h]:16.4f}"
            else:
                row += f" | {'---':>16}"
        print(row)

    # 存活率
    print("\n存活率:")
    for h in ROLLOUT_HORIZONS:
        row = f"{h:>5}"
        for i, (m, r) in enumerate(zip(methods, results_list)):
            if isinstance(r, dict) and h in r:
                if isinstance(r[h], dict):
                    row += f" | {r[h]['survival_rate']:15.0%}%"
                else:
                    row += f" | {'100%':>16}"
            else:
                row += f" | {'---':>16}"
        print(row)

    # 保存
    log_dir = Path('D:/系统辨识作业/sindy_bicycle/continuation_stage_v9/logs')
    log_dir.mkdir(exist_ok=True)
    out_path = log_dir / 'per_state_residual_results.json'

    all_results = {}
    for h in ROLLOUT_HORIZONS:
        all_results[str(h)] = {
            'v9_ref': v9_ref.get(h, None),
            'pure_node': results_v9[h]['nmae_mean'],
            'auto': results_a[h]['nmae_mean'],
            'node_ey_epsi': results_b[h]['nmae_mean'],
            'node_ey_epsi_thd': results_c[h]['nmae_mean'],
            'pure_gp': results_d[h]['nmae_mean'],
        }

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            'results': all_results,
            'groups': {STATE_NAMES_7D[j]: groups[j] for j in range(STATE_DIM)},
        }, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {out_path}")
    print("Done.")


if __name__ == '__main__':
    main()
