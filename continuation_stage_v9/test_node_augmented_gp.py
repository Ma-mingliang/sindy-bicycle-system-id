"""最终方案: NODE 增强 GP。

GP 的输入不仅有 (s, a)，还有 NODE 的预测 s_next_node。
GP 学习: (s, a, s_next_node) → s_next_real

这样 GP 可以利用 NODE 的预测作为"锚点"，
只需要学习 NODE 的残差，而残差比原始 delta 小得多。

用法:
    python test_node_augmented_gp.py
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
# Neural ODE
# ============================================================
class ODEFunc(nn.Module):
    def __init__(self, hidden=64, depth=3):
        super().__init__()
        layers = [nn.Linear(STATE_DIM + ACTION_DIM, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
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
        self._model = ODEFunc(hidden=64, depth=3).to(DEVICE)
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        self._model.load_state_dict(ckpt['model_state'])
        self._model.eval()

    def predict(self, s, tau):
        s_norm = torch.FloatTensor(s / self._state_std).unsqueeze(0).to(DEVICE)
        a_norm = torch.FloatTensor([tau / self._action_std]).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            dsdt_norm = self._model(s_norm, a_norm).cpu().numpy()[0]
        return s + dsdt_norm * self._delta_std * DT


# ============================================================
# NODE 增强 GP
# ============================================================
class NODEAugmentedGP:
    """GP 输入: [s_norm(7), a_norm(1), s_next_node_norm(7)] = 15D
    GP 输出: s_next_real_norm(7) 或 residual_norm(7)

    GP 学习: 给定当前状态、动作和 NODE 预测，输出修正后的下一状态。
    """

    def __init__(self, node, max_samples=5000):
        self._node = node
        self._max_samples = max_samples
        self._gps = None
        self._x_mean = None
        self._x_std = None

    def train(self, states, actions, deltas, state_std, action_std, delta_std):
        from sklearn.exceptions import ConvergenceWarning
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern

        self._state_std = state_std
        self._action_std = action_std
        self._delta_std = delta_std

        print("    Computing NODE predictions for training set...")
        # 计算 NODE 预测
        node_nexts = np.array([
            self._node.predict(states[i], actions[i])
            for i in range(len(states))
        ])
        node_deltas = node_nexts - states
        real_deltas = deltas

        # 残差 = 真实 - NODE
        residuals = real_deltas - node_deltas
        res_std = np.std(residuals, axis=0)
        print(f"    Residual std / delta_std: {res_std / delta_std}")

        # 构建增强输入: [s_norm, a_norm, s_next_node_norm]
        s_norm = states / state_std
        a_norm = actions.reshape(-1, 1) / action_std
        node_next_norm = node_nexts / state_std
        X_aug = np.column_stack([s_norm, a_norm, node_next_norm])  # (n, 15)

        # 目标: 残差（归一化）
        Y = residuals / delta_std  # (n, 7)

        # 下采样
        n = len(X_aug)
        if n > self._max_samples:
            rng = np.random.RandomState(SEED)
            idx = rng.choice(n, self._max_samples, replace=False)
        else:
            idx = np.arange(n)

        X = X_aug[idx]
        Y = Y[idx]

        self._x_mean = X.mean(axis=0)
        self._x_std = X.std(axis=0) + 1e-8
        X_scaled = (X - self._x_mean) / self._x_std

        self._gps = {}
        kernel = Matern(nu=2.5, length_scale=1.0)
        for col in range(STATE_DIM):
            t0 = time.time()
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1, alpha=1e-3)
            gp.fit(X_scaled, Y[:, col])
            self._gps[col] = gp
            print(f"    GP dim {col} ({STATE_NAMES_7D[col]}) done ({time.time()-t0:.1f}s)")

    def predict(self, s, tau):
        # NODE 预测
        s_next_node = self._node.predict(s, tau)

        # 构建增强输入
        s_norm = s / self._state_std
        a_norm = tau / self._action_std
        node_next_norm = s_next_node / self._state_std
        x = np.concatenate([s_norm, [a_norm], node_next_norm]).reshape(1, -1)
        x_scaled = (x - self._x_mean) / self._x_std

        # GP 预测残差
        correction = np.zeros(STATE_DIM)
        for col, gp in self._gps.items():
            correction[col] = gp.predict(x_scaled)[0] * self._delta_std[col]

        return s_next_node + correction


# ============================================================
# 对比: 标准 GP (无 NODE 增强)
# ============================================================
class StandardGP:
    """标准 GP: 输入 [s_norm, a_norm] = 8D，输出残差。"""

    def __init__(self, node, max_samples=5000):
        self._node = node
        self._max_samples = max_samples
        self._gps = None
        self._x_mean = None
        self._x_std = None

    def train(self, states, actions, deltas, state_std, action_std, delta_std):
        from sklearn.exceptions import ConvergenceWarning
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern

        self._state_std = state_std
        self._action_std = action_std
        self._delta_std = delta_std

        # 计算 NODE 残差
        node_nexts = np.array([self._node.predict(states[i], actions[i]) for i in range(len(states))])
        residuals = deltas - (node_nexts - states)

        # 标准输入: [s_norm, a_norm]
        X = np.column_stack([states / state_std, actions.reshape(-1, 1) / action_std])
        Y = residuals / delta_std

        n = len(X)
        if n > self._max_samples:
            rng = np.random.RandomState(SEED)
            idx = rng.choice(n, self._max_samples, replace=False)
        else:
            idx = np.arange(n)

        X, Y = X[idx], Y[idx]
        self._x_mean = X.mean(axis=0)
        self._x_std = X.std(axis=0) + 1e-8
        X_scaled = (X - self._x_mean) / self._x_std

        self._gps = {}
        kernel = Matern(nu=2.5, length_scale=1.0)
        for col in range(STATE_DIM):
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1, alpha=1e-3)
            gp.fit(X_scaled, Y[:, col])
            self._gps[col] = gp

    def predict(self, s, tau):
        s_next_node = self._node.predict(s, tau)
        s_norm = s / self._state_std
        a_norm = tau / self._action_std
        x = np.concatenate([s_norm, [a_norm]]).reshape(1, -1)
        x_scaled = (x - self._x_mean) / self._x_std
        correction = np.zeros(STATE_DIM)
        for col, gp in self._gps.items():
            correction[col] = gp.predict(x_scaled)[0] * self._delta_std[col]
        return s_next_node + correction


# ============================================================
# 主函数
# ============================================================
def main():
    print("=" * 80)
    print("最终方案: NODE 增强 GP vs 标准 GP")
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

    # 加载 V9 NODE
    print("[1] Loading V9 Neural ODE checkpoint...")
    node = NeuralODE7D(state_std, action_std, delta_std)
    ckpt_path = Path('D:/系统辨识作业/sindy_bicycle/continuation_stage_v9/checkpoints/BEST_NEURAL_ODE_V9.pt')
    node.load_checkpoint(ckpt_path)

    # 评估 V9 NODE
    print("[2] Evaluating V9 NODE baseline...")
    results_node = multi_step_evaluate(node, segments, state_std, ROLLOUT_HORIZONS)

    # 训练 NODE 增强 GP
    print("\n[3] Training NODE-augmented GP...")
    t0 = time.time()
    aug_gp = NODEAugmentedGP(node, max_samples=5000)
    aug_gp.train(train_s, train_a, train_d, state_std, action_std, delta_std)
    print(f"  Done in {time.time()-t0:.1f}s")

    # 评估 NODE 增强 GP
    print("\n[4] Evaluating NODE-augmented GP...")
    results_aug = multi_step_evaluate(aug_gp, segments, state_std, ROLLOUT_HORIZONS)

    # 训练标准 GP (对比)
    print("\n[5] Training standard GP (no NODE augmentation)...")
    t0 = time.time()
    std_gp = StandardGP(node, max_samples=5000)
    std_gp.train(train_s, train_a, train_d, state_std, action_std, delta_std)
    print(f"  Done in {time.time()-t0:.1f}s")

    # 评估标准 GP
    print("\n[6] Evaluating standard GP...")
    results_std = multi_step_evaluate(std_gp, segments, state_std, ROLLOUT_HORIZONS)

    # Pure GP 基线
    print("\n[7] Training pure GP baseline...")
    from test_improved_gp_v2 import PureGP
    pure_gp = PureGP(max_samples=2000)
    pure_gp.train(train_s, train_a, train_d, state_std, action_std, delta_std)
    results_pure_gp = multi_step_evaluate(pure_gp, segments, state_std, ROLLOUT_HORIZONS)

    # ============================================================
    # 输出对比表
    # ============================================================
    print("\n" + "=" * 100)
    print("综合对比: NODE 增强 GP")
    print("=" * 100)

    v9_ref = {
        1: 0.0051, 5: 0.0170, 10: 0.0628, 20: 0.1817,
        50: 0.4557, 100: 0.5064, 200: 0.4737, 500: 0.5529, 1000: 0.6443,
    }

    methods = ['V9(ref)', 'NODE', 'PureGP', 'StdGP+NODE', 'AugGP+NODE']
    results_list = [v9_ref, results_node, results_pure_gp, results_std, results_aug]

    header = f"{'H':>5}"
    for m in methods:
        header += f" | {m:>12}"
    print(header)
    print("-" * (5 + 15 * len(methods)))

    for h in ROLLOUT_HORIZONS:
        row = f"{h:>5}"
        for r in results_list:
            if isinstance(r, dict) and h in r:
                if isinstance(r[h], dict):
                    row += f" | {r[h]['nmae_mean']:12.4f}"
                else:
                    row += f" | {r[h]:12.4f}"
            else:
                row += f" | {'---':>12}"
        print(row)

    # 存活率
    print("\n存活率:")
    for h in ROLLOUT_HORIZONS:
        row = f"{h:>5}"
        for r in results_list:
            if isinstance(r, dict) and h in r:
                if isinstance(r[h], dict):
                    row += f" | {r[h]['survival_rate']:11.0%}%"
                else:
                    row += f" | {'100%':>12}"
            else:
                row += f" | {'---':>12}"
        print(row)

    # 改善百分比
    print("\n改善百分比 (vs V9 NODE):")
    for h in ROLLOUT_HORIZONS:
        v9_val = v9_ref.get(h, None)
        if v9_val is None:
            continue
        row = f"{h:>5}"
        for m, r in zip(methods, results_list):
            if isinstance(r, dict) and h in r:
                if isinstance(r[h], dict):
                    val = r[h]['nmae_mean']
                else:
                    val = r[h]
                imp = (val - v9_val) / v9_val * 100
                row += f" | {imp:>+11.1f}%"
            else:
                row += f" | {'---':>12}"
        print(row)

    # 保存
    log_dir = Path('D:/系统辨识作业/sindy_bicycle/continuation_stage_v9/logs')
    log_dir.mkdir(exist_ok=True)
    out_path = log_dir / 'node_augmented_gp_results.json'

    save_results = {}
    for h in ROLLOUT_HORIZONS:
        save_results[str(h)] = {}
        for m, r in zip(methods, results_list):
            if isinstance(r, dict) and h in r:
                if isinstance(r[h], dict):
                    save_results[str(h)][m] = r[h]['nmae_mean']
                else:
                    save_results[str(h)][m] = r[h]

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({'results': save_results}, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {out_path}")
    print("Done.")


if __name__ == '__main__':
    main()
