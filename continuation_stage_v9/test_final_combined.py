"""最终组合方案: 分状态 + NODE 增强 GP。

- 对 NODE 预测准的状态 (v, theta, delta): 直接用 NODE
- 对 NODE 预测差的状态 (e_y, e_psi, theta_dot, delta_dot): 用 NODE 增强 GP

用法:
    python test_final_combined.py
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
# 分状态 NODE 增强 GP
# ============================================================
class PerStateAugmentedGP:
    """对不同状态用不同策略：
    - NODE 预测准的 (v, theta, delta): 直接用 NODE
    - NODE 预测差的 (e_y, e_psi, theta_dot, delta_dot): 用 NODE 增强 GP
    """

    def __init__(self, node, node_dims, gp_dims, max_samples=5000):
        self._node = node
        self._node_dims = node_dims  # 直接用 NODE 的维度
        self._gp_dims = gp_dims      # 用 GP 修正的维度
        self._max_samples = max_samples
        self._gps = {}
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

        print("    Computing NODE predictions...")
        node_nexts = np.array([self._node.predict(states[i], actions[i]) for i in range(len(states))])
        node_deltas = node_nexts - states
        residuals = deltas - node_deltas

        # 构建增强输入: [s_norm, a_norm, s_next_node_norm]
        s_norm = states / state_std
        a_norm = actions.reshape(-1, 1) / action_std
        node_next_norm = node_nexts / state_std
        X_aug = np.column_stack([s_norm, a_norm, node_next_norm])  # (n, 15)

        # 下采样
        n = len(X_aug)
        if n > self._max_samples:
            rng = np.random.RandomState(SEED)
            idx = rng.choice(n, self._max_samples, replace=False)
        else:
            idx = np.arange(n)

        X = X_aug[idx]
        self._x_mean = X.mean(axis=0)
        self._x_std = X.std(axis=0) + 1e-8
        X_scaled = (X - self._x_mean) / self._x_std

        kernel = Matern(nu=2.5, length_scale=1.0)
        for col in self._gp_dims:
            Y = residuals[idx, col] / delta_std[col]
            t0 = time.time()
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1, alpha=1e-3)
            gp.fit(X_scaled, Y)
            self._gps[col] = gp
            print(f"    GP for {STATE_NAMES_7D[col]} done ({time.time()-t0:.1f}s)")

    def predict(self, s, tau):
        s_next_node = self._node.predict(s, tau)

        # 构建增强输入
        s_norm = s / self._state_std
        a_norm = tau / self._action_std
        node_next_norm = s_next_node / self._state_std
        x = np.concatenate([s_norm, [a_norm], node_next_norm]).reshape(1, -1)
        x_scaled = (x - self._x_mean) / self._x_std

        # 组合: NODE 直接 + GP 修正
        result = np.zeros(STATE_DIM)
        for dim in range(STATE_DIM):
            if dim in self._gp_dims and dim in self._gps:
                correction = self._gps[dim].predict(x_scaled)[0] * self._delta_std[dim]
                result[dim] = s_next_node[dim] + correction
            else:
                result[dim] = s_next_node[dim]

        return result


# ============================================================
# 主函数
# ============================================================
def main():
    print("=" * 80)
    print("最终组合: 分状态 + NODE 增强 GP")
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

    # 最佳组合: NODE 用于 v, theta, delta; AugGP 用于 e_y, e_psi, theta_dot, delta_dot
    print("\n[3] Training PerState AugmentedGP (NODE for v,theta,delta; AugGP for rest)...")
    t0 = time.time()
    combined = PerStateAugmentedGP(
        node,
        node_dims=[2, 3, 5],     # v, theta, delta
        gp_dims=[0, 1, 4, 6],    # e_y, e_psi, theta_dot, delta_dot
        max_samples=5000,
    )
    combined.train(train_s, train_a, train_d, state_std, action_std, delta_std)
    print(f"  Done in {time.time()-t0:.1f}s")

    print("\n[4] Evaluating combined model...")
    results_combined = multi_step_evaluate(combined, segments, state_std, ROLLOUT_HORIZONS)

    # 对比其他方案
    print("\n[5] Loading other results for comparison...")
    from test_improved_gp_v2 import PureGP
    pure_gp = PureGP(max_samples=2000)
    pure_gp.train(train_s, train_a, train_d, state_std, action_std, delta_std)
    results_pure_gp = multi_step_evaluate(pure_gp, segments, state_std, ROLLOUT_HORIZONS)

    # 方案 5 PerState (标准 GP)
    from test_per_state_residual import PerStateHybridModel
    perstate = PerStateHybridModel(node, state_std, action_std, delta_std,
                                    node_dims=[2, 3, 5], gp_dims=[0, 1, 4, 6])
    perstate.train_gp(train_s, train_a, train_d)
    results_perstate = multi_step_evaluate(perstate, segments, state_std, ROLLOUT_HORIZONS)

    # NODE 增强 GP (所有状态)
    from test_node_augmented_gp import NODEAugmentedGP
    aug_gp = NODEAugmentedGP(node, max_samples=5000)
    aug_gp.train(train_s, train_a, train_d, state_std, action_std, delta_std)
    results_aug = multi_step_evaluate(aug_gp, segments, state_std, ROLLOUT_HORIZONS)

    # ============================================================
    # 输出对比表
    # ============================================================
    print("\n" + "=" * 110)
    print("最终对比: 所有方案")
    print("=" * 110)

    v9_ref = {
        1: 0.0051, 5: 0.0170, 10: 0.0628, 20: 0.1817,
        50: 0.4557, 100: 0.5064, 200: 0.4737, 500: 0.5529, 1000: 0.6443,
    }

    methods = ['V9(ref)', 'NODE', 'PureGP', 'PerState', 'AugGP', 'Combined']
    results_list = [v9_ref, results_node, results_pure_gp, results_perstate, results_aug, results_combined]

    header = f"{'H':>5}"
    for m in methods:
        header += f" | {m:>10}"
    print(header)
    print("-" * (5 + 13 * len(methods)))

    for h in ROLLOUT_HORIZONS:
        row = f"{h:>5}"
        for r in results_list:
            if isinstance(r, dict) and h in r:
                if isinstance(r[h], dict):
                    row += f" | {r[h]['nmae_mean']:10.4f}"
                else:
                    row += f" | {r[h]:10.4f}"
            else:
                row += f" | {'---':>10}"
        print(row)

    # 存活率
    print("\n存活率:")
    for h in ROLLOUT_HORIZONS:
        row = f"{h:>5}"
        for r in results_list:
            if isinstance(r, dict) and h in r:
                if isinstance(r[h], dict):
                    row += f" | {r[h]['survival_rate']:9.0%}%"
                else:
                    row += f" | {'100%':>10}"
            else:
                row += f" | {'---':>10}"
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
                row += f" | {imp:>+9.1f}%"
            else:
                row += f" | {'---':>10}"
        print(row)

    # 保存
    log_dir = Path('D:/系统辨识作业/sindy_bicycle/continuation_stage_v9/logs')
    log_dir.mkdir(exist_ok=True)
    out_path = log_dir / 'final_combined_results.json'

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
