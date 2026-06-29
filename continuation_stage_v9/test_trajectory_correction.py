"""方案 5b: NODE + 轨迹级 GP 修正。

核心改变：GP 不是替代 NODE 的某些状态，而是修正 NODE 的预测误差。
GP 在真实的 NODE rollout 轨迹上训练，学习 NODE 在哪些区域会出错。

用法:
    python test_trajectory_correction.py
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
# 轨迹级 GP 修正
# ============================================================
class TrajectoryCorrectionGP:
    """GP 学习 NODE 的预测误差（在轨迹空间中）。

    训练数据来源：
    1. 训练集中的连续轨迹段
    2. 对每段轨迹做 NODE rollout
    3. 记录 NODE 预测 vs 真实的误差
    4. GP 学习 (s, tau) → NODE_error

    推理时：
    s_next = NODE(s, tau) + GP_correction(s, tau)
    """

    def __init__(self, node, max_samples=5000):
        self._node = node
        self._max_samples = max_samples
        self._gps = None
        self._x_mean = None
        self._x_std = None

    def train(self, data, state_std, action_std, delta_std):
        from sklearn.exceptions import ConvergenceWarning
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern

        self._state_std = state_std
        self._action_std = action_std
        self._delta_std = delta_std

        # 从训练集的连续轨迹段中收集 NODE 误差
        print("    Collecting NODE errors from trajectory segments...")
        train_mask = data['train_mask']
        all_states = data['all_states']
        all_actions = data['all_actions']
        all_deltas = data['all_deltas']

        train_indices = np.where(train_mask)[0]

        # 找连续段
        runs = []
        run_start = train_indices[0]
        for i in range(1, len(train_indices)):
            if train_indices[i] != train_indices[i - 1] + 1:
                runs.append((run_start, train_indices[i - 1] + 1))
                run_start = train_indices[i]
        runs.append((run_start, train_indices[-1] + 1))

        # 从每个段中采样轨迹，做 NODE rollout，记录误差
        correction_inputs = []  # (s, a) pairs
        correction_targets = []  # NODE error = real_delta - node_delta

        rng = np.random.RandomState(SEED)
        seg_len = 50  # 短轨迹段
        n_segments = 200  # 采样 200 个段

        for _ in range(n_segments):
            valid_runs = [(s, e) for s, e in runs if e - s >= seg_len + 1]
            if not valid_runs:
                continue
            r_start, r_end = valid_runs[rng.randint(len(valid_runs))]
            offset = rng.randint(r_start, r_end - seg_len)

            # NODE rollout
            s_cur = all_states[offset].copy()
            for step in range(seg_len):
                tau = all_actions[offset + step]
                real_next = all_states[offset + step + 1]
                real_delta = all_deltas[offset + step]

                # NODE 预测
                node_next = self._node.predict(s_cur, tau)
                node_delta = node_next - s_cur

                # 记录误差
                error = real_delta - node_delta
                correction_inputs.append(np.concatenate([
                    s_cur / state_std, [tau / action_std]
                ]))
                correction_targets.append(error / delta_std)

                # 用 NODE 预测继续 rollout（自回归）
                s_cur = node_next

        correction_inputs = np.array(correction_inputs)
        correction_targets = np.array(correction_targets)

        print(f"    Collected {len(correction_inputs)} correction samples")

        # 误差统计
        err_std = np.std(correction_targets, axis=0)
        print(f"    Correction target std (normalized): {err_std}")

        # 下采样
        n = len(correction_inputs)
        if n > self._max_samples:
            idx = rng.choice(n, self._max_samples, replace=False)
        else:
            idx = np.arange(n)

        X = correction_inputs[idx]
        Y = correction_targets[idx]

        self._x_mean = X.mean(axis=0)
        self._x_std = X.std(axis=0) + 1e-8
        X_scaled = (X - self._x_mean) / self._x_std

        # 只对 NODE 误差大的状态训练 GP
        # 先检查哪些状态的误差值得修正
        print(f"    Training GP for each state dimension...")
        self._gps = {}
        kernel = Matern(nu=2.5, length_scale=1.0)

        for col in range(STATE_DIM):
            # 只有当误差 std > 0.01 时才训练 GP
            if err_std[col] < 0.01:
                print(f"    {STATE_NAMES_7D[col]}: err_std={err_std[col]:.4f} < 0.01, skipping")
                continue

            t0 = time.time()
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1, alpha=1e-3)
            gp.fit(X_scaled, Y[:, col])
            self._gps[col] = gp
            print(f"    {STATE_NAMES_7D[col]}: GP trained ({time.time()-t0:.1f}s), err_std={err_std[col]:.4f}")

    def predict(self, s, tau):
        # NODE 预测
        s_next_node = self._node.predict(s, tau)

        # GP 修正
        if not self._gps:
            return s_next_node

        s_norm = s / self._state_std
        a_norm = tau / self._action_std
        x = np.concatenate([s_norm, [a_norm]]).reshape(1, -1)
        x_scaled = (x - self._x_mean) / self._x_std

        correction = np.zeros(STATE_DIM)
        for col, gp in self._gps.items():
            correction[col] = gp.predict(x_scaled)[0] * self._delta_std[col]

        return s_next_node + correction


# ============================================================
# 基线对比
# ============================================================
def main():
    print("=" * 80)
    print("方案 5b: NODE + 轨迹级 GP 修正")
    print("=" * 80)
    print(f"Device: {DEVICE}\n")

    # 加载数据
    print("Loading data...")
    cfg = DataConfig(
        data_path='D:/系统辨识作业/sindy_bicycle/data/stage2_dataset_150k.npz',
        seed=SEED,
    )
    data = load_7d_data(cfg)
    state_std = data['state_std']
    action_std = data['action_std']
    delta_std = data['delta_std']
    segments = get_test_segments(data, n_segments=N_EVAL_SEGMENTS,
                                 segment_length=SEGMENT_LENGTH, seed=SEED)
    print(f"  Train: {len(data['train_states'])}, Segments: {len(segments)}\n")

    # 加载 V9 NODE
    print("[1] Loading V9 Neural ODE checkpoint...")
    node = NeuralODE7D(state_std, action_std, delta_std)
    ckpt_path = Path('D:/系统辨识作业/sindy_bicycle/continuation_stage_v9/checkpoints/BEST_NEURAL_ODE_V9.pt')
    node.load_checkpoint(ckpt_path)

    # 评估 V9 NODE
    print("[2] Evaluating V9 NODE baseline...")
    results_node = multi_step_evaluate(node, segments, state_std, ROLLOUT_HORIZONS)

    # 训练轨迹级 GP 修正
    print("\n[3] Training trajectory correction GP...")
    t0 = time.time()
    correction_gp = TrajectoryCorrectionGP(node, max_samples=5000)
    correction_gp.train(data, state_std, action_std, delta_std)
    print(f"  Done in {time.time()-t0:.1f}s")

    # 评估 NODE + GP 修正
    print("\n[4] Evaluating NODE + Trajectory GP correction...")
    results_corrected = multi_step_evaluate(correction_gp, segments, state_std, ROLLOUT_HORIZONS)

    # 对比不同修正策略
    print("\n[5] Testing different correction strategies...")

    # 策略 A: 只修正 e_y 和 e_psi
    class SelectiveCorrection:
        def __init__(self, node, gp_model, correct_dims):
            self._node = node
            self._gp = gp_model
            self._correct_dims = correct_dims

        def predict(self, s, tau):
            s_next_node = self._node.predict(s, tau)
            if not self._gp._gps:
                return s_next_node

            s_norm = s / self._gp._state_std
            a_norm = tau / self._gp._action_std
            x = np.concatenate([s_norm, [a_norm]]).reshape(1, -1)
            x_scaled = (x - self._gp._x_mean) / self._gp._x_std

            correction = np.zeros(STATE_DIM)
            for col in self._correct_dims:
                if col in self._gp._gps:
                    correction[col] = self._gp._gps[col].predict(x_scaled)[0] * self._gp._delta_std[col]

            return s_next_node + correction

    # 策略 B: 修正所有状态
    results_corrected_all = results_corrected

    # 策略 C: 只修正 e_y, e_psi
    correct_ey_epsi = SelectiveCorrection(node, correction_gp, [0, 1])
    results_corrected_ey_epsi = multi_step_evaluate(correct_ey_epsi, segments, state_std, ROLLOUT_HORIZONS)

    # 策略 D: 修正 e_y, e_psi, theta_dot, delta_dot
    correct_dynamic = SelectiveCorrection(node, correction_gp, [0, 1, 4, 6])
    results_corrected_dynamic = multi_step_evaluate(correct_dynamic, segments, state_std, ROLLOUT_HORIZONS)

    # Pure GP 基线
    print("\n[6] Training pure GP baseline...")
    from test_improved_gp_v2 import PureGP
    pure_gp = PureGP(max_samples=2000)
    pure_gp.train(data['train_states'], data['train_actions'], data['train_deltas'],
                   state_std, action_std, delta_std)
    results_pure_gp = multi_step_evaluate(pure_gp, segments, state_std, ROLLOUT_HORIZONS)

    # 方案 5 PerState 基线
    from test_per_state_residual import PerStateHybridModel
    print("\n[7] Training PerState baseline...")
    perstate = PerStateHybridModel(node, state_std, action_std, delta_std,
                                    node_dims=[2, 3, 5], gp_dims=[0, 1, 4, 6])
    perstate.train_gp(data['train_states'], data['train_actions'], data['train_deltas'])
    results_perstate = multi_step_evaluate(perstate, segments, state_std, ROLLOUT_HORIZONS)

    # ============================================================
    # 输出对比表
    # ============================================================
    print("\n" + "=" * 120)
    print("综合对比: 轨迹级修正 vs 其他方案")
    print("=" * 120)

    v9_ref = {
        1: 0.0051, 5: 0.0170, 10: 0.0628, 20: 0.1817,
        50: 0.4557, 100: 0.5064, 200: 0.4737, 500: 0.5529, 1000: 0.6443,
    }

    methods = ['V9(ref)', 'NODE', 'PureGP', 'PerState', 'CorrAll', 'Corr(ey,epsi)', 'Corr(dynamic)']
    results_list = [v9_ref, results_node, results_pure_gp, results_perstate,
                    results_corrected_all, results_corrected_ey_epsi, results_corrected_dynamic]

    header = f"{'H':>5}"
    for m in methods:
        header += f" | {m:>14}"
    print(header)
    print("-" * (5 + 17 * len(methods)))

    for h in ROLLOUT_HORIZONS:
        row = f"{h:>5}"
        for r in results_list:
            if isinstance(r, dict) and h in r:
                if isinstance(r[h], dict):
                    row += f" | {r[h]['nmae_mean']:14.4f}"
                else:
                    row += f" | {r[h]:14.4f}"
            else:
                row += f" | {'---':>14}"
        print(row)

    # 存活率
    print("\n存活率:")
    for h in ROLLOUT_HORIZONS:
        row = f"{h:>5}"
        for r in results_list:
            if isinstance(r, dict) and h in r:
                if isinstance(r[h], dict):
                    row += f" | {r[h]['survival_rate']:13.0%}%"
                else:
                    row += f" | {'100%':>14}"
            else:
                row += f" | {'---':>14}"
        print(row)

    # 保存
    log_dir = Path('D:/系统辨识作业/sindy_bicycle/continuation_stage_v9/logs')
    log_dir.mkdir(exist_ok=True)
    out_path = log_dir / 'trajectory_correction_results.json'

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
