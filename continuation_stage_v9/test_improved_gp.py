"""改进 GP 方案：方案1 (GP+物理先验残差) + 方案3 (短期NODE+长期GP混合)。

用法:
    python test_improved_gp.py
"""
import sys
sys.path.insert(0, 'D:/系统辨识作业/sindy_bicycle')
sys.path.insert(0, 'D:/系统辨识作业/sindy_bicycle/continuation_stage_v8')
sys.stdout.reconfigure(line_buffering=True)

import json
import math
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
# 物理先验：线性化动力学
# ============================================================
class LinearPhysicsPrior:
    """用最小二乘拟合的线性模型作为物理先验。

    delta ≈ A_lin @ [s, tau] + b_lin
    """

    def __init__(self):
        self._coef = None  # (8, 7) — 7D state + 1D action -> 7D delta
        self._bias = None  # (7,)
        self._state_std = None
        self._action_std = None
        self._delta_std = None

    def train(self, states, actions, deltas, state_std, action_std, delta_std):
        self._state_std = state_std
        self._action_std = action_std
        self._delta_std = delta_std

        # 拟合 delta = X @ coef
        X = np.column_stack([states, actions.reshape(-1, 1)])  # (n, 8)
        coef, residuals, rank, sv = np.linalg.lstsq(X, deltas, rcond=None)
        self._coef = coef  # (8, 7)

        # 计算拟合误差
        pred = X @ coef
        rmse = np.sqrt(np.mean((pred - deltas) ** 2, axis=0))
        print(f"    Linear prior RMSE per state: {rmse}")
        print(f"    Linear prior RMSE / delta_std: {rmse / delta_std}")

    def predict_delta(self, s, tau):
        """返回物理空间的 delta。"""
        x = np.concatenate([s, [tau]])
        return x @ self._coef

    def predict(self, s, tau):
        return s + self.predict_delta(s, tau)


# ============================================================
# GP 残差学习器
# ============================================================
class GPResidual:
    """GP 学习物理先验的残差：delta_residual = delta_real - delta_physics。"""

    def __init__(self, physics_prior, max_samples=2000):
        self._physics = physics_prior
        self._max_samples = max_samples
        self._gps = None
        self._x_mean = None
        self._x_std = None
        self._state_std = None
        self._action_std = None
        self._delta_std = None

    def train(self, states, actions, deltas, state_std, action_std, delta_std):
        from sklearn.exceptions import ConvergenceWarning
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern

        self._state_std = state_std
        self._action_std = action_std
        self._delta_std = delta_std

        # 计算物理先验的 delta
        physics_deltas = np.array([
            self._physics.predict_delta(states[i], actions[i])
            for i in range(len(states))
        ])
        # 残差 = 真实 - 物理
        residuals = deltas - physics_deltas

        # 下采样
        n = len(states)
        if n > self._max_samples:
            rng = np.random.RandomState(SEED)
            idx = rng.choice(n, self._max_samples, replace=False)
        else:
            idx = np.arange(n)

        # GP 输入：归一化的 [s, a]
        X = np.column_stack([
            states[idx] / state_std,
            actions[idx].reshape(-1, 1) / action_std,
        ])
        # GP 输出：归一化的残差
        Y = residuals[idx] / delta_std

        self._x_mean = X.mean(axis=0)
        self._x_std = X.std(axis=0) + 1e-8
        X_scaled = (X - self._x_mean) / self._x_std

        self._gps = []
        kernel = Matern(nu=2.5, length_scale=1.0)
        for col in range(STATE_DIM):
            t0 = time.time()
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1, alpha=1e-3)
            gp.fit(X_scaled, Y[:, col])
            self._gps.append(gp)
            print(f"    GP residual dim {col} ({STATE_NAMES_7D[col]}) done ({time.time()-t0:.1f}s)")

    def predict(self, s, tau):
        # 物理先验预测
        s_next_physics = self._physics.predict(s, tau)

        # GP 残差预测
        s_norm = s / self._state_std
        a_norm = tau / self._action_std
        x = np.concatenate([s_norm, [a_norm]]).reshape(1, -1)
        x_scaled = (x - self._x_mean) / self._x_std
        residual_norm = np.array([gp.predict(x_scaled)[0] for gp in self._gps])
        residual = residual_norm * self._delta_std

        return s_next_physics + residual

    def predict_with_residual_info(self, s, tau):
        """返回详细信息用于诊断。"""
        delta_physics = self._physics.predict_delta(s, tau)
        s_next_physics = s + delta_physics

        s_norm = s / self._state_std
        a_norm = tau / self._action_std
        x = np.concatenate([s_norm, [a_norm]]).reshape(1, -1)
        x_scaled = (x - self._x_mean) / self._x_std
        residual_norm = np.array([gp.predict(x_scaled)[0] for gp in self._gps])
        residual = residual_norm * self._delta_std

        return s_next_physics + residual, delta_physics, residual


# ============================================================
# Neural ODE (复现 V9 最佳配置)
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

    def train_from_data(self, train_s, train_a, train_d, n_epochs=100, lr=1e-3):
        from torch.utils.data import DataLoader, TensorDataset
        import torch.optim as optim

        s_norm = train_s / self._state_std
        a_norm = train_a / self._action_std
        derivs = train_d / (self._delta_std * DT)

        train_x = torch.FloatTensor(np.column_stack([s_norm, a_norm.reshape(-1, 1)])).to(DEVICE)
        train_y = torch.FloatTensor(derivs).to(DEVICE)

        self._model = ODEFunc(hidden=64, depth=3, activation='tanh').to(DEVICE)
        opt = optim.Adam(self._model.parameters(), lr=lr)
        crit = nn.MSELoss()
        ds = TensorDataset(train_x, train_y)
        loader = DataLoader(ds, batch_size=256, shuffle=True)

        self._model.train()
        for epoch in range(n_epochs):
            total_loss = 0
            for xb, yb in loader:
                pred = self._model(xb[:, :STATE_DIM], xb[:, STATE_DIM:STATE_DIM+1])
                loss = crit(pred, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item()
            if epoch % 20 == 0:
                print(f"    Neural ODE epoch {epoch}: loss={total_loss/len(loader):.6f}")
        self._model.eval()

    def predict(self, s, tau):
        s_norm = torch.FloatTensor(s / self._state_std).unsqueeze(0).to(DEVICE)
        a_norm = torch.FloatTensor([tau / self._action_std]).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            dsdt_norm = self._model(s_norm, a_norm).cpu().numpy()[0]
        dsdt = dsdt_norm * self._delta_std * DT
        return s + dsdt


# ============================================================
# 方案 3: 混合模型 (短期 NODE + 长期 GP)
# ============================================================
class HybridModel:
    """短期用 Neural ODE，长期用 GP+Physics，渐变过渡。"""

    def __init__(self, neural_ode, gp_residual, blend_start=20, blend_end=50):
        self._node = neural_ode
        self._gp = gp_residual
        self._blend_start = blend_start
        self._blend_end = blend_end
        self._step = 0

    def reset(self):
        self._step = 0

    def predict(self, s, tau):
        s_next_node = self._node.predict(s, tau)
        s_next_gp = self._gp.predict(s, tau)

        if self._step < self._blend_start:
            result = s_next_node
        elif self._step < self._blend_end:
            alpha = (self._step - self._blend_start) / (self._blend_end - self._blend_start)
            result = (1 - alpha) * s_next_node + alpha * s_next_gp
        else:
            result = s_next_gp

        self._step += 1
        return result


# ============================================================
# 训练和评估
# ============================================================
def main():
    print("=" * 80)
    print("改进 GP 方案：物理先验残差 + 混合模型")
    print("=" * 80)
    print(f"Device: {DEVICE}\n")

    # 加载数据
    print("Loading data...")
    t0 = time.time()
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
    print(f"  Train: {len(train_s)}, Test: {len(data['test_states'])}")
    print(f"  Loaded in {time.time()-t0:.1f}s\n")

    # 获取测试段
    segments = get_test_segments(data, n_segments=N_EVAL_SEGMENTS,
                                 segment_length=SEGMENT_LENGTH, seed=SEED)

    # ============================================================
    # 方案 1: GP + 物理先验残差
    # ============================================================
    print("=" * 80)
    print("方案 1: GP + 物理先验残差")
    print("=" * 80)

    # 1a: 训练物理先验
    print("\n[1a] Training linear physics prior...")
    physics = LinearPhysicsPrior()
    physics.train(train_s, train_a, train_d, state_std, action_std, delta_std)

    # 1b: 训练 GP 残差
    print("\n[1b] Training GP residual on physics prior residuals...")
    t0 = time.time()
    gp_res = GPResidual(physics, max_samples=2000)
    gp_res.train(train_s, train_a, train_d, state_std, action_std, delta_std)
    print(f"  GP residual training done in {time.time()-t0:.1f}s")

    # 1c: 评估
    print("\n[1c] Evaluating GP + Physics Prior...")
    results_gp_physics = multi_step_evaluate(gp_res, segments, state_std, ROLLOUT_HORIZONS)

    # ============================================================
    # 方案 3: 混合模型 (短期 NODE + 长期 GP)
    # ============================================================
    print("\n" + "=" * 80)
    print("方案 3: 混合模型 (短期 Neural ODE + 长期 GP+Physics)")
    print("=" * 80)

    # 3a: 训练 Neural ODE
    print("\n[3a] Training Neural ODE (V9 config)...")
    t0 = time.time()
    node = NeuralODE7D(state_std, action_std, delta_std)
    node.train_from_data(train_s, train_a, train_d, n_epochs=100, lr=1e-3)
    print(f"  Neural ODE training done in {time.time()-t0:.1f}s")

    # 3b: 评估纯 Neural ODE
    print("\n[3b] Evaluating pure Neural ODE...")
    results_node = multi_step_evaluate(node, segments, state_std, ROLLOUT_HORIZONS)

    # 3c: 评估混合模型 (不同过渡参数)
    blend_configs = [
        (10, 30, "Hybrid(10-30)"),
        (20, 50, "Hybrid(20-50)"),
        (20, 100, "Hybrid(20-100)"),
        (50, 200, "Hybrid(50-200)"),
    ]

    hybrid_results = {}
    for blend_start, blend_end, name in blend_configs:
        print(f"\n[3c] Evaluating {name}...")
        hybrid = HybridModel(node, gp_res, blend_start=blend_start, blend_end=blend_end)
        # 需要每个 segment 评估前 reset
        results_h = multi_step_evaluate_with_reset(hybrid, segments, state_std, ROLLOUT_HORIZONS)
        hybrid_results[name] = results_h

    # ============================================================
    # 输出对比表
    # ============================================================
    print("\n" + "=" * 100)
    print("综合对比结果")
    print("=" * 100)

    v9_baseline = {
        1: 0.0051, 5: 0.0170, 10: 0.0628, 20: 0.1817,
        50: 0.4557, 100: 0.5064, 200: 0.4737, 500: 0.5529, 1000: 0.6443,
    }

    # 表头
    header = f"{'H':>5}"
    methods = ['V9 NODE', 'NODE(retrained)', 'GP+Physics', 'Hybrid(20-50)']
    for m in methods:
        header += f" | {m:>14}"
    print(header)
    print("-" * (5 + 17 * len(methods)))

    for h in ROLLOUT_HORIZONS:
        row = f"{h:>5}"
        row += f" | {v9_baseline.get(h, float('nan')):14.4f}"
        row += f" | {results_node[h]['nmae_mean']:14.4f}"
        row += f" | {results_gp_physics[h]['nmae_mean']:14.4f}"
        row += f" | {hybrid_results['Hybrid(20-50)'][h]['nmae_mean']:14.4f}"
        print(row)

    # 存活率
    print("\n存活率:")
    for h in ROLLOUT_HORIZONS:
        row = f"{h:>5}"
        row += f" | {'100%':>14}"
        row += f" | {results_node[h]['survival_rate']:13.0%}%"
        row += f" | {results_gp_physics[h]['survival_rate']:13.0%}%"
        row += f" | {hybrid_results['Hybrid(20-50)'][h]['survival_rate']:13.0%}%"
        print(row)

    # 混合模型对比
    print("\n混合模型不同过渡参数对比:")
    print(f"{'H':>5}", end="")
    for _, _, name in blend_configs:
        print(f" | {name:>14}", end="")
    print()
    print("-" * (5 + 17 * len(blend_configs)))
    for h in ROLLOUT_HORIZONS:
        print(f"{h:>5}", end="")
        for _, _, name in blend_configs:
            print(f" | {hybrid_results[name][h]['nmae_mean']:14.4f}", end="")
        print()

    # 保存结果
    log_dir = Path('D:/系统辨识作业/sindy_bicycle/continuation_stage_v9/logs')
    log_dir.mkdir(exist_ok=True)
    out_path = log_dir / 'improved_gp_results.json'

    all_results = {}
    for h in ROLLOUT_HORIZONS:
        all_results[str(h)] = {
            'v9_node': v9_baseline.get(h, None),
            'node_retrained': results_node[h]['nmae_mean'],
            'gp_physics': results_gp_physics[h]['nmae_mean'],
            'gp_physics_survival': results_gp_physics[h]['survival_rate'],
        }
        for _, _, name in blend_configs:
            all_results[str(h)][name] = hybrid_results[name][h]['nmae_mean']

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            'methods': {
                'gp_physics': 'GP + Linear Physics Prior Residual',
                'node': 'Neural ODE (V9 config, retrained)',
                'hybrid': 'Short-term NODE + Long-term GP+Physics',
            },
            'results': all_results,
        }, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {out_path}")
    print("Done.")


def multi_step_evaluate_with_reset(model, segments, state_std, horizons,
                                    survival_mode='physical'):
    """评估前 reset step 计数器的评估函数。"""
    from canonical_node.evaluation_v9 import compute_nmae, check_survival, compute_survival

    results = {}
    for h in horizons:
        nmae_list = []
        survival_list = []

        for seg in segments:
            model.reset()  # 重置 step 计数器
            s0 = seg['states'][0].copy()
            actions_seg = seg['actions']
            real_states = seg['states']
            n = min(h, len(actions_seg))
            predicted = [s0.copy()]
            s_cur = s0.copy()
            survived = True

            for step in range(n):
                try:
                    s_next = model.predict(s_cur, actions_seg[step])
                    if np.any(np.isnan(s_next)) or np.any(np.isinf(s_next)):
                        survived = False
                        break
                    if not check_survival(s_next, mode=survival_mode):
                        survived = False
                        break
                    predicted.append(s_next.copy())
                    s_cur = s_next
                except Exception:
                    survived = False
                    break

            predicted = np.array(predicted)
            real = real_states[:len(predicted)]
            n_valid = min(len(predicted), len(real)) - 1
            if n_valid > 0:
                nmae = compute_nmae(predicted[1:n_valid+1], real[1:n_valid+1], state_std)
                nmae_list.append(nmae['overall'])
            else:
                nmae_list.append(float('nan'))
            survival_list.append(survived and n_valid >= n - 1)

        valid_nmae = [x for x in nmae_list if not np.isnan(x)]
        results[h] = {
            'nmae_mean': float(np.nanmean(valid_nmae)) if valid_nmae else float('nan'),
            'nmae_std': float(np.nanstd(valid_nmae)) if valid_nmae else float('nan'),
            'survival_rate': compute_survival(survival_list),
        }

    return results


if __name__ == '__main__':
    main()
