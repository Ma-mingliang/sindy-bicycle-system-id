"""改进 GP 方案 v2：用 V9 Neural ODE checkpoint 作为先验。

方案 1: GP + Neural ODE 先验残差
方案 3: 短期 NODE + 长期 GP 混合 (用 V9 checkpoint)

用法:
    python test_improved_gp_v2.py
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
# Neural ODE (V9 架构)
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
        """加载 V9 checkpoint。"""
        self._model = ODEFunc(hidden=64, depth=3, activation='tanh').to(DEVICE)
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        # V9 checkpoint 可能有不同结构
        if isinstance(ckpt, dict):
            if 'model_state' in ckpt:
                state_dict = ckpt['model_state']
            elif 'model_state_dict' in ckpt:
                state_dict = ckpt['model_state_dict']
            elif 'state_dict' in ckpt:
                state_dict = ckpt['state_dict']
            else:
                state_dict = ckpt
        else:
            state_dict = ckpt
        self._model.load_state_dict(state_dict)
        self._model.eval()
        print(f"    Loaded V9 checkpoint from {path}")

    def train_from_data(self, train_s, train_a, train_d, n_epochs=100, lr=1e-3):
        """从头训练（无课程学习，仅用于对比）。"""
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
                print(f"    NODE epoch {epoch}: loss={total_loss/len(loader):.6f}")
        self._model.eval()

    def predict(self, s, tau):
        s_norm = torch.FloatTensor(s / self._state_std).unsqueeze(0).to(DEVICE)
        a_norm = torch.FloatTensor([tau / self._action_std]).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            dsdt_norm = self._model(s_norm, a_norm).cpu().numpy()[0]
        dsdt = dsdt_norm * self._delta_std * DT
        return s + dsdt

    def predict_delta(self, s, tau):
        s_norm = torch.FloatTensor(s / self._state_std).unsqueeze(0).to(DEVICE)
        a_norm = torch.FloatTensor([tau / self._action_std]).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            dsdt_norm = self._model(s_norm, a_norm).cpu().numpy()[0]
        return dsdt_norm * self._delta_std * DT


# ============================================================
# 方案 1: GP + Neural ODE 先验残差
# ============================================================
class GPWithNODEPrior:
    """GP 学习 Neural ODE 的残差：residual = delta_real - delta_NODE。

    Neural ODE 提供动力学先验（比线性模型强得多），
    GP 只学残差（比原始 delta 小，更容易学）。
    """

    def __init__(self, node_prior, max_samples=2000):
        self._node = node_prior
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

        # 计算 Neural ODE 的 delta
        print("    Computing NODE deltas for training set...")
        node_deltas = np.array([
            self._node.predict_delta(states[i], actions[i])
            for i in range(len(states))
        ])
        residuals = deltas - node_deltas

        # 残差统计
        res_std = np.std(residuals, axis=0)
        orig_std = np.std(deltas, axis=0)
        print(f"    Residual std / delta_std: {res_std / delta_std}")
        print(f"    Residual reduction: {np.mean(1 - res_std/orig_std):.1%}")

        # 下采样
        n = len(states)
        if n > self._max_samples:
            rng = np.random.RandomState(SEED)
            idx = rng.choice(n, self._max_samples, replace=False)
        else:
            idx = np.arange(n)

        X = np.column_stack([
            states[idx] / state_std,
            actions[idx].reshape(-1, 1) / action_std,
        ])
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
        s_next_node = self._node.predict(s, tau)

        s_norm = s / self._state_std
        a_norm = tau / self._action_std
        x = np.concatenate([s_norm, [a_norm]]).reshape(1, -1)
        x_scaled = (x - self._x_mean) / self._x_std
        residual_norm = np.array([gp.predict(x_scaled)[0] for gp in self._gps])
        residual = residual_norm * self._delta_std

        return s_next_node + residual


# ============================================================
# 方案 3: 混合模型 (短期 NODE + 长期 GP)
# ============================================================
class HybridModel:
    """短期用 Neural ODE，长期用 GP+NODE先验，渐变过渡。"""

    def __init__(self, neural_ode, gp_model, blend_start=20, blend_end=50):
        self._node = neural_ode
        self._gp = gp_model
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
# 纯 GP 基线 (复现)
# ============================================================
class PureGP:
    """纯 GP，预测 delta ≈ 0 的基线。"""

    def __init__(self, max_samples=2000):
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

        n = len(states)
        if n > self._max_samples:
            rng = np.random.RandomState(SEED)
            idx = rng.choice(n, self._max_samples, replace=False)
        else:
            idx = np.arange(n)

        X = np.column_stack([states[idx] / state_std, actions[idx].reshape(-1, 1) / action_std])
        Y = deltas[idx] / delta_std

        self._x_mean = X.mean(axis=0)
        self._x_std = X.std(axis=0) + 1e-8
        X_scaled = (X - self._x_mean) / self._x_std

        self._gps = []
        kernel = Matern(nu=2.5, length_scale=1.0)
        for col in range(STATE_DIM):
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1, alpha=1e-3)
            gp.fit(X_scaled, Y[:, col])
            self._gps.append(gp)

    def predict(self, s, tau):
        s_norm = s / self._state_std
        a_norm = tau / self._action_std
        x = np.concatenate([s_norm, [a_norm]]).reshape(1, -1)
        x_scaled = (x - self._x_mean) / self._x_std
        delta_norm = np.array([gp.predict(x_scaled)[0] for gp in self._gps])
        return s + delta_norm * self._delta_std


# ============================================================
# 评估函数 (支持 reset)
# ============================================================
def multi_step_evaluate_with_reset(model, segments, state_std, horizons,
                                    survival_mode='physical'):
    from canonical_node.evaluation_v9 import compute_nmae, check_survival, compute_survival

    results = {}
    for h in horizons:
        nmae_list = []
        survival_list = []

        for seg in segments:
            if hasattr(model, 'reset'):
                model.reset()
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


# ============================================================
# 主函数
# ============================================================
def main():
    print("=" * 80)
    print("改进 GP v2: 用 V9 Neural ODE 作为先验")
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

    # ============================================================
    # 加载 V9 Neural ODE checkpoint
    # ============================================================
    print("[1] Loading V9 Neural ODE checkpoint...")
    v9_node = NeuralODE7D(state_std, action_std, delta_std)
    ckpt_path = Path('D:/系统辨识作业/sindy_bicycle/continuation_stage_v9/checkpoints/BEST_NEURAL_ODE_V9.pt')

    if ckpt_path.exists():
        v9_node.load_checkpoint(ckpt_path)
    else:
        print("    V9 checkpoint not found, training from scratch...")
        v9_node.train_from_data(train_s, train_a, train_d, n_epochs=100)

    # 评估 V9 NODE
    print("\n[2] Evaluating V9 Neural ODE baseline...")
    results_v9node = multi_step_evaluate(v9_node, segments, state_std, ROLLOUT_HORIZONS)

    # ============================================================
    # 方案 1: GP + Neural ODE 先验残差
    # ============================================================
    print("\n" + "=" * 80)
    print("方案 1: GP + Neural ODE 先验残差")
    print("=" * 80)

    print("\n[3] Training GP residual on NODE residuals...")
    t0 = time.time()
    gp_node = GPWithNODEPrior(v9_node, max_samples=2000)
    gp_node.train(train_s, train_a, train_d, state_std, action_std, delta_std)
    print(f"  Done in {time.time()-t0:.1f}s")

    print("\n[4] Evaluating GP + NODE Prior...")
    results_gp_node = multi_step_evaluate(gp_node, segments, state_std, ROLLOUT_HORIZONS)

    # ============================================================
    # 纯 GP 基线 (复现)
    # ============================================================
    print("\n[5] Training pure GP baseline...")
    t0 = time.time()
    pure_gp = PureGP(max_samples=2000)
    pure_gp.train(train_s, train_a, train_d, state_std, action_std, delta_std)
    print(f"  Done in {time.time()-t0:.1f}s")

    print("\n[6] Evaluating pure GP...")
    results_pure_gp = multi_step_evaluate(pure_gp, segments, state_std, ROLLOUT_HORIZONS)

    # ============================================================
    # 方案 3: 混合模型 (不同过渡参数)
    # ============================================================
    print("\n" + "=" * 80)
    print("方案 3: 混合模型 (短期 NODE + 长期 GP+NODE)")
    print("=" * 80)

    blend_configs = [
        (10, 30, "Hybrid(10-30)"),
        (20, 50, "Hybrid(20-50)"),
        (20, 100, "Hybrid(20-100)"),
        (50, 200, "Hybrid(50-200)"),
    ]

    hybrid_results = {}
    for blend_start, blend_end, name in blend_configs:
        print(f"\n[7] Evaluating {name}...")
        hybrid = HybridModel(v9_node, gp_node, blend_start=blend_start, blend_end=blend_end)
        results_h = multi_step_evaluate_with_reset(hybrid, segments, state_std, ROLLOUT_HORIZONS)
        hybrid_results[name] = results_h

    # ============================================================
    # 输出对比表
    # ============================================================
    print("\n" + "=" * 110)
    print("综合对比结果")
    print("=" * 110)

    v9_ref = {
        1: 0.0051, 5: 0.0170, 10: 0.0628, 20: 0.1817,
        50: 0.4557, 100: 0.5064, 200: 0.4737, 500: 0.5529, 1000: 0.6443,
    }

    methods = ['V9(ref)', 'NODE', 'PureGP', 'GP+NODE', 'Hybrid(20-50)', 'Hybrid(50-200)']
    header = f"{'H':>5}"
    for m in methods:
        header += f" | {m:>13}"
    print(header)
    print("-" * (5 + 16 * len(methods)))

    for h in ROLLOUT_HORIZONS:
        row = f"{h:>5}"
        row += f" | {v9_ref.get(h, float('nan')):13.4f}"
        row += f" | {results_v9node[h]['nmae_mean']:13.4f}"
        row += f" | {results_pure_gp[h]['nmae_mean']:13.4f}"
        row += f" | {results_gp_node[h]['nmae_mean']:13.4f}"
        row += f" | {hybrid_results['Hybrid(20-50)'][h]['nmae_mean']:13.4f}"
        row += f" | {hybrid_results['Hybrid(50-200)'][h]['nmae_mean']:13.4f}"
        print(row)

    # 存活率
    print("\n存活率:")
    for h in ROLLOUT_HORIZONS:
        row = f"{h:>5}"
        row += f" | {'100%':>13}"
        row += f" | {results_v9node[h]['survival_rate']:12.0%}%"
        row += f" | {results_pure_gp[h]['survival_rate']:12.0%}%"
        row += f" | {results_gp_node[h]['survival_rate']:12.0%}%"
        row += f" | {hybrid_results['Hybrid(20-50)'][h]['survival_rate']:12.0%}%"
        row += f" | {hybrid_results['Hybrid(50-200)'][h]['survival_rate']:12.0%}%"
        print(row)

    # 保存
    log_dir = Path('D:/系统辨识作业/sindy_bicycle/continuation_stage_v9/logs')
    log_dir.mkdir(exist_ok=True)
    out_path = log_dir / 'improved_gp_v2_results.json'

    all_results = {}
    for h in ROLLOUT_HORIZONS:
        all_results[str(h)] = {
            'v9_ref': v9_ref.get(h, None),
            'node': results_v9node[h]['nmae_mean'],
            'pure_gp': results_pure_gp[h]['nmae_mean'],
            'gp_node_prior': results_gp_node[h]['nmae_mean'],
            'hybrid_20_50': hybrid_results['Hybrid(20-50)'][h]['nmae_mean'],
            'hybrid_50_200': hybrid_results['Hybrid(50-200)'][h]['nmae_mean'],
        }

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({'results': all_results}, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {out_path}")
    print("Done.")


if __name__ == '__main__':
    main()
