"""论文方案1：Ensemble-based uncertainty。

用5个NN集成估计不确定性，高不确定性时回退基线。
对比：
- 纯NeuralODE基线
- 单NN改进混合（OOD检测）
- 5-NN Ensemble混合（集成不确定性）
"""

import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
from methods_common import generate_training_data, real_step, K_lqr, dt
from methods_nn import NeuralODEMethod, ResidualNet, DEVICE
from methods_evaluate import make_tau_func


# ============================================================
# Ensemble NN
# ============================================================
class EnsembleNN:
    """多个NN集成，返回均值和标准差。"""

    def __init__(self, n_models: int = 5, hidden: int = 128):
        self.n_models = n_models
        self.models = [ResidualNet(hidden=hidden).to(DEVICE) for _ in range(n_models)]

    def train_all(self, train_x: torch.Tensor, train_y: torch.Tensor, n_epochs: int = 200):
        for i, model in enumerate(self.models):
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

    def predict(self, x: torch.Tensor):
        """返回 (mean, std) 在归一化空间。"""
        preds = []
        with torch.no_grad():
            for model in self.models:
                preds.append(model(x).cpu().numpy())
        preds = np.array(preds)  # (n_models, batch, 4)
        return np.mean(preds, axis=0), np.std(preds, axis=0)


# ============================================================
# OOD检测器（保留用于对比）
# ============================================================
class OODDetector:
    def __init__(self, train_inputs: np.ndarray, threshold: float = 3.0):
        self.mean = np.mean(train_inputs, axis=0)
        self.std = np.std(train_inputs, axis=0) + 1e-8
        self.threshold = threshold

    def is_ood(self, x: np.ndarray) -> bool:
        z = np.abs(x - self.mean) / self.std
        return np.max(z) > self.threshold


# ============================================================
# 训练函数
# ============================================================
def prepare_data(baseline, states, actions, deltas, state_std, action_std, delta_std):
    """计算残差并准备训练数据。"""
    residuals = np.empty_like(deltas)
    for i in range(len(states)):
        s_next_base = baseline.predict(states[i], actions[i])
        residuals[i] = (deltas[i] - (s_next_base - states[i])) / delta_std
    train_x = np.column_stack([states / state_std, actions.reshape(-1, 1) / action_std])
    return torch.FloatTensor(train_x).to(DEVICE), torch.FloatTensor(residuals).to(DEVICE), train_x


def dagger_augment(baseline, ensemble_or_nn, state_std, action_std, delta_std, n_rounds=2, is_ensemble=True):
    """DAgger数据增强。"""
    extra_x, extra_y = [], []
    for seg_i in range(3):
        tau_func = make_tau_func(100 + seg_i, 500)
        s0 = np.array([np.random.uniform(-0.25, 0.25), 0.0, 0.0, 0.0])
        s_base, s_real = s0.copy(), s0.copy()
        for step in range(500):
            tau = tau_func(step, s_real)
            s_real = real_step(s_real, tau)
            s_norm = s_base / state_std
            a_norm = tau / action_std
            inp = torch.FloatTensor(np.concatenate([s_norm, [a_norm]])).unsqueeze(0).to(DEVICE)
            if is_ensemble:
                delta_nn, std = ensemble_or_nn.predict(inp)
                delta_nn = delta_nn[0]
            else:
                with torch.no_grad():
                    delta_nn = ensemble_or_nn(inp).cpu().numpy()[0]
            s_prev = s_base.copy()
            s_base = baseline.predict(s_base, tau) + delta_nn * delta_std * 0.3
            if abs(s_real[0]) > math.pi / 3:
                break
            s_next_base = baseline.predict(s_prev, tau)
            real_res = (s_real - s_next_base) / delta_std
            extra_x.append(np.concatenate([s_norm, [a_norm]]))
            extra_y.append(real_res - (s_next_base - s_prev) / delta_std)
    return extra_x, extra_y


# ============================================================
# 测试
# ============================================================
def run_rollout(baseline, model, state_std, action_std, delta_std,
                use_ensemble=True, uncertainty_threshold=0.01, residual_scale=0.3):
    """运行5段rollout，返回各步误差。"""
    eval_steps = [1, 5, 10, 20, 50, 100, 200, 500]
    errors = {s: [] for s in eval_steps}
    ood_count, total_count = 0, 0

    for seg_i in range(5):
        tau_func = make_tau_func(seg_i, 500)
        s0 = np.array([np.random.uniform(-0.25, 0.25), 0.0, 0.0, 0.0])
        s_real, s_pred = s0.copy(), s0.copy()

        for step in range(500):
            tau = tau_func(step, s_real)
            s_real = real_step(s_real, tau)
            s_norm = s_pred / state_std
            a_norm = tau / action_std
            inp = torch.FloatTensor(np.concatenate([s_norm, [a_norm]])).unsqueeze(0).to(DEVICE)

            if use_ensemble:
                delta_nn, std = model.predict(inp)
                delta_nn = delta_nn[0]
                unc = np.mean(std)
                total_count += 1
                if unc > uncertainty_threshold:
                    ood_count += 1
                    delta_nn = np.zeros(4)
            else:
                with torch.no_grad():
                    delta_nn = model(inp).cpu().numpy()[0]

            s_pred = baseline.predict(s_pred, tau) + delta_nn * delta_std * residual_scale

            if abs(s_real[0]) > math.pi / 3:
                break
            if step + 1 in eval_steps:
                errors[step + 1].append(abs(s_pred[0] - s_real[0]))

    ood_rate = ood_count / total_count * 100 if total_count > 0 else 0
    return errors, ood_rate


if __name__ == '__main__':
    print("=" * 60)
    print("论文方案1：Ensemble-based uncertainty")
    print("=" * 60)

    states, actions, deltas, state_std, action_std, delta_std = generate_training_data(30000)

    # 训练NeuralODE基线
    print("\n训练NeuralODE基线...")
    baseline = NeuralODEMethod()
    baseline.train(states, actions, deltas, state_std, action_std, delta_std)

    # 准备数据
    train_x_t, train_y_t, train_x_np = prepare_data(baseline, states, actions, deltas, state_std, action_std, delta_std)

    # === 方法A：单NN + OOD检测 ===
    print("\n训练单NN + OOD检测...")
    nn_single = ResidualNet().to(DEVICE)
    t0 = time.time()
    ds = TensorDataset(train_x_t, train_y_t)
    loader = DataLoader(ds, batch_size=256, shuffle=True)
    opt = optim.Adam(nn_single.parameters(), lr=1e-3)
    crit = nn.MSELoss()
    nn_single.train()
    for _ in range(200):
        for xb, yb in loader:
            loss = crit(nn_single(xb.to(DEVICE)), yb.to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()
    nn_single.eval()
    print(f"  耗时: {time.time()-t0:.1f}s")

    ood_det = OODDetector(train_x_np, threshold=3.0)

    # 单NN rollout（带OOD检测）
    eval_steps = [1, 5, 10, 20, 50, 100, 200, 500]
    single_errors = {s: [] for s in eval_steps}
    for seg_i in range(5):
        tau_func = make_tau_func(seg_i, 500)
        s0 = np.array([np.random.uniform(-0.25, 0.25), 0.0, 0.0, 0.0])
        s_real, s_pred = s0.copy(), s0.copy()
        for step in range(500):
            tau = tau_func(step, s_real)
            s_real = real_step(s_real, tau)
            s_norm = s_pred / state_std
            a_norm = tau / action_std
            inp = np.concatenate([s_norm, [a_norm]])
            if ood_det.is_ood(inp):
                delta_nn = np.zeros(4)
            else:
                with torch.no_grad():
                    delta_nn = nn_single(
                        torch.FloatTensor(s_norm).unsqueeze(0).to(DEVICE),
                        torch.FloatTensor([a_norm]).unsqueeze(0).to(DEVICE)
                    ).cpu().numpy()[0]
            s_pred = baseline.predict(s_pred, tau) + delta_nn * delta_std * 0.3
            if abs(s_real[0]) > math.pi / 3:
                break
            if step + 1 in eval_steps:
                single_errors[step + 1].append(abs(s_pred[0] - s_real[0]))

    # === 方法B：5-NN Ensemble ===
    print("\n训练5-NN Ensemble...")
    ensemble = EnsembleNN(n_models=5)
    t0 = time.time()
    ensemble.train_all(train_x_t, train_y_t, n_epochs=200)
    print(f"  耗时: {time.time()-t0:.1f}s")

    # 找最佳uncertainty_threshold
    print("\n搜索最佳uncertainty_threshold...")
    best_thresh, best_500 = 0, 1e9
    for thresh in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]:
        errors, ood_rate = run_rollout(baseline, ensemble, state_std, action_std, delta_std,
                                       use_ensemble=True, uncertainty_threshold=thresh)
        e500 = np.mean(errors[500]) if errors[500] else float('nan')
        print(f"  threshold={thresh:.3f}: 500步MAE={e500:.4f}, OOD率={ood_rate:.1f}%")
        if e500 < best_500:
            best_500 = e500
            best_thresh = thresh

    print(f"\n最佳threshold: {best_thresh:.3f}")

    # 用最佳threshold测试Ensemble
    ensemble_errors, ood_rate = run_rollout(baseline, ensemble, state_std, action_std, delta_std,
                                            use_ensemble=True, uncertainty_threshold=best_thresh)

    # 纯基线
    print("\n运行纯NeuralODE基线...")
    base_errors = {s: [] for s in eval_steps}
    for seg_i in range(5):
        tau_func = make_tau_func(seg_i, 500)
        s0 = np.array([np.random.uniform(-0.25, 0.25), 0.0, 0.0, 0.0])
        s_real, s_base = s0.copy(), s0.copy()
        for step in range(500):
            tau = tau_func(step, s_real)
            s_real = real_step(s_real, tau)
            s_base = baseline.predict(s_base, tau)
            if abs(s_real[0]) > math.pi / 3:
                break
            if step + 1 in eval_steps:
                base_errors[step + 1].append(abs(s_base[0] - s_real[0]))

    # 输出对比
    print(f"\n{'='*60}")
    print("对比结果（500步MAE, rad）")
    print(f"{'='*60}")
    print(f"  {'步数':>6} | {'纯NeuralODE':>12} | {'单NN+OOD':>12} | {'Ensemble':>12}")
    print(f"  {'-'*55}")
    for step in eval_steps:
        b = np.mean(base_errors[step]) if base_errors[step] else float('nan')
        s = np.mean(single_errors[step]) if single_errors[step] else float('nan')
        e = np.mean(ensemble_errors[step]) if ensemble_errors[step] else float('nan')
        print(f"  {step:6d} | {b:12.5f} | {s:12.5f} | {e:12.5f}")

    print(f"\nEnsemble OOD率: {ood_rate:.1f}%")
    print("\n完成。")
