"""论文方案2：Domain Randomization。

训练NN残差时，在输入上加噪声扰动，让NN对分布偏移更鲁棒。
对比：
- 纯NeuralODE基线
- 单NN+OOD（当前最佳）
- Domain Rand NN（输入加高斯噪声训练）
- Domain Rand + OOD（噪声训练 + OOD检测）
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


class OODDetector:
    def __init__(self, train_inputs, threshold=3.0):
        self.mean = np.mean(train_inputs, axis=0)
        self.std = np.std(train_inputs, axis=0) + 1e-8
        self.threshold = threshold

    def is_ood(self, x):
        return np.max(np.abs(x - self.mean) / self.std) > self.threshold


def train_with_noise(model, train_x, train_y, n_epochs=200, noise_std=0.5):
    """训练时在输入上加高斯噪声。"""
    ds = TensorDataset(train_x, train_y)
    loader = DataLoader(ds, batch_size=256, shuffle=True)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.MSELoss()
    model.train()
    for _ in range(n_epochs):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            # 加噪声
            noise = torch.randn_like(xb) * noise_std
            xb_noisy = xb + noise
            loss = crit(model(xb_noisy), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()


def run_test(name, baseline, model, state_std, action_std, delta_std,
             use_ood=False, ood_det=None, residual_scale=0.3):
    """运行5段rollout。"""
    eval_steps = [1, 5, 10, 20, 50, 100, 200, 500]
    errors = {s: [] for s in eval_steps}

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

            delta_nn = np.zeros(4)
            if use_ood and ood_det is not None and ood_det.is_ood(inp):
                pass  # OOD, 用零残差
            else:
                with torch.no_grad():
                    delta_nn = model(
                        torch.FloatTensor(s_norm).unsqueeze(0).to(DEVICE),
                        torch.FloatTensor([a_norm]).unsqueeze(0).to(DEVICE)
                    ).cpu().numpy()[0]

            s_pred = baseline.predict(s_pred, tau) + delta_nn * delta_std * residual_scale

            if abs(s_real[0]) > math.pi / 3:
                break
            if step + 1 in eval_steps:
                errors[step + 1].append(abs(s_pred[0] - s_real[0]))

    return errors


if __name__ == '__main__':
    print("=" * 60)
    print("论文方案2：Domain Randomization")
    print("=" * 60)

    states, actions, deltas, state_std, action_std, delta_std = generate_training_data(30000)

    # 基线
    print("\n训练NeuralODE基线...")
    baseline = NeuralODEMethod()
    baseline.train(states, actions, deltas, state_std, action_std, delta_std)

    # 准备数据
    residuals = np.empty_like(deltas)
    for i in range(len(states)):
        s_next = baseline.predict(states[i], actions[i])
        residuals[i] = (deltas[i] - (s_next - states[i])) / delta_std
    train_x_np = np.column_stack([states / state_std, actions.reshape(-1, 1) / action_std])
    train_x = torch.FloatTensor(train_x_np).to(DEVICE)
    train_y = torch.FloatTensor(residuals).to(DEVICE)

    ood_det = OODDetector(train_x_np, threshold=3.0)

    # === 方法A：标准训练 + OOD（当前最佳）===
    print("\n训练A: 标准NN + OOD...")
    nn_a = ResidualNet().to(DEVICE)
    t0 = time.time()
    ds = TensorDataset(train_x, train_y)
    loader = DataLoader(ds, batch_size=256, shuffle=True)
    opt = optim.Adam(nn_a.parameters(), lr=1e-3)
    crit = nn.MSELoss()
    nn_a.train()
    for _ in range(200):
        for xb, yb in loader:
            loss = crit(nn_a(xb.to(DEVICE)), yb.to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()
    nn_a.eval()
    print(f"  耗时: {time.time()-t0:.1f}s")

    # === 方法B：Domain Rand（不同噪声级别）===
    results_a = run_test("标准+OOD", baseline, nn_a, state_std, action_std, delta_std, use_ood=True, ood_det=ood_det)

    all_results = {"标准+OOD": results_a}

    for noise_std in [0.1, 0.3, 0.5, 1.0]:
        print(f"\n训练B: Domain Rand (noise={noise_std})...")
        nn_b = ResidualNet().to(DEVICE)
        t0 = time.time()
        train_with_noise(nn_b, train_x, train_y, n_epochs=200, noise_std=noise_std)
        print(f"  耗时: {time.time()-t0:.1f}s")

        # 不带OOD
        err_no_ood = run_test(f"DR(noise={noise_std})", baseline, nn_b, state_std, action_std, delta_std)
        all_results[f"DR({noise_std})"] = err_no_ood

        # 带OOD
        err_ood = run_test(f"DR({noise_std})+OOD", baseline, nn_b, state_std, action_std, delta_std,
                           use_ood=True, ood_det=ood_det)
        all_results[f"DR({noise_std})+OOD"] = err_ood

    # 纯基线
    eval_steps = [1, 5, 10, 20, 50, 100, 200, 500]
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
    all_results["纯NeuralODE"] = base_errors

    # 输出对比
    print(f"\n{'='*70}")
    print("500步MAE对比 (rad)")
    print(f"{'='*70}")
    print(f"  {'方法':<22} | {'50步':>8} | {'100步':>8} | {'200步':>8} | {'500步':>8}")
    print(f"  {'-'*65}")
    for name in all_results:
        err = all_results[name]
        e50 = np.mean(err[50]) if err[50] else float('nan')
        e100 = np.mean(err[100]) if err[100] else float('nan')
        e200 = np.mean(err[200]) if err[200] else float('nan')
        e500 = np.mean(err[500]) if err[500] else float('nan')
        print(f"  {name:<22} | {e50:8.4f} | {e100:8.4f} | {e200:8.4f} | {e500:8.4f}")

    print("\n完成。")
