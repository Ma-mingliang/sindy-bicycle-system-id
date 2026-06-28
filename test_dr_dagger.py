"""论文方案2增强版：Domain Rand + OOD + DAgger 三者结合。

当前最佳：NeuralODE + OOD + DAgger + 残差缩放0.3 = 0.17 rad
测试：NeuralODE + DR(1.0) + OOD + DAgger + 残差缩放0.3
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


def train_with_noise(model, train_x, train_y, n_epochs=200, noise_std=1.0):
    ds = TensorDataset(train_x, train_y)
    loader = DataLoader(ds, batch_size=256, shuffle=True)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.MSELoss()
    model.train()
    for _ in range(n_epochs):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            xb_noisy = xb + torch.randn_like(xb) * noise_std
            loss = crit(model(xb_noisy), yb)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()


def dagger_augment(baseline, nn_model, state_std, action_std, delta_std, ood_det, residual_scale=0.3):
    """DAgger: rollout收集新数据。"""
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
            inp = np.concatenate([s_norm, [a_norm]])
            with torch.no_grad():
                delta_nn = nn_model(
                    torch.FloatTensor(s_norm).unsqueeze(0).to(DEVICE),
                    torch.FloatTensor([a_norm]).unsqueeze(0).to(DEVICE)
                ).cpu().numpy()[0]
            if ood_det.is_ood(inp):
                delta_nn = np.zeros(4)
            s_prev = s_base.copy()
            s_base = baseline.predict(s_base, tau) + delta_nn * delta_std * residual_scale
            if abs(s_real[0]) > math.pi / 3:
                break
            s_next_base = baseline.predict(s_prev, tau)
            real_res = (s_real - s_next_base) / delta_std
            extra_x.append(inp)
            extra_y.append(real_res - (s_next_base - s_prev) / delta_std)
    return extra_x, extra_y


def run_rollout(baseline, nn_model, state_std, action_std, delta_std, ood_det, residual_scale=0.3):
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
            if ood_det.is_ood(inp):
                delta_nn = np.zeros(4)
            else:
                with torch.no_grad():
                    delta_nn = nn_model(
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
    print("Domain Rand + OOD + DAgger 三合一")
    print("=" * 60)

    states, actions, deltas, state_std, action_std, delta_std = generate_training_data(30000)
    eval_steps = [1, 5, 10, 20, 50, 100, 200, 500]

    # 基线
    print("\n训练NeuralODE基线...")
    baseline = NeuralODEMethod()
    baseline.train(states, actions, deltas, state_std, action_std, delta_std)

    # 残差
    residuals = np.empty_like(deltas)
    for i in range(len(states)):
        s_next = baseline.predict(states[i], actions[i])
        residuals[i] = (deltas[i] - (s_next - states[i])) / delta_std
    train_x_np = np.column_stack([states / state_std, actions.reshape(-1, 1) / action_std])
    base_x = torch.FloatTensor(train_x_np).to(DEVICE)
    base_y = torch.FloatTensor(residuals).to(DEVICE)

    # === 方法A：标准 + OOD + DAgger（当前最佳）===
    print("\n训练A: 标准 + OOD + DAgger...")
    all_x_a = [train_x_np.copy()]
    all_y_a = [residuals.copy()]
    nn_a = None
    ood_a = None
    for round_i in range(3):
        tx = torch.FloatTensor(np.vstack(all_x_a)).to(DEVICE)
        ty = torch.FloatTensor(np.vstack(all_y_a)).to(DEVICE)
        nn_a = ResidualNet().to(DEVICE)
        ds = TensorDataset(tx, ty)
        loader = DataLoader(ds, batch_size=256, shuffle=True)
        opt = optim.Adam(nn_a.parameters(), lr=1e-3)
        crit = nn.MSELoss()
        nn_a.train()
        for _ in range(200):
            for xb, yb in loader:
                loss = crit(nn_a(xb.to(DEVICE)), yb.to(DEVICE))
                opt.zero_grad(); loss.backward(); opt.step()
        nn_a.eval()
        ood_a = OODDetector(np.vstack(all_x_a), threshold=3.0)
        if round_i < 2:
            ex, ey = dagger_augment(baseline, nn_a, state_std, action_std, delta_std, ood_a)
            if ex:
                all_x_a.append(np.array(ex))
                all_y_a.append(np.array(ey))
    err_a = run_rollout(baseline, nn_a, state_std, action_std, delta_std, ood_a)

    # === 方法B：DR(1.0) + OOD + DAgger ===
    print("\n训练B: DR(1.0) + OOD + DAgger...")
    all_x_b = [train_x_np.copy()]
    all_y_b = [residuals.copy()]
    nn_b = None
    ood_b = None
    for round_i in range(3):
        tx = torch.FloatTensor(np.vstack(all_x_b)).to(DEVICE)
        ty = torch.FloatTensor(np.vstack(all_y_b)).to(DEVICE)
        nn_b = ResidualNet().to(DEVICE)
        train_with_noise(nn_b, tx, ty, n_epochs=200, noise_std=1.0)
        ood_b = OODDetector(np.vstack(all_x_b), threshold=3.0)
        if round_i < 2:
            ex, ey = dagger_augment(baseline, nn_b, state_std, action_std, delta_std, ood_b)
            if ex:
                all_x_b.append(np.array(ex))
                all_y_b.append(np.array(ey))
    err_b = run_rollout(baseline, nn_b, state_std, action_std, delta_std, ood_b)

    # === 方法C：DR(0.5) + OOD + DAgger ===
    print("\n训练C: DR(0.5) + OOD + DAgger...")
    all_x_c = [train_x_np.copy()]
    all_y_c = [residuals.copy()]
    nn_c = None
    ood_c = None
    for round_i in range(3):
        tx = torch.FloatTensor(np.vstack(all_x_c)).to(DEVICE)
        ty = torch.FloatTensor(np.vstack(all_y_c)).to(DEVICE)
        nn_c = ResidualNet().to(DEVICE)
        train_with_noise(nn_c, tx, ty, n_epochs=200, noise_std=0.5)
        ood_c = OODDetector(np.vstack(all_x_c), threshold=3.0)
        if round_i < 2:
            ex, ey = dagger_augment(baseline, nn_c, state_std, action_std, delta_std, ood_c)
            if ex:
                all_x_c.append(np.array(ex))
                all_y_c.append(np.array(ey))
    err_c = run_rollout(baseline, nn_c, state_std, action_std, delta_std, ood_c)

    # 纯基线
    print("\n运行纯NeuralODE基线...")
    base_err = {s: [] for s in eval_steps}
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
                base_err[step + 1].append(abs(s_base[0] - s_real[0]))

    # 输出
    print(f"\n{'='*70}")
    print("三合一方案对比 (MAE, rad)")
    print(f"{'='*70}")
    print(f"  {'步数':>6} | {'纯ODE':>10} | {'标准+OOD+D':>12} | {'DR1.0+OOD+D':>12} | {'DR0.5+OOD+D':>12}")
    print(f"  {'-'*65}")
    for step in eval_steps:
        b = np.mean(base_err[step]) if base_err[step] else float('nan')
        a = np.mean(err_a[step]) if err_a[step] else float('nan')
        b500 = np.mean(err_b[step]) if err_b[step] else float('nan')
        c = np.mean(err_c[step]) if err_c[step] else float('nan')
        print(f"  {step:6d} | {b:10.5f} | {a:12.5f} | {b500:12.5f} | {c:12.5f}")

    print("\n完成。")
