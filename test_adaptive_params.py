"""自适应残差缩放参数优化。

测试不同参数组合：
- max_scale: [0.2, 0.3, 0.4, 0.5]
- k (sigmoid陡峭度): [2.0, 3.0, 4.0, 5.0]
- threshold (OOD阈值): [2.0, 3.0, 4.0]

找出最佳参数组合。
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from methods_common import generate_training_data, real_step, K_lqr, dt
from methods_nn import NeuralODEMethod, ResidualNet, DEVICE
from methods_classic import GPMethod, ParamIDMethod
from methods_evaluate import make_tau_func


# ============================================================
# 自适应OOD检测器
# ============================================================
class AdaptiveOODDetector:
    def __init__(self, train_inputs, threshold=3.0, max_scale=0.5, k=3.0):
        self.mean = np.mean(train_inputs, axis=0)
        self.std = np.std(train_inputs, axis=0) + 1e-8
        self.threshold = threshold
        self.max_scale = max_scale
        self.k = k

    def compute_scale(self, x):
        z = np.abs(x - self.mean) / self.std
        mahal_dist = np.max(z)
        normalized_dist = (mahal_dist - 1.0) / (self.threshold - 1.0)
        clipped_input = max(-10.0, min(10.0, self.k * normalized_dist * 4))
        sigmoid = 1.0 / (1.0 + math.exp(clipped_input))
        return sigmoid * self.max_scale

    def is_ood(self, x):
        z = np.abs(x - self.mean) / self.std
        return np.max(z) > self.threshold


# ============================================================
# 训练自适应混合方法
# ============================================================
def train_adaptive(baseline_cls, baseline_kwargs=None, n_epochs=100,
                   dagger_rounds=3, max_scale=0.5, ood_threshold=3.0, k=3.0):
    if baseline_kwargs is None:
        baseline_kwargs = {}

    states, actions, deltas, state_std, action_std, delta_std = generate_training_data(30000)
    baseline = baseline_cls(**baseline_kwargs)
    baseline.train(states, actions, deltas, state_std, action_std, delta_std)

    residuals = np.empty_like(deltas)
    for i in range(len(states)):
        s_next_base = baseline.predict(states[i], actions[i])
        residuals[i] = (deltas[i] - (s_next_base - states[i])) / delta_std

    train_inputs_base = np.column_stack([states / state_std, actions.reshape(-1, 1) / action_std])
    all_train_inputs = [train_inputs_base.copy()]
    all_train_residuals = [residuals.copy()]

    for round_i in range(dagger_rounds):
        train_inputs = np.vstack(all_train_inputs)
        train_residuals = np.vstack(all_train_residuals)

        nn_model = ResidualNet().to(DEVICE)
        train_x = torch.FloatTensor(train_inputs).to(DEVICE)
        train_y = torch.FloatTensor(train_residuals).to(DEVICE)
        ds = TensorDataset(train_x, train_y)
        loader = DataLoader(ds, batch_size=256, shuffle=True)
        opt = optim.Adam(nn_model.parameters(), lr=1e-3)
        crit = nn.MSELoss()
        nn_model.train()
        for _ in range(n_epochs):
            for xb, yb in loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                loss = crit(nn_model(xb), yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
        nn_model.eval()

        ood_det = AdaptiveOODDetector(train_inputs, threshold=ood_threshold, max_scale=max_scale, k=k)

        if round_i < dagger_rounds - 1:
            new_inputs = []
            new_residuals = []
            for seg_i in range(3):
                tau_func = make_tau_func(100 + seg_i, 500)
                phi_init = np.random.uniform(-0.25, 0.25)
                s0 = np.array([phi_init, 0.0, 0.0, 0.0])
                s_base = s0.copy()
                s_real = s0.copy()
                for step in range(500):
                    tau = tau_func(step, s_real)
                    s_real = real_step(s_real, tau)
                    s_norm = s_base / state_std
                    a_norm = tau / action_std
                    inp = np.concatenate([s_norm, [a_norm]])
                    with torch.no_grad():
                        s_t = torch.FloatTensor(s_norm).unsqueeze(0).to(DEVICE)
                        a_t = torch.FloatTensor([a_norm]).unsqueeze(0).to(DEVICE)
                        delta_nn = nn_model(s_t, a_t).cpu().numpy()[0]
                    adaptive_scale = ood_det.compute_scale(inp)
                    s_base = baseline.predict(s_base, tau) + delta_nn * delta_std * adaptive_scale
                    if abs(s_real[0]) > math.pi / 3:
                        break
                    new_inputs.append(inp)
                    s_next_base = baseline.predict(s_base - delta_nn * delta_std * adaptive_scale, tau)
                    base_delta = s_next_base - (s_base - delta_nn * delta_std * adaptive_scale)
                    real_delta = s_real - (s_base - delta_nn * delta_std * adaptive_scale)
                    new_residuals.append((real_delta - base_delta) / delta_std)

            if new_inputs:
                all_train_inputs.append(np.array(new_inputs))
                all_train_residuals.append(np.array(new_residuals))

    return baseline, nn_model, ood_det, (state_std, action_std, delta_std)


# ============================================================
# 混合预测
# ============================================================
def adaptive_predict(baseline, nn_model, ood_detector, s, tau,
                     state_std, action_std, delta_std):
    s_next_base = baseline.predict(s, tau)
    s_norm = s / state_std
    a_norm = tau / action_std
    inp = np.concatenate([s_norm, [a_norm]])
    adaptive_scale = ood_detector.compute_scale(inp)
    if adaptive_scale < 0.01:
        return s_next_base
    with torch.no_grad():
        s_t = torch.FloatTensor(s_norm).unsqueeze(0).to(DEVICE)
        a_t = torch.FloatTensor([a_norm]).unsqueeze(0).to(DEVICE)
        delta_nn = nn_model(s_t, a_t).cpu().numpy()[0] * delta_std * adaptive_scale
    return s_next_base + delta_nn


# ============================================================
# 评估
# ============================================================
def evaluate(baseline, nn_model, ood_det, norms):
    state_std, action_std, delta_std = norms
    eval_steps = [500]
    n_segments = 5
    all_errors = {step: [] for step in eval_steps}

    for seg_i in range(n_segments):
        tau_func = make_tau_func(seg_i, 500)
        phi_init = np.random.uniform(-0.25, 0.25)
        s0 = np.array([phi_init, 0.0, 0.0, 0.0])
        s_real = s0.copy()
        s_hybrid = s0.copy()

        for step in range(500):
            tau = tau_func(step, s_real)
            s_real = real_step(s_real, tau)
            s_hybrid = adaptive_predict(
                baseline, nn_model, ood_det, s_hybrid, tau,
                state_std, action_std, delta_std
            )
            if abs(s_real[0]) > math.pi / 3:
                break
            if step + 1 in eval_steps:
                all_errors[step + 1].append(abs(s_hybrid[0] - s_real[0]))

    return np.mean(all_errors[500]) if all_errors[500] else float('nan')


# ============================================================
# 主测试
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("自适应残差缩放参数优化")
    print("=" * 60)

    # 参数网格
    max_scales = [0.2, 0.3, 0.4, 0.5]
    ks = [2.0, 3.0, 4.0, 5.0]
    thresholds = [2.0, 3.0, 4.0]

    results = []

    for max_scale in max_scales:
        for k in ks:
            for threshold in thresholds:
                print(f"\n测试: max_scale={max_scale}, k={k}, threshold={threshold}")
                baseline, nn_model, ood_det, norms = train_adaptive(
                    NeuralODEMethod,
                    max_scale=max_scale, ood_threshold=threshold, k=k
                )
                mae = evaluate(baseline, nn_model, ood_det, norms)
                results.append({
                    'max_scale': max_scale,
                    'k': k,
                    'threshold': threshold,
                    'mae': mae
                })
                print(f"  500步MAE: {mae:.5f}")

    # 排序结果
    results.sort(key=lambda x: x['mae'])

    print(f"\n{'='*60}")
    print("参数优化结果（按500步MAE排序）")
    print(f"{'='*60}")
    print(f"\n  {'max_scale':>10} | {'k':>5} | {'threshold':>10} | {'500步MAE':>15}")
    print(f"  {'-'*50}")
    for r in results[:10]:  # 显示前10个
        print(f"  {r['max_scale']:10.1f} | {r['k']:5.1f} | {r['threshold']:10.1f} | {r['mae']:15.5f}")

    print(f"\n最佳参数:")
    best = results[0]
    print(f"  max_scale={best['max_scale']}, k={best['k']}, threshold={best['threshold']}")
    print(f"  500步MAE: {best['mae']:.5f}")

    print("\n完成。")
