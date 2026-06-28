"""组合改进：自适应残差缩放 + 多步损失训练。

将第一轮和第三轮的最佳改进组合：
- 自适应残差缩放（第一轮，GP基线改善46%）
- 多步损失训练（第三轮，NeuralODE基线改善80%）

组合方式：
- 训练时使用多步损失
- 推理时使用自适应残差缩放
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
    def __init__(self, train_inputs, threshold=3.0, max_scale=0.5):
        self.mean = np.mean(train_inputs, axis=0)
        self.std = np.std(train_inputs, axis=0) + 1e-8
        self.threshold = threshold
        self.max_scale = max_scale

    def compute_scale(self, x):
        z = np.abs(x - self.mean) / self.std
        mahal_dist = np.max(z)
        k = 3.0
        normalized_dist = (mahal_dist - 1.0) / (self.threshold - 1.0)
        clipped_input = max(-10.0, min(10.0, k * normalized_dist * 4))
        sigmoid = 1.0 / (1.0 + math.exp(clipped_input))
        return sigmoid * self.max_scale

    def is_ood(self, x):
        z = np.abs(x - self.mean) / self.std
        return np.max(z) > self.threshold


# ============================================================
# 组合训练：多步损失 + 自适应残差缩放
# ============================================================
def train_combined(baseline_cls, baseline_kwargs=None, n_epochs=100,
                   dagger_rounds=3, max_scale=0.5, ood_threshold=3.0,
                   multistep_length=10, multistep_weight=0.5):
    if baseline_kwargs is None:
        baseline_kwargs = {}

    states, actions, deltas, state_std, action_std, delta_std = generate_training_data(30000)
    baseline = baseline_cls(**baseline_kwargs)
    baseline.train(states, actions, deltas, state_std, action_std, delta_std)

    residuals = np.empty_like(deltas)
    for i in range(len(states)):
        s_next_base = baseline.predict(states[i], actions[i])
        residuals[i] = (deltas[i] - (s_next_base - states[i])) / delta_std
    residual_std = np.std(residuals, axis=0)
    print(f"    残差std: {residual_std}, 占比: {np.mean(residual_std):.4f}")

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

        ood_det = AdaptiveOODDetector(train_inputs, threshold=ood_threshold, max_scale=max_scale)

        nn_model.train()
        for epoch in range(n_epochs):
            for xb, yb in loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = nn_model(xb)
                single_loss = crit(pred, yb)

                multistep_loss = torch.tensor(0.0, device=DEVICE)
                n_multistep = 0
                for _ in range(2):
                    idx = np.random.randint(0, len(states))
                    s_start = states[idx]
                    tau_start = actions[idx]
                    s_current = s_start.copy()
                    rollout_loss = torch.tensor(0.0, device=DEVICE)
                    for step in range(multistep_length):
                        s_next_base = baseline.predict(s_current, tau_start)
                        s_norm = s_current / state_std
                        a_norm = tau_start / action_std
                        inp = np.concatenate([s_norm, [a_norm]])
                        if ood_det.is_ood(inp):
                            break
                        adaptive_scale = ood_det.compute_scale(inp)
                        with torch.no_grad():
                            s_t = torch.FloatTensor(s_norm).unsqueeze(0).to(DEVICE)
                            a_t = torch.FloatTensor([a_norm]).unsqueeze(0).to(DEVICE)
                            delta_nn = nn_model(s_t, a_t).cpu().numpy()[0]
                        s_next = s_next_base + delta_nn * delta_std * adaptive_scale
                        s_real = real_step(s_current, tau_start)
                        error = np.sum((s_next - s_real) ** 2)
                        rollout_loss += torch.tensor(error, device=DEVICE)
                        s_current = s_next
                        tau_start = -K_lqr @ s_current
                    multistep_loss += rollout_loss / multistep_length
                    n_multistep += 1

                if n_multistep > 0:
                    multistep_loss = multistep_loss / n_multistep
                total_loss = single_loss + multistep_weight * multistep_loss
                opt.zero_grad()
                total_loss.backward()
                opt.step()
        nn_model.eval()

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
                print(f"    DAgger轮{round_i+1}: 收集{len(new_inputs)}个新样本")

    return baseline, nn_model, ood_det, (state_std, action_std, delta_std)


# ============================================================
# 组合预测：自适应残差缩放
# ============================================================
def combined_predict(baseline, nn_model, ood_detector, s, tau,
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
# 测试
# ============================================================
def test_combined(name, baseline_cls, baseline_kwargs=None,
                  max_scale=0.5, multistep_length=10, multistep_weight=0.5):
    print(f"\n{'='*60}")
    print(f"测试: {name}")
    print(f"{'='*60}")

    baseline, nn_model, ood_det, (state_std, action_std, delta_std) = train_combined(
        baseline_cls, baseline_kwargs,
        n_epochs=100, dagger_rounds=3, max_scale=max_scale, ood_threshold=3.0,
        multistep_length=multistep_length, multistep_weight=multistep_weight
    )

    eval_steps = [1, 5, 10, 20, 50, 100, 200, 500]
    n_segments = 5
    all_errors = {step: [] for step in eval_steps}
    baseline_errors = {step: [] for step in eval_steps}

    for seg_i in range(n_segments):
        tau_func = make_tau_func(seg_i, 500)
        phi_init = np.random.uniform(-0.25, 0.25)
        s0 = np.array([phi_init, 0.0, 0.0, 0.0])
        s_real = s0.copy()
        s_base = s0.copy()
        s_hybrid = s0.copy()

        for step in range(500):
            tau = tau_func(step, s_real)
            s_real = real_step(s_real, tau)
            s_base = baseline.predict(s_base, tau)
            s_hybrid = combined_predict(
                baseline, nn_model, ood_det, s_hybrid, tau,
                state_std, action_std, delta_std
            )
            if abs(s_real[0]) > math.pi / 3:
                break
            if step + 1 in eval_steps:
                all_errors[step + 1].append(abs(s_hybrid[0] - s_real[0]))
                baseline_errors[step + 1].append(abs(s_base[0] - s_real[0]))

    print(f"\n  {'步数':>6} | {'纯基线':>10} | {'组合改进':>10} | {'改进倍率':>10}")
    print(f"  {'-'*50}")
    for step in eval_steps:
        b_err = np.mean(baseline_errors[step]) if baseline_errors[step] else float('nan')
        h_err = np.mean(all_errors[step]) if all_errors[step] else float('nan')
        ratio = h_err / b_err if b_err > 0 and not math.isnan(h_err) else float('nan')
        print(f"  {step:6d} | {b_err:10.5f} | {h_err:10.5f} | {ratio:10.2f}x")

    return baseline_errors, all_errors


if __name__ == '__main__':
    print("=" * 60)
    print("组合改进：自适应残差缩放 + 多步损失训练")
    print("=" * 60)
    print("训练时使用多步损失，推理时使用自适应残差缩放")
    print("=" * 60)

    # 测试NeuralODE + 组合改进
    test_combined(
        "NeuralODE + 组合改进(自适应+多步损失)", NeuralODEMethod,
        max_scale=0.5, multistep_length=10, multistep_weight=0.5
    )

    # 测试ParamID + 组合改进
    test_combined(
        "ParamID + 组合改进(自适应+多步损失)", ParamIDMethod,
        max_scale=0.5, multistep_length=10, multistep_weight=0.5
    )

    print("\n完成。")
