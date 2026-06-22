"""域随机化改进：训练时使用扰动的物理参数。

思路：
- 训练时对状态添加噪声，模拟物理参数不确定性
- 提高残差网络对参数变化的鲁棒性
- 类似于数据增强，但针对物理系统

测试配置：
- 噪声水平：[0.01, 0.05, 0.1, 0.2]
- 基线：NeuralODE
- 对比自适应残差缩放
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
# 域随机化训练
# ============================================================
def train_with_domain_randomization(baseline_cls, baseline_kwargs=None,
                                     noise_level=0.05, n_epochs=100,
                                     dagger_rounds=3, residual_scale=0.3):
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

        # 添加域随机化噪声
        noisy_inputs = train_inputs.copy()
        for i in range(len(noisy_inputs)):
            # 对状态部分添加噪声
            state_noise = np.random.randn(4) * noise_level
            noisy_inputs[i, :4] += state_noise
            # 对动作部分添加较小噪声
            action_noise = np.random.randn(1) * noise_level * 0.1
            noisy_inputs[i, 4] += action_noise[0]

        # 合并原始和噪声数据
        augmented_inputs = np.vstack([train_inputs, noisy_inputs])
        augmented_residuals = np.vstack([train_residuals, train_residuals])

        nn_model = ResidualNet().to(DEVICE)
        train_x = torch.FloatTensor(augmented_inputs).to(DEVICE)
        train_y = torch.FloatTensor(augmented_residuals).to(DEVICE)
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
                    with torch.no_grad():
                        s_t = torch.FloatTensor(s_norm).unsqueeze(0).to(DEVICE)
                        a_t = torch.FloatTensor([a_norm]).unsqueeze(0).to(DEVICE)
                        delta_nn = nn_model(s_t, a_t).cpu().numpy()[0]
                    s_base = baseline.predict(s_base, tau) + delta_nn * delta_std * residual_scale
                    if abs(s_real[0]) > math.pi / 3:
                        break
                    new_inputs.append(np.concatenate([s_norm, [a_norm]]))
                    s_next_base = baseline.predict(s_base - delta_nn * delta_std * residual_scale, tau)
                    base_delta = s_next_base - (s_base - delta_nn * delta_std * residual_scale)
                    real_delta = s_real - (s_base - delta_nn * delta_std * residual_scale)
                    new_residuals.append((real_delta - base_delta) / delta_std)

            if new_inputs:
                all_train_inputs.append(np.array(new_inputs))
                all_train_residuals.append(np.array(new_residuals))
                print(f"    DAgger轮{round_i+1}: 收集{len(new_inputs)}个新样本")

    return baseline, nn_model, (state_std, action_std, delta_std)


# ============================================================
# 预测
# ============================================================
def dr_predict(baseline, nn_model, s, tau,
               state_std, action_std, delta_std, residual_scale=0.3):
    s_next_base = baseline.predict(s, tau)
    s_norm = s / state_std
    a_norm = tau / action_std
    with torch.no_grad():
        s_t = torch.FloatTensor(s_norm).unsqueeze(0).to(DEVICE)
        a_t = torch.FloatTensor([a_norm]).unsqueeze(0).to(DEVICE)
        delta_nn = nn_model(s_t, a_t).cpu().numpy()[0]
    return s_next_base + delta_nn * delta_std * residual_scale


# ============================================================
# 测试
# ============================================================
def test_domain_randomization(name, baseline_cls, baseline_kwargs=None,
                              noise_level=0.05, residual_scale=0.3):
    print(f"\n{'='*60}")
    print(f"测试: {name}")
    print(f"{'='*60}")

    baseline, nn_model, (state_std, action_std, delta_std) = train_with_domain_randomization(
        baseline_cls, baseline_kwargs,
        noise_level=noise_level, n_epochs=100, dagger_rounds=3, residual_scale=residual_scale
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
        s_dr = s0.copy()

        for step in range(500):
            tau = tau_func(step, s_real)
            s_real = real_step(s_real, tau)
            s_base = baseline.predict(s_base, tau)
            s_dr = dr_predict(
                baseline, nn_model, s_dr, tau,
                state_std, action_std, delta_std, residual_scale
            )
            if abs(s_real[0]) > math.pi / 3:
                break
            if step + 1 in eval_steps:
                all_errors[step + 1].append(abs(s_dr[0] - s_real[0]))
                baseline_errors[step + 1].append(abs(s_base[0] - s_real[0]))

    print(f"\n  {'步数':>6} | {'纯基线':>10} | {'域随机化':>10} | {'改进倍率':>10}")
    print(f"  {'-'*50}")
    for step in eval_steps:
        b_err = np.mean(baseline_errors[step]) if baseline_errors[step] else float('nan')
        h_err = np.mean(all_errors[step]) if all_errors[step] else float('nan')
        ratio = h_err / b_err if b_err > 0 and not math.isnan(h_err) else float('nan')
        print(f"  {step:6d} | {b_err:10.5f} | {h_err:10.5f} | {ratio:10.2f}x")

    return baseline_errors, all_errors


# ============================================================
# 噪声水平对比测试
# ============================================================
def test_noise_levels(baseline_cls, baseline_kwargs=None):
    print(f"\n{'='*60}")
    print(f"测试: 噪声水平对比")
    print(f"{'='*60}")

    noise_levels = [0.01, 0.05, 0.1, 0.2]
    results = {}

    for noise in noise_levels:
        print(f"\n--- 噪声水平: {noise} ---")
        baseline, nn_model, (state_std, action_std, delta_std) = train_with_domain_randomization(
            baseline_cls, baseline_kwargs,
            noise_level=noise, n_epochs=100, dagger_rounds=3, residual_scale=0.3
        )

        n_segments = 5
        errors_500 = []
        for seg_i in range(n_segments):
            tau_func = make_tau_func(seg_i, 500)
            phi_init = np.random.uniform(-0.25, 0.25)
            s0 = np.array([phi_init, 0.0, 0.0, 0.0])
            s_real = s0.copy()
            s_dr = s0.copy()

            for step in range(500):
                tau = tau_func(step, s_real)
                s_real = real_step(s_real, tau)
                s_dr = dr_predict(
                    baseline, nn_model, s_dr, tau,
                    state_std, action_std, delta_std, 0.3
                )
                if abs(s_real[0]) > math.pi / 3:
                    break
                if step == 499:
                    errors_500.append(abs(s_dr[0] - s_real[0]))

        results[noise] = np.mean(errors_500) if errors_500 else float('nan')
        print(f"  噪声{noise}: 500步MAE = {results[noise]:.5f}")

    return results


if __name__ == '__main__':
    print("=" * 60)
    print("域随机化改进测试")
    print("=" * 60)

    # 测试NeuralODE + 域随机化
    test_domain_randomization(
        "NeuralODE + 域随机化(噪声=0.05)", NeuralODEMethod,
        noise_level=0.05, residual_scale=0.3
    )

    # 测试不同噪声水平
    test_noise_levels(NeuralODEMethod)

    print("\n完成。")
