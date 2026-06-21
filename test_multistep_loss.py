"""第三轮改进：多步损失训练。

改进点：
- 当前：只训练单步残差
- 改进：用多步rollout loss训练NN
  - 不只训单步残差，而是用多步rollout的误差来训练
  - 这样可以让NN学习到长期的影响，减少分布偏移

思路：
- 用当前NN进行多步rollout
- 计算rollout的累积误差
- 用累积误差来更新NN的参数
- 这样可以让NN学习到长期的影响
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
# 多步损失训练
# ============================================================
def train_multistep_loss(baseline_cls, baseline_kwargs=None, n_epochs=100,
                         dagger_rounds=3, residual_scale=0.3, ood_threshold=3.0,
                         multistep_length=10, multistep_weight=0.5):
    """训练多步损失混合方法。

    Args:
        baseline_cls: 基线方法类
        baseline_kwargs: 基线构造参数
        n_epochs: 每轮训练epoch数
        dagger_rounds: DAgger迭代轮数
        residual_scale: 残差缩放系数
        ood_threshold: OOD检测阈值(σ)
        multistep_length: 多步rollout的长度
        multistep_weight: 多步损失的权重（0-1）

    Returns:
        baseline, nn_model, ood_detector, (state_std, action_std, delta_std)
    """
    if baseline_kwargs is None:
        baseline_kwargs = {}

    # 生成训练数据
    states, actions, deltas, state_std, action_std, delta_std = generate_training_data(30000)

    # 训练基线
    baseline = baseline_cls(**baseline_kwargs)
    baseline.train(states, actions, deltas, state_std, action_std, delta_std)

    # 计算残差
    residuals = np.empty_like(deltas)
    for i in range(len(states)):
        s_next_base = baseline.predict(states[i], actions[i])
        residuals[i] = (deltas[i] - (s_next_base - states[i])) / delta_std
    residual_std = np.std(residuals, axis=0)
    print(f"    残差std: {residual_std}, 占比: {np.mean(residual_std):.4f}")

    # 初始训练输入（用于OOD检测）
    train_inputs_base = np.column_stack([states / state_std, actions.reshape(-1, 1) / action_std])

    # DAgger迭代
    all_train_inputs = [train_inputs_base.copy()]
    all_train_residuals = [residuals.copy()]

    for round_i in range(dagger_rounds):
        # 合并所有训练数据
        train_inputs = np.vstack(all_train_inputs)
        train_residuals = np.vstack(all_train_residuals)

        # 训练NN（带多步损失）
        nn_model = ResidualNet().to(DEVICE)
        train_x = torch.FloatTensor(train_inputs).to(DEVICE)
        train_y = torch.FloatTensor(train_residuals).to(DEVICE)
        ds = TensorDataset(train_x, train_y)
        loader = DataLoader(ds, batch_size=256, shuffle=True)
        opt = optim.Adam(nn_model.parameters(), lr=1e-3)
        crit = nn.MSELoss()

        # OOD检测器
        class SimpleOODDetector:
            def __init__(self, train_inputs, threshold=3.0):
                self.mean = np.mean(train_inputs, axis=0)
                self.std = np.std(train_inputs, axis=0) + 1e-8
                self.threshold = threshold

            def is_ood(self, x):
                z = np.abs(x - self.mean) / self.std
                return np.max(z) > self.threshold

        ood_det = SimpleOODDetector(train_inputs, threshold=ood_threshold)

        nn_model.train()
        for epoch in range(n_epochs):
            for xb, yb in loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)

                # 单步损失
                pred = nn_model(xb)
                single_loss = crit(pred, yb)

                # 多步损失（随机采样一些起点进行rollout）
                multistep_loss = torch.tensor(0.0, device=DEVICE)
                n_multistep = 0

                # 随机采样2个起点进行多步rollout
                for _ in range(2):
                    # 随机选择一个训练样本作为起点
                    idx = np.random.randint(0, len(states))
                    s_start = states[idx]
                    tau_start = actions[idx]

                    # 进行多步rollout
                    s_current = s_start.copy()
                    rollout_loss = torch.tensor(0.0, device=DEVICE)

                    for step in range(multistep_length):
                        # 基线预测
                        s_next_base = baseline.predict(s_current, tau_start)

                        # NN残差
                        s_norm = s_current / state_std
                        a_norm = tau_start / action_std
                        inp = np.concatenate([s_norm, [a_norm]])

                        # OOD检测
                        if ood_det.is_ood(inp):
                            break

                        with torch.no_grad():
                            s_t = torch.FloatTensor(s_norm).unsqueeze(0).to(DEVICE)
                            a_t = torch.FloatTensor([a_norm]).unsqueeze(0).to(DEVICE)
                            delta_nn = nn_model(s_t, a_t).cpu().numpy()[0]

                        # 混合预测
                        s_next = s_next_base + delta_nn * delta_std * residual_scale

                        # 真实下一步（用真实模型）
                        s_real = real_step(s_current, tau_start)

                        # 计算误差
                        error = np.sum((s_next - s_real) ** 2)
                        rollout_loss += torch.tensor(error, device=DEVICE)

                        # 更新状态
                        s_current = s_next
                        tau_start = -K_lqr @ s_current  # LQR控制

                    multistep_loss += rollout_loss / multistep_length
                    n_multistep += 1

                if n_multistep > 0:
                    multistep_loss = multistep_loss / n_multistep

                # 总损失 = 单步损失 + 多步损失 * 权重
                total_loss = single_loss + multistep_weight * multistep_loss

                opt.zero_grad()
                total_loss.backward()
                opt.step()
        nn_model.eval()

        # DAgger: 用当前模型rollout，收集新数据
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
                    # 带残差的基线预测
                    s_norm = s_base / state_std
                    a_norm = tau / action_std
                    inp = np.concatenate([s_norm, [a_norm]])
                    with torch.no_grad():
                        s_t = torch.FloatTensor(s_norm).unsqueeze(0).to(DEVICE)
                        a_t = torch.FloatTensor([a_norm]).unsqueeze(0).to(DEVICE)
                        delta_nn = nn_model(s_t, a_t).cpu().numpy()[0]
                    if ood_det.is_ood(inp):
                        delta_nn = np.zeros(4)
                    s_base = baseline.predict(s_base, tau) + delta_nn * delta_std * residual_scale
                    if abs(s_real[0]) > math.pi / 3:
                        break
                    # 收集新残差
                    s_next_base = baseline.predict(s_base - delta_nn * delta_std * residual_scale, tau)
                    base_delta = s_next_base - (s_base - delta_nn * delta_std * residual_scale)
                    real_delta = s_real - (s_base - delta_nn * delta_std * residual_scale)
                    new_res = (real_delta - base_delta) / delta_std
                    new_inputs.append(inp)
                    new_residuals.append(new_res)

            if new_inputs:
                all_train_inputs.append(np.array(new_inputs))
                all_train_residuals.append(np.array(new_residuals))
                print(f"    DAgger轮{round_i+1}: 收集{len(new_inputs)}个新样本")

    return baseline, nn_model, ood_det, (state_std, action_std, delta_std)


# ============================================================
# 混合预测
# ============================================================
def multistep_hybrid_predict(baseline, nn_model, ood_detector, s, tau,
                             state_std, action_std, delta_std, residual_scale=0.3):
    """多步损失训练的混合预测。"""
    s_next_base = baseline.predict(s, tau)
    s_norm = s / state_std
    a_norm = tau / action_std
    inp = np.concatenate([s_norm, [a_norm]])

    if ood_detector.is_ood(inp):
        return s_next_base

    with torch.no_grad():
        s_t = torch.FloatTensor(s_norm).unsqueeze(0).to(DEVICE)
        a_t = torch.FloatTensor([a_norm]).unsqueeze(0).to(DEVICE)
        delta_nn = nn_model(s_t, a_t).cpu().numpy()[0] * delta_std * residual_scale
    return s_next_base + delta_nn


# ============================================================
# 测试
# ============================================================
def test_multistep_loss(name, baseline_cls, baseline_kwargs=None,
                        residual_scale=0.3, multistep_length=10, multistep_weight=0.5):
    print(f"\n{'='*60}")
    print(f"测试: {name}")
    print(f"{'='*60}")

    baseline, nn_model, ood_det, (state_std, action_std, delta_std) = train_multistep_loss(
        baseline_cls, baseline_kwargs,
        n_epochs=100, dagger_rounds=3, residual_scale=residual_scale, ood_threshold=3.0,
        multistep_length=multistep_length, multistep_weight=multistep_weight
    )

    # 多段rollout测试
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
            s_hybrid = multistep_hybrid_predict(
                baseline, nn_model, ood_det, s_hybrid, tau,
                state_std, action_std, delta_std, residual_scale
            )

            if abs(s_real[0]) > math.pi / 3:
                break

            if step + 1 in eval_steps:
                all_errors[step + 1].append(abs(s_hybrid[0] - s_real[0]))
                baseline_errors[step + 1].append(abs(s_base[0] - s_real[0]))

    # 输出结果
    print(f"\n  {'步数':>6} | {'纯基线':>10} | {'多步损失':>10} | {'改进倍率':>10}")
    print(f"  {'-'*50}")
    for step in eval_steps:
        b_err = np.mean(baseline_errors[step]) if baseline_errors[step] else float('nan')
        h_err = np.mean(all_errors[step]) if all_errors[step] else float('nan')
        ratio = h_err / b_err if b_err > 0 and not math.isnan(h_err) else float('nan')
        print(f"  {step:6d} | {b_err:10.5f} | {h_err:10.5f} | {ratio:10.2f}x")

    return baseline_errors, all_errors


if __name__ == '__main__':
    print("=" * 60)
    print("第三轮改进：多步损失训练")
    print("=" * 60)
    print("改进点：用多步rollout loss训练NN")
    print("  - 不只训单步残差，而是用多步rollout的误差来训练")
    print("  - 这样可以让NN学习到长期的影响，减少分布偏移")
    print("=" * 60)

    # 测试NeuralODE + 多步损失（NeuralODE预测快，适合多步训练）
    test_multistep_loss(
        "NeuralODE + 多步损失(10步)", NeuralODEMethod,
        residual_scale=0.3, multistep_length=10, multistep_weight=0.5
    )

    print("\n完成。")
