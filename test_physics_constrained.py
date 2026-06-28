"""第四轮改进：物理约束残差。

改进点：
- 当前：NN残差无约束，可以输出任意值
- 改进：给NN残差添加物理约束
  1. 能量约束：残差不能无限增加系统能量
  2. 有界约束：残差输出限制在合理范围内
  3. 平滑约束：相邻时间步的残差变化应平滑

思路：
- 在NN输出后添加约束层
- 用物理先验知识限制残差的范围
- 这样可以避免残差输出不合理的值
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
# 物理约束残差网络
# ============================================================
class PhysicsConstrainedResidualNet(nn.Module):
    """带物理约束的残差网络。

    约束：
    1. 输出有界：使用tanh限制输出范围
    2. 能量约束：限制残差不能无限增加系统能量
    3. 平滑约束：输出应平滑变化
    """

    def __init__(self, state_dim=4, action_dim=1, hidden=128,
                 max_residual=0.1, energy_weight=0.01):
        super().__init__()
        self.max_residual = max_residual
        self.energy_weight = energy_weight

        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, state_dim)
        )

    def forward(self, s, a=None):
        """前向传播，带物理约束。

        Args:
            s: 状态 (batch, state_dim) 或 (state_dim,)
            a: 动作 (batch, action_dim) 或 (action_dim,)

        Returns:
            残差 (batch, state_dim) 或 (state_dim,)
        """
        if a is None:
            # 单输入模式：拼接s和a
            x = s
        else:
            # 双输入模式
            if s.dim() == 1:
                x = torch.cat([s, a.unsqueeze(0)])
            else:
                x = torch.cat([s, a], dim=-1)

        # 原始残差
        raw_residual = self.net(x)

        # 约束1：有界约束（tanh限制范围）
        bounded_residual = torch.tanh(raw_residual) * self.max_residual

        return bounded_residual


# ============================================================
# 物理约束混合预测
# ============================================================
def physics_constrained_predict(baseline, nn_model, ood_detector, s, tau,
                                state_std, action_std, delta_std, residual_scale=0.3):
    """物理约束混合预测。"""
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
# 训练物理约束混合方法
# ============================================================
def train_physics_constrained(baseline_cls, baseline_kwargs=None, n_epochs=100,
                              dagger_rounds=3, residual_scale=0.3, ood_threshold=3.0,
                              max_residual=0.1, energy_weight=0.01):
    """训练物理约束混合方法。"""
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

    # 初始训练输入
    train_inputs_base = np.column_stack([states / state_std, actions.reshape(-1, 1) / action_std])

    # DAgger迭代
    all_train_inputs = [train_inputs_base.copy()]
    all_train_residuals = [residuals.copy()]

    for round_i in range(dagger_rounds):
        train_inputs = np.vstack(all_train_inputs)
        train_residuals = np.vstack(all_train_residuals)

        # 训练物理约束NN
        nn_model = PhysicsConstrainedResidualNet(
            max_residual=max_residual, energy_weight=energy_weight
        ).to(DEVICE)
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

                pred = nn_model(xb)

                # 主损失：MSE
                mse_loss = crit(pred, yb)

                # 物理约束损失
                # 1. 能量约束：残差不应增加系统能量（简化：限制残差范数）
                energy_loss = torch.mean(torch.norm(pred, dim=-1) ** 2)

                # 2. 平滑约束：批次内残差应平滑（减少突变）
                if pred.shape[0] > 1:
                    smooth_loss = torch.mean(torch.norm(pred[1:] - pred[:-1], dim=-1) ** 2)
                else:
                    smooth_loss = torch.tensor(0.0, device=DEVICE)

                # 总损失
                total_loss = mse_loss + energy_weight * energy_loss + 0.001 * smooth_loss

                opt.zero_grad()
                total_loss.backward()
                opt.step()
        nn_model.eval()

        # DAgger
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
                    if ood_det.is_ood(inp):
                        delta_nn = np.zeros(4)
                    s_base = baseline.predict(s_base, tau) + delta_nn * delta_std * residual_scale
                    if abs(s_real[0]) > math.pi / 3:
                        break
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
# 测试
# ============================================================
def test_physics_constrained(name, baseline_cls, baseline_kwargs=None,
                             residual_scale=0.3, max_residual=0.1, energy_weight=0.01):
    print(f"\n{'='*60}")
    print(f"测试: {name}")
    print(f"{'='*60}")

    baseline, nn_model, ood_det, (state_std, action_std, delta_std) = train_physics_constrained(
        baseline_cls, baseline_kwargs,
        n_epochs=100, dagger_rounds=3, residual_scale=residual_scale, ood_threshold=3.0,
        max_residual=max_residual, energy_weight=energy_weight
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
            s_hybrid = physics_constrained_predict(
                baseline, nn_model, ood_det, s_hybrid, tau,
                state_std, action_std, delta_std, residual_scale
            )

            if abs(s_real[0]) > math.pi / 3:
                break

            if step + 1 in eval_steps:
                all_errors[step + 1].append(abs(s_hybrid[0] - s_real[0]))
                baseline_errors[step + 1].append(abs(s_base[0] - s_real[0]))

    print(f"\n  {'步数':>6} | {'纯基线':>10} | {'物理约束':>10} | {'改进倍率':>10}")
    print(f"  {'-'*50}")
    for step in eval_steps:
        b_err = np.mean(baseline_errors[step]) if baseline_errors[step] else float('nan')
        h_err = np.mean(all_errors[step]) if all_errors[step] else float('nan')
        ratio = h_err / b_err if b_err > 0 and not math.isnan(h_err) else float('nan')
        print(f"  {step:6d} | {b_err:10.5f} | {h_err:10.5f} | {ratio:10.2f}x")

    return baseline_errors, all_errors


if __name__ == '__main__':
    print("=" * 60)
    print("第四轮改进：物理约束残差")
    print("=" * 60)
    print("改进点：给NN残差添加物理约束")
    print("  1. 有界约束：tanh限制输出范围")
    print("  2. 能量约束：限制残差范数")
    print("  3. 平滑约束：减少残差突变")
    print("=" * 60)

    # 测试NeuralODE + 物理约束残差
    test_physics_constrained(
        "NeuralODE + 物理约束残差", NeuralODEMethod,
        residual_scale=0.3, max_residual=0.1, energy_weight=0.01
    )

    print("\n完成。")
