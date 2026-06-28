"""第五轮改进：贝叶斯神经网络（MC Dropout）。

改进点：
- 当前：标准NN，无不确定性估计
- 改进：用MC Dropout近似贝叶斯神经网络
  - 训练时使用Dropout
  - 推理时保持Dropout开启，进行多次前向传播
  - 用多次预测的均值和方差作为最终预测和不确定性

思路：
- MC Dropout是贝叶斯神经网络的实用近似
- 通过多次采样获得不确定性估计
- 不确定性高时减少残差权重
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
from methods_nn import NeuralODEMethod, DEVICE
from methods_classic import GPMethod, ParamIDMethod
from methods_evaluate import make_tau_func


# ============================================================
# MC Dropout残差网络
# ============================================================
class MCDropoutResidualNet(nn.Module):
    """MC Dropout残差网络。

    训练时使用Dropout，推理时保持Dropout开启进行多次采样。
    """

    def __init__(self, state_dim=4, action_dim=1, hidden=128, dropout_rate=0.1):
        super().__init__()
        self.dropout_rate = dropout_rate

        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden, state_dim)
        )

    def forward(self, s, a=None):
        if a is None:
            x = s
        else:
            if s.dim() == 1:
                x = torch.cat([s, a.unsqueeze(0)])
            else:
                x = torch.cat([s, a], dim=-1)
        return self.net(x)

    def predict_with_uncertainty(self, s, a=None, n_samples=10):
        """多次采样获得不确定性估计。

        Args:
            s: 状态
            a: 动作
            n_samples: 采样次数

        Returns:
            mean: 预测均值
            std: 预测标准差（不确定性）
        """
        self.train()  # 保持Dropout开启
        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(s, a)
                predictions.append(pred.cpu().numpy())
        self.eval()

        predictions = np.array(predictions)
        mean = np.mean(predictions, axis=0)
        std = np.std(predictions, axis=0)
        return mean, std


# ============================================================
# MC Dropout混合预测
# ============================================================
def mc_dropout_predict(baseline, nn_model, ood_detector, s, tau,
                       state_std, action_std, delta_std,
                       residual_scale=0.3, n_samples=10, uncertainty_threshold=0.5):
    """MC Dropout混合预测。

    不确定性高时减少残差权重。
    """
    s_next_base = baseline.predict(s, tau)
    s_norm = s / state_std
    a_norm = tau / action_std
    inp = np.concatenate([s_norm, [a_norm]])

    if ood_detector.is_ood(inp):
        return s_next_base, 0.0

    s_t = torch.FloatTensor(s_norm).unsqueeze(0).to(DEVICE)
    a_t = torch.FloatTensor([a_norm]).unsqueeze(0).to(DEVICE)

    # MC Dropout多次采样
    mean, std = nn_model.predict_with_uncertainty(s_t, a_t, n_samples=n_samples)

    # 计算不确定性（归一化）
    uncertainty = np.mean(std) / np.mean(delta_std)

    # 不确定性高时减少残差权重
    if uncertainty > uncertainty_threshold:
        adaptive_scale = residual_scale * (1.0 - (uncertainty - uncertainty_threshold) / uncertainty_threshold)
        adaptive_scale = max(0.0, adaptive_scale)
    else:
        adaptive_scale = residual_scale

    delta_nn = mean[0] * delta_std * adaptive_scale
    return s_next_base + delta_nn, uncertainty


# ============================================================
# 训练MC Dropout混合方法
# ============================================================
def train_mc_dropout(baseline_cls, baseline_kwargs=None, n_epochs=100,
                     dagger_rounds=3, residual_scale=0.3, ood_threshold=3.0,
                     dropout_rate=0.1, n_samples=10, uncertainty_threshold=0.5):
    """训练MC Dropout混合方法。"""
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

        nn_model = MCDropoutResidualNet(dropout_rate=dropout_rate).to(DEVICE)
        train_x = torch.FloatTensor(train_inputs).to(DEVICE)
        train_y = torch.FloatTensor(train_residuals).to(DEVICE)
        ds = TensorDataset(train_x, train_y)
        loader = DataLoader(ds, batch_size=256, shuffle=True)
        opt = optim.Adam(nn_model.parameters(), lr=1e-3)
        crit = nn.MSELoss()

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
                loss = crit(pred, yb)
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
                    inp = np.concatenate([s_norm, [a_norm]])
                    s_t = torch.FloatTensor(s_norm).unsqueeze(0).to(DEVICE)
                    a_t = torch.FloatTensor([a_norm]).unsqueeze(0).to(DEVICE)
                    mean, _ = nn_model.predict_with_uncertainty(s_t, a_t, n_samples=n_samples)
                    delta_nn = mean[0]
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
def test_mc_dropout(name, baseline_cls, baseline_kwargs=None,
                    residual_scale=0.3, dropout_rate=0.1, n_samples=10, uncertainty_threshold=0.5):
    print(f"\n{'='*60}")
    print(f"测试: {name}")
    print(f"{'='*60}")

    baseline, nn_model, ood_det, (state_std, action_std, delta_std) = train_mc_dropout(
        baseline_cls, baseline_kwargs,
        n_epochs=100, dagger_rounds=3, residual_scale=residual_scale, ood_threshold=3.0,
        dropout_rate=dropout_rate, n_samples=n_samples, uncertainty_threshold=uncertainty_threshold
    )

    eval_steps = [1, 5, 10, 20, 50, 100, 200, 500]
    n_segments = 5
    all_errors = {step: [] for step in eval_steps}
    baseline_errors = {step: [] for step in eval_steps}
    uncertainty_values = []

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
            s_hybrid, uncertainty = mc_dropout_predict(
                baseline, nn_model, ood_det, s_hybrid, tau,
                state_std, action_std, delta_std,
                residual_scale=residual_scale, n_samples=n_samples,
                uncertainty_threshold=uncertainty_threshold
            )
            uncertainty_values.append(uncertainty)

            if abs(s_real[0]) > math.pi / 3:
                break

            if step + 1 in eval_steps:
                all_errors[step + 1].append(abs(s_hybrid[0] - s_real[0]))
                baseline_errors[step + 1].append(abs(s_base[0] - s_real[0]))

    print(f"\n  {'步数':>6} | {'纯基线':>10} | {'MC Dropout':>10} | {'改进倍率':>10}")
    print(f"  {'-'*50}")
    for step in eval_steps:
        b_err = np.mean(baseline_errors[step]) if baseline_errors[step] else float('nan')
        h_err = np.mean(all_errors[step]) if all_errors[step] else float('nan')
        ratio = h_err / b_err if b_err > 0 and not math.isnan(h_err) else float('nan')
        print(f"  {step:6d} | {b_err:10.5f} | {h_err:10.5f} | {ratio:10.2f}x")

    unc_arr = np.array(uncertainty_values)
    print(f"\n  不确定性统计:")
    print(f"    平均: {np.mean(unc_arr):.4f}")
    print(f"    中位: {np.median(unc_arr):.4f}")
    print(f"    最小: {np.min(unc_arr):.4f}")
    print(f"    最大: {np.max(unc_arr):.4f}")

    return baseline_errors, all_errors


if __name__ == '__main__':
    print("=" * 60)
    print("第五轮改进：贝叶斯神经网络（MC Dropout）")
    print("=" * 60)
    print("改进点：用MC Dropout近似贝叶斯神经网络")
    print("  - 训练时使用Dropout")
    print("  - 推理时保持Dropout开启，进行多次前向传播")
    print("  - 用多次预测的均值和方差作为最终预测和不确定性")
    print("=" * 60)

    # 测试NeuralODE + MC Dropout
    test_mc_dropout(
        "NeuralODE + MC Dropout", NeuralODEMethod,
        residual_scale=0.3, dropout_rate=0.1, n_samples=10, uncertainty_threshold=0.5
    )

    print("\n完成。")
