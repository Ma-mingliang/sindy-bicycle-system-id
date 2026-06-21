"""第二轮改进：GP + NN更深度融合。

改进点：
- 当前：GP和NN独立训练，NN只是学习残差
- 改进：用GP的不确定性指导NN残差的置信度
  - GP预测时返回均值和方差
  - 方差大时说明GP不确定，此时NN残差的置信度应该降低
  - 用GP的方差来调整NN残差的缩放系数

思路：
- GP的方差反映了预测的不确定性
- 当GP方差大时，说明基线预测不可靠，此时NN残差可能也不可靠
- 可以用GP方差来调整NN残差的权重：方差大→权重小，方差小→权重大
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
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel


# ============================================================
# 带不确定性的GP
# ============================================================
class GPUncertainty:
    """带不确定性估计的GP方法。

    不仅返回均值预测，还返回方差。
    """

    def __init__(self):
        self.models = []
        self.state_std = None
        self.action_std = None
        self.delta_std = None

    def train(self, states, actions, deltas, state_std, action_std, delta_std):
        """训练GP模型。"""
        self.state_std = state_std
        self.action_std = action_std
        self.delta_std = delta_std

        X = np.column_stack([states / state_std, actions.reshape(-1, 1) / action_std])
        Y = deltas / delta_std

        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)

        for i in range(4):
            print(f"    GP输出{i} 训练中...", end="", flush=True)
            gp = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=2,
                alpha=0.01,
                random_state=42
            )
            gp.fit(X, Y[:, i])
            self.models.append(gp)
            print(f" 完成")

    def predict_with_uncertainty(self, s, tau):
        """预测状态转移，返回均值和方差。

        Args:
            s: 当前状态
            tau: 控制输入

        Returns:
            s_next: 预测的下一状态
            variance: 预测的方差（4维）
        """
        x = np.concatenate([s / self.state_std, [tau / self.action_std]]).reshape(1, -1)

        means = []
        variances = []
        for i, gp in enumerate(self.models):
            mean, std = gp.predict(x, return_std=True)
            means.append(mean[0])
            variances.append(std[0] ** 2)

        means = np.array(means) * self.delta_std + s
        variances = np.array(variances) * (self.delta_std ** 2)

        return means, variances

    def predict(self, s, tau):
        """兼容接口，只返回均值。"""
        s_next, _ = self.predict_with_uncertainty(s, tau)
        return s_next


# ============================================================
# 不确定性感知的混合预测
# ============================================================
def uncertainty_aware_predict(baseline, nn_model, ood_detector, s, tau,
                              state_std, action_std, delta_std,
                              base_scale=0.3, uncertainty_weight=0.5):
    """不确定性感知的混合预测。

    用GP的不确定性来调整NN残差的权重：
    - GP方差小→基线预测可靠→NN残差权重正常
    - GP方差大→基线预测不可靠→NN残差权重降低
    """
    s_next_base, variance = baseline.predict_with_uncertainty(s, tau)

    # 计算归一化方差（相对于delta_std^2）
    normalized_variance = np.mean(variance) / (np.mean(delta_std ** 2))

    # 用方差调整scale：方差大→scale小，方差小→scale大
    # sigmoid映射：normalized_variance=0时scale=base_scale，normalized_variance=1时scale=0
    k = 5.0  # 控制sigmoid陡峭程度
    clipped_input = max(-10.0, min(10.0, k * (normalized_variance - 0.5)))
    variance_scale = 1.0 / (1.0 + math.exp(clipped_input))
    adaptive_scale = base_scale * variance_scale

    # 计算OOD检测
    s_norm = s / state_std
    a_norm = tau / action_std
    inp = np.concatenate([s_norm, [a_norm]])

    if ood_detector.is_ood(inp):
        adaptive_scale = 0.0

    # 如果scale太小，直接返回基线预测
    if adaptive_scale < 0.01:
        return s_next_base, adaptive_scale, normalized_variance

    with torch.no_grad():
        s_t = torch.FloatTensor(s_norm).unsqueeze(0).to(DEVICE)
        a_t = torch.FloatTensor([a_norm]).unsqueeze(0).to(DEVICE)
        delta_nn = nn_model(s_t, a_t).cpu().numpy()[0] * delta_std * adaptive_scale
    return s_next_base + delta_nn, adaptive_scale, normalized_variance


# ============================================================
# 训练不确定性感知混合方法
# ============================================================
def train_uncertainty_aware(baseline_cls, baseline_kwargs=None, n_epochs=100,
                            dagger_rounds=3, base_scale=0.3, ood_threshold=3.0):
    """训练不确定性感知混合方法。"""
    if baseline_kwargs is None:
        baseline_kwargs = {}

    # 生成训练数据（减少到5000避免内存不足）
    states, actions, deltas, state_std, action_std, delta_std = generate_training_data(5000)

    # 训练带不确定性的GP
    baseline = GPUncertainty()
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

        # 训练NN
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

                    # 不确定性感知预测
                    s_base, _, _ = uncertainty_aware_predict(
                        baseline, nn_model, ood_det, s_base, tau,
                        state_std, action_std, delta_std,
                        base_scale=base_scale, uncertainty_weight=0.5
                    )

                    if abs(s_real[0]) > math.pi / 3:
                        break
                    # 收集新残差
                    s_next_base, _ = baseline.predict_with_uncertainty(s_base - delta_std * 0.0, tau)  # 简化
                    new_res = (s_real - s_next_base) / delta_std
                    new_inputs.append(np.concatenate([s_base / state_std, [tau / action_std]]))
                    new_residuals.append(new_res)

            if new_inputs:
                all_train_inputs.append(np.array(new_inputs))
                all_train_residuals.append(np.array(new_residuals))
                print(f"    DAgger轮{round_i+1}: 收集{len(new_inputs)}个新样本")

    return baseline, nn_model, ood_det, (state_std, action_std, delta_std)


# ============================================================
# 测试
# ============================================================
def test_uncertainty_aware(name, base_scale=0.3):
    print(f"\n{'='*60}")
    print(f"测试: {name}")
    print(f"{'='*60}")

    baseline, nn_model, ood_det, (state_std, action_std, delta_std) = train_uncertainty_aware(
        GPMethod,
        n_epochs=100, dagger_rounds=3, base_scale=base_scale, ood_threshold=3.0
    )

    # 多段rollout测试
    eval_steps = [1, 5, 10, 20, 50, 100, 200, 500]
    n_segments = 5
    all_errors = {step: [] for step in eval_steps}
    baseline_errors = {step: [] for step in eval_steps}
    scale_values = []
    variance_values = []

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
            s_hybrid, adaptive_scale, norm_var = uncertainty_aware_predict(
                baseline, nn_model, ood_det, s_hybrid, tau,
                state_std, action_std, delta_std,
                base_scale=base_scale, uncertainty_weight=0.5
            )
            scale_values.append(adaptive_scale)
            variance_values.append(norm_var)

            if abs(s_real[0]) > math.pi / 3:
                break

            if step + 1 in eval_steps:
                all_errors[step + 1].append(abs(s_hybrid[0] - s_real[0]))
                baseline_errors[step + 1].append(abs(s_base[0] - s_real[0]))

    # 输出结果
    print(f"\n  {'步数':>6} | {'纯基线':>10} | {'不确定性感知':>10} | {'改进倍率':>10}")
    print(f"  {'-'*50}")
    for step in eval_steps:
        b_err = np.mean(baseline_errors[step]) if baseline_errors[step] else float('nan')
        h_err = np.mean(all_errors[step]) if all_errors[step] else float('nan')
        ratio = h_err / b_err if b_err > 0 and not math.isnan(h_err) else float('nan')
        print(f"  {step:6d} | {b_err:10.5f} | {h_err:10.5f} | {ratio:10.2f}x")

    # 统计scale分布
    scale_arr = np.array(scale_values)
    var_arr = np.array(variance_values)
    print(f"\n  自适应scale统计:")
    print(f"    平均: {np.mean(scale_arr):.4f}")
    print(f"    中位: {np.median(scale_arr):.4f}")
    print(f"    最小: {np.min(scale_arr):.4f}")
    print(f"    最大: {np.max(scale_arr):.4f}")
    print(f"    >0.1的比例: {np.mean(scale_arr > 0.1):.2%}")
    print(f"\n  GP不确定性统计:")
    print(f"    平均归一化方差: {np.mean(var_arr):.4f}")
    print(f"    中位归一化方差: {np.median(var_arr):.4f}")

    return baseline_errors, all_errors


if __name__ == '__main__':
    print("=" * 60)
    print("第二轮改进：GP + NN更深度融合")
    print("=" * 60)
    print("改进点：用GP的不确定性指导NN残差的置信度")
    print("  - GP方差大→基线预测不可靠→NN残差权重降低")
    print("  - GP方差小→基线预测可靠→NN残差权重正常")
    print("=" * 60)

    # 测试GP + 不确定性感知残差
    test_uncertainty_aware("GP + 不确定性感知残差", base_scale=0.3)

    print("\n完成。")
