"""第一轮改进：自适应残差缩放。

改进点：
- 当前：残差缩放固定0.3
- 改进：根据OOD距离动态调整scale（近分布→大scale，远分布→小scale）

思路：
- 用sigmoid函数将Mahalanobis距离映射到[0, 1]
- 距离<1σ时scale接近max_scale（0.5）
- 距离>3σ时scale接近0
- 这样在分布内可以充分利用残差，在分布外更保守
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
    """基于Mahalanobis距离的自适应OOD检测。

    返回连续的scale值而非二元判断：
    - 距离<1σ时scale接近max_scale
    - 距离>threshold时scale接近0
    """

    def __init__(self, train_inputs: np.ndarray, threshold: float = 3.0, max_scale: float = 0.5):
        self.mean = np.mean(train_inputs, axis=0)
        self.std = np.std(train_inputs, axis=0) + 1e-8
        self.threshold = threshold
        self.max_scale = max_scale

    def compute_scale(self, x: np.ndarray) -> float:
        """计算自适应残差缩放系数。

        Args:
            x: 输入向量（归一化后）

        Returns:
            scale值（0到max_scale之间）
        """
        # 计算Mahalanobis距离（简化版：用max绝对z-score）
        z = np.abs(x - self.mean) / self.std
        mahal_dist = np.max(z)

        # 用sigmoid函数映射到[0, 1]
        # 当dist=0时scale≈max_scale
        # 当dist=threshold时scale≈0
        # 限制输入范围防止溢出
        k = 3.0  # 控制sigmoid陡峭程度
        normalized_dist = (mahal_dist - 1.0) / (self.threshold - 1.0)
        # 限制输入范围在[-10, 10]防止溢出
        clipped_input = max(-10.0, min(10.0, k * normalized_dist * 4))
        sigmoid = 1.0 / (1.0 + math.exp(clipped_input))
        scale = sigmoid * self.max_scale

        return scale

    def is_ood(self, x: np.ndarray) -> bool:
        """保留二元判断接口，用于兼容。"""
        z = np.abs(x - self.mean) / self.std
        return np.max(z) > self.threshold


# ============================================================
# 自适应混合预测
# ============================================================
def adaptive_hybrid_predict(baseline, nn_model, ood_detector, s, tau,
                            state_std, action_std, delta_std):
    """自适应混合预测：根据OOD距离动态调整残差缩放。"""
    s_next_base = baseline.predict(s, tau)
    s_norm = s / state_std
    a_norm = tau / action_std
    inp = np.concatenate([s_norm, [a_norm]])

    # 计算自适应scale
    adaptive_scale = ood_detector.compute_scale(inp)

    # 如果scale太小，直接返回基线预测
    if adaptive_scale < 0.01:
        return s_next_base, 0.0

    with torch.no_grad():
        s_t = torch.FloatTensor(s_norm).unsqueeze(0).to(DEVICE)
        a_t = torch.FloatTensor([a_norm]).unsqueeze(0).to(DEVICE)
        delta_nn = nn_model(s_t, a_t).cpu().numpy()[0] * delta_std * adaptive_scale
    return s_next_base + delta_nn, adaptive_scale


# ============================================================
# 训练自适应混合方法
# ============================================================
def train_adaptive_hybrid(baseline_cls, baseline_kwargs=None, n_epochs=100,
                          dagger_rounds=3, max_scale=0.5, ood_threshold=3.0):
    """训练自适应混合方法。

    Args:
        baseline_cls: 基线方法类
        baseline_kwargs: 基线构造参数
        n_epochs: 每轮训练epoch数
        dagger_rounds: DAgger迭代轮数
        max_scale: 最大残差缩放系数
        ood_threshold: OOD检测阈值(σ)

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

        # 自适应OOD检测器
        ood_det = AdaptiveOODDetector(train_inputs, threshold=ood_threshold, max_scale=max_scale)

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

                    # 自适应scale
                    adaptive_scale = ood_det.compute_scale(inp)
                    s_base = baseline.predict(s_base, tau) + delta_nn * delta_std * adaptive_scale

                    if abs(s_real[0]) > math.pi / 3:
                        break
                    # 收集新残差
                    s_next_base = baseline.predict(s_base - delta_nn * delta_std * adaptive_scale, tau)
                    base_delta = s_next_base - (s_base - delta_nn * delta_std * adaptive_scale)
                    real_delta = s_real - (s_base - delta_nn * delta_std * adaptive_scale)
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
def test_adaptive_hybrid(name, baseline_cls, baseline_kwargs=None, max_scale=0.5):
    print(f"\n{'='*60}")
    print(f"测试: {name}")
    print(f"{'='*60}")

    baseline, nn_model, ood_det, (state_std, action_std, delta_std) = train_adaptive_hybrid(
        baseline_cls, baseline_kwargs,
        n_epochs=100, dagger_rounds=3, max_scale=max_scale, ood_threshold=3.0
    )

    # 多段rollout测试
    eval_steps = [1, 5, 10, 20, 50, 100, 200, 500]
    n_segments = 5
    all_errors = {step: [] for step in eval_steps}
    baseline_errors = {step: [] for step in eval_steps}
    scale_values = []

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
            s_hybrid, adaptive_scale = adaptive_hybrid_predict(
                baseline, nn_model, ood_det, s_hybrid, tau,
                state_std, action_std, delta_std
            )
            scale_values.append(adaptive_scale)

            if abs(s_real[0]) > math.pi / 3:
                break

            if step + 1 in eval_steps:
                all_errors[step + 1].append(abs(s_hybrid[0] - s_real[0]))
                baseline_errors[step + 1].append(abs(s_base[0] - s_real[0]))

    # 输出结果
    print(f"\n  {'步数':>6} | {'纯基线':>10} | {'自适应混合':>10} | {'改进倍率':>10}")
    print(f"  {'-'*50}")
    for step in eval_steps:
        b_err = np.mean(baseline_errors[step]) if baseline_errors[step] else float('nan')
        h_err = np.mean(all_errors[step]) if all_errors[step] else float('nan')
        ratio = h_err / b_err if b_err > 0 and not math.isnan(h_err) else float('nan')
        print(f"  {step:6d} | {b_err:10.5f} | {h_err:10.5f} | {ratio:10.2f}x")

    # 统计scale分布
    scale_arr = np.array(scale_values)
    print(f"\n  自适应scale统计:")
    print(f"    平均: {np.mean(scale_arr):.4f}")
    print(f"    中位: {np.median(scale_arr):.4f}")
    print(f"    最小: {np.min(scale_arr):.4f}")
    print(f"    最大: {np.max(scale_arr):.4f}")
    print(f"    >0.1的比例: {np.mean(scale_arr > 0.1):.2%}")

    return baseline_errors, all_errors


if __name__ == '__main__':
    print("=" * 60)
    print("第一轮改进：自适应残差缩放")
    print("=" * 60)
    print("改进点：根据OOD距离动态调整残差缩放系数")
    print("  - 距离<1σ时scale接近0.5")
    print("  - 距离>3σ时scale接近0")
    print("=" * 60)

    # 测试GP + 自适应残差
    test_adaptive_hybrid("GP + 自适应残差缩放", GPMethod, max_scale=0.5)

    # 对比：固定scale=0.3
    print("\n" + "=" * 60)
    print("对比：固定scale=0.3（基线）")
    print("=" * 60)

    # 测试NeuralODE + 自适应残差
    test_adaptive_hybrid("NeuralODE + 自适应残差缩放", NeuralODEMethod, max_scale=0.5)

    print("\n完成。")
