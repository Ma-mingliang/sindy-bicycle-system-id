"""改进版混合方法全面测试：SINDy+NN / NeuralODE+NN / NNE2E+NN / ParamID+NN。

基线：改进版混合方法（OOD检测 + DAgger + 残差缩放0.3）
目标：找到最佳基线方法，再叠加论文改进方案。
"""

import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
from methods_common import generate_training_data, real_step, K_lqr, dt
from methods_sindy import SINDyPoly, SINDyTrig, SINDyTrigExp
from methods_nn import NeuralODEMethod, NNE2E, ResidualNet, _train_nn, DEVICE
from methods_classic import ParamIDMethod
from methods_evaluate import make_tau_func


# ============================================================
# OOD检测器
# ============================================================
class OODDetector:
    def __init__(self, train_inputs: np.ndarray, threshold: float = 3.0):
        self.mean = np.mean(train_inputs, axis=0)
        self.std = np.std(train_inputs, axis=0) + 1e-8
        self.threshold = threshold

    def is_ood(self, x: np.ndarray) -> bool:
        z = np.abs(x - self.mean) / self.std
        return np.max(z) > self.threshold


# ============================================================
# 改进版混合训练（通用）
# ============================================================
def train_improved(baseline, states, actions, deltas, state_std, action_std, delta_std,
                   n_epochs=200, dagger_rounds=3, residual_scale=0.3, ood_threshold=3.0):
    """训练改进版混合方法，返回(nn_model, ood_detector)。"""

    # 计算残差
    residuals = np.empty_like(deltas)
    for i in range(len(states)):
        s_next_base = baseline.predict(states[i], actions[i])
        residuals[i] = (deltas[i] - (s_next_base - states[i])) / delta_std

    train_inputs_base = np.column_stack([states / state_std, actions.reshape(-1, 1) / action_std])
    all_train_inputs = [train_inputs_base.copy()]
    all_train_residuals = [residuals.copy()]

    nn_model = None
    ood_det = None

    for round_i in range(dagger_rounds):
        train_inputs = np.vstack(all_train_inputs)
        train_residuals = np.vstack(all_train_residuals)

        nn_model = ResidualNet().to(DEVICE)
        train_x = torch.FloatTensor(train_inputs).to(DEVICE)
        train_y = torch.FloatTensor(train_residuals).to(DEVICE)
        _train_nn(nn_model, train_x, train_y, n_epochs=n_epochs)
        ood_det = OODDetector(train_inputs, threshold=ood_threshold)

        # DAgger: rollout收集新数据
        if round_i < dagger_rounds - 1:
            new_inputs, new_residuals = [], []
            for seg_i in range(3):
                tau_func = make_tau_func(100 + seg_i, 500)
                s0 = np.array([np.random.uniform(-0.25, 0.25), 0.0, 0.0, 0.0])
                s_base = s0.copy()
                s_real = s0.copy()
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
                    base_res = (s_next_base - s_prev) / delta_std
                    new_inputs.append(inp)
                    new_residuals.append(real_res - base_res)

            if new_inputs:
                all_train_inputs.append(np.array(new_inputs))
                all_train_residuals.append(np.array(new_residuals))

    return nn_model, ood_det


def hybrid_predict(baseline, nn_model, ood_det, s, tau, state_std, action_std, delta_std, residual_scale=0.3):
    s_next_base = baseline.predict(s, tau)
    s_norm = s / state_std
    a_norm = tau / action_std
    inp = np.concatenate([s_norm, [a_norm]])
    if ood_det.is_ood(inp):
        return s_next_base
    with torch.no_grad():
        delta_nn = nn_model(
            torch.FloatTensor(s_norm).unsqueeze(0).to(DEVICE),
            torch.FloatTensor([a_norm]).unsqueeze(0).to(DEVICE)
        ).cpu().numpy()[0] * delta_std * residual_scale
    return s_next_base + delta_nn


# ============================================================
# 测试函数
# ============================================================
def test_method(name, baseline, nn_model, ood_det, state_std, action_std, delta_std):
    eval_steps = [1, 5, 10, 20, 50, 100, 200, 500]
    base_errors = {s: [] for s in eval_steps}
    hybrid_errors = {s: [] for s in eval_steps}

    for seg_i in range(5):
        tau_func = make_tau_func(seg_i, 500)
        s0 = np.array([np.random.uniform(-0.25, 0.25), 0.0, 0.0, 0.0])
        s_real, s_base, s_hybrid = s0.copy(), s0.copy(), s0.copy()

        for step in range(500):
            tau = tau_func(step, s_real)
            s_real = real_step(s_real, tau)
            s_base = baseline.predict(s_base, tau)
            s_hybrid = hybrid_predict(baseline, nn_model, ood_det, s_hybrid, tau,
                                      state_std, action_std, delta_std)
            if abs(s_real[0]) > math.pi / 3:
                break
            if step + 1 in eval_steps:
                base_errors[step + 1].append(abs(s_base[0] - s_real[0]))
                hybrid_errors[step + 1].append(abs(s_hybrid[0] - s_real[0]))

    print(f"\n  {name}")
    print(f"  {'步数':>6} | {'纯基线':>10} | {'改进混合':>10} | {'倍率':>8}")
    print(f"  {'-'*45}")
    for step in eval_steps:
        b = np.mean(base_errors[step]) if base_errors[step] else float('nan')
        h = np.mean(hybrid_errors[step]) if hybrid_errors[step] else float('nan')
        r = h / b if b > 1e-10 else float('nan')
        print(f"  {step:6d} | {b:10.5f} | {h:10.5f} | {r:8.2f}x")

    return base_errors, hybrid_errors


# ============================================================
# 主测试
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("改进版混合方法全面测试")
    print("=" * 60)

    # 生成训练数据
    print("\n生成训练数据...")
    states, actions, deltas, state_std, action_std, delta_std = generate_training_data(30000)

    # 定义要测试的基线方法
    baselines = [
        ("SINDyPoly", SINDyPoly(), {}),
        ("SINDyTrig", SINDyTrig(), {}),
        ("NeuralODE", NeuralODEMethod(), {}),
        ("NNE2E", NNE2E(), {}),
        ("ParamID", ParamIDMethod(), {}),
    ]

    all_results = {}

    for name, baseline, kwargs in baselines:
        print(f"\n{'='*60}")
        print(f"训练基线: {name}")
        print(f"{'='*60}")
        t0 = time.time()
        baseline.train(states, actions, deltas, state_std, action_std, delta_std)
        print(f"  基线训练耗时: {time.time()-t0:.1f}s")

        t0 = time.time()
        nn_model, ood_det = train_improved(
            baseline, states, actions, deltas, state_std, action_std, delta_std,
            n_epochs=200, dagger_rounds=3, residual_scale=0.3
        )
        print(f"  NN残差训练耗时: {time.time()-t0:.1f}s")

        base_err, hybrid_err = test_method(f"{name} + 改进NN残差", baseline, nn_model, ood_det,
                                           state_std, action_std, delta_std)
        all_results[name] = (base_err, hybrid_err)

    # 总结对比表
    print(f"\n{'='*60}")
    print("总结：500步MAE对比 (rad)")
    print(f"{'='*60}")
    print(f"  {'方法':<25} | {'纯基线':>10} | {'改进混合':>10} | {'倍率':>8}")
    print(f"  {'-'*60}")
    for name in all_results:
        base_err, hybrid_err = all_results[name]
        b500 = np.mean(base_err[500]) if base_err[500] else float('nan')
        h500 = np.mean(hybrid_err[500]) if hybrid_err[500] else float('nan')
        r = h500 / b500 if b500 > 1e-10 else float('nan')
        print(f"  {name:<25} | {b500:10.4f} | {h500:10.4f} | {r:8.2f}x")

    print("\n完成。")
