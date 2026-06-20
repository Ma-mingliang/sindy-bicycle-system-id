"""诊断混合方法：为什么加NN残差反而变差？

测试：
1. 残差分布分析（残差std vs delta_std）
2. 增加训练epoch（200→500→1000）对比
3. 分布偏移检测（rollout中的NN输入 vs 训练输入）
4. 单步精度 vs 多步rollout精度对比
"""

import numpy as np
import math
import torch
import time
from methods_common import generate_training_data, real_step, K_lqr, dt
from methods_nn import NeuralODEMethod, ResidualNet, _train_nn, DEVICE
from methods_classic import GPMethod, ParamIDMethod
from methods_evaluate import make_tau_func, run_trajectory

# ============================================================
# 1. 生成训练数据
# ============================================================
print("=" * 70)
print("混合方法诊断")
print("=" * 70)

print("\n[1] 生成训练数据...")
states, actions, deltas, state_std, action_std, delta_std = generate_training_data(30000)
print(f"  delta_std (物理空间): {delta_std}")
print(f"  state_std: {state_std}")

# ============================================================
# 2. 训练基线模型，分析残差分布
# ============================================================
print("\n[2] 残差分布分析...")

baselines = {
    'NeuralODE': NeuralODEMethod(),
    'ParamID': ParamIDMethod(),
}

for name, baseline in baselines.items():
    baseline.train(states, actions, deltas, state_std, action_std, delta_std)

    # 计算基线残差
    residuals = np.empty_like(deltas)
    for i in range(len(states)):
        s_next_base = baseline.predict(states[i], actions[i])
        residuals[i] = deltas[i] - (s_next_base - states[i])

    residual_std = np.std(residuals, axis=0)
    ratio = residual_std / delta_std
    print(f"\n  {name} 基线:")
    print(f"    残差std:  {residual_std}")
    print(f"    delta_std: {delta_std}")
    print(f"    残差/总 比例: {ratio}")
    print(f"    → 残差占总变化的 {np.mean(ratio)*100:.1f}%")

# ============================================================
# 3. 训练NeuralODE+NN，不同epoch对比
# ============================================================
print("\n[3] NeuralODE+NN 不同训练epoch对比...")

# 先训练NeuralODE基线
neuralode = NeuralODEMethod()
neuralode.train(states, actions, deltas, state_std, action_std, delta_std)

# 计算残差
residuals_nodenn = np.empty_like(deltas)
for i in range(len(states)):
    s_next_base = neuralode.predict(states[i], actions[i])
    residuals_nodenn[i] = (deltas[i] - (s_next_base - states[i])) / delta_std

train_x = torch.FloatTensor(np.column_stack([states / state_std, actions.reshape(-1, 1) / action_std])).to(DEVICE)
train_y = torch.FloatTensor(residuals_nodenn).to(DEVICE)

for n_epochs in [200, 500, 1000]:
    nn = ResidualNet().to(DEVICE)
    _train_nn(nn, train_x, train_y, n_epochs=n_epochs)

    # 单步精度测试
    single_step_errs = []
    for i in range(1000):
        s = states[i]
        tau = actions[i]
        s_real = s + deltas[i]
        s_base = neuralode.predict(s, tau)
        s_norm = s / state_std
        a_norm = tau / action_std
        with torch.no_grad():
            s_t = torch.FloatTensor(s_norm).unsqueeze(0).to(DEVICE)
            a_t = torch.FloatTensor([a_norm]).unsqueeze(0).to(DEVICE)
            delta_nn = nn(s_t, a_t).cpu().numpy()[0] * delta_std
        s_hybrid = s_base + delta_nn
        single_step_errs.append(np.abs(s_hybrid[0] - s_real[0]))

    # 多步rollout
    tau_func = make_tau_func(0, 500)
    s0 = np.array([0.15, 0.0, 0.0, 0.0])

    # 纯NeuralODE rollout
    s_ode = s0.copy()
    s_hybrid = s0.copy()
    s_real = s0.copy()
    ode_500 = None
    hybrid_500 = None
    for step in range(500):
        tau = tau_func(step, s_real)
        s_real = real_step(s_real, tau)
        s_ode = neuralode.predict(s_ode, tau)

        s_norm = s_hybrid / state_std
        a_norm = tau / action_std
        with torch.no_grad():
            s_t = torch.FloatTensor(s_norm).unsqueeze(0).to(DEVICE)
            a_t = torch.FloatTensor([a_norm]).unsqueeze(0).to(DEVICE)
            delta_nn = nn(s_t, a_t).cpu().numpy()[0] * delta_std
        s_hybrid = neuralode.predict(s_hybrid, tau) + delta_nn

        if abs(s_real[0]) > math.pi / 3:
            break
        if step == 499:
            ode_500 = abs(s_ode[0] - s_real[0])
            hybrid_500 = abs(s_hybrid[0] - s_real[0])

    ode_str = f"{ode_500:.4f}" if ode_500 is not None else "N/A"
    hybrid_str = f"{hybrid_500:.4f}" if hybrid_500 is not None else "N/A"
    print(f"  epoch={n_epochs:4d}: 单步MAE={np.mean(single_step_errs):.6f}, "
          f"500步ODE误差={ode_str}, 500步ODE+NN误差={hybrid_str}")

# ============================================================
# 4. 分布偏移检测
# ============================================================
print("\n[4] 分布偏移检测...")
print("  比较rollout中NN输入的分布 vs 训练分布")

# 训练输入分布
train_inputs = np.column_stack([states / state_std, actions.reshape(-1, 1) / action_std])
train_mean = np.mean(train_inputs, axis=0)
train_std = np.std(train_inputs, axis=0)
print(f"  训练输入 mean: {train_mean}")
print(f"  训练输入 std:  {train_std}")

# NeuralODE rollout中收集输入
nn = ResidualNet().to(DEVICE)
_train_nn(nn, train_x, train_y, n_epochs=500)

rollout_inputs = []
tau_func = make_tau_func(1, 500)
s0 = np.array([0.15, 0.0, 0.0, 0.0])
s_hybrid = s0.copy()
s_real = s0.copy()
for step in range(500):
    tau = tau_func(step, s_real)
    s_real = real_step(s_real, tau)
    s_norm = s_hybrid / state_std
    a_norm = tau / action_std
    rollout_inputs.append(np.concatenate([s_norm, [a_norm]]))
    with torch.no_grad():
        s_t = torch.FloatTensor(s_norm).unsqueeze(0).to(DEVICE)
        a_t = torch.FloatTensor([a_norm]).unsqueeze(0).to(DEVICE)
        delta_nn = nn(s_t, a_t).cpu().numpy()[0] * delta_std
    s_hybrid = neuralode.predict(s_hybrid, tau) + delta_nn
    if abs(s_real[0]) > math.pi / 3:
        break

rollout_inputs = np.array(rollout_inputs)
rollout_mean = np.mean(rollout_inputs, axis=0)
rollout_std = np.std(rollout_inputs, axis=0)
print(f"  Rollout输入 mean: {rollout_mean}")
print(f"  Rollout输入 std:  {rollout_std}")
print(f"  均值偏移: {np.abs(rollout_mean - train_mean) / (train_std + 1e-8)}")
print(f"  → 大于2σ表示严重分布偏移")

# ============================================================
# 5. 残差修正的符号一致性
# ============================================================
print("\n[5] 残差修正方向一致性...")
# 检查NN预测的残差方向是否与真实残差方向一致
correct_sign = 0
total = 0
for i in range(5000):
    s = states[i]
    tau = actions[i]
    real_delta = deltas[i]
    base_delta = neuralode.predict(s, tau) - s
    real_residual = real_delta - base_delta

    s_norm = s / state_std
    a_norm = tau / action_std
    with torch.no_grad():
        s_t = torch.FloatTensor(s_norm).unsqueeze(0).to(DEVICE)
        a_t = torch.FloatTensor([a_norm]).unsqueeze(0).to(DEVICE)
        pred_residual = nn(s_t, a_t).cpu().numpy()[0] * delta_std

    correct_sign += np.sum(np.sign(real_residual) == np.sign(pred_residual))
    total += len(real_residual)

print(f"  符号一致率: {correct_sign/total*100:.1f}%")
print(f"  → 低于80%说明NN学到的残差方向不可靠")

print("\n诊断完成。")
