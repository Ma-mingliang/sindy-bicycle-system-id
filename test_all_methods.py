"""12种系统辨识方法对比测试 — 主入口。

评估方案C：真实模型+LQR生成参考轨迹，所有模型接收相同动作序列。

运行：
    "E:/Anaconda/envs/DL/python.exe" "D:/系统辨识作业/sindy_bicycle/test_all_methods.py"
"""

import numpy as np
import math
import time
from methods_common import generate_training_data
from methods_sindy import SINDyPoly, SINDyTrig, SINDyTrigExp
from methods_nn import SINDyBestNN, NNE2E, NeuralODEMethod
from methods_classic import GPMethod, PINNMethod, ParamIDMethod
from methods_hybrid import NeuralODE_NN, GP_NN, ParamID_NN
from methods_evaluate import make_tau_func, run_trajectory, compute_rmse_at_steps, print_results_table

# ============================================================
# 配置
# ============================================================
N_SEGMENTS = 5
N_STEPS = 500
EVAL_STEPS = [1, 5, 10, 20, 50, 100, 200, 500]

# 所有12种方法（注释掉耗时较长的方法可加速调试）
ALL_METHODS = [
    SINDyPoly(),
    SINDyTrig(),
    SINDyTrigExp(),
    SINDyBestNN(),
    NNE2E(),
    NeuralODEMethod(),
    GPMethod(),
    PINNMethod(),
    ParamIDMethod(),
    NeuralODE_NN(),
    GP_NN(),
    ParamID_NN(),
]

# ============================================================
# 1. 生成训练数据
# ============================================================
print("=" * 70)
print("12种系统辨识方法对比测试")
print("=" * 70)
print(f"\n[1/4] 生成训练数据...")
t0 = time.time()
states, actions, deltas, state_std, action_std, delta_std = generate_training_data(n_samples=30000)
print(f"  样本数: {len(states)}, 耗时: {time.time()-t0:.1f}s")

# ============================================================
# 2. 训练所有方法
# ============================================================
print(f"\n[2/4] 训练所有方法...")
for i, method in enumerate(ALL_METHODS):
    t0 = time.time()
    print(f"\n  [{i+1}/{len(ALL_METHODS)}] 训练 {method.name}...")
    method.train(states, actions, deltas, state_std, action_std, delta_std)
    print(f"    完成，耗时: {time.time()-t0:.1f}s")

# ============================================================
# 3. 运行轨迹测试
# ============================================================
print(f"\n[3/4] 运行{N_SEGMENTS}段×{N_STEPS}步轨迹测试...")
all_trajs = []
for seg_i in range(N_SEGMENTS):
    t0 = time.time()
    phi_init = np.random.uniform(-0.25, 0.25)
    s0 = np.array([phi_init, 0.0, 0.0, 0.0])
    tau_func = make_tau_func(seg_i, N_STEPS)
    trajs = run_trajectory(s0, N_STEPS, tau_func, ALL_METHODS)
    all_trajs.append(trajs)
    valid = np.sum(~np.isnan(trajs['real'][:, 0])) - 1
    print(f"  段{seg_i+1}: phi_init={math.degrees(phi_init):+.1f}°, 有效步数={valid}, 耗时={time.time()-t0:.1f}s")

# ============================================================
# 4. 计算RMSE并输出结果
# ============================================================
print(f"\n[4/4] 计算RMSE...")
method_names = [m.name for m in ALL_METHODS]
rmse_results = compute_rmse_at_steps(all_trajs, EVAL_STEPS, method_names)

print(f"\n{'='*70}")
print("RMSE表（theta角度，单位：rad）")
print("=" * 70)
print_results_table(rmse_results, EVAL_STEPS, method_names)

print(f"\n{'='*70}")
print("测试完成。")
print("=" * 70)
