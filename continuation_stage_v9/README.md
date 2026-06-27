# V9: Neural ODE 优化、残差机制重构与长时域验证

## 项目概述

本项目基于 7D 自行车动力学模型，使用 Neural ODE（神经常微分方程）学习自行车的连续时间动力学，并进行系统性的残差方法比较和长时域稳定性验证。

**最终结论：纯 Neural ODE 优于所有残差增强方法。**

## 问题描述

### 背景
在自动驾驶和机器人路径跟踪中，需要一个准确的"世界模型"来预测车辆的未来状态。传统方法使用物理推导的动力学方程（如自行车模型），但这些模型存在建模误差。Neural ODE 可以从数据中学习更准确的动力学模型。

### 核心问题
1. Neural ODE 能否准确预测自行车的长期行为（H=1 到 H=1000 步）？
2. 在 Neural ODE 基础上添加"残差修正"能否进一步提升预测精度？
3. 学到的模型能否用于 MPC（模型预测控制）规划？

### 状态空间（7维）
| 状态 | 含义 | 单位 |
|------|------|------|
| e_y | 横向位置误差 | m |
| e_psi | 航向角误差 | rad |
| v | 速度 | m/s |
| theta | 曲率 | rad |
| theta_dot | 曲率变化率 | rad/s |
| delta | 前轮转角 | rad |
| delta_dot | 转角变化率 | rad/s |

控制输入：`delta_cmd`（方向盘指令）

### 数据集
- **文件**: `stage2_dataset_150k.npz`
- **规模**: 150k 样本，59 个 episode
- **控制频率**: 30Hz（dt = 0.0333s）
- **训练/测试**: 75/25 按 episode 划分

## 方法论

### V9 实验流程（7 个阶段）

```
Phase 1A: Agent I 结果复现（3 seeds）
Phase 1B: 最佳 Neural ODE 超参数搜索（6 configs × 3 seeds）
Phase 2C: 残差目标比较（15 种残差变体）
Phase 2D: 门控与集成方法（decay, state-gated, ensemble）
Phase 3E: 长时域稳定性分析（H=1~1000）
Phase 3F: MPC/MPPI 规划验证（200 候选序列）
Phase 4G: 独立审查与最终裁决
```

### Neural ODE 架构

```
dx/dt = f_θ(x, u)

输入: [x(7维), u(1维)] = 8维
隐藏层1: Linear(8, 64) + Tanh
隐藏层2: Linear(64, 64) + Tanh
隐藏层3: Linear(64, 64) + Tanh
输出层: Linear(64, 7)

参数量: 9,351
积分方法: 固定步长 RK4 (dt=0.0333s)
```

### 训练策略
- **多步 rollout 课程学习**: 从 1 步逐步增加到 20 步
- **一致性损失**: 鼓励 theta_dot 和 theta 之间的物理关系
- **Jacobian 正则化**: 平滑动力学函数
- **学习率调度**: 余弦退火
- **梯度裁剪**: max_norm=1.0
- **训练轮数**: 200 epochs

### 残差方法（15 种变体）

| 类别 | 方法 | 描述 |
|------|------|------|
| 基础 | state_residual | x_{t+1}^true - x_{t+1}^NODE |
| 基础 | derivative_residual | (x_{t+1}-x_t)/dt - f_NODE(x_t, u_t) |
| 多步 | endpoint_residual_H5/H10 | H步端点误差 |
| 分状态 | per_state_ey_epsi | 仅修正 e_y, e_psi |
| 分状态 | per_state_theta | 仅修正 theta, theta_dot |
| 分状态 | per_state_delta | 仅修正 delta, delta_dot |
| 短时 | short_time_K5/K10/K20 | 仅前K步有残差 |
| 缩放 | state_residual_scale0.05~0.3 | 不同残差权重 |
| 门控 | decay_gamma0.9~0.98 | 指数衰减残差 |
| 门控 | state_gated | 学习门控网络 |
| 集成 | ensemble_mean/median/trimmed | 5成员集成 |

## 关键结果

### 最佳模型配置

```
config3_seed43: hidden=64, depth=3, tanh, lr=1e-3
综合评分: 0.9921 (首次突破 1.0)
```

### 长时域预测性能

| Horizon | NMAE | 物理存活率 |
|---------|------|-----------|
| H=1 | 0.0051 | 100% |
| H=10 | 0.0628 | 100% |
| H=50 | 0.4557 | 100% |
| H=100 | 0.5064 | 100% |
| H=200 | 0.4737 | 100% |
| H=500 | 0.5529 | 100% |
| H=1000 | 0.6443 | 80% |

### 各状态误差分布（H=100）

| 状态 | NMAE | 说明 |
|------|------|------|
| e_y | 0.7214 | 横向位置，误差最大 |
| e_psi | 0.8883 | 航向角，误差最大 |
| v | 0.6041 | 速度，中等误差 |
| theta_dot | 0.3809 | 曲率变化率 |
| delta | 0.2855 | 转向角，误差较小 |
| delta_dot | 0.4113 | 转角变化率 |
| theta | 0.2536 | 曲率，误差最小 |

### 核心发现

1. **depth=3 显著优于 depth=2**（score 0.99 vs 1.20）
2. **残差方法无显著收益** — 15 种残差变体均未超越纯 Neural ODE
3. **门控/集成未能改善长期性能**
4. **误差不是单调增长** — H=200 附近存在自然平台
5. **MPC 可行** — H=10 预测误差 < 16%，排序相关性完美

### MPC 规划验证

| Horizon | Spearman | Kendall | Top-3 Overlap | Regret |
|---------|----------|---------|---------------|--------|
| H=10 | 1.000 | 1.000 | 1.000 | 0.000 |
| H=20 | 1.000 | 1.000 | 1.000 | 0.000 |
| H=50 | 1.000 | 1.000 | 1.000 | 0.000 |

### 运行时性能

| 指标 | 值 |
|------|-----|
| 单步推理 | < 0.03ms |
| 20步 rollout | 1.6ms |
| 50步 rollout | 4.0ms |
| 200候选规划 | 330ms |

## 局限性

1. **横向状态误差大**: e_y 和 e_psi 在 H=100+ 时 NMAE > 0.7
2. **误差波动大**: 变异系数 60-83%，难以预测误差大小
3. **H≈400 误差跳跃**: 约13秒处误差突然增大
4. **规划验证局限**: 使用同模型评估，Spearman=1.000 不代表真实环境
5. **数据集有限**: 仅 59 个 episode，可能不足以覆盖所有工况

## 下一步建议

1. **MPC 实车验证**: 用 H=10~15 规划窗口做实际路径跟踪
2. **误差感知规划**: 加入不确定性量化，不只用点预测
3. **更多数据**: 收集更多长 episode 以改善 H=500+ 预测
4. **物理约束嵌入**: 能量守恒、角度周期性等先验知识
5. **残差方法改进**: 需要更好的训练策略以避免长期误差累积

## 文件结构

```
continuation_stage_v9/
├── README.md                          # 本文档
├── FINAL_EXECUTION_OUTPUT_V9.md       # 完整实验报告
├── canonical_node/
│   ├── config_v9.py                   # 配置与超参数网格
│   ├── data_loader_v9.py              # 数据加载
│   ├── neural_ode_v9.py               # Neural ODE 模型
│   └── evaluation_v9.py               # 评估工具
├── residual_methods/
│   ├── __init__.py
│   └── residual_models.py             # 9种残差模型
├── checkpoints/
│   └── BEST_NEURAL_ODE_V9.pt          # 最佳模型权重
├── raw_results/
│   ├── PHASE_1A_REPRODUCE.json        # Agent I 复现
│   ├── PHASE_1B_NEURAL_ODE_SEARCH.json # 超参数搜索
│   ├── PHASE_2C_RESIDUAL_TARGETS.json # 残差比较
│   ├── PHASE_2D_GATING_ENSEMBLE.json  # 门控与集成
│   ├── PHASE_3E_LONG_HORIZON.json     # 长时域分析
│   ├── PHASE_3F_PLANNING.json         # 规划验证
│   ├── PHASE_4G_VERIFICATION.json     # 独立审查
│   └── ALL_RESULTS.json               # 全部结果汇总
├── run_v9_experiments.py              # 完整实验脚本
├── run_remaining.py                   # 增量实验脚本
└── generate_report.py                 # 报告生成器
```

## 复现步骤

```bash
# 1. 确保数据文件存在
ls stage2_dataset_150k.npz

# 2. 运行完整实验（约5小时）
python run_v9_experiments.py

# 3. 或运行剩余阶段（如已有Phase 1A）
python run_remaining.py

# 4. 生成报告
python generate_report.py
```

## 环境依赖

- Python 3.8+
- PyTorch >= 1.10
- NumPy
- SciPy

## 最终模型

**选择**: 纯 Neural ODE（无残差增强）

**理由**:
1. 在 H=100~1000 表现最稳定
2. 残差方法长期累积引入额外误差
3. 运行时高效（单步 < 0.03ms）
4. 物理存活率高

**交接指令**: `继续纯 Neural ODE`
