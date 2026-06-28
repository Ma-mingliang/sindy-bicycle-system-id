# V9 Neural ODE 代码审计报告

## 审计目标

对 7D 自行车模型 Neural ODE 基线（V9）及其变体（V10/V11/V12）进行全面代码审计，定位 1-500 步 rollout 预测退化的根因，并设计分层解决方案。

## 审计范围

| 模块 | 文件 | 行数 | 审计重点 |
|------|------|------|----------|
| V9 核心 | `neural_ode_v9.py` | 241 | 归一化、损失函数、训练循环 |
| V12 修复 | `neural_ode_v12.py` | 1144 | SequentialDataset、RK4、课程 |
| 数据管线 | `data_loader_v9.py` + V8 `data_loader.py` | 160 | Episode 检测、train/test 划分 |
| 评估 | `evaluation_v9.py` | 130 | NMAE、存活率、多步评估 |
| 配置 | `config_v9.py` | 43 | 物理限制、超参数 |

## 审计产物

```
audit_v9/
├── 00_inventory/
│   └── A_project_structure.md          # 项目结构与依赖
├── 01_env_state_dynamics/
│   └── B_state_normalization.md        # 状态归一化与动力学
├── 02_reward_termination/
│   └── C_data_pipeline.md              # 数据管线审计
├── 03_algorithm_network_loss/
│   └── D_algorithm_network_loss.md     # 算法与损失函数
├── 04_train_loop_data_eval/
│   └── E_train_data_eval.md            # 训练与评估管线
├── 05_interface_boundary/
│   └── F_interface_boundary.md         # 接口与边界条件
├── 06_problem_attribution/
│   ├── G_problem_attribution.md        # 问题归因
│   └── H_solution_design.md            # 解决方案设计
├── 07_final_review/
│   └── I_expert_challenge.md           # 专家挑战与反驳
└── FINAL_V9_CODE_AUDIT_README.md       # 本文件
```

## 核心发现

### 1. 四个根因

| # | 根因 | 严重度 | 状态 |
|---|------|--------|------|
| 1 | **多步损失失效**: shuffle=True 破坏时间连续性 | CRITICAL | V12 已修复 |
| 2 | **训练-评估 Gap**: 训练 max 20 步 vs 评估 500 步 | CRITICAL | V12 部分修复 |
| 3 | **归一化不匹配**: 多步 rollout 增量被缩小 ~100 倍 | HIGH | V12 已修复 |
| 4 | **常数预测崩溃**: 长 rollout 训练导致零增量输出 | HIGH | 未解决 |

### 2. V9 "存活"的真相

V9 在 H=500 保持 100% 存活率，但这**不是**模型能力：

- 归一化 bug 使增量被缩小 ~100 倍
- 预测几乎不动，自然不超出物理限制
- 修复 bug 后（V12 seq_only），存活率立即降到 0%

### 3. V12 "失败"的真相

V12 修复了 bug 但暴露了根本问题：

- 模型无法学习长 horizon 动力学
- 常数预测是长 rollout MSE 损失的全局最优解
- 收缩性正则化恢复存活率但不改善 NMAE

### 4. 实验验证

| 配置 | H=1 | H=50 | H=500 | Surv@500 |
|------|-----|------|-------|----------|
| v9_baseline | 0.007 | 0.561 | 0.875 | 100% |
| v12_seq_only | 0.030 | 0.456 | 1.146 | 0% |
| v12_ext_rollout | 0.044 | 0.674 | 0.674 | 0% |
| v12_full | 0.044 | 0.610 | 0.610 | 100% |

## 解决方案

### Tier 1: 保守方案（最小风险）

| 变更 | 描述 | 预期效果 |
|------|------|----------|
| 存活感知损失 | 物理限制违反时施加二次惩罚 | Surv@500: 0% → 40-60% |
| 课程上限 H=30 | 防止常数预测崩溃 | 消除退化 |
| 课程回归检测 | 误差增加 50% 时回退 | 防止过冲 |

**目标**: H=50 NMAE < 0.40, H=500 < 0.50, Surv@500 > 50%

### Tier 2: 稳健方案（中等变更）

| 变更 | 描述 | 预期效果 |
|------|------|----------|
| 多 horizon 采样 | 70% 基础 + 30% 随机更短 | NMAE@500 改善 5-10% |
| 增强收缩性 | lambda=0.15, warmup=10 | NMAE@500 改善 3-5% |
| 推理时状态裁剪 | 裁剪到物理限制的 95% | Surv@500: 0% → 80% |
| 早停 | 基于验证 H=1 误差 | 防止过拟合 |

**目标**: H=50 NMAE < 0.30, H=500 < 0.45, Surv@500 > 80%

### Tier 3: 激进方案（重大变更）

| 变更 | 描述 | 预期效果 |
|------|------|----------|
| 自适应积分步长 | 学习 dt_ratio ∈ [0.13, 1.0] | 高曲率区域精度提升 |
| 多尺度架构 | 快/慢动力学分离网络头 | 长 horizon 泛化改善 |
| 模型集成 | N=3 不同种子，平均预测 | 不确定性估计 |
| 存活门控课程 | 存活率 > 80% 时升级 | 防止过冲 |

**目标**: H=50 NMAE < 0.20, H=500 < 0.35, Surv@500 > 90%

## 实施优先级

```
Tier 1（从此开始）:
  变更 2（课程上限）→ 变更 1（存活损失）→ 变更 3（回归保护）

Tier 2（Tier 1 验证后）:
  变更 6（裁剪）→ 变更 5（收缩性）→ 变更 4（多 horizon）

Tier 3（Tier 2 验证后）:
  变更 8+9（自适应步长+多尺度）→ 变更 10（集成）
```

## 建议

从 **Tier 1 变更 1+2**（存活损失 + 课程上限 H=30）开始。这两个变更以最小代码改动和低风险解决两个最有影响力的问题。

## 附录：关键代码位置

| 问题 | 文件 | 行 |
|------|------|-----|
| shuffle=True | neural_ode_v9.py | 96 |
| 归一化速率计算 | neural_ode_v9.py | 92 |
| 多步 rollout（broken） | neural_ode_v9.py | 128-138 |
| 推理去归一化 | neural_ode_v9.py | 197 |
| 物理限制定义 | config_v9.py | 14-22 |
| Episode 检测 | data_loader.py | 32-44 |
| 存活率检查 | evaluation_v9.py | 35-50 |
| SequentialSegmentDataset | neural_ode_v12.py | ~100 |
| RK4 积分 | neural_ode_v12.py | ~300 |
| 收缩性正则化 | neural_ode_contractive.py | ~150 |
