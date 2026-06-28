# 实验计划（第 17 章）

## 实验 1: 多种子验证（验证可复现性）

### 配置
- 模型: v9_fixed_seq (SequentialSegmentDataset + 正确归一化)
- 种子: [43, 44, 45]
- 评估: 标准 horizons

### 目的
- 验证现有结果是否在合理误差范围内可复现
- 量化种子对结果的影响

### 预期
- H=500 NMAE 标准差 < 0.05
- 存活率一致（0% 或 100%）

---

## 实验 2: 状态相关收缩性（方案 S5）

### 配置
```python
# 在 V12Config 基础上修改
lambda_contract_state_dependent = True
contract_margin = 0.6  # 物理限制的 60%
contract_scale = 0.1   # sigmoid 温度
```

### 实现步骤
1. 修改 `_contractivity_loss` 为状态相关版本
2. 添加 sigmoid 权重: `w = sigmoid((|s_norm| - margin) / scale)`
3. 收缩损失 = w * relu(max_eigenvalue)

### 评估
- 标准 horizons + 存活率
- 与 v9_fixed_seq_contract 对比

---

## 实验 3: 投影式约束（方案 S7）

### 配置
```python
# 训练时在 rollout 每步后投影
projection_enabled = True
projection_limits = PHYSICAL_LIMITS  # 使用标准物理限制
```

### 实现步骤
1. 在 multi-step rollout 循环中添加投影步骤
2. `s_cur = torch.clamp(s_cur, -limits_norm, limits_norm)`
3. 确保梯度可以通过 clamp 流动

### 评估
- 标准 horizons + 存活率
- 与无投影版本对比

---

## 实验 4: 扩展课程（方案 S3）

### 配置
```python
rollout_curriculum = '1,5,10,20,50,100'
n_epochs = 400
```

### 评估
- 标准 horizons + 存活率
- 训练曲线分析

---

## 实验优先级

| 优先级 | 实验 | 预计时间 | 依赖 |
|--------|------|----------|------|
| 1 | 多种子验证 | 10 min | 无 |
| 2 | 状态相关收缩性 | 20 min | 实验 1 |
| 3 | 投影式约束 | 20 min | 实验 1 |
| 4 | 扩展课程 | 30 min | 实验 1 |

---

## 验收标准

| 标准 | 目标 | 当前最优 |
|------|------|----------|
| H=50 NMAE | < 0.548 | 0.637 (safety_blend) |
| H=500 NMAE | < 0.998 | 0.946 (safety_blend) |
| Surv@500 | 100% | 100% (safety_blend) |
| H=1 NMAE | < 0.020 | 0.035 (safety_blend) |

---

*日期: 2026-06-28*
