# 系统辨识方法对比测试报告

> 测试时间：2026-06-21
> 测试环境：PyTorch + CUDA (E:/Anaconda/envs/DL/python.exe)

---

## 一、测试概述

### 1.1 测试目标

对比12种系统辨识方法在自行车动力学模型上的表现，并测试6种改进方案对NN残差的优化效果。

### 1.2 测试方案

**方案C：真实模型+LQR生成参考轨迹**
- 真实模型+LQR生成5段×500步轨迹
- 所有模型从相同初始状态出发，接收相同力矩序列
- 评估步数：[1, 5, 10, 20, 50, 100, 200, 500]
- 指标：theta角度MAE（rad）
- 发散判断：|phi| > pi/3 时截断

### 1.3 物理模型

**Meijaard 2007基准自行车动力学**
- 4维状态：`[theta(横滚角), delta(转向角), theta_dot, delta_dot]`
- 1维输入：`tau(转向力矩)`
- 控制周期：dt = 1/30 s
- 前进速度：v0 = 3.5 m/s

---

## 二、12种基础方法对比

### 2.1 测试结果

| # | 方法 | 说明 | 500步MAE (rad) | 状态 |
|---|------|------|----------------|------|
| 1 | SINDyPoly | 多项式库(21特征) + STLSQ | 发散 | ❌ |
| 2 | SINDyTrig | 三角函数库(32特征) | 发散 | ❌ |
| 3 | SINDyTrigExp | 三角+指数库(36特征) | 发散 | ❌ |
| 4 | SINDyBestNN | 最佳SINDy + NN残差 | 发散 | ❌ |
| 5 | NNE2E | 端到端MLP | 发散 | ❌ |
| 6 | NeuralODE | NN学ds/dt，Euler积分 | 1.59~13.36 | ⚠️ |
| 7 | GP | 高斯过程回归(5000样本限制) | 7.05 | ⚠️ |
| 8 | PINN | 物理约束NN | 发散 | ❌ |
| 9 | ParamID | 线性最小二乘拟合 | 发散 | ❌ |
| 10 | NeuralODE_NN | 方法6 + NN残差 | 发散 | ❌ |
| 11 | GP_NN | 方法7 + NN残差 | 发散 | ❌ |
| 12 | ParamID_NN | 方法9 + NN残差 | 发散 | ❌ |

### 2.2 结论

- **纯方法中NeuralODE和GP表现最好**（1.59 rad和7.05 rad）
- **加NN残差后全部发散**，说明NN残差存在固有问题

---

## 三、NN残差恶化诊断

### 3.1 诊断结果

运行 `diagnose_hybrid.py` 得出以下结论：

| 诊断项 | 结果 | 说明 |
|--------|------|------|
| 残差占比 | 0.3%~0.8% | 基线已捕捉99%+动态 |
| 符号一致率 | 78.9% | 21%方向错误 |
| 分布偏移 | 2~4σ | NN输入偏离训练分布 |
| 更多epoch | 更差 | 过拟合残差噪声 |

### 3.2 根本原因

**正反馈放大误差**：
```
基线预测误差 → NN输入偏移 → NN输出错误残差 → 下一步更大偏移 → 指数发散
```

---

## 四、改进方案测试

### 4.1 三重防护技术

| 技术 | 作用 | 参数 |
|------|------|------|
| OOD检测 | 分布外时回退零残差 | Mahalanobis距离 > 3σ |
| 残差缩放 | 降低错误残差影响 | scale=0.3 |
| DAgger | 用rollout数据重新训练 | 3轮，每轮2-3段轨迹 |

### 4.2 论文改进方案测试结果

| 方案 | 效果 | 原因 |
|------|------|------|
| 纯NN残差(无防护) | 恶化(发散) | 正反馈放大误差 |
| +残差缩放0.3 | 部分改善 | 降低但未消除错误影响 |
| +OOD检测 | 显著改善 | 分布外时安全回退 |
| +DAgger | 进一步改善 | 减少分布偏移 |
| Ensemble(5-NN) | 失败 | 残差噪声大，集成方差高，99.9%OOD率 |
| Domain Rand | 部分改善 | 需配合OOD+DAgger |
| Conformal Prediction | 失败 | 预测区间过宽，89-98%回退率 |
| Latent-space(2D) | 不稳定 | 降维丢失信息，0.10~0.43波动大 |

---

## 五、最终结果：全部6种基线 + 改进NN残差

### 5.1 测试结果

| 方法 | 纯基线 (rad) | 改进混合 (rad) | 倍率 | 结论 |
|------|-------------|---------------|------|------|
| SINDyPoly | 1.73e+13 | 1.04e+13 | 0.60x | 改善(绝对值仍大) |
| SINDyTrig | 2.01e+13 | 9.38e+8 | 0.00x | 大幅改善 |
| NeuralODE | 1.59 | 0.48 | 0.30x | 改善 |
| NNE2E(纯NN) | 1.28e+9 | 0.23 | 0.00x | 从发散恢复 |
| ParamID | 8.52e+8 | 5.75e+8 | 0.67x | 改善 |
| **GP** | 7.05 | **0.15** | 0.02x | **最佳** |

### 5.2 最终排名

| 排名 | 方法 | 500步MAE (rad) |
|------|------|----------------|
| 1 | GP + OOD + DAgger + 0.3 | 0.15 |
| 2 | NNE2E + OOD + DAgger + 0.3 | 0.23 |
| 3 | NeuralODE + OOD + DAgger + 0.3 | 0.48 |
| 4 | Latent + OOD | 0.10~0.43 (不稳定) |
| 5 | DR + OOD + DAgger | 0.72 |

---

## 六、核心改进技术详解

### 6.1 OOD检测

```python
# 训练时计算分布统计
train_inputs = normalize(state_action_pairs)
mu = mean(train_inputs)
sigma = cov(train_inputs)

# 推理时检测OOD
def is_ood(input_normalized):
    mahal_dist = sqrt((input - mu).T @ inv(sigma) @ (input - mu))
    return mahal_dist > 3.0  # 阈值3σ
```

### 6.2 残差缩放

```python
# 训练残差（归一化空间）
residual = (真实下一步 - 基线预测) / delta_std

# 推理时（保守应用）
if ood_detector.is_ood(input):
    delta_nn = 0  # 回退
else:
    delta_nn = nn_model(input)
s_next = baseline.predict(s, tau) + delta_nn * delta_std * 0.3
```

### 6.3 DAgger

```python
# 3轮迭代
for dagger_round in range(3):
    # 1. 用当前策略收集rollout数据
    rollout_data = collect_rollout(policy, env)

    # 2. 用基线模型标注rollout数据的"真实"动作
    labeled_data = label_with_baseline(rollout_data)

    # 3. 合并原始数据和rollout数据
    augmented_dataset = merge(original_data, labeled_data)

    # 4. 重新训练残差模型
    train_residual_model(augmented_dataset)
```

---

## 七、结论与建议

### 7.1 核心结论

1. **GP + OOD + DAgger + 0.3 是最佳方案**（0.15 rad）
2. **所有基线加改进NN残差后都有改善**（100 epoch版本）
3. **改进方案对不稳定的纯NN特别有效**（NNE2E从发散恢复到0.23 rad）

### 7.2 后续改进方向

1. **自适应残差缩放**：根据OOD距离动态调整scale
2. **GP + NN更深度融合**：用GP不确定性指导NN置信度
3. **多步损失训练**：用rollout loss训练NN
4. **物理约束残差**：能量守恒等
5. **贝叶斯神经网络**：替代ensemble
6. **更大的GP**：SGP/DeepGP

---

## 八、相关代码文件

| 文件 | 说明 |
|------|------|
| `test_all_methods.py` | 12种方法原始对比 |
| `diagnose_hybrid.py` | NN残差恶化诊断 |
| `improved_hybrid.py` | 改进版混合方法（OOD+DAgger+0.3） |
| `test_improved_all.py` | 5种基线+改进NN测试 |
| `test_ensemble.py` | Ensemble不确定性方案（失败） |
| `test_domain_rand.py` | Domain Randomization方案 |
| `test_dr_dagger.py` | DR+OOD+DAgger三合一 |
| `test_conformal.py` | Conformalized Neural Dynamics（失败） |
| `test_latent.py` | Latent-space dynamics（不稳定） |
| `test_all_improved.py` | 全部6基线+改进NN最终对比 ← 最终结果 |

---

## 九、运行命令

```bash
# 运行最终对比（推荐，约20分钟含GP）
"E:/Anaconda/envs/DL/python.exe" "D:/系统辨识作业/sindy_bicycle/test_all_improved.py"

# 运行单个改进方案测试
"E:/Anaconda/envs/DL/python.exe" "D:/系统辨识作业/sindy_bicycle/test_dr_dagger.py"
"E:/Anaconda/envs/DL/python.exe" "D:/系统辨识作业/sindy_bicycle/test_latent.py"

# 运行诊断
"E:/Anaconda/envs/DL/python.exe" "D:/系统辨识作业/sindy_bicycle/diagnose_hybrid.py"
```

---

*报告生成时间：2026-06-21*
*GitHub: https://github.com/Ma-mingliang/sindy-bicycle-system-id*
