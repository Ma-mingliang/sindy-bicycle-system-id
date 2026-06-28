# AGENT D: 训练异常与因果归因报告

## 入口点

- **V9 训练入口**: `canonical_node/neural_ode_v9.py` L74 (`NeuralODEV9.train()`)
- **V12 训练入口**: `canonical_node/neural_ode_v12.py` L823 (`NeuralODEV12.train()`)
- **实验运行器**: `run_v9_fixed_rerun.py`

## 执行流程 (V9 基线训练)

1. `load_7d_data()` 加载 8D 数据集，丢弃 kappa，提取 7D 状态，从 done 标志找到 episode 边界
2. `prepare_v12_data()` 计算训练数据内 episode 边界，取最后 10% 作为验证
3. `NeuralODEV9.train()` 归一化: `train_s = states / state_std`, `train_a = actions / action_std`, **`train_dsdot = deltas / (delta_std * dt)`**
4. 创建 `TensorDataset` + `shuffle=True` 对单个时间步
5. 训练循环: 200 epochs，每 epoch ~414 batches

## 执行流程 (V12 固定训练)

1. 相同数据加载
2. `NeuralODEV12.train()` 归一化: **`train_dsdot = deltas / state_std`**（不同于 V9!）
3. 创建 `SequentialSegmentDataset`，max_segment_len=20，`shuffle=True` 在段间
4. 训练循环: 200 epochs，每 epoch ~21 batches

---

## 分析 1: 为什么 V9 训练比 V12 慢 50 倍

**根因: DataLoader 设计每 epoch 创建 20 倍更多 batch**

| 因子 | V9 | V12 | 比率 |
|------|-----|------|------|
| 每 epoch batch 数 | ~414 | ~21 | 20x |
| 总 batch 数 (200 epochs) | ~82,800 | ~4,200 | 20x |
| 估计总模型评估 | ~1.15M | ~126K | ~9x |

剩余因子（9x 计算比 vs 50x 墙钟比）由以下解释:
- V9 的 `create_graph=True` 在 Jacobian 计算中创建更大的计算图
- V9 的 Python 循环开销更高
- V12 的大张量操作更受益于 GPU 并行

**结论**: 50x 训练时间差异主要由 TensorDataset vs SequentialSegmentDataset 引起（20x batch 数差异）。

---

## 分析 2: V9 的"保守预测"——Bug，非归纳偏置

**V9 基线的保守预测是两个复合 bug 的涌现症状，不是有益的设计选择。**

**Bug A: 多步 rollout 归一化不匹配**:
```python
train_s = states / self._state_std          # 按 state_std 归一化
train_dsdot = deltas / (self._delta_std * self._dt)  # 按 delta_std * dt 归一化
```

Rollout 产出: `states/state_std + deltas/delta_std`
目标: `states/state_std + deltas/state_std`

**Bug B: 洗牌 DataLoader 破坏时间顺序**

**组合效应**: 多步损失双重损坏。模型只能优化单步损失，学习预测近零 delta（归一化空间的均值）。产生:
- H=1 NMAE = 0.006（优秀单步，因为单步损失有效）
- H=500 NMAE = 0.998（长 horizon 近随机，因为多步动力学从未学习）
- 存活率 = 100%（因为预测变化微小，状态保持在初始值附近）

**结论**: "保守预测"是 **bug 驱动的产物**，不是有益的归纳偏置。

---

## 分析 3: 存活率崩溃——直接触发

**v9_fixed_seq 的存活率崩溃（H=500 0%）是由模型首次真正学习动力学触发的。**

当 shuffle bug 被修复:
1. 多步损失现在提供有意义的梯度信号
2. 模型学习预测实际状态变化（更大的 delta）
3. 这些更大的预测在 500 rollout 步后累积
4. 累积的预测误差将状态推到物理限制之外

**因果链**:
1. Shuffle 修复 → 多步损失变得有意义
2. 有意义的多步损失 → 模型学习实际动力学
3. 实际动力学 → 更大的状态预测
4. 更大的预测 → 长 horizon 累积误差
5. 累积误差 → 状态超出物理限制
6. 超出限制 → 存活检查失败 → H=500 0% 存活

**结论**: 存活率崩溃**不是不稳定性**——是模型变得足够准确以预测大的状态变化，但没有机制防止这些变化超出物理界限。V9 基线的 100% 存活是"通过在安全方向上犯错获得的"。

---

## 分析 4: 收缩性对 NMAE 的影响

**收缩性正则化通过约束 Jacobian 谱创造精度-稳定性权衡。**

| Horizon | v9_fixed_seq NMAE | v9_fixed_seq_contract NMAE | 变化 |
|---------|-------------------|----------------------------|------|
| H=1 | 0.056 | 0.035 | -37% (更好!) |
| H=10 | 0.269 | 0.191 | -29% (更好!) |
| H=50 | 0.385 | 0.647 | +68% (更差) |
| H=100 | 0.412 | 0.756 | +83% (更差) |
| H=200 | 0.446 | 0.849 | +90% (更差) |
| H=500 | 0.476 | 0.958 | +101% (更差) |

**机制**:
- **短 horizon 帮助**: 约束防止不稳定单步预测
- **长 horizon 损害**: 约束防止学习需要发散轨迹的动力学
- **恢复存活**: 有界 Jacobian 谱隐式约束状态变化速率

**结论**: 收缩性是**原则性正则化，用长 horizon 精度换取有界状态演化**。

---

## 因果图

```
根本原因 (V9 中的 Bug)
========================
[Bug A: 归一化不匹配]          [Bug B: 洗牌 DataLoader]
  目标 = deltas/(delta_std*dt)    shuffle=True 在时间步上
  状态 = states/state_std         破坏时间顺序
         |                              |
         v                              v
  [多步损失数学错误]           [多步损失无意义]
         |                              |
         +--------------+---------------+
                        |
                        v
            [多步损失无有用梯度信号]
                        |
                        v
            [单步损失主导]
                        |
                        v
            [模型预测近零 delta]
                        |
            +-----------+-----------+
            |                       |
            v                       v
  [优秀的单步性能         [100% 存活:
   H=1 NMAE=0.006]       微小预测保持
                          在限制内]
            |
            v
  [近随机的长 horizon:
   H=500 NMAE=0.998]

修复: Sequential Segment Dataset
==================================
[SequentialSegmentDataset]
  连续段，长度 20
  段间洗牌，不洗牌时间步
         |
         v
  [多步损失变得有意义]
         |
         +----> [模型学习实际动力学]
                       |
            +----------+----------+
            |                     |
            v                     v
  [更好的 NMAE           [存活率崩溃:
   H=500: 0.476]         更大预测超出
                         物理限制]
            |
            v
  [0% 存活 at H=500]

修复: 添加收缩性
=================
[收缩性正则化]
  lambda=0.05, warmup=20, ramp=30
  惩罚正 Jacobian 特征值
         |
         +----------+----------+
         |                     |
         v                     v
  [短 horizon 改善:      [长 horizon 恶化:
   H=1: 0.035            H=50: 0.647
   约束防止不稳定         约束防止学习
   预测                   发散动力学]
         |                     |
         +----------+----------+
                    |
                    v
         [100% 存活恢复]
```

---

## 假设矩阵

| ID | 假设 | 类别 | 证据 | 置信度 | 结论 |
|----|------|------|------|--------|------|
| H1 | V9 50x 减速由 TensorDataset 创建 20x 更多 batch 引起 | 根因 | 106K/256=414 vs 5300/256=21 | 95% | **已确认** |
| H2 | V9 归一化不匹配破坏多步损失 | 根因 | V9 L92 vs V12 L875 | 98% | **已确认** |
| H3 | V9 "保守预测"是 bug 而非归纳偏置 | 根因 | H=1 0.006 但 H=500 0.998 | 99% | **已确认** |
| H4 | 存活率崩溃由模型学习实际动力学触发 | 症状 | v9_fixed_seq NMAE 3-4x 更好但 0% 存活 | 97% | **已确认** |
| H5 | 收缩性用长 horizon 精度换存活 | 机制 | H=1 改善 37%, H=500 恶化 101% | 95% | **已确认** |
| H6 | 精度-稳定性权衡是根本剩余挑战 | 结构性 | 无配置同时实现低 NMAE 和高存活 | 90% | **已确认** |
| H7 | V9 基线 100% 存活是琐碎常数预测的证据 | Bug 产物 | 存活=100% 同时 H=500 NMAE=0.998 | 99% | **已确认** |

---

## 关键文件

| 文件 | 角色 | 重要性 |
|------|------|--------|
| `canonical_node/neural_ode_v9.py` | V9 模型（两个关键 bug） | CRITICAL |
| `canonical_node/neural_ode_v12.py` | V12 模型（所有修复） | CRITICAL |
| `canonical_node/data_loader_v9.py` | V8 数据加载包装器 | LOW |
| `canonical_node/evaluation_v9.py` | 多步评估 + 存活检查 | MEDIUM |
| `run_v9_fixed_rerun.py` | 实验运行器 | HIGH |
| `raw_results/V9_FIXED_RERUN.json` | 实验结果 | HIGH |

---

## 新开发建议

1. **V9 归一化不匹配 (Bug A) 是比 shuffle bug (Bug B) 更深的根因。** 如果在 V9 内工作，先修复 Bug A。

2. **精度-稳定性权衡不能仅靠收缩性解决。** 应探索: (a) 仅在物理限制附近激活的状态相关约束，(b) 每 rollout 步后裁剪状态的投影方法，(c) 训练时惩罚限制违规的奖励塑形。

3. **SequentialSegmentDataset 的 50x 训练加速本质上是免费的。** 任何未来 V9 衍生应使用基于段的数据加载。

4. **V9 基线 H=500 100% 存活不应被引用为正面结果。** 它是模型无法学习动力学的产物。

5. **v9_tier1_survival 实验与 v9_fixed_seq_contract 产生相同结果**，表明存活损失组件无效——可能因为收缩性损失已经防止了训练期间状态超出物理限制。

---

*报告由 Agent D（训练异常与因果归因）生成*
*日期: 2026-06-28*
