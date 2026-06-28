# 循环 2 总结

**RUN_ID**: 20260628_175824_neural_ode_72h
**循环时间**: 2026-06-28 19:00 - 20:00
**有效研究时间**: 1 小时

---

## 1. 新证据

### 关键发现

1. **GP 突破性发现**
   - GP 在长期预测上全面优于 Neural ODE
   - H=500: GP 0.4325 vs Neural ODE 0.5529 (改善 21.8%)
   - PrimaryLongHorizonScore: GP 0.4372 vs Neural ODE 0.5110 (改善 14.4%)
   - GP 的 NMAE 在 H=50 后趋于稳定（平台期）

2. **所有 Neural ODE 改进都失败**
   - v9_fixed: 修复后恶化
   - Direct Delta: 长期恶化
   - History Window: 长期恶化
   - RK4: 无改善
   - Physics Loss: 长期恶化
   - Deep Koopman: 不如 Neural ODE
   - Long Curriculum: 所有 horizon 恶化
   - Ensemble 残差: 长期恶化

3. **"BUG" 是正则化**
   - 原始 v9 的"BUGGY"多步损失实际上起到了正则化作用
   - 修复后模型过拟合训练数据
   - 原始 v9 基线是最佳 Neural ODE 模型

---

## 2. 新实验

| 实验 | 状态 | 结果 |
|------|------|------|
| Long Curriculum 50 | 完成 | 所有 horizon 恶化 |
| Long Curriculum 100 | 完成 | 待分析 |
| GP 优化 | 运行中 | 6 种核函数比较 |
| GP 多 seed 验证 | 运行中 | 5 个 seed 统计验证 |
| GP 消融实验 | 运行中 | 6 种配置比较 |
| GP + 物理约束 | 运行中 | 子代理实现 |
| GP OOD 测试 | 运行中 | 子代理实现 |
| Ensemble | 运行中 | 5 个模型集成 |

---

## 3. 被推翻假设

| 假设 | 状态 | 原因 |
|------|------|------|
| H1: 积分误差主导 | 已排除 | RK4 无改善 |
| H2: 分布偏移 | 已排除 | v9_fixed 反而恶化 |
| H6: 物理先验 | 已排除 | Physics Loss 反而恶化 |
| H7: Markov 性 | 已排除 | History Window 反而恶化 |
| H10: 直接离散预测 | 已排除 | Direct Delta 反而恶化 |

---

## 4. 保留假设

| 假设 | 状态 | 下一步 |
|------|------|--------|
| H3: 数据覆盖度 | 待验证 | GP 消融实验 |
| H4: 角度周期性 | 待验证 | 未测试 |
| H5: 模型容量 | 待验证 | GP 消融实验 |
| H8: 需要集成 | 待验证 | Ensemble 训练 |
| H9: 速度条件化 | 待验证 | 未测试 |

---

## 5. 淘汰算法

| 算法 | 原因 |
|------|------|
| v9_fixed | 修复后恶化 |
| Direct Delta | 长期恶化 |
| History Window | 长期恶化 |
| RK4 积分 | 无改善 |
| Physics Loss | 长期恶化 |
| Deep Koopman | 不如 Neural ODE |
| Long Curriculum | 所有 horizon 恶化 |
| Ensemble 残差 | 长期恶化 |

---

## 6. 当前最好模型

- **模型**: 纯 GP
- **PrimaryLongHorizonScore**: 0.4372
- **相对 v9 改善**: 14.4%

---

## 7. 当前置信度

**40/100**

理由:
- GP 突破性发现，长期预测改善显著
- 但还需要多 seed 验证、消融实验、OOD 测试
- 还需要独立复核和 GitHub 交付

---

## 8. 后台任务

1. Ensemble 训练 (运行中)
2. GP 优化 (运行中)
3. GP 多 seed 验证 (运行中)
4. GP 消融实验 (运行中)
5. 5 个子代理分析 (运行中)

---

## 9. 下一步

1. 等待后台任务完成
2. 分析 GP 优化结果
3. 分析 GP 多 seed 验证结果
4. 分析 GP 消融实验结果
5. 实现 GP + 物理约束
6. OOD 测试
7. 独立复核
8. GitHub 交付
