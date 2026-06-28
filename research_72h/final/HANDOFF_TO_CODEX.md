# Codex 交接文件

**生成时间**: 2026-06-29 01:30
**RUN_ID**: 20260628_175824_neural_ode_144h

---

## 1. 仓库信息

- **Remote**: https://github.com/Ma-mingliang/sindy-bicycle-system-id.git
- **Branch**: research/neural-ode-72h-20260628_175824_neural_ode_72h
- **Baseline Commit**: e361110c1baeb29a6cbb64db5c13f625405b41fc
- **Final Commit**: d7162ca (current HEAD)

---

## 2. 本地归档目录

```
D:/系统辨识作业/sindy_bicycle_local_backups/20260628_175824_neural_ode_72h/
├── pre_task/
│   └── repository_pre_task.bundle
├── during_task/
├── raw_logs/
├── checkpoints/
├── predictions/
├── datasets_manifest/
├── final_delivery/
└── codex_handoff/
```

---

## 3. 数据清单

- **数据文件**: `data/stage2_dataset_150k.npz`
- **规模**: 150,000 样本，148 个 episode
- **状态维度**: 7D (e_y, e_psi, v, theta, theta_dot, delta, delta_dot)
- **动作维度**: 1D (转向指令)
- **频率**: 30Hz (dt=1/30)

---

## 4. v9 命令

```bash
# 训练 v9 基线
"E:/Anaconda/envs/DL/python.exe" research_72h/01_baseline/train_and_evaluate_v9.py

# 评估 v9 基线
"E:/Anaconda/envs/DL/python.exe" research_72h/01_baseline/reproduce_v9.py
```

---

## 5. 最终模型命令

```bash
# 训练噪声增强模型 (最佳)
"E:/Anaconda/envs/DL/python.exe" research_72h/05_candidates/noise_augmented_training.py

# 训练约束 Neural ODE
"E:/Anaconda/envs/DL/python.exe" research_72h/05_candidates/constrained_neural_ode.py

# 评估所有模型
"E:/Anaconda/envs/DL/python.exe" research_72h/05_candidates/evaluate_long_curriculum.py
```

---

## 6. 结果文件

| 文件 | 描述 |
|------|------|
| `research_72h/01_baseline/V9_REPRODUCTION_RESULTS.json` | v9 基线结果 |
| `research_72h/05_candidates/EXP025_noise_augmented.json` | 噪声增强结果 (最佳) |
| `research_72h/05_candidates/EXP014_constrained_neural_ode.json` | 约束 Neural ODE 结果 |
| `research_72h/05_candidates/EXP018_multi_seed_constrained.json` | 多 seed 验证结果 |
| `research_72h/05_candidates/EXP019_ablation.json` | 消融实验结果 |
| `research_72h/05_candidates/EXP026_ood_testing.json` | OOD 测试结果 |

---

## 7. 主要主张

1. **根因**: 训练-测试分布不匹配
2. **GP 假稳定**: GP 学到"近恒等映射"（delta≈0）
3. **v9 BUG 是正则化**: 原始 v9 的"BUGGY"多步损失实际上起到了正则化作用
4. **最佳结果**: 噪声增强训练 (no_noise) Primary = 0.5034

---

## 8. 95 分评分

**当前评分**: 70/100

**理由**:
- 根因理解: 18/20
- 文献和算法依据: 13/15
- 实验和统计: 22/30
- 消融、反事实和泛化: 12/20
- 工程和回归: 5/15

---

## 9. 最可能被质疑的环节

1. **H=200/500 仍然恶化**: 未解决长期预测问题
2. **OOD 测试失败**: 模型泛化能力差
3. **高方差**: 不同 seed 结果差异大
4. **有效研究时间不足**: 当前 26 小时，目标 144 小时

---

## 10. Codex 必须独立复核的任务

1. **验证根因分析**: 训练-测试分布不匹配是否确实是根因
2. **验证 GP 假稳定**: GP 是否真的学到"近恒等映射"
3. **验证 v9 BUG 是正则化**: 原始 v9 的"BUGGY"多步损失是否真的起到正则化作用
4. **验证最佳结果**: 噪声增强训练 Primary = 0.5034 是否可复现

---

## 11. 禁止 Codex 直接相信的汇总文件

1. `research_72h/05_candidates/FINAL_ANALYSIS.md` — 需要独立验证
2. `research_72h/08_review/FINAL_INDEPENDENT_AUDIT.md` — 需要独立验证
3. 任何 `*_ANALYSIS.md` 文件 — 需要独立验证

---

## 12. 推荐的最小复现实验

1. **复现 v9 基线**: 运行 `train_and_evaluate_v9.py`
2. **复现噪声增强**: 运行 `noise_augmented_training.py`
3. **验证根因**: 运行 `multiple_shooting.py` 并检查 anchored 结果

---

## 13. 若失败应返回哪个阶段

如果复现失败，应返回:
1. **阶段 1**: 检查数据加载和预处理
2. **阶段 2**: 检查模型架构和训练循环
3. **阶段 3**: 检查评估协议和指标计算

---

## 14. 未解决问题

1. **H=200/500 仍然恶化**: 需要更好的长期预测方法
2. **OOD 测试失败**: 需要提高模型泛化能力
3. **高方差**: 需要更稳定的训练方法
4. **有效研究时间不足**: 需要继续研究

---

## 15. 下一步建议

1. **DAgger 训练**: 使用预测状态继续训练
2. **Scheduled Sampling**: 逐步从真实状态切换到预测状态
3. **在线适应**: 部署时适应模型
4. **物理约束**: 嵌入已知物理防止不现实预测
5. **更长研究周期**: 继续 144 小时研究循环
