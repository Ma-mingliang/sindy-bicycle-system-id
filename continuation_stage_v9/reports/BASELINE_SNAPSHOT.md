# Baseline Snapshot

## 当前基线结果

来源: `raw_results/V9_FIXED_RERUN.json` (2026-06-28)

| 配置 | H=1 | H=10 | H=50 | H=100 | H=200 | H=500 | Surv@500 |
|------|-----|------|------|-------|-------|-------|----------|
| v9_baseline | 0.006 | 0.066 | 0.548 | 0.672 | 0.719 | 0.998 | 100% |
| v9_fixed_seq | 0.056 | 0.269 | 0.385 | 0.412 | 0.446 | 0.476 | 0% |
| v9_fixed_seq_contract | 0.035 | 0.191 | 0.647 | 0.756 | 0.849 | 0.958 | 100% |

## 关键检查点

- `checkpoints/V9_FIXED_v9_fixed_seq.pt`
- `checkpoints/V9_FIXED_v9_fixed_seq_contract.pt`
- `checkpoints/V9_FIXED_v9_tier1_survival.pt`

## 配置

- 随机种子: 43
- hidden=64, depth=3, tanh, lr=1e-3, n_epochs=200
- rollout_curriculum='1,5,10,20'
- 评估 horizons: [1, 10, 50, 100, 200, 500]
- 评估 segments: 5, 每段 1101 步

## 未提交修改

- `canonical_node/neural_ode_v12.py`: 添加了存活感知损失参数和计算（无效）
- `run_v9_fixed_rerun.py`: 新增实验脚本
- `reports/`: 新增报告目录

## 验收标准

修复成功的定义：
- H=50 NMAE < 0.548（优于 V9 baseline）
- H=500 NMAE < 0.998（优于 V9 baseline）
- Surv@500 = 100%
- H=1 NMAE < 0.020（不严重退步）
