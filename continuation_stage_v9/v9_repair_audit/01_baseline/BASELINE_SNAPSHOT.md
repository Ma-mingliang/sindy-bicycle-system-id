# V9 基线快照

## 环境信息
- **项目目录**: `D:/系统辨识作业/sindy_bicycle/continuation_stage_v9`
- **Git 分支**: main
- **最新 commit**: b9cffe8 (docs: 闭环修复报告)
- **工作区状态**: 有未提交修改（canonical_node/__init__.py, 3 个 json 文件）
- **Python**: E:/Anaconda/python.exe
- **GPU**: NVIDIA GeForce RTX 5060 Laptop GPU (CUDA available)
- **随机种子**: 43

## V9 配置
```python
NeuralODEConfig(
    hidden=64, depth=3, activation='tanh', lr=1e-3,
    weight_decay=0.0, n_epochs=200, batch_size=256,
    dt=1/30, rollout_curriculum='1,5,10,20',
    lambda_multi=0.3, lambda_consistency=0.1,
    lambda_jacobian=0.01, seed=43,
)
```

## 评估配置
- Horizons: [1, 10, 50, 100, 200, 500]
- 评估段数: 5，每段 1101 步
- 存活模式: physical（检查物理限制）
- 物理限制: e_y=5, e_psi=π, v=5, delta=π/2, delta_dot=10

## 关键文件
| 文件 | 作用 |
|------|------|
| `canonical_node/neural_ode_v9.py` | V9 原始模型（未修改） |
| `canonical_node/neural_ode_v12.py` | V12 修复模型（已修改：添加存活损失） |
| `canonical_node/evaluation_v9.py` | 多步评估函数 |
| `canonical_node/config_v9.py` | 配置常量 |
| `canonical_node/data_loader_v9.py` | 数据加载 |
| `run_v9_fixed_rerun.py` | V9 修复验证实验脚本 |
| `run_repair_experiments.py` | 修复实验脚本 |
| `run_hybrid_experiments.py` | 混合实验脚本 |

## 检查点
| 检查点 | 状态 |
|--------|------|
| `checkpoints/V9_FIXED_v9_fixed_seq.pt` | 最佳 NMAE，0% 存活 |
| `checkpoints/V9_FIXED_v9_fixed_seq_contract.pt` | 100% 存活，中等 NMAE |
| `checkpoints/REPAIR_soft_contract.pt` | 软收缩性 |

## 基线结果
来源: `raw_results/V9_FIXED_RERUN.json` (2026-06-28)
| 配置 | H=1 | H=10 | H=50 | H=100 | H=200 | H=500 | Surv@500 |
|------|-----|------|------|-------|-------|-------|----------|
| v9_baseline | 0.006 | 0.066 | 0.548 | 0.672 | 0.719 | 0.998 | 100% |
| v9_fixed_seq | 0.056 | 0.269 | 0.385 | 0.412 | 0.446 | 0.476 | 0% |
| v9_fixed_seq_contract | 0.035 | 0.191 | 0.647 | 0.756 | 0.849 | 0.958 | 100% |
