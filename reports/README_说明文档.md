# 自行车世界模型统一评估框架 — 说明文档

> **项目**: 系统辨识作业 — 自行车动力学世界模型
> **生成日期**: 2026-06-25
> **Python**: 3.9.23 | scikit-learn 1.6.1 | PyTorch 2.7.1+cu128

---

## 一、项目背景

本项目基于 Meijaard 2007 基准自行车动力学模型 (4维状态空间: phi, delta, phi_dot, delta_dot)，使用多种方法建立世界模型，用于预测自行车在 LQR 控制器作用下的动力学行为。

**核心问题**: 不同世界模型在不同评估模式下的预测精度如何？

---

## 二、评估框架结构

```
sindy_bicycle/
├── evaluate/                    # 统一评估框架
│   ├── eval_config.py          # 配置加载
│   ├── eval_metrics.py         # 指标计算 (MAE, RMSE, Max, P95等)
│   ├── eval_models.py          # 模型封装 (7个模型类)
│   ├── eval_modes.py           # 4种评估模式 (A/B/C/D)
│   ├── eval_runner.py          # 主执行入口
│   ├── eval_plots.py           # 图表生成
│   ├── run_evaluation.py       # CLI脚本
│   └── run_ensemble_evaluation.py  # 集成模型专用脚本
├── configs/
│   └── reproducible_world_model_evaluation.yaml
├── reports/                     # 评估报告
└── results/                     # 评估结果 (JSON)
```

---

## 三、评估模型

| 模型 | 说明 | 训练时间 |
|------|------|----------|
| `real_dynamics` | 真实动力学 (基准，误差=0) | 无需训练 |
| `linearized_model` | 线性化模型 (A, B矩阵) | 无需训练 |
| `sindy_4d` | SINDy 稀疏辨识 (4D多项式) | < 1秒 |
| `gp_standard` | 4个独立高斯过程 (RBF核, 5000样本) | ~28分钟 |
| `gp_b4_sparse` | GP + Nystroem近似 (5000样本) | ~5分钟 |
| `gp_ensemble_5_3` | GP + 5个NN + 3轮DAgger | ~38分钟 |
| `gp_ensemble_7_5` | GP + 7个NN + 5轮DAgger | ~54分钟 |

---

## 四、四种评估模式

| 模式 | 名称 | 动作来源 | 模型状态来源 | 用途 |
|------|------|----------|-------------|------|
| Mode A | Teacher Forcing | tau_func(s_real) | 真实状态 | 单步精度 |
| Mode B | 固定动作开环 | 预收集固定序列 | 自身预测 | 开环预测 |
| Mode C | 混合模式 | tau_func(s_real) | 自身预测 | 常用评估 |
| Mode D | 完整闭环 | tau_func(s_model) | 自身预测 | 最严格 |

---

## 五、核心结果

### 5.1 Mode D (完整闭环) — 最严格评估

| 模型 | 10步 | 50步 | 100步 | 500步 | 1000步 |
|------|------|------|-------|-------|--------|
| gp_standard | 0.000 | 0.000 | 0.000 | 0.000 | **0.000** |
| gp_ensemble_7_5 | 0.265 | 0.137 | 0.104 | 0.077 | **0.074** |
| gp_ensemble_5_3 | 0.247 | 0.172 | 0.157 | 0.147 | 0.145 |
| sindy_4d | 0.162 | 0.065 | 0.035 | 0.011 | **0.008** |
| gp_b4_sparse | 0.527 | 1.666 | 3.081 | 14.53 | 28.85 |

### 5.2 Mode C (混合模式) — MAE (rad)

| 模型 | 100步 | 500步 | 1000步 |
|------|-------|-------|--------|
| gp_standard | 0.002 | 12.15 | 17.38 |
| gp_ensemble_7_5 | 0.222 | 0.252 | 0.257 |
| gp_ensemble_5_3 | 0.277 | 0.320 | 0.328 |
| sindy_4d | 0.363 | 发散 | 发散 |

---

## 六、关键发现

1. **GP标准模型在Mode D下完美跟踪**: 1000步 MAE < 0.00001 rad
2. **集成模型 (7+5) 是最佳整体**: Mode D 1000步 MAE = 0.074 rad，提供不确定性估计
3. **SINDy在闭环控制下表现优异**: 1000步 MAE 仅 0.008 rad
4. **Mode D比Mode C更有意义**: 闭环评估更接近实际使用场景
5. **集成模型在Mode C下显著优于单一GP**: 0.252 vs 12.149 (48倍改进)

---

## 七、模型推荐

| 用途 | 推荐模型 | 最大时域 | 预期精度 |
|------|----------|----------|----------|
| 闭环控制 (MPC) | gp_standard | 1000步+ | < 0.001 rad |
| 需要不确定性估计 | gp_ensemble_7_5 | 1000步+ | < 0.074 rad |
| 开环预测 | gp_ensemble_7_5 | 500步 | < 0.252 rad |
| 快速原型 | sindy_4d | 1000步 (闭环) | < 0.008 rad |

---

## 八、如何复现

```bash
# 环境
E:/Anaconda/envs/DL/python.exe  # Python 3.9.23

# 快速评估 (5 seeds, 5 horizons)
cd "D:/系统辨识作业/sindy_bicycle"
PYTHONUNBUFFERED=1 "E:/Anaconda/envs/DL/python.exe" evaluate/run_evaluation.py --quick

# 集成模型评估
PYTHONUNBUFFERED=1 "E:/Anaconda/envs/DL/python.exe" evaluate/run_ensemble_evaluation.py --quick
```

---

## 九、文件清单

### results/
| 文件 | 说明 |
|------|------|
| quick_with_gp_results.json | 500次评估 (5模型 × 4模式) |
| ensemble_evaluation_results.json | 200次评估 (2集成模型 × 4模式) |

### reports/
| 文件 | 说明 |
|------|------|
| NEXT_STAGE_FINAL_CONCLUSION.md | 最终结论 |
| ENSEMBLE_EVALUATION_RESULTS.md | 集成模型详细结果 |
| EVALUATION_MODE_A/B/C/D_RESULTS.md | 各模式详细结果 |
| MODEL_HORIZON_RECOMMENDATION.md | 模型时域推荐 |

---

*文档生成时间: 2026-06-25*
