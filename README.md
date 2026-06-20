# SINDy 无人自行车系统辨识

复现论文《基于 SINDy 与 TD-MPC2 的无人自行车系统辨识与模型强化学习研究》中的 SINDy 稀疏辨识部分。

## 方法概述

### 状态定义 (8维外环状态)

| 符号 | 含义 | 归一化 |
|------|------|--------|
| ey | 横向误差 (m) | /10 |
| eψ | 航向误差 (rad) | /1.57 |
| v | 纵向速度 (m/s) | /5 |
| θ | 横滚角 (rad) | /1.57 |
| θ̇ | 横滚角速度 (rad/s) | /10 |
| k | 参考曲率 (1/m) | ×8 |
| δ | 转向角 (rad) | /0.785 |
| δ̇ | 转向角速度 (rad/s) | /3 |

### SINDy 辨识流程

1. **数据采集**: 分层采样 (直线/曲率连续变化/急弯), 混合激励
2. **字典构造**: 二阶多项式 (常数+线性+二次, 共55项)
3. **稀疏回归**: STLSQ (阈值筛选+重估计)
4. **模型筛选**: 误差-稀疏性-稳定性折中

### 辨识目标

```
s_{t+1} = s_t + f_SINDy(s_t, a_t) + ξ_t
```

## 辨识结果

| 状态 | R² | 活跃项 | 关键物理关系 |
|------|-----|--------|-------------|
| Δey | - | 0 | (常数假设) |
| Δeψ | 0.969 | 1 | v*δ 驱动航向变化 |
| Δv | 1.000 | 0 | (常数假设) |
| Δθ | 0.966 | 1 | = θ̇ (运动学) |
| Δθ̇ | 0.860 | 1 | v*δ 驱动横滚加速度 |
| Δk | 1.000 | 0 | (常数假设) |
| Δδ | 0.998 | 6 | 横滚-转向耦合 + 速度效应 |
| Δδ̇ | 0.985 | 31 | 最复杂的转向动力学 |

**平均 R² = 0.847, 平均 RMSE = 0.033**

## 文件结构

```
sindy_bicycle/
├── bicycle_dynamics.py          # Whipple 自行车动力学模型
├── data_collector.py            # 数据采集 (分层采样)
├── sindy_identification.py      # SINDy 辨识核心 (STLSQ)
├── world_model.py               # 世界模型类 (rollout + 不确定性)
├── realistic_bicycle_params.py  # 开源自行车参数 (Moore et al.)
├── visualization.py             # 可视化工具
├── main.py                      # 主流程
├── run_experiment.py            # 改进实验
├── requirements.txt             # 依赖
├── summary.txt                  # 完整实验报告
└── README.md                    # 本文件
```

## 运行

```bash
cd sindy_bicycle
pip install -r requirements.txt
python main.py          # 基础流程
python run_experiment.py  # 改进实验 (更大数据量, 网格搜索)
```

## 输出

- `bicycle_data_improved.npz` - 仿真数据 (2083 样本)
- `sindy_model_improved.npz` - SINDy 模型系数
- `fig_*.png` - 可视化图表
- `summary.txt` - 完整实验报告

## 开源数据资源

| 资源 | 说明 |
|------|------|
| [BicycleParameters](https://github.com/moorepants/BicycleParameters) | 实测自行车物理参数 |
| [bicycle](https://github.com/moorepants/bicycle) | Whipple 模型方程推导 |
| [bicycle-data](https://github.com/moorepants/bicycle-data) | 实验测量数据集 |
| [PyDy](https://pydy.org) | 多体动力学仿真框架 |

## 参考文献

1. Brunton S L, et al. "Discovering governing equations from data by sparse identification of nonlinear dynamical systems." PNAS, 2016.
2. Hansen N, et al. "TD-MPC2: Scalable, Robust World Models for Continuous Control." ICLR, 2024.
3. Moore J K. "Human Control of a Bicycle." PhD dissertation, UC Davis, 2012.
