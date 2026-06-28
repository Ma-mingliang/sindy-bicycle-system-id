# A: 项目结构与依赖审计

## 1. 文件清单

### V9 核心包: `canonical_node/`

| 文件 | 行数 | 角色 | 关键类/函数 |
|------|------|------|-------------|
| `config_v9.py` | 43 | 中央配置常量 | `STATE_NAMES_7D`, `STATE_DIM=7`, `ACTION_DIM=1`, `IDX_7D_FROM_8D`, `DATA_PATH`, `ROLLOUT_HORIZONS`, `PHYSICAL_LIMITS` |
| `data_loader_v9.py` | 19 | V8 数据加载器的薄封装 | `load_7d_data()`, `get_test_segments()` |
| `neural_ode_v9.py` | 241 | V9 核心 Neural ODE 基线 | `NeuralODEConfig`, `ODEFunc`, `NeuralODEV9` |
| `evaluation_v9.py` | 130 | 评估指标 | `compute_nmae()`, `check_survival()`, `compute_survival()`, `multi_step_evaluate()` |
| `neural_ode_v12.py` | 1144 | V12 根因修复版 | `V12Config`, `SequentialSegmentDataset`, `NeuralODEV12` |
| `neural_ode_contractive.py` | 425 | V11 收缩性正则化 | `ContractiveODEConfig`, `compute_jacobian()`, `contractivity_loss()` |
| `neural_ode_physics.py` | 281 | V10 物理约束损失 | `PhysicsODEConfig`, `PhysicsNeuralODE` |
| `neural_ode_feedback.py` | 307 | V10 反馈校正 | `FeedbackODEConfig`, `FeedbackNeuralODE` |
| `neural_ode_attractor.py` | 622 | V10 吸引子稳定 | `AttractorODEConfig`, `AttractorNeuralODE` |
| `neural_ode_adaptive_curriculum.py` | 432 | V10 自适应课程 | `AdaptiveCurriculumConfig`, `AdaptiveCurriculumNeuralODE` |
| `neural_ode_freq_curriculum.py` | 379 | V10 频域课程 | `FreqCurriculumConfig`, `FreqCurriculumNeuralODE` |
| `__init__.py` | 12 | 包初始化，重导出 | 重导出所有公共符号（缺少 `neural_ode_physics`） |

### V9 残差方法: `residual_methods/`

| 文件 | 行数 | 角色 |
|------|------|------|
| `residual_models.py` | 559 | 9 种残差校正方法 |

### V9 运行脚本（根目录）

| 文件 | 行数 | 角色 |
|------|------|------|
| `run_v9_experiments.py` | 913 | 主实验管线 Phase 1A-4G |
| `run_remaining.py` | 710 | 增量实验（崩溃恢复） |
| `run_v10_physics.py` | ~180 | V10 物理约束实验 |
| `run_v10_curriculum.py` | ~150 | V10 扩展课程实验 |
| `run_v10_feedback.py` | ~160 | V10 反馈校正实验 |
| `run_v10_attractor.py` | ~295 | V10 吸引子稳定实验 |
| `run_adaptive_curriculum.py` | ~270 | V10 自适应+频域课程实验 |
| `run_v11_contractive.py` | ~255 | V11 收缩性正则化实验 |
| `run_v12_rootfix.py` | 484 | V12 根因修复消融实验 |
| `generate_report.py` | 358 | 从 JSON 生成报告 |

## 2. 依赖图

```
config_v9.py  <-- (无内部依赖，叶模块)
    │
    ▼
data_loader_v9.py ──► V8/canonical_7d/data_loader.py
    │                  V8/canonical_7d/config.py (DataConfig)
    ▼
neural_ode_v9.py ──► config_v9.py (STATE_DIM, ACTION_DIM)
    │
    ▼
neural_ode_v12.py ──► config_v9.py, neural_ode_v9.py (ODEFunc, NeuralODEConfig),
                       neural_ode_contractive.py (compute_jacobian, max_eigenvalue_symmetric)
    │
    ▼
evaluation_v9.py ──► config_v9.py (STATE_DIM, STATE_NAMES_7D, STATE_LIMIT, PHYSICAL_LIMITS)
    │
    ▼
__init__.py ──► 以上所有模块（重导出）
```

### 跨版本依赖: V9 → V8

**[CONFIRMED]** V9 通过唯一通道依赖 V8：
- `data_loader_v9.py:3`: `sys.path.insert(0, 'D:/系统辨识作业/sindy_bicycle/continuation_stage_v8')`
- `data_loader_v9.py:6`: `from canonical_7d.data_loader import load_7d_data as _v8_load`
- 所有运行脚本在顶部插入 V8 和 V9 路径

### 配置继承链

```
NeuralODEConfig (neural_ode_v9.py:15)
    ├── AttractorODEConfig (neural_ode_attractor.py:41)
    ├── ContractiveODEConfig (neural_ode_contractive.py:46)
    ├── FeedbackODEConfig (neural_ode_feedback.py:23)
    ├── AdaptiveCurriculumConfig (neural_ode_adaptive_curriculum.py:34)
    ├── FreqCurriculumConfig (neural_ode_freq_curriculum.py:63)
    ├── PhysicsODEConfig (neural_ode_physics.py:20)
    └── V12Config (neural_ode_v12.py:57)
```

## 3. 外部依赖

| 包 | 用途 |
|----|------|
| `torch` | 神经网络训练、自动微分 |
| `numpy` | 数值运算、数据处理 |
| `scipy.stats` | Spearman/Kendall 相关（Phase 3F） |
| `sklearn.gaussian_process` | GP 模型（仅 V8 基线） |

## 4. 死代码发现

### 4.1 未使用的导入 **[CONFIRMED]**

| 文件 | 行 | 导入 | 状态 |
|------|-----|------|------|
| `neural_ode_v9.py` | 3 | `import json` | 未使用 |
| `neural_ode_v9.py` | 6 | `from typing import Optional, List` | 未使用 |
| `neural_ode_feedback.py` | 14 | `from typing import Optional` | 未使用 |
| `neural_ode_attractor.py` | 28 | `from typing import Optional, List` | 未使用 |
| `neural_ode_contractive.py` | 36 | `from typing import Optional, Literal` | 未使用 |
| `neural_ode_freq_curriculum.py` | 31 | `from typing import Optional` | 未使用 |
| `residual_methods/residual_models.py` | 8 | `from typing import Optional, List` | 未使用 |

### 4.2 未导出的模块 **[CONFIRMED]**

`neural_ode_physics.py` 未在 `__init__.py` 中导出，与其他所有模型变体不一致。

### 4.3 重复常量 **[CONFIRMED]**

`residual_methods/residual_models.py:14-15` 硬编码 `STATE_DIM=7, ACTION_DIM=1`，与 `config_v9.py` 重复。

### 4.4 大规模代码重复 **[CONFIRMED]**

训练循环在 7 个模型类中近乎完全复制：
- `NeuralODEV9.train()` (neural_ode_v9.py:74-188)
- `AttractorNeuralODE.train()` (neural_ode_attractor.py:375-543)
- `NeuralODEContractive.train()` (neural_ode_contractive.py:221-376)
- `FeedbackNeuralODE.train()` (neural_ode_feedback.py:131-260)
- `AdaptiveCurriculumNeuralODE.train()` (neural_ode_adaptive_curriculum.py:228-384)
- `FreqCurriculumNeuralODE.train()` (neural_ode_freq_curriculum.py:191-334)
- `PhysicsNeuralODE.train()` (neural_ode_physics.py:105-234)

`predict()`, `predict_batch()`, `save()`, `load()` 方法同样在所有类中逐字复制。

### 4.5 桩函数 **[CONFIRMED]**

`neural_ode_v9.py:210-211`: `predict_with_uncertainty` 返回 `(self.predict(s, tau), None)`，不确定性始终为 None。

## 5. 硬编码路径 **[CONFIRMED]**

| 文件 | 行 | 路径 |
|------|-----|------|
| `config_v9.py` | 9 | `DATA_PATH = 'D:/系统辨识作业/sindy_bicycle/data/stage2_dataset_150k.npz'` |
| `config_v9.py` | 10 | `V8_PATH = 'D:/系统辨识作业/sindy_bicycle/continuation_stage_v8'` |
| `config_v9.py` | 11 | `V9_PATH = 'D:/系统辨识作业/sindy_bicycle/continuation_stage_v9'` |
| `data_loader_v9.py` | 3 | `sys.path.insert(0, 'D:/系统辨识作业/sindy_bicycle/continuation_stage_v8')` |
| 所有运行脚本 | 顶部 | `sys.path.insert(0, 'D:/...')` |

这些路径为 Windows 特定且机器特定，移动项目会破坏所有脚本。

## 6. 共享 ODEFunc 骨干网络 **[CONFIRMED]**

所有模型变体共享 `neural_ode_v9.py:41-59` 定义的 `ODEFunc`：
- 输入: `STATE_DIM + ACTION_DIM = 8`（拼接状态+动作）
- 隐藏层: 可配置深度和宽度，可配置激活函数
- 输出: `STATE_DIM = 7`（状态导数）
- 末层初始化: Xavier gain=0.1, 零偏置
