# F: 接口与边界条件审计

## 1. 模块间接口

### 1.1 config_v9.py → 所有模块

**导出符号**:
- `STATE_NAMES_7D`: 7 元素列表
- `STATE_DIM = 7`
- `ACTION_DIM = 1`
- `IDX_7D_FROM_8D`: 8→7 映射索引
- `DATA_PATH`: 硬编码绝对路径
- `V8_PATH`, `V9_PATH`: 跨版本路径
- `ROLLOUT_HORIZONS`: [1, 5, 10, 50, 100, 200, 500]
- `PHYSICAL_LIMITS`: 7 维字典

**问题**: 硬编码路径，机器特定，移动项目会破坏。

### 1.2 data_loader_v9.py → V8 data_loader.py

```python
# data_loader_v9.py:3
sys.path.insert(0, 'D:/系统辨识作业/sindy_bicycle/continuation_stage_v8')
from canonical_7d.data_loader import load_7d_data as _v8_load
```

**接口**: `load_7d_data(data_path, seed) → (train_states, train_actions, train_deltas, test_states, test_actions, test_deltas, state_std, action_std, delta_std)`

**问题**: 硬编码 sys.path，跨版本依赖脆弱。

### 1.3 neural_ode_v9.py → config_v9.py

```python
from config_v9 import STATE_DIM, ACTION_DIM, STATE_NAMES_7D
```

**接口**: 仅使用维度常量，未使用物理限制。

### 1.4 neural_ode_v12.py → neural_ode_v9.py

```python
from neural_ode_v9 import ODEFunc, NeuralODEConfig
```

**接口**: 继承 ODEFunc 骨干和 NeuralODEConfig 基类。

### 1.5 neural_ode_v12.py → neural_ode_contractive.py

```python
from neural_ode_contractive import compute_jacobian, max_eigenvalue_symmetric
```

**接口**: 复用 Jacobian 计算工具。

### 1.6 evaluation_v9.py → config_v9.py

```python
from config_v9 import STATE_DIM, STATE_NAMES_7D, PHYSICAL_LIMITS
```

**接口**: 使用物理限制进行存活检查。

## 2. 公共 API 边界

### 2.1 NeuralODEV9 类 (neural_ode_v9.py)

**公共方法**:
- `train(dataloader, n_epochs, ...)`: 训练模型
- `predict(state, action)`: 单步预测
- `predict_batch(states, actions)`: 批量预测
- `predict_with_uncertainty(state, action)`: 桩函数，返回 (pred, None)
- `save(path)`, `load(path)`: 序列化

**边界验证**: 无输入形状验证，无 NaN 检查。

### 2.2 multi_step_evaluate 函数 (evaluation_v9.py)

**接口**: `(model, segment, horizons) → results_dict`

**边界验证**:
- 段长度 ≥ max(horizons) + 1: 无显式检查
- 存活检测: 每步检查物理限制
- 异常: 静默捕获所有异常

## 3. __init__.py 导出 (canonical_node/__init__.py)

```python
from .config_v9 import *
from .data_loader_v9 import *
from .neural_ode_v9 import *
from .evaluation_v9 import *
from .neural_ode_v12 import *
from .neural_ode_contractive import *
from .neural_ode_feedback import *
from .neural_ode_attractor import *
from .neural_ode_adaptive_curriculum import *
from .neural_ode_freq_curriculum import *
# MISSING: neural_ode_physics
```

**问题**: `neural_ode_physics.py` 未导出，与其他模型变体不一致。

## 4. 边界条件测试

### 4.1 未测试的边界

| 边界 | 当前状态 | 风险 |
|------|----------|------|
| 空数据集 | 未处理 | 崩溃 |
| NaN/Inf 输入 | 未检测 | 静默传播 |
| 超出物理范围的状态 | 仅评估时检查 | 训练时无约束 |
| 零 std 维度 | 静默替换 | 掩盖问题 |
| 短 episode (<1101) | 静默排除 | 数据丢失 |
| 段长度 > episode 长度 | 未检查 | 可能索引越界 |

### 4.2 数值稳定性

- **梯度爆炸**: Jacobian 正则化部分缓解
- **梯度消失**: tanh 激活可能导致
- **NaN 传播**: 无检测机制，静默异常捕获掩盖问题

## 5. 硬编码值汇总

| 文件 | 行 | 值 | 含义 |
|------|-----|-----|------|
| config_v9.py:9 | DATA_PATH | 'D:/...' | 数据文件路径 |
| config_v9.py:10 | V8_PATH | 'D:/...' | V8 代码路径 |
| config_v9.py:11 | V9_PATH | 'D:/...' | V9 代码路径 |
| data_loader_v9.py:3 | sys.path | 'D:/...' | V8 导入路径 |
| neural_ode_v9.py:15 | dt=1/30 | 0.0333 | 时间步长 |
| neural_ode_v9.py:15 | hidden=128 | 128 | 隐藏层宽度 |
| neural_ode_v9.py:15 | depth=2 | 2 | 网络深度 |
| neural_ode_v9.py:100 | lambda_multi | 0.3 | 多步损失权重 |
| neural_ode_v9.py:101 | lambda_consistency | 0.1 | 一致性损失权重 |
| neural_ode_v9.py:102 | lambda_jacobian | 0.01 | Jacobian 正则化权重 |

## 6. 总结

| 问题 | 严重度 | 建议 |
|------|--------|------|
| 硬编码路径 | HIGH | 使用相对路径或环境变量 |
| 无输入验证 | HIGH | 添加形状和 NaN 检查 |
| 静默异常捕获 | HIGH | 记录异常或重新抛出 |
| neural_ode_physics 未导出 | LOW | 添加到 __init__.py |
| 无边界条件测试 | MEDIUM | 添加单元测试 |
| 超参数硬编码 | MEDIUM | 移至配置文件 |
