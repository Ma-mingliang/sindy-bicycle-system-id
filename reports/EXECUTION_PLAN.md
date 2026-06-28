# 统一评估执行方案

> **生成时间**: 2026-06-24
> **目的**: 精确到每一步的执行计划

---

## 1. 代码结构

```
evaluate/
├── __init__.py
├── eval_config.py      # 配置加载 (从YAML读取)
├── eval_models.py      # 6种模型定义
├── eval_modes.py       # Mode A/B/C/D 评估逻辑
├── eval_metrics.py     # 指标计算
├── eval_runner.py      # 主执行入口
└── eval_plots.py       # 图表生成
```

---

## 2. 执行步骤

### 步骤1: 创建 eval_config.py

从 YAML 加载所有配置，返回配置字典。

```python
def load_config(yaml_path) -> dict
def get_seeds(config) -> list[int]
def get_horizons(config) -> list[int]
def get_model_configs(config) -> list[dict]
```

### 步骤2: 创建 eval_models.py

定义6种模型的统一接口：

```python
class BaseModel(ABC):
    def predict(self, s: np.ndarray, tau: float) -> np.ndarray
    def name(self) -> str

class RealDynamics(BaseModel):
    # 调用 methods_common.real_step

class LinearizedModel(BaseModel):
    # 使用 A, B 矩阵线性预测

class GPStandard(BaseModel):
    # 训练GP, 预测

class GPB4Sparse(BaseModel):
    # Nystroem近似GP

class SINDy4D(BaseModel):
    # 加载 meijaard_sindy_v35.npz

class GPEEnsemble(BaseModel):
    # GP + NN集成 + DAgger
```

### 步骤3: 创建 eval_modes.py

4种评估模式，统一输入输出：

```python
def run_mode_a(model, s0, tau_func, n_steps, real_dynamics) -> dict
def run_mode_b(model, s0, tau_func, n_steps, real_dynamics) -> dict
def run_mode_c(model, s0, tau_func, n_steps, real_dynamics) -> dict
def run_mode_d(model, s0, n_steps, real_dynamics, K_lqr, rng) -> dict
```

每个函数返回:
```python
{
    'states_model': (n+1, 4),
    'states_real': (n+1, 4),
    'actions': (n,),
    'actions_real': (n,),  # Mode D only
    'survival_steps': int,
    'instability_detected': bool,
}
```

### 步骤4: 创建 eval_metrics.py

```python
def compute_all_metrics(states_model, states_real, actions, state_std) -> dict
def compute_per_state_metrics(errors, state_name) -> dict
def compute_stability_metrics(survival_steps, max_steps) -> dict
def compute_control_metrics(actions_model, actions_real) -> dict
def compute_uncertainty_metrics(ensemble_stds, errors) -> dict
```

指标列表:
- MAE, RMSE, Max, Median, P95, 终点误差
- 归一化误差 (除以 training_std)
- 首次超过 1σ/2σ/3σ 的步数
- 控制力矩误差
- 失稳率, 存活步数

### 步骤5: 创建 eval_runner.py (主入口)

```python
def train_all_models(config) -> dict  # 训练所有模型, 返回模型字典
def run_single_evaluation(model, mode, seed, horizon, config) -> dict
def run_full_evaluation(config) -> dict  # 30 seed × 9 horizon × 6 model × 4 mode
def save_results(results, config)
```

执行流程:
1. 训练所有模型 (一次性)
2. 对每个 seed:
   a. 生成初始状态和扰动
   b. 对每个模型:
      - 对每个模式 (A/B/C/D):
        - 对每个时域 (1,5,10,20,50,100,200,500,1000):
          - 运行评估
          - 计算指标
3. 汇总统计
4. 保存结果

### 步骤6: 创建 eval_plots.py

```python
def plot_error_vs_time(results, config)
def plot_state_trajectories(results, config)
def plot_model_comparison(results, config)
def plot_uncertainty(results, config)
```

### 步骤7: 数据版本审计 (Task #22)

```python
def audit_data_files() -> dict
def compute_file_hashes() -> dict
def verify_statistics() -> dict
```

输出: `reports/DATA_PROVENANCE_AND_HASH_AUDIT.md`

---

## 3. 执行顺序

| 步骤 | 文件 | 依赖 | 预计耗时 |
|------|------|------|----------|
| 1 | eval_config.py | 无 | 5分钟 |
| 2 | eval_models.py | 步骤1 | 15分钟 |
| 3 | eval_modes.py | 步骤2 | 10分钟 |
| 4 | eval_metrics.py | 无 | 10分钟 |
| 5 | eval_runner.py | 步骤1-4 | 15分钟 |
| 6 | eval_plots.py | 步骤5 | 10分钟 |
| 7 | 运行评估 | 步骤5 | ~60分钟 |
| 8 | 生成报告 | 步骤7 | 15分钟 |

---

## 4. 关键约束

1. 所有随机操作使用显式 RNG (seed 写入配置)
2. 所有模型使用相同初始状态和扰动
3. 全状态评估 (phi, delta, phi_dot, delta_dot)
4. 不修改原有代码
5. 新代码放入 evaluate/ 目录

---

*文档生成时间: 2026-06-24*
