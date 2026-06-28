# Agent C: Neural ODE 数值积分方法分析

**生成时间**: 2026-06-28
**分析对象**: V9 Neural ODE (`continuation_stage_v9/canonical_node/neural_ode_v9.py`)
**参考实现**: V12 Neural ODE (`continuation_stage_v9/canonical_node/neural_ode_v12.py`)

---

## 1. 数值积分误差分析

### 1.1 V9 Euler 方法的数学分析

V9 使用前向 Euler 积分：

```
x_{t+1} = x_t + dt * f_θ(x_t, u_t)
```

其中 dt = 1/30 ≈ 0.0333s。

**局部截断误差 (LTE)**:
- Euler 方法的 LTE = O(dt²) = O((1/30)²) = O(0.00111)
- 精确形式: LTE ≈ (dt²/2) * x''(ξ)，其中 ξ ∈ [t, t+dt]
- 对于本系统，x'' 的量级取决于状态二阶导数的大小

**全局截断误差 (GTE)**:
- GTE = O(dt) = O(1/30) ≈ 0.0333
- 经过 N 步积分后，误差累积: ε_total ≈ N * C * dt²
- 对于 H=500 步: ε_total ≈ 500 * C * (1/30)² ≈ 0.556 * C

**关键问题**: dt=1/30 是否过大？

对于自行车动力学系统:
- 系统特征时间尺度: 转向响应 ~0.1-0.5s，横滚动力学 ~0.2-1.0s
- Nyquist 采样: dt < T_min/2 ≈ 0.05s → dt=1/30 勉强满足
- 但 Euler 方法需要更小步长才能保证精度: dt < T_min/10 ≈ 0.01s

**结论**: dt=1/30 对于 Euler 方法偏大，特别是对于长时间积分。

### 1.2 刚性问题分析

自行车动力学系统的 Jacobian 矩阵特征值分布:

```
状态: [e_y, e_psi, v, theta, theta_dot, delta, delta_dot]
```

- **快动态**: delta_dot（转向角速度），时间常数 ~0.01-0.05s
- **中动态**: theta_dot（横滚角速度），时间常数 ~0.1-0.5s
- **慢动态**: e_y（横向位置），时间常数 ~1-5s

**刚性比估计**:
- λ_max/λ_min ≈ 50-500（取决于状态点）
- 刚性系统定义: 刚性比 > 10
- **结论**: 系统具有中等刚性，Euler 方法的稳定性域有限

**Euler 方法稳定性域**:
- 复平面左半圆，半径 = 2/dt = 60
- 对于刚性系统，需要 |λ*dt| < 2 → |λ| < 60
- 如果 Jacobian 特征值 |λ| > 60，Euler 方法会不稳定

### 1.3 V9 训练中的数值问题

V9 代码中的关键细节:

```python
# neural_ode_v9.py 第 92 行
train_dsdot = torch.FloatTensor(deltas / (self._delta_std * self._dt))
```

这意味着模型输出已经是 `rate * dt`（已乘以 dt），而不是纯粹的导数。

```python
# neural_ode_v9.py 第 135 行 (多步 rollout)
s_cur = s_cur + dsdt * self._dt  # 这里又乘了一次 dt！
```

**BUG**: 模型输出已经是 `delta_normalized = delta / (delta_std * dt)`，但在多步 rollout 中又乘以 `self._dt`，导致:
- 实际更新: `s_next = s + (delta / (delta_std * dt)) * dt = s + delta / delta_std`
- 这是正确的归一化更新

但在 `predict` 方法中:
```python
# neural_ode_v9.py 第 197 行
dsdt = dsdt_norm * self._delta_std * self._dt
return s + dsdt
```
- `dsdt_norm = model(s_norm, a_norm)` 输出是 `delta / (delta_std * dt)`
- `dsdt = (delta / (delta_std * dt)) * delta_std * dt = delta`
- `s + dsdt = s + delta` ✓ 正确

**结论**: V9 的归一化处理是自洽的，但代码可读性差，容易引入 bug。

---

## 2. 高阶积分方法分析

### 2.1 RK4 方法

**理论改进**:
- RK4 局部截断误差: O(dt⁵) vs Euler 的 O(dt²)
- RK4 全局截断误差: O(dt⁴) vs Euler 的 O(dt)
- 对于 dt=1/30: RK4 误差 ≈ (1/30)⁴ ≈ 1.2e-6 vs Euler ≈ (1/30) ≈ 0.033

**V12 中的 RK4 实现** (neural_ode_v12.py 第 320-339 行):

```python
def _integrate_rk4(model, s, a, dt):
    k1 = model(s, a)
    k2 = model(s + k1 / 2.0, a)
    k3 = model(s + k2 / 2.0, a)
    k4 = model(s + k3, a)
    return s + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
```

**注意**: 由于模型输出已经是 `rate * dt`，RK4 的实现需要特别处理。V12 的实现假设模型输出是 `f(s,a) * dt`，因此:
- k1 = f(s,a) * dt
- k2 = f(s + k1/2, a) * dt（注意: s + k1/2 是半步状态）
- 这种实现是正确的，因为 RK4 的权重 (1,2,2,1)/6 自动处理了 dt 缩放

**RK4 的计算成本**: 4x 模型前向传播 vs Euler 的 1x

**RK4 的改进潜力**:
- 对于 dt=1/30，RK4 可以将积分误差降低 ~1000 倍
- 但 RK4 对于刚性系统的稳定性改善有限（仍然是显式方法）
- 计算成本增加 4 倍，但对于 7D 系统是可接受的

### 2.2 自适应步长方法

**Dormand-Prince (RK45)**:
- 嵌入式 RK 方法，同时计算 4 阶和 5 阶解
- 误差估计: |y5 - y4| ≈ local error
- 自动调整步长: dt_new = dt * (tol/error)^(1/5)

**优势**:
- 在平滑区域使用大步长（dt > 1/30）
- 在快速变化区域使用小步长（dt < 1/30）
- 总体计算量可能更少

**实现挑战**:
- 需要可微分的步长控制（对于反向传播）
- 需要处理事件检测（如状态边界）
- 对于实时控制，计算时间不确定

**适用场景**:
- 离线仿真和验证
- 训练时的长 rollout（如 V12 的 100 步课程）
- 不适合实时控制（计算时间不确定）

### 2.3 隐式方法

**后向 Euler**:
```
x_{t+1} = x_t + dt * f(x_{t+1}, u_t)
```
- A-稳定（无条件稳定）
- 适合刚性系统
- 需要求解非线性方程（Newton 迭代）

**Trapezoidal 方法**:
```
x_{t+1} = x_t + (dt/2) * [f(x_t, u_t) + f(x_{t+1}, u_t)]
```
- A-稳定
- 二阶精度
- 也需要求解非线性方程

**对于 Neural ODE 的特殊考虑**:
- f_θ 是神经网络，求解隐式方程需要迭代
- 每次迭代需要前向传播，计算成本高
- 可以使用不动点迭代或 Newton-Krylov 方法

**适用场景**:
- 系统刚性比 > 100
- 需要长时间积分（H > 1000）
- 可以接受更高计算成本

---

## 3. Neural ODE 特定数值问题

### 3.1 学到的 f_θ 是否光滑？

**架构分析**:
- V9 使用 Tanh 激活函数（C∞ 光滑）
- 2-3 层全连接网络
- 最后一层 Xavier 初始化（gain=0.1）

**光滑性保证**:
- Tanh 是光滑的: tanh'(x) = 1 - tanh²(x)
- 但梯度可能在饱和区域接近零（梯度消失）
- 多层组合可能导致梯度爆炸/消失

**实际光滑性**:
- 在训练数据分布内，f_θ 应该是光滑的
- 在分布外（OOD）区域，f_θ 可能不光滑或不准确
- Jacobian 正则化（λ_jacobian=0.01）鼓励光滑性

### 3.2 Jacobian 的条件数

**V9 的 Jacobian 正则化**:
```python
# neural_ode_v9.py 第 152-163 行
for i in range(STATE_DIM):
    grad = torch.autograd.grad(dsdt[:, i].sum(), s_req, create_graph=True)[0]
    jac_norm = jac_norm + grad.pow(2).sum()
loss_jacobian = jac_norm / (STATE_DIM * min(32, len(sb)))
```

这是 Frobenius 范数正则化，不是条件数正则化。

**条件数问题**:
- 条件数 κ(J) = σ_max / σ_min
- 高条件数意味着系统对扰动敏感
- V9 没有直接控制条件数

**V12 的改进**:
- 添加了 contractivity 正则化（neural_ode_v12.py 第 285-303 行）
- 惩罚 Jacobian 对称部分的正特征值
- 鼓励局部收缩性（附近轨迹收敛）

### 3.3 长时间积分的稳定性

**误差累积模型**:
- 每步误差: ε_step = ε_model + ε_integrator
- N 步后总误差: ε_total ≈ N * ε_step * exp(λ_max * N * dt)
- 如果 λ_max > 0（不稳定），误差指数增长

**V9 的观测现象** (KNOWN_LIMITS.md):
- H=10: NMAE=6.3%（可用）
- H=100: NMAE=50.6%（需谨慎）
- H=500: NMAE=55.3%（不可靠）
- H≈400 时误差突然跳跃

**误差跳跃的可能原因**:
1. **数值不稳定**: Jacobian 特征值导致指数增长
2. **模型误差累积**: 单步误差线性累积
3. **分布偏移**: 长 rollout 导致状态偏离训练分布
4. **积分误差**: Euler 方法的截断误差累积

**V12 的改进**:
- Contractivity 正则化（鼓励 λ_max < 0）
- 顺序数据采样（修复 shuffled DataLoader bug）
- 扩展课程学习（训练到 100 步）

---

## 4. 误差分离实验设计

### 4.1 误差来源分解

总预测误差可以分解为:

```
ε_total = ε_model + ε_integrator + ε_interaction
```

其中:
- **ε_model**: 模型误差（f_θ ≠ f_true）
- **ε_integrator**: 积分方法误差（Euler vs RK4 vs 精确解）
- **ε_interaction**: 模型误差和积分误差的耦合项

### 4.2 实验设计

**实验 1: 真实导数 + 不同积分方法**

目标: 纯积分误差，无模型误差

```python
def experiment_true_derivative_integration():
    """
    使用真实导数（从数据计算）比较不同积分方法

    方法:
    1. 真实导数 + Euler (dt=1/30)
    2. 真实导数 + RK4 (dt=1/30)
    3. 真实导数 + Euler (dt=1/60)  # 更小步长
    4. 真实导数 + 精确解（如果可用）

    指标:
    - 各状态的 NMAE
    - 误差随 horizon 的增长曲线
    - 误差的自相关（判断是否系统性偏差）
    """
    # 从数据计算真实导数
    true_deltas = np.diff(states, axis=0) / dt  # 真实导数

    # 定义积分器
    def integrate_euler(s0, actions, dt, n_steps):
        s = s0.copy()
        trajectory = [s]
        for i in range(n_steps):
            dsdt = true_deltas[i]  # 使用真实导数
            s = s + dsdt * dt
            trajectory.append(s)
        return np.array(trajectory)

    def integrate_rk4(s0, actions, dt, n_steps):
        s = s0.copy()
        trajectory = [s]
        for i in range(n_steps):
            # 使用真实导数的 RK4
            k1 = true_deltas[i]
            k2 = true_deltas[i]  # 简化: 假设导数在半步不变
            k3 = true_deltas[i]
            k4 = true_deltas[i]
            s = s + (k1 + 2*k2 + 2*k3 + k4) / 6 * dt
            trajectory.append(s)
        return np.array(trajectory)

    # 比较结果
    # ...
```

**实验 2: V9 模型导数 + 不同积分方法**

目标: 分离模型误差和积分误差

```python
def experiment_v9_model_integration():
    """
    使用 V9 模型导数比较不同积分方法

    方法:
    1. V9 模型 + Euler (dt=1/30)  # 基线
    2. V9 模型 + RK4 (dt=1/30)
    3. V9 模型 + Euler (dt=1/60)
    4. V9 模型 + 自适应步长

    指标:
    - 各状态的 NMAE
    - 存活率
    - 误差随 horizon 的增长曲线
    """
    # 加载 V9 模型
    model = NeuralODEV9()
    model.load('path/to/v9/model.pt')

    # 定义积分器
    def integrate_v9_euler(s0, actions, dt, n_steps):
        s = s0.copy()
        trajectory = [s]
        for i in range(n_steps):
            s_next = model.predict(s, actions[i])
            s = s_next
            trajectory.append(s)
        return np.array(trajectory)

    def integrate_v9_rk4(s0, actions, dt, n_steps):
        s = s0.copy()
        trajectory = [s]
        for i in range(n_steps):
            # RK4 需要多次模型调用
            k1 = model.predict(s, actions[i]) - s
            k2 = model.predict(s + k1/2, actions[i]) - (s + k1/2)
            k3 = model.predict(s + k2/2, actions[i]) - (s + k2/2)
            k4 = model.predict(s + k3, actions[i]) - (s + k3)
            s = s + (k1 + 2*k2 + 2*k3 + k4) / 6
            trajectory.append(s)
        return np.array(trajectory)

    # 比较结果
    # ...
```

**实验 3: 误差分解量化**

```python
def experiment_error_decomposition():
    """
    量化分解总误差为模型误差和积分误差

    假设:
    - ε_total(H) = ε_model(H) + ε_integrator(H) + ε_interaction(H)
    - ε_integrator(H) 可以通过实验 1 估计
    - ε_model(H) 可以通过单步误差估计

    方法:
    1. 计算单步模型误差: ε_model(1) = |f_θ(x) - f_true(x)|
    2. 计算纯积分误差: ε_integrator(H) from 实验 1
    3. 计算总误差: ε_total(H) from 实验 2
    4. 计算交互项: ε_interaction = ε_total - ε_model*H - ε_integrator
    """
    # 单步模型误差
    model_single_error = np.mean(np.abs(model_pred - true_delta))

    # 纯积分误差（从实验 1）
    integrator_error = ...  # 从实验 1 结果

    # 总误差（从实验 2）
    total_error = ...  # 从实验 2 结果

    # 交互项
    interaction_error = total_error - model_single_error * H - integrator_error

    # 分析交互项的符号和大小
    # 如果 interaction > 0: 模型误差和积分误差相互放大
    # 如果 interaction < 0: 模型误差和积分误差相互抵消
    # 如果 |interaction| << |total|: 误差近似可加
```

### 4.3 关键指标

**误差增长率**:
```python
def compute_error_growth_rate(errors, horizons):
    """
    计算误差随 horizon 的增长率

    如果误差线性增长: ε(H) ≈ a * H + b
    如果误差指数增长: ε(H) ≈ a * exp(b * H)
    如果误差平方根增长: ε(H) ≈ a * sqrt(H) + b

    返回:
    - growth_type: 'linear', 'exponential', 'sqrt'
    - growth_rate: 增长率参数
    - R²: 拟合优度
    """
    # 拟合不同模型
    from scipy.optimize import curve_fit

    def linear(H, a, b):
        return a * H + b

    def exponential(H, a, b):
        return a * np.exp(b * H)

    def sqrt_growth(H, a, b):
        return a * np.sqrt(H) + b

    # 比较拟合优度
    # ...
```

**误差自相关**:
```python
def compute_error_autocorrelation(errors, max_lag=50):
    """
    计算误差的自相关函数

    如果误差是白噪声: 自相关快速衰减
    如果误差有系统性偏差: 自相关缓慢衰减
    如果误差有周期性: 自相关有周期性峰值

    返回:
    - autocorr: 自相关函数
    - significant_lags: 显著非零的滞后
    """
    from statsmodels.tsa.stattools import acf

    autocorr = acf(errors, nlags=max_lag)
    significant_lags = np.where(np.abs(autocorr) > 1.96 / np.sqrt(len(errors)))[0]

    return autocorr, significant_lags
```

---

## 5. 推荐的改进方案

### 5.1 短期改进（低成本）

**方案 A: RK4 积分**
- 实现成本: 低（V12 已有实现）
- 预期收益: 积分误差降低 ~1000 倍
- 适用场景: 所有 horizon

```python
# 在 V9 中启用 RK4
config = NeuralODEConfig()
config.integration_method = 'rk4'  # 需要添加此选项
```

**方案 B: Euler 子步长**
- 实现成本: 极低
- 预期收益: 积分误差降低 ~N² 倍（N 为子步数）
- 适用场景: 中等 horizon

```python
# 在 V9 中使用子步长
def predict_substep(s, tau, n_substeps=2):
    dt_sub = dt / n_substeps
    s_cur = s
    for _ in range(n_substeps):
        dsdt = model(s_cur, tau)
        s_cur = s_cur + dsdt * dt_sub
    return s_cur
```

### 5.2 中期改进（中等成本）

**方案 C: Contractivity 正则化**
- 实现成本: 中等（V12 已有实现）
- 预期收益: 长 horizon 稳定性提升
- 适用场景: H > 50

```python
# 在 V9 中添加 contractivity 正则化
def contractivity_loss(model, s, a):
    J = compute_jacobian(model, s, a)
    J_sym = 0.5 * (J + J.transpose(-1, -2))
    eigenvalues = torch.linalg.eigvalsh(J_sym)
    max_eig = eigenvalues[:, -1]
    return torch.relu(max_eig).mean()
```

**方案 D: 顺序数据采样**
- 实现成本: 低
- 预期收益: 多步损失更准确
- 适用场景: 训练阶段

```python
# 修复 V9 的 shuffled DataLoader
# 参见 V12 的 SequentialSegmentDataset
```

### 5.3 长期改进（高成本）

**方案 E: 自适应步长 RK45**
- 实现成本: 高
- 预期收益: 自动平衡精度和计算量
- 适用场景: 离线仿真和验证

**方案 F: 隐式积分方法**
- 实现成本: 高
- 预期收益: 刚性系统稳定性
- 适用场景: 高刚性比区域

**方案 G: 混合积分方法**
- 实现成本: 高
- 预期收益: 自动选择最优积分方法
- 适用场景: 全状态空间

---

## 6. 实验优先级

### 优先级 1: 误差分离实验（立即执行）

```python
# 实验 1: 真实导数积分误差
# 目标: 量化纯积分误差的上限
# 预期: 确认 Euler 是否是瓶颈

# 实验 2: V9 模型 + RK4
# 目标: 量化积分方法改进的收益
# 预期: 如果收益显著，说明积分误差是主要瓶颈

# 实验 3: 误差分解
# 目标: 定量分离模型误差和积分误差
# 预期: 指导后续改进方向
```

### 优先级 2: V9 + RK4 集成（1-2 天）

```python
# 修改 V9 的 predict 方法支持 RK4
# 在多个 horizon 上比较 Euler vs RK4
# 分析计算成本增加是否可接受
```

### 优先级 3: Contractivity 正则化（3-5 天）

```python
# 实现 contractivity 损失
# 在训练中添加 contractivity 正则化
# 分析长 horizon 稳定性改善
```

---

## 7. 关键代码路径

### V9 核心文件
- `continuation_stage_v9/canonical_node/neural_ode_v9.py`: 主模型
- `continuation_stage_v9/canonical_node/evaluation_v9.py`: 评估函数
- `continuation_stage_v9/canonical_node/config_v9.py`: 配置

### V12 参考实现
- `continuation_stage_v9/canonical_node/neural_ode_v12.py`: RK4 实现、顺序数据、contractivity

### Contractivity 实现
- `continuation_stage_v9/canonical_node/neural_ode_contractive.py`: Jacobian 计算、特征值分析

---

## 8. 总结

### 关键发现

1. **dt=1/30 对 Euler 方法偏大**: 局部截断误差 O(dt²) ≈ 0.001，全局误差 O(dt) ≈ 0.033
2. **系统具有中等刚性**: 刚性比 ~50-500，Euler 稳定性域有限
3. **V9 的归一化处理自洽**: 但代码可读性差，容易引入 bug
4. **V12 已解决部分问题**: RK4、顺序数据、contractivity

### 误差来源估计

- **模型误差**（f_θ ≠ f_true）: 可能占主导，特别是长 horizon
- **积分误差**（Euler vs 精确解）: 对于 H<100 可能不是主要瓶颈
- **误差累积**: 指数增长（如果 λ_max > 0）或线性增长（如果 λ_max < 0）

### 推荐行动

1. **立即**: 执行误差分离实验，量化积分误差的贡献
2. **短期**: 在 V9 中集成 RK4，比较性能改善
3. **中期**: 添加 contractivity 正则化，改善长 horizon 稳定性
4. **长期**: 考虑自适应步长或隐式方法

---

## 附录 A: 数学推导

### A.1 Euler 方法误差分析

**局部截断误差**:
```
x(t+dt) = x(t) + dt * x'(t) + (dt²/2) * x''(ξ)
x_{n+1} = x_n + dt * f(x_n, u_n)
LTE = x(t+dt) - x_{n+1} = (dt²/2) * x''(ξ)
```

**全局截断误差**:
```
假设 |LTE| ≤ C * dt²
N 步后: |GTE| ≤ N * C * dt² = (T/dt) * C * dt² = C * T * dt
```

### A.2 RK4 方法误差分析

**局部截断误差**:
```
LTE = (dt⁵/120) * x⁵(ξ) = O(dt⁵)
```

**全局截断误差**:
```
|GTE| ≤ N * C * dt⁵ = (T/dt) * C * dt⁵ = C * T * dt⁴
```

### A.3 稳定性分析

**Euler 方法稳定性域**:
```
|1 + λ*dt| < 1
→ 复平面左半圆，半径 = 2/dt
```

**RK4 方法稳定性域**:
```
|1 + λ*dt + (λ*dt)²/2 + (λ*dt)³/6 + (λ*dt)⁴/24| < 1
→ 复平面左半区域，比 Euler 更大
```

**隐式方法稳定性域**:
```
后向 Euler: |1/(1 - λ*dt)| < 1
→ 整个左半平面（A-稳定）
```

---

## 附录 B: 实验代码模板

### B.1 真实导数积分实验

```python
import numpy as np

def true_derivative_integration_experiment(states, actions, dt, horizons):
    """
    使用真实导数比较不同积分方法

    Args:
        states: [N, 7] 真实状态序列
        actions: [N-1] 动作序列
        dt: 时间步长 (1/30)
        horizons: 评估 horizon 列表

    Returns:
        results: dict[horizon] -> dict[method] -> NMAE
    """
    true_deltas = np.diff(states, axis=0)  # 真实状态变化

    results = {}

    for h in horizons:
        results[h] = {}

        # 方法 1: Euler (dt=1/30)
        errors_euler = []
        for i in range(len(states) - h):
            s0 = states[i]
            s_true = states[i:i+h+1]

            # Euler 积分
            s = s0.copy()
            trajectory = [s]
            for j in range(h):
                dsdt = true_deltas[i+j] / dt
                s = s + dsdt * dt
                trajectory.append(s)

            trajectory = np.array(trajectory)
            error = np.mean(np.abs(trajectory - s_true) / state_std)
            errors_euler.append(error)

        results[h]['euler'] = np.mean(errors_euler)

        # 方法 2: RK4 (dt=1/30)
        errors_rk4 = []
        for i in range(len(states) - h):
            s0 = states[i]
            s_true = states[i:i+h+1]

            # RK4 积分
            s = s0.copy()
            trajectory = [s]
            for j in range(h):
                k1 = true_deltas[i+j] / dt
                k2 = true_deltas[i+j] / dt  # 简化
                k3 = true_deltas[i+j] / dt
                k4 = true_deltas[i+j] / dt
                s = s + (k1 + 2*k2 + 2*k3 + k4) / 6 * dt
                trajectory.append(s)

            trajectory = np.array(trajectory)
            error = np.mean(np.abs(trajectory - s_true) / state_std)
            errors_rk4.append(error)

        results[h]['rk4'] = np.mean(errors_rk4)

    return results
```

### B.2 V9 模型积分实验

```python
def v9_model_integration_experiment(model, test_segments, horizons):
    """
    使用 V9 模型比较不同积分方法

    Args:
        model: NeuralODEV9 实例
        test_segments: 测试数据段
        horizons: 评估 horizon 列表

    Returns:
        results: dict[horizon] -> dict[method] -> NMAE
    """
    results = {}

    for h in horizons:
        results[h] = {}

        # 方法 1: Euler (原始 V9)
        errors_euler = []
        for seg in test_segments:
            s0 = seg['states'][0]
            actions = seg['actions'][:h]
            s_true = seg['states'][:h+1]

            # Euler 积分
            s = s0.copy()
            trajectory = [s]
            for a in actions:
                s_next = model.predict(s, a)
                s = s_next
                trajectory.append(s)

            trajectory = np.array(trajectory)
            error = np.mean(np.abs(trajectory - s_true) / state_std)
            errors_euler.append(error)

        results[h]['euler'] = np.mean(errors_euler)

        # 方法 2: RK4
        errors_rk4 = []
        for seg in test_segments:
            s0 = seg['states'][0]
            actions = seg['actions'][:h]
            s_true = seg['states'][:h+1]

            # RK4 积分
            s = s0.copy()
            trajectory = [s]
            for a in actions:
                k1 = model.predict(s, a) - s
                k2 = model.predict(s + k1/2, a) - (s + k1/2)
                k3 = model.predict(s + k2/2, a) - (s + k2/2)
                k4 = model.predict(s + k3, a) - (s + k3)
                s = s + (k1 + 2*k2 + 2*k3 + k4) / 6
                trajectory.append(s)

            trajectory = np.array(trajectory)
            error = np.mean(np.abs(trajectory - s_true) / state_std)
            errors_rk4.append(error)

        results[h]['rk4'] = np.mean(errors_rk4)

    return results
```

### B.3 误差分解函数

```python
def error_decomposition(total_error, integrator_error, model_single_error, H):
    """
    分解总误差为模型误差和积分误差

    Args:
        total_error: 总误差 (from V9 model + Euler)
        integrator_error: 纯积分误差 (from true derivative + Euler)
        model_single_error: 单步模型误差
        H: horizon

    Returns:
        decomposition: dict with error components
    """
    # 线性累积的模型误差
    model_cumulative = model_single_error * H

    # 积分误差
    integrator = integrator_error

    # 交互项
    interaction = total_error - model_cumulative - integrator

    # 误差比例
    total = model_cumulative + integrator + interaction

    decomposition = {
        'model_error': model_cumulative,
        'integrator_error': integrator,
        'interaction_error': interaction,
        'total_error': total,
        'model_fraction': model_cumulative / total if total > 0 else 0,
        'integrator_fraction': integrator / total if total > 0 else 0,
        'interaction_fraction': interaction / total if total > 0 else 0,
    }

    return decomposition
```

---

**文档版本**: 1.0
**最后更新**: 2026-06-28
**状态**: 待实验验证
