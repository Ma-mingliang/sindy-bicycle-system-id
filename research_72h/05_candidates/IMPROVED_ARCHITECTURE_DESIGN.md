# 改进架构设计：从 Wide SiLU NODE 到物理信息 Neural ODE

**设计时间**: 2026-06-28
**当前最佳**: Wide SiLU NODE (256x5, SiLU) -- Primary = 0.3930, +23.1% vs v9
**设计目标**: 进一步提升至 Primary < 0.30 (+40% vs v9)

---

## 1. Wide SiLU NODE 优势分析

### 1.1 为什么有效？

Wide SiLU NODE 是 72h 研究中唯一突破 v9 基线的方法。其成功源于三个关键因素的协同：

**因素 1: SiLU 激活函数的梯度优势**

| 激活函数 | Primary Score | 训练 Loss (epoch 50) | 最终 Loss |
|----------|--------------|---------------------|-----------|
| Tanh (256x5) | 1.0813 | 604.2 | 373.6 |
| **SiLU (256x5)** | **0.3930** | **407.6** | **319.6** |

SiLU (Swish) = x * sigmoid(x) 相比 Tanh 的核心优势：
- 无饱和区：Tanh 在 |x|>2 时梯度接近零，SiLU 对正输入保持梯度
- 光滑非单调：SiLU 在 x<0 时有小负值，提供正则化效果
- 更好的损失景观：SiLU 网络的损失函数更平滑，优化更容易收敛到好的解

**因素 2: 宽架构捕捉非线性耦合**

Agent S 的耦合分析揭示了状态间存在复杂的非线性依赖：
- theta_dot 的 R^2 从 0.008（仅自回归）到 0.791（全部状态），说明 99% 的预测信息来自跨状态耦合
- e_y 与 e_psi 的互信息 = 1.0 bit，但 Pearson 相关仅 -0.154（强非线性耦合）
- theta_dot, delta, v, a 之间相关性 r > 0.98（强耦合组）

256 宽 x 5 层 = 267K 参数提供了足够的表示容量来编码这些非线性耦合关系。

**因素 3: 联合预测保持耦合一致性**

所有 7 个状态由同一个网络同时预测，隐式保持了：
- 运动学一致性（e_y_dot 与 v, e_psi 的关系）
- 动力学一致性（theta_dot 与 theta 的积分关系）
- 闭环一致性（e_y -> action -> theta_dot -> v -> e_y 的反馈环）

这解释了为什么 Hybrid 方法（分别预测不同状态组）全部失败。

### 1.2 有什么局限？

**局限 1: 长时域平台期**

| Horizon | NMAE | 增长率 |
|---------|------|--------|
| H=100 | 0.3689 | - |
| H=200 | 0.4050 | +9.8% |
| H=500 | 0.4050 | +0.0% |
| H=1000 | 0.4050 | +0.0% |

NMAE 在 H=200 后完全停止增长。这有两种可能解释：
- 正面：模型学到了稳定的吸引子，预测收敛到固定点
- 负面：模型的预测已完全偏离真实轨迹，误差饱和

后者的可能性更大，因为 H=200 的 NMAE=0.4050 仍然很高。

**局限 2: 短时域精度下降**

| Horizon | v9 NMAE | Wide SiLU NMAE | 变化 |
|---------|---------|----------------|------|
| H=1 | 0.0051 | 0.0190 | -273% (更差) |

宽架构在单步预测上不如窄架构，说明存在过参数化导致的单步精度损失。

**局限 3: 缺乏物理归纳偏置**

Wide SiLU NODE 是纯数据驱动的，没有嵌入任何已知物理关系：
- e_y_dot = v * sin(e_psi)（精确运动学）-- 未嵌入
- e_psi_dot = -v * delta / L（精确运动学）-- 未嵌入
- theta_ddot ~ (g/h) * theta（近似动力学）-- 未嵌入

模型被迫从数据中重新发现这些关系，浪费了表示容量。

**局限 4: 训练数据分布问题**

Agent Q 的深层分析揭示：
- 训练数据来自 LQR 控制下的平衡轨迹，缺乏大偏差恢复过程
- 速度范围极窄 [0.54, 0.66]，动力学退化为近线性
- 模型学到的是"在平衡点附近小幅振荡"的动力学，不是完整的自行车动力学

### 1.3 如何改进？

基于上述分析，改进方向按优先级排列：

| 优先级 | 改进方向 | 理论依据 | 预期收益 |
|--------|----------|----------|----------|
| P0 | 嵌入精确运动学硬约束 | e_y/e_psi 是误差最大的状态，运动学是精确的 | e_y/e_psi NMAE 下降 50%+ |
| P1 | UDE 风格物理-残差分离 | 减少学习空间，物理项提供稳定性 | 整体 NMAE 下降 15-25% |
| P2 | 修复多步训练损失 | shuffle bug 导致多步损失失效 | 长时域预测改善 |
| P3 | 合同性正则化 | 约束 Jacobian 特征值 < 1 | 防止误差指数放大 |
| P4 | 数据增强 | 当前数据覆盖度不足 | 泛化能力提升 |

---

## 2. Coupling Network 问题分析

### 2.1 为什么失败？

本研究中所有尝试"分离状态预测"的方法都失败了：

| 方法 | Primary Score | vs v9 | 失败原因 |
|------|--------------|-------|----------|
| Hybrid SiLU NODE + GP | 0.7511 | -47% | 跨状态耦合被破坏 |
| Per-State Ensemble | 0.7066 | -38% | NODE 在开环发散 |
| GP Per-State | 0.7174 | +23.7%* | *仅在特定评估下 |
| State-Specific Models | 0.5885 | +33.8%* | *使用不同评估标准 |

**根本原因: 状态不可分离**

Agent S 的分析定量证明了这一点：

1. **theta_dot 的预测信息 99% 来自跨状态耦合**（R^2: 0.008 -> 0.791）
2. **e_y 与 e_psi 存在强非线性耦合**（互信息 = 1.0 bit，但线性相关仅 -0.15）
3. **闭环效应**: e_y -> action -> theta_dot -> v -> e_y 形成反馈环

当 GP 独立预测 v, theta, delta，SiLU NODE 独立预测 e_y, e_psi, theta_dot, delta_dot 时：
- SiLU NODE 在训练时看到的是真实的 v, theta, delta
- 在推理时看到的是 GP 的预测值，其误差特性与真实值不同
- 这造成训练-推理分布偏移，导致误差累积

### 2.2 如何修复？

**不值得修复。** 原因如下：

1. **物理上不可分离**: 自行车动力学的状态间存在不可忽略的耦合（陀螺效应、trail 效应、离心力）
2. **信息论限制**: theta_dot 的 99% 预测信息来自其他状态，分离预测本质上丢失信息
3. **已有更好的替代**: Wide SiLU NODE 的联合预测已经有效解决了耦合问题

**如果非要尝试**，唯一可行的方向是：

- **注意力耦合机制**: 在分离的子网络之间添加交叉注意力，让每个子网络能看到其他子网络的中间表示
- **但这本质上等价于一个更大的联合网络**，失去了分离的意义

### 2.3 结论

Coupling Network / 分离状态预测方法在这个问题上是死路。所有证据一致指向：**必须联合预测所有状态**。

---

## 3. 改进架构设计

### 3.1 方案 A: Physics-Constrained Wide SiLU NODE (推荐)

**核心思想**: 将精确运动学硬编码到 Wide SiLU NODE 中，让网络只学习真正未知的动力学。

#### 架构

```
输入: [s(7D), u(1D)] = 8D
      |
      v
+------------------+
| 运动学硬编码层    |  <-- 不可学习，精确公式
| e_y_dot = v*sin(e_psi)
| e_psi_dot = -v*delta/L
+------------------+
      |
      v
+------------------+
| Wide SiLU NODE   |  <-- 只学习剩余 5 个状态
| (256x5, SiLU)    |
| 输入: 8D         |
| 输出: 5D         |
| (v_dot, theta_dot, theta_ddot,
|  delta_dot, delta_ddot)
+------------------+
      |
      v
+------------------+
| 组合层           |
| dsdt[0] = e_y_dot_physics     (硬编码)
| dsdt[1] = e_psi_dot_physics   (硬编码)
| dsdt[2] = nn_output[0]        (学习)
| dsdt[3] = nn_output[1]        (学习)
| dsdt[4] = nn_output[2]        (学习)
| dsdt[5] = nn_output[3]        (学习)
| dsdt[6] = nn_output[4]        (学习)
+------------------+
```

#### 实现

```python
class PhysicsConstrainedWideNODE(nn.Module):
    def __init__(self, state_dim=7, action_dim=1, hidden=256, depth=5):
        super().__init__()
        self.wheelbase = 1.0  # L

        # 只学习 5 个状态的导数（v, theta, theta_dot, delta, delta_dot）
        self.net = self._build_mlp(state_dim + action_dim, 5, hidden, depth)

    def _build_mlp(self, in_dim, out_dim, hidden, depth):
        layers = [nn.Linear(in_dim, hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, out_dim)]
        # 最后一层零初始化：从恒等映射开始
        nn.init.zeros_(layers[-1].weight)
        nn.init.zeros_(layers[-1].bias)
        return nn.Sequential(*layers)

    def forward(self, s, a):
        # s: [batch, 7], a: [batch, 1]
        # 物理空间中的状态（未归一化）
        v = s[:, 2]
        e_psi = s[:, 1]
        delta = s[:, 5]  # 7D 索引中的 delta

        # 精确运动学（不可学习）
        e_y_dot = v * torch.sin(e_psi)
        e_psi_dot = -v * delta / self.wheelbase

        # NN 学习剩余 5 个状态
        nn_input = torch.cat([s, a], dim=-1)
        nn_output = self.net(nn_input)  # [batch, 5]

        # 组合
        dsdt = torch.zeros(s.shape[0], 7, device=s.device)
        dsdt[:, 0] = e_y_dot          # 精确
        dsdt[:, 1] = e_psi_dot        # 精确
        dsdt[:, 2:] = nn_output       # 学习

        return dsdt
```

#### 预期效果

| 组件 | 当前 NMAE (H=100) | 预期 NMAE | 改善 |
|------|-------------------|-----------|------|
| e_y | 0.5141 | ~0.05 | -90% (精确运动学) |
| e_psi | 0.4298 | ~0.05 | -88% (精确运动学) |
| v | 0.0203 | ~0.02 | 持平 |
| theta | 0.3865 | ~0.30 | -22% (学习负担减少) |
| theta_dot | 0.2957 | ~0.22 | -25% (学习负担减少) |
| delta | 0.7160 | ~0.55 | -23% (学习负担减少) |
| delta_dot | 0.2201 | ~0.18 | -18% (学习负担减少) |
| **Primary** | **0.3930** | **~0.25** | **~36%** |

**理论依据**: e_y 和 e_psi 是误差最大的两个状态（合计贡献约 40% 的总误差），且它们的运动学关系是精确的。硬编码后：
1. 这两个状态的模型误差降为零
2. 网络的学习空间从 7D 减少到 5D
3. 更多容量用于学习真正困难的动力学

**风险**: 
- 物理空间 vs 归一化空间的转换需要仔细处理
- 如果 PyBullet 仿真中的运动学与理论公式有偏差（如 Agent D 指出的 trail/wheelbase 差异），硬编码可能引入系统误差

---

### 3.2 方案 B: Enhanced Wide SiLU NODE (保守改进)

**核心思想**: 在不改变架构的情况下，通过训练策略和正则化改进 Wide SiLU NODE。

#### 改进点

**B1: 合同性正则化 (Contractivity Regularization)**

约束动力学 Jacobian 的最大特征值 < 1，防止误差指数放大：

```python
def contractivity_loss(model, s, a, spectral_bound=0.98):
    """惩罚 Jacobian 对称部分的正特征值"""
    s_req = s.detach().requires_grad_(True)
    dsdt = model(s_req, a)

    jac_norm = 0
    for i in range(7):
        grad = torch.autograd.grad(dsdt[:, i].sum(), s_req,
                                     create_graph=True)[0]
        jac_norm += grad.pow(2).sum()

    # 更精确的实现：计算 Jacobian 对称部分的特征值
    # J_sym = 0.5 * (J + J^T)
    # loss = sum(relu(eig(J_sym) - spectral_bound))
    return jac_norm / (7 * s.shape[0])
```

**B2: 物理约束损失 (软约束)**

在损失函数中添加运动学一致性项：

```python
def physics_constraint_loss(s_pred, s_cur, dt, wheelbase=1.0):
    """软约束：预测的导数应与运动学一致"""
    # 归一化空间中的状态
    e_y_pred = s_pred[:, 0]
    e_psi_pred = s_pred[:, 1]
    v_cur = s_cur[:, 2]
    e_psi_cur = s_cur[:, 1]
    delta_cur = s_cur[:, 5]

    # 运动学导数（在物理空间中）
    e_y_dot_physics = v_cur * torch.sin(e_psi_cur)
    e_psi_dot_physics = -v_cur * delta_cur / wheelbase

    # 预测的导数
    e_y_dot_pred = (e_y_pred - s_cur[:, 0]) / dt
    e_psi_dot_pred = (e_psi_pred - s_cur[:, 1]) / dt

    # 损失
    loss_ey = F.mse_loss(e_y_dot_pred, e_y_dot_physics)
    loss_epsi = F.mse_loss(e_psi_dot_pred, e_psi_dot_physics)

    return loss_ey + loss_epsi
```

**B3: 多步训练修复**

使用 SequentialSegmentDataset 替代 shuffled DataLoader：

```python
class SequentialSegmentDataset(Dataset):
    """确保多步训练使用连续轨迹段"""
    def __init__(self, states, actions, segment_len=20):
        self.states = states
        self.actions = actions
        self.segment_len = segment_len
        self.n_segments = len(states) - segment_len

    def __len__(self):
        return self.n_segments

    def __getitem__(self, idx):
        s = self.states[idx:idx+self.segment_len+1]
        a = self.actions[idx:idx+self.segment_len]
        return s, a
```

**B4: 学习率与训练时间优化**

| 参数 | 当前值 | 建议值 | 理由 |
|------|--------|--------|------|
| Epochs | 200 | 500 | SiLU 收敛更慢但最终更好 |
| LR | 1e-3 | 5e-4 | 更稳定的训练 |
| Batch size | 256 | 128 | 更多梯度更新 |
| Weight decay | 0 | 1e-5 | 轻微正则化 |

#### 预期效果

| 改进 | 预期 Primary 改善 | 置信度 |
|------|-------------------|--------|
| 合同性正则化 | -5 to -10% | 中 |
| 物理约束损失 | -3 to -8% | 中 |
| 多步训练修复 | -5 to -15% | 高 |
| 训练时间优化 | -2 to -5% | 高 |
| **综合** | **~0.28-0.32** | **中** |

**优势**: 实现简单，风险低
**劣势**: 改善幅度有限，无法从根本上解决物理先验缺失问题

---

### 3.3 方案 C: UDE-Enhanced Wide SiLU NODE (混合方法)

**核心思想**: 将已知物理方程硬编码为骨架，Wide SiLU NODE 只学习残差。这是方案 A 的推广。

#### 架构

```
输入: [s(7D), u(1D)] = 8D
      |
      v
+------------------+
| 物理骨架层       |  <-- 可学习参数的物理方程
| e_y_dot = v*sin(e_psi)
| e_psi_dot = -v*delta/L
| theta_ddot = g_over_h*theta - damping*theta_dot
| delta_ddot = -wn2*delta + trail*v*theta - zeta*delta_dot
| v_dot = -0.1*(v - v_target)
+------------------+
      |
      v
+------------------+
| Wide SiLU NODE   |  <-- 学习残差修正
| (128x4, SiLU)    |
| 输入: 8D         |
| 输出: 7D         |
| (残差修正)
+------------------+
      |
      v
+------------------+
| 组合层           |
| dsdt = physics + residual * scale
+------------------+
```

#### 实现

```python
class UDEEnhancedWideNODE(nn.Module):
    def __init__(self, state_dim=7, action_dim=1, hidden=128, depth=4):
        super().__init__()

        # 可学习的物理参数
        self.g_over_h = nn.Parameter(torch.tensor(12.2625))
        self.damping_theta = nn.Parameter(torch.tensor(2.0))
        self.omega_n_sq = nn.Parameter(torch.tensor(25.0))
        self.zeta_wn = nn.Parameter(torch.tensor(5.0))
        self.trail_coeff = nn.Parameter(torch.tensor(0.225))  # wn2 * trail / L
        self.v_damping = nn.Parameter(torch.tensor(0.1))
        self.wheelbase = 1.0

        # 残差网络
        self.residual_net = self._build_mlp(
            state_dim + action_dim, state_dim, hidden, depth
        )
        # 残差缩放因子（可学习）
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def _build_mlp(self, in_dim, out_dim, hidden, depth):
        layers = [nn.Linear(in_dim, hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, out_dim)]
        nn.init.zeros_(layers[-1].weight)
        nn.init.zeros_(layers[-1].bias)
        return nn.Sequential(*layers)

    def forward(self, s, a):
        # 解包状态（在物理空间中）
        e_y = s[:, 0]
        e_psi = s[:, 1]
        v = s[:, 2]
        theta = s[:, 3]
        theta_dot = s[:, 4]
        delta = s[:, 5]
        delta_dot = s[:, 6]

        # 物理骨架
        physics_dsdt = torch.zeros(s.shape[0], 7, device=s.device)
        physics_dsdt[:, 0] = v * torch.sin(e_psi)              # e_y: 运动学
        physics_dsdt[:, 1] = -v * delta / self.wheelbase        # e_psi: 运动学
        physics_dsdt[:, 2] = -self.v_damping * (v - 0.6)        # v: 速度恢复
        physics_dsdt[:, 3] = theta_dot                           # theta: 定义
        physics_dsdt[:, 4] = (self.g_over_h * theta              # theta_dot: 动力学
                              - self.damping_theta * theta_dot)
        physics_dsdt[:, 5] = delta_dot                           # delta: 定义
        physics_dsdt[:, 6] = (-self.omega_n_sq * delta           # delta_dot: 动力学
                              + self.trail_coeff * v * theta
                              - self.zeta_wn * delta_dot)

        # NN 残差
        nn_input = torch.cat([s, a], dim=-1)
        residual = self.residual_net(nn_input)

        # 组合：物理 + 缩放的残差
        dsdt = physics_dsdt + residual * self.residual_scale

        return dsdt
```

#### 预期效果

| 组件 | 贡献 | 说明 |
|------|------|------|
| 运动学硬编码 | e_y/e_psi NMAE -90% | 精确，不需要学习 |
| 动力学骨架 | theta_dot/delta_dot NMAE -30% | 提供正确的归纳偏置 |
| 可学习参数 | 微调物理参数 | 适应 PyBullet 仿真的特定参数 |
| SiLU 残差 | 捕捉未建模效应 | 非线性修正、耦合项 |

**预期 Primary**: ~0.20-0.28

**风险**:
- 物理参数的初始值可能不准确（trail=0.15 vs Meijaard=0.069）
- 残差网络可能学到错误的补偿
- 可学习参数可能发散（需要约束范围）

---

### 3.4 方案 D: Ensemble of Physics-Constrained NODEs

**核心思想**: 训练多个方案 A 的变体，取平均预测以减少方差。

#### 架构

```
Model 1: Physics-Constrained Wide NODE (seed=42)
Model 2: Physics-Constrained Wide NODE (seed=43)
Model 3: Physics-Constrained Wide NODE (seed=44)
Model 4: Physics-Constrained Wide NODE (seed=45)
Model 5: Physics-Constrained Wide NODE (seed=46)
      |
      v
+------------------+
| 集成平均         |
| s_pred = mean(s_pred_1, ..., s_pred_5)
+------------------+
```

#### 预期效果

- 方差减少: sqrt(5) ~ 2.2x
- 如果模型间误差不相关，Primary 可改善 10-20%
- 训练成本: 5x

---

## 4. 方案比较与推荐

### 4.1 理论上限分析

**信息论下限**: 如果所有精确物理关系都被嵌入，且残差网络完美学习，Primary 的理论下限约为 0.05-0.10（主要受限于数据噪声和未建模效应）。

**当前瓶颈**: 
1. e_y/e_psi 的运动学误差（可完全消除）
2. theta_dot/delta 的动力学学习误差（可部分减少）
3. 数据覆盖度不足（需要数据增强）
4. 物理不稳定性放大误差（需要合同性约束）

### 4.2 方案对比

| 方案 | 预期 Primary | 改善 vs 当前最佳 | 实现难度 | 风险 | 训练时间 |
|------|-------------|-----------------|----------|------|----------|
| A: Physics-Constrained NODE | 0.22-0.28 | 29-44% | 低 | 低 | ~400s |
| B: Enhanced NODE (软约束) | 0.28-0.32 | 18-29% | 极低 | 极低 | ~500s |
| C: UDE-Enhanced NODE | 0.20-0.28 | 29-49% | 中 | 中 | ~600s |
| D: Ensemble of A | 0.18-0.25 | 36-54% | 低 | 低 | ~2000s |

### 4.3 推荐路径

**Phase 1 (立即)**: 方案 A -- Physics-Constrained Wide SiLU NODE
- 实现最简单，预期收益最高（风险调整后）
- 核心改动：修改 ODEFunc.forward()，硬编码 e_y_dot 和 e_psi_dot

**Phase 2 (如果 Phase 1 成功)**: 方案 C -- UDE-Enhanced NODE
- 在方案 A 基础上添加可学习物理参数和动力学骨架
- 进一步减少学习空间

**Phase 3 (如果 Phase 2 成功)**: 方案 D -- Ensemble
- 训练 5 个方案 C 的变体
- 取平均预测

**Phase 4 (并行)**: 方案 B 的部分改进
- 合同性正则化可以与任何方案组合
- 多步训练修复是必须的独立任务

---

## 5. 实现细节

### 5.1 关键注意事项

**5.1.1 归一化空间 vs 物理空间**

当前 v9 在归一化空间中训练，但物理方程在物理空间中成立。需要：

```python
# 方案 1: 在物理空间中计算物理约束，然后归一化
def forward_physics_space(self, s_phys, a_phys):
    # s_phys: 物理空间中的状态
    e_y_dot = s_phys[:, 2] * torch.sin(s_phys[:, 1])  # v * sin(e_psi)
    # ... 归一化后传给 NN ...

# 方案 2: 在归一化空间中等价变换
def forward_normalized(self, s_norm, a_norm):
    # s_norm = s_phys / state_std
    v_phys = s_norm[:, 2] * state_std[2]
    e_psi_phys = s_norm[:, 1] * state_std[1]
    e_y_dot_phys = v_phys * torch.sin(e_psi_phys)
    e_y_dot_norm = e_y_dot_phys / state_std[0]
    # ...
```

推荐方案 1（物理空间），更清晰且不易出错。

**5.1.2 积分方法**

保持 Euler 积分（与 v9 一致），避免引入额外变量。RK4 可以作为后续改进。

**5.1.3 训练策略**

```python
# 课程学习（与 v9 一致）
curriculum = [
    (0, 49, 1),    # epoch 0-49: rollout=1
    (50, 99, 5),   # epoch 50-99: rollout=5
    (100, 149, 10), # epoch 100-149: rollout=10
    (150, 299, 20), # epoch 150-299: rollout=20
    (300, 499, 50), # epoch 300-499: rollout=50 (新增)
]

# 损失函数
loss = (loss_single
        + 0.3 * loss_multi
        + 0.1 * loss_physics_constraint
        + 0.01 * loss_contractivity)
```

### 5.2 验证清单

- [ ] 运动学硬编码在物理空间中正确实现
- [ ] 归一化/反归一化转换正确
- [ ] 梯度可以通过硬编码层反向传播（用于 NN 部分）
- [ ] 单步预测与 v9 一致（H=1 应该接近）
- [ ] 长时域预测改善（H=100, 200, 500）
- [ ] 多 seed 验证（至少 3 个 seed）
- [ ] 存活率 100%（至少到 H=500）

---

## 6. 预期结果总结

### 6.1 最佳方案预期

**方案 A (Physics-Constrained NODE)**:

| Horizon | 当前 Wide SiLU | 预期 | 改善 |
|---------|----------------|------|------|
| H=1 | 0.0190 | 0.015 | +21% |
| H=10 | 0.0372 | 0.025 | +33% |
| H=50 | 0.1913 | 0.10 | +48% |
| H=100 | 0.3689 | 0.18 | +51% |
| H=200 | 0.4050 | 0.25 | +38% |
| H=500 | 0.4050 | 0.30 | +26% |
| **Primary** | **0.3930** | **~0.24** | **~39%** |

**方案 C (UDE-Enhanced NODE)**:

| Horizon | 预期 | 改善 vs Wide SiLU |
|---------|------|-------------------|
| H=100 | 0.15 | +59% |
| H=200 | 0.22 | +46% |
| H=500 | 0.28 | +31% |
| **Primary** | **~0.22** | **~44%** |

### 6.2 与 v9 基线比较

| 方法 | Primary | vs v9 |
|------|---------|-------|
| v9 baseline | 0.5110 | - |
| Wide SiLU NODE | 0.3930 | +23.1% |
| **方案 A 预期** | **0.24** | **+53%** |
| **方案 C 预期** | **0.22** | **+57%** |
| **方案 D 预期** | **0.20** | **+61%** |

### 6.3 理论上限

如果所有改进都成功实现：
- 运动学硬编码消除 e_y/e_psi 误差: Primary -15%
- 动力学骨架减少学习负担: Primary -10%
- 合同性正则化防止发散: Primary -5%
- 集成减少方差: Primary -10%
- **综合理论下限**: Primary ~0.15-0.18 (+65-70% vs v9)

---

## 7. 风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| 运动学方程与 PyBullet 不匹配 | 中 | 高 | 先验证 r = v*sin(e_psi) 在数据中成立 |
| 归一化空间转换错误 | 低 | 高 | 编写单元测试验证转换 |
| 可学习物理参数发散 | 中 | 中 | 添加参数范围约束 |
| 过拟合（容量减少后） | 低 | 中 | 保持 weight_decay=1e-5 |
| 训练时间增加 | 低 | 低 | 使用 GPU 加速 |

---

## 8. 结论

### 核心洞察

1. **Wide SiLU NODE 有效因为**: SiLU 激活 + 宽架构 + 联合预测三者协同
2. **Coupling Network 失败因为**: 状态间存在不可分离的非线性耦合
3. **最大改进空间**: 嵌入精确运动学硬约束（消除 e_y/e_psi 的模型误差）
4. **推荐路径**: 方案 A -> 方案 C -> 方案 D

### 下一步行动

1. **立即**: 实现方案 A (Physics-Constrained Wide SiLU NODE)
2. **验证**: 确认运动学方程在 PyBullet 数据中成立
3. **训练**: 使用 500 epochs + 多步训练修复
4. **评估**: 3+ seeds, 5+ test segments
5. **迭代**: 如果成功，实现方案 C 和 D

---

*本设计基于 72h 研究的全部实验结果、Agent 分析报告和对抗式审查结论。*
*关键参考: AGENT_D (物理分析), AGENT_S (耦合分析), AGENT_Q (深层根因), EXP045 (Wide NODE), EXP048 (Hybrid 失败)*
