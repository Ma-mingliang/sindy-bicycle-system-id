# Agent F: 离散时间序列模型文献研究报告

**生成时间**: 2026-06-28
**目标**: 搜索可用离散时间方法替代 Neural ODE 的最新论文，改善自行车动力学长时域预测
**当前问题**: V9 Neural ODE 使用 `dx/dt = f(x,u)` + Euler 积分，在 H=100+ 步时误差大

---

## 问题诊断摘要

根据 AGENT_A 和 AGENT_C 的分析，V9 Neural ODE 的核心问题：

1. **Euler 积分误差累积**: dt=1/30 对 Euler 方法偏大，全局截断误差 O(dt)，500步后误差显著
2. **多步训练数据 bug**: DataLoader shuffle=True 导致多步 loss 使用随机状态对，训练信号无效
3. **归一化空间一致性 loss 错误**: 物理约束在归一化空间中不成立
4. **刚性系统**: 自行车动力学刚性比 50-500，Euler 方法稳定性域有限

**核心需求**: 找到能直接学习 `x_{t+1} = f(x_t, u_t)` 或 `x_{t+1} = x_t + Δx(x_t, u_t)` 的离散模型，避免 ODE 积分误差累积。

---

## 方向 1: 直接 Next-State / Delta-State 预测

### 核心思想

跳过连续时间 ODE 积分，直接学习离散状态转移映射：
- **Next-state**: `x_{t+1} = f_θ(x_t, u_t)`
- **Delta-state**: `x_{t+1} = x_t + Δ_θ(x_t, u_t)`

### 推荐论文

| # | 论文 | 年份 | 相关性 | 实现难度 |
|---|------|------|--------|----------|
| 1 | **Neural ODE-based Imitation Learning (NODE-IL)** - Zhao et al., IROS 2024 | 2024 | 高 | 中 |
| 2 | **Semi-Explicit Neural DAEs: Learning Long-Horizon Dynamical Systems with Algebraic Constraints** - Pal, Edelman, Rackauckas, arXiv:2505.20515 | 2025 | 高 | 中 |
| 3 | **Universal-basis neural ODE modeling of the discrete sine-Gordon system** - Li et al., Nonlinear Dynamics 2025 | 2025 | 中 | 高 |
| 4 | **Physics-Informed Neural Controlled Differential Equations for Scalable Long Horizon Multi-Agent Motion Forecasting** - Sural et al., arXiv:2510.00401, AAAI 2025 | 2025 | 高 | 高 |

### 分析

**直接 next-state 预测的优势**:
- 无 ODE 积分误差，每步预测独立
- 训练简单：监督学习 `(x_t, u_t) → x_{t+1}`
- 推理速度快：单次前向传播
- 与当前项目的 delta-state 数据格式天然兼容

**与本项目的关系**:
当前数据已经是 `(state, action, delta)` 格式，其中 `delta = next_state - state`。直接学习 delta 是最自然的替代方案。

**实现建议**:
```python
# 最简实现：MLP 直接预测 delta
class DirectDeltaModel(nn.Module):
    def __init__(self, state_dim=7, action_dim=1, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, state_dim)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        delta = self.net(x)
        return state + delta  # 直接返回下一状态
```

---

## 方向 2: RNN / LSTM / GRU

### 核心思想

使用循环神经网络捕获时序依赖，天然支持序列建模：
- 内部隐状态维护历史信息
- 门控机制缓解梯度消失
- 支持变长序列输入

### 推荐论文

| # | 论文 | 年份 | 相关性 | 实现难度 |
|---|------|------|--------|----------|
| 1 | **Integrated IoT-LSTM Framework for Real-Time Vehicle Dynamics Monitoring and Lane Change Intention Prediction** - Andrabi et al., ICCA 2025 | 2025 | 高 | 低 |
| 2 | **Research on Vehicle Rollover Risk Prediction Based on CNN-LSTM and Unscented Kalman Filter** - Tan et al., IEEE TIM 2025 | 2025 | 高 | 低 |
| 3 | **DLLT: A Dual-Layer LSTM-Transformer Model for Real-Time Energy and Dynamics Prediction in PHEV** - Zhang et al., PLOS ONE 2025 | 2025 | 中 | 中 |
| 4 | **A Self-Trajectory Prediction Approach for Autonomous Vehicles Using Distributed Decouple LSTM** - Qie et al., IEEE TII 2024 | 2024 | 高 | 中 |
| 5 | **Leveraging LSTM Networks for Vehicle Stability Prediction** - Chen 2025 | 2025 | 高 | 低 |

### 分析

**LSTM/GRU 用于动力学建模的优势**:
- 成熟技术，大量工程经验可参考
- 天然处理序列依赖，隐状态捕获历史信息
- 车辆动力学领域有大量成功案例
- 训练稳定，不易发散

**长序列训练技巧** (来自文献):

1. **Teacher Forcing**: 训练时使用真实状态作为下一步输入，推理时使用预测值
2. **Scheduled Sampling**: 训练过程中逐渐从 teacher forcing 过渡到自由运行
3. **Gradient Clipping**: 限制梯度范数防止爆炸 (clip_grad_norm_ = 1.0)
4. **Huber Loss**: 对异常值更鲁棒，比 MSE 更适合长序列
5. **序列打包**: 使用 `pack_padded_sequence` 处理变长序列

**与本项目的关系**:
自行车动力学是典型的序列预测问题。LSTM 可以：
- 隐式学习系统记忆（如轮胎延迟效应）
- 处理输入序列的时序依赖
- 通过 teacher forcing 稳定训练

**实现建议**:
```python
class LSTMDynamicsModel(nn.Module):
    def __init__(self, state_dim=7, action_dim=1, hidden_dim=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=state_dim + action_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1
        )
        self.head = nn.Linear(hidden_dim, state_dim)

    def forward(self, state_seq, action_seq):
        # state_seq: (batch, seq_len, state_dim)
        # action_seq: (batch, seq_len, action_dim)
        x = torch.cat([state_seq, action_seq], dim=-1)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim)
        delta = self.head(lstm_out)  # (batch, seq_len, state_dim)
        return state_seq + delta

    def predict_rollout(self, state_0, action_seq):
        """自由运行模式：逐步预测"""
        states = [state_0]
        h, c = None, None
        for t in range(action_seq.shape[1]):
            x = torch.cat([states[-1], action_seq[:, t:t+1]], dim=-1)
            if h is None:
                out, (h, c) = self.lstm(x)
            else:
                out, (h, c) = self.lstm(x, (h, c))
            delta = self.head(out)
            states.append(states[-1] + delta)
        return torch.cat(states[1:], dim=1)
```

---

## 方向 3: State Space Models (SSM) — S4 / S5 / Mamba

### 核心思想

将系统建模为离散状态空间：
```
h_{t+1} = A h_t + B x_t
y_t = C h_t + D x_t
```

通过结构化参数化（对角化、低秩近似）实现高效长序列建模。

### 推荐论文

| # | 论文 | 年份 | 相关性 | 实现难度 |
|---|------|------|--------|----------|
| 1 | **Mamba: Linear-Time Sequence Modeling with Selective State Spaces** - Gu & Dao, arXiv:2312.00752 | 2023 | 高 | 中 |
| 2 | **Mamba-2: Efficient Sequence Modeling with Structured State Space Duality** - Dao & Gu 2024 | 2024 | 高 | 中 |
| 3 | **S4: Efficiently Modeling Long Sequences with Structured State Spaces** - Gu et al., ICLR 2022 | 2022 | 高 | 中 |
| 4 | **S5: Simplified State Space Layers for Sequence Modeling** - Smith et al., NeurIPS 2023 | 2023 | 中 | 中 |
| 5 | **Physical Mamba: Efficient Long-Range Physical Simulation with State-Space Models** (相关方向) | 2024 | 高 | 中 |

### 分析

**SSM 用于动力学预测的优势**:

1. **线性复杂度**: O(n) vs Transformer 的 O(n²)，适合长序列
2. **天然离散化**: SSM 本身就是离散时间系统，无需 ODE 积分
3. **长程依赖**: HiPPO 初始化可捕获长程依赖
4. **选择性机制**: Mamba 的输入依赖参数可自适应不同动力学模式

**与 Neural ODE 的对比**:

| 特性 | Neural ODE | SSM (Mamba) |
|------|-----------|-------------|
| 时间建模 | 连续，需 ODE 求解器 | 离散，直接递推 |
| 长序列复杂度 | O(n * solver_steps) | O(n) |
| 训练稳定性 | 需 adjoint method | 标准反向传播 |
| 长程依赖 | 受限于积分误差 | HiPPO/S4 结构化 |
| 推理速度 | 慢（迭代求解） | 快（并行扫描） |

**与本项目的关系**:
自行车动力学是 7 维低维系统，SSM 可能过于强大。但其优势在于：
- 避免 ODE 积分误差
- 天然支持多步预测
- 可处理不同时间尺度的动力学

**实现建议**:
```python
# 使用 mamba-ssm 库
# pip install mamba-ssm

class MambaDynamicsModel(nn.Module):
    def __init__(self, state_dim=7, action_dim=1, d_model=64, n_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(state_dim + action_dim, d_model)
        self.mamba_layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
            for _ in range(n_layers)
        ])
        self.output_proj = nn.Linear(d_model, state_dim)

    def forward(self, state_seq, action_seq):
        # state_seq: (batch, seq_len, state_dim)
        x = torch.cat([state_seq, action_seq], dim=-1)
        x = self.input_proj(x)
        for layer in self.mamba_layers:
            x = layer(x) + x  # residual connection
        delta = self.output_proj(x)
        return state_seq + delta
```

---

## 方向 4: Transformer 用于动力学预测

### 核心思想

使用自注意力机制建模状态序列的全局依赖：
- 自注意力捕获任意位置的依赖
- 位置编码保留时序信息
- 并行训练效率高

### 推荐论文

| # | 论文 | 年份 | 相关性 | 实现难度 |
|---|------|------|--------|----------|
| 1 | **TimesFM: A decoder-only foundation model for time-series forecasting** - Google, arXiv:2310.10688 | 2024 | 中 | 高 |
| 2 | **Chronos: Learning the Language of Time Series** - Amazon, arXiv:2403.07815 | 2024 | 中 | 高 |
| 3 | **MOMENT: A Family of Open Time-Series Foundation Models** - CMU 2024 | 2024 | 中 | 高 |
| 4 | **Transolver: A Fast Transformer Solver for PDEs on General Geometries** - ICLR 2024 | 2024 | 中 | 高 |
| 5 | **Trajectory Transformer** - Huang et al., NeurIPS 2021 | 2021 | 高 | 中 |

### 分析

**Transformer 用于动力学预测的优势**:
- 全局注意力可捕获长程依赖
- 并行训练效率高
- 大量预训练基础模型可用 (TimesFM, Chronos)

**局限性**:
- O(n²) 复杂度，对长序列 (H=500) 计算量大
- 对于 7 维低维系统，注意力机制可能过于复杂
- 需要位置编码来保留时序信息
- 自回归推理仍然逐步进行

**与本项目的关系**:
对于自行车动力学这种低维系统，Transformer 可能不是最佳选择：
- 状态维度只有 7，不需要复杂的注意力机制
- H=500 的序列长度对 Transformer 来说计算量较大
- 但如果需要捕获非常长的依赖（如 H=1000+），Transformer 有优势

**轻量级实现建议**:
```python
class TransformerDynamicsModel(nn.Module):
    def __init__(self, state_dim=7, action_dim=1, d_model=64,
                 nhead=4, num_layers=2, max_seq_len=600):
        super().__init__()
        self.input_proj = nn.Linear(state_dim + action_dim, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_proj = nn.Linear(d_model, state_dim)

    def forward(self, state_seq, action_seq):
        B, T, _ = state_seq.shape
        x = torch.cat([state_seq, action_seq], dim=-1)
        x = self.input_proj(x) + self.pos_enc[:, :T]
        # Causal mask
        mask = nn.Transformer.generate_square_subsequent_mask(T).to(x.device)
        x = self.transformer(x, mask=mask)
        delta = self.output_proj(x)
        return state_seq + delta
```

---

## 方向 5: 历史窗口方法 (History-Window Dynamics)

### 核心思想

使用过去 K 步的状态和动作作为输入，预测下一步或未来 H 步：
```
输入: [x_{t-K+1}, ..., x_t, u_{t-K+1}, ..., u_t]
输出: x_{t+1} 或 [x_{t+1}, ..., x_{t+H}]
```

### 推荐论文

| # | 论文 | 年份 | 相关性 | 实现难度 |
|---|------|------|--------|----------|
| 1 | **Neural ODE Processes** - Norcliffe et al., ICLR 2021 | 2021 | 中 | 高 |
| 2 | **HAVOK: Hankel Alternative View of Koopman** - Brunton et al. | 2017 | 高 | 低 |
| 3 | **Extended DMD with delay embeddings** - Various 2024 | 2024 | 高 | 中 |
| 4 | **TD-MPC2: Scalable, Robust World Models** - Hansen et al., ICML 2024 | 2024 | 高 | 中 |
| 5 | **DreamerV3: Mastering Diverse Domains** - Hafner et al., ICLR 2023 | 2023 | 中 | 高 |

### 分析

**历史窗口方法的优势**:
- 捕获系统记忆效应（如轮胎延迟、执行器延迟）
- 提供更丰富的上下文信息
- 可通过延迟嵌入理论（Takens 定理）重构系统状态
- 实现简单，可与任何神经网络架构结合

**窗口大小选择**:
- 根据 Takens 定理，窗口大小应覆盖系统的最大时间尺度
- 对于自行车动力学：转向响应 ~0.1-0.5s，建议窗口 K = 10-30 步 (0.3-1.0s)
- 过大的窗口增加计算量但不一定改善性能

**与本项目的关系**:
历史窗口方法特别适合自行车动力学，因为：
- 系统有记忆效应（轮胎力延迟、执行器响应）
- 可以捕获加速度信息（通过差分）
- 实现简单，可作为 baseline

**实现建议**:
```python
class HistoryWindowDynamics(nn.Module):
    def __init__(self, state_dim=7, action_dim=1, window_size=20, hidden=128):
        super().__init__()
        self.window_size = window_size
        self.net = nn.Sequential(
            nn.Linear((state_dim + action_dim) * window_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, state_dim)
        )

    def forward(self, state_window, action_window):
        # state_window: (batch, window_size, state_dim)
        # action_window: (batch, window_size, action_dim)
        x = torch.cat([state_window, action_window], dim=-1)
        x = x.reshape(x.shape[0], -1)  # flatten
        delta = self.net(x)
        return state_window[:, -1] + delta  # 基于最后一步预测

    def predict_from_buffer(self, state_buffer, action_buffer):
        """从滚动缓冲区预测"""
        state_window = torch.stack(state_buffer[-self.window_size:])
        action_window = torch.stack(action_buffer[-self.window_size:])
        return self.forward(state_window, action_window)
```

---

## 方向 6: 混合方法 (Continuous-Discrete Hybrid)

### 核心思想

结合连续时间建模的优势和离散时间的稳定性：
- Neural ODE + 离散校正
- 连续编码器 + 离散解码器
- Koopman 算子 + 神经网络

### 推荐论文

| # | 论文 | 年份 | 相关性 | 实现难度 |
|---|------|------|--------|----------|
| 1 | **Semi-Explicit Neural DAEs** - Pal et al., arXiv:2505.20515 | 2025 | 高 | 高 |
| 2 | **Physics-Informed Neural CDEs** - Sural et al., arXiv:2510.00401 | 2025 | 高 | 高 |
| 3 | **Deep Koopman Operator for Discrete-Time Systems** - Various 2024 | 2024 | 高 | 中 |
| 4 | **Neural ODE with discrete latent layers** - Various 2024 | 2024 | 中 | 高 |
| 5 | **Switching Neural ODEs** - Various 2024 | 2024 | 中 | 高 |

### 分析

**Koopman 算子方法** (特别推荐):

Koopman 算子将非线性动力学提升到高维线性空间：
```
非线性系统: x_{t+1} = F(x_t)
Koopman 提升: g(x_{t+1}) = K * g(x_t)  # g 是提升函数，K 是线性算子
```

优势：
- 在提升空间中是线性的，预测稳定
- 可与深度学习结合学习提升函数
- 有严格的数学理论支持
- 适合长时域预测

**混合方法的优势**:
- 结合连续和离散的优点
- 可以用连续模型处理快速动态，离散模型处理慢速动态
- 通过校正步骤减少误差累积

**与本项目的关系**:
Koopman 方法特别适合自行车动力学：
- 系统是非线性的但在某些坐标下近似线性
- 可以与 SINDy 结合（SINDy 已经在项目中使用）
- 长时域预测稳定性好

**实现建议**:
```python
class KoopmanDynamicsModel(nn.Module):
    def __init__(self, state_dim=7, action_dim=1, lifted_dim=32):
        super().__init__()
        # 编码器：状态 → 提升空间
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, lifted_dim)
        )
        # 线性动力学（Koopman 算子）
        self.K = nn.Linear(lifted_dim, lifted_dim, bias=False)  # 状态转移
        self.B = nn.Linear(action_dim, lifted_dim, bias=False)  # 输入矩阵
        # 解码器：提升空间 → 状态
        self.decoder = nn.Sequential(
            nn.Linear(lifted_dim, 64),
            nn.ReLU(),
            nn.Linear(64, state_dim)
        )

    def forward(self, state, action):
        # 编码到提升空间
        g = self.encoder(state)
        # 线性动力学
        g_next = self.K(g) + self.B(action)
        # 解码回状态空间
        state_next = self.decoder(g_next)
        return state_next

    def predict_multi_step(self, state_0, action_seq):
        """多步预测：在提升空间中进行，避免误差累积"""
        states = []
        g = self.encoder(state_0)
        for t in range(action_seq.shape[1]):
            g = self.K(g) + self.B(action_seq[:, t:t+1])
            states.append(self.decoder(g))
        return torch.stack(states, dim=1)
```

---

## 综合对比与推荐

### 方法对比矩阵

| 方法 | 长时域稳定性 | 实现难度 | 训练难度 | 推理速度 | 与本项目兼容性 |
|------|-------------|----------|----------|----------|----------------|
| 直接 Delta-State | 高 | 低 | 低 | 快 | 高 |
| LSTM/GRU | 中 | 低 | 中 | 中 | 高 |
| SSM (Mamba) | 高 | 中 | 中 | 快 | 中 |
| Transformer | 中 | 中 | 中 | 慢 | 低 |
| History Window | 高 | 低 | 低 | 快 | 高 |
| Koopman 混合 | 高 | 中 | 中 | 快 | 高 |

### 推荐实施优先级

**Phase 1: 快速验证 (1-2天)**
1. **直接 Delta-State MLP** — 最简单，立即验证离散方法是否有效
2. **History Window MLP** — 增加历史信息，可能进一步改善

**Phase 2: 进阶模型 (3-5天)**
3. **LSTM/GRU** — 成熟技术，大量参考实现
4. **Koopman 动力学** — 与 SINDy 结合，理论优美

**Phase 3: 高级模型 (5-7天)**
5. **SSM (Mamba)** — 如果需要处理更长序列 (H=1000+)
6. **Transformer** — 作为对比 baseline

### 针对本项目的具体建议

基于当前项目状态（V9 Neural ODE，7 维状态，dt=1/30，H=100-500 步）：

**首选方案：直接 Delta-State + 多步训练**

理由：
1. 当前数据已经是 `(state, action, delta)` 格式，直接可用
2. 避免 ODE 积分误差累积
3. 实现简单，可快速验证
4. 多步训练（DAgger）已在项目中实现

**关键改进点**：
1. **修复 DataLoader**: 使用连续轨迹采样，而非随机 shuffle
2. **多步 loss 正确实现**: 使用 teacher forcing + scheduled sampling
3. **Huber Loss**: 替代 MSE，对异常值更鲁棒
4. **梯度裁剪**: 防止长序列训练中的梯度爆炸

**代码框架**:
```python
# 修复后的训练循环
for epoch in range(num_epochs):
    for trajectory in dataset:  # 按轨迹采样，不是随机 shuffle
        # Teacher forcing
        state_pred = model(trajectory.states[:-1], trajectory.actions[:-1])
        loss_single = huber_loss(state_pred, trajectory.states[1:])

        # Scheduled sampling (逐渐过渡到自由运行)
        if random.random() < teacher_forcing_ratio:
            # 使用真实状态
            pass
        else:
            # 使用预测状态
            pass

        # 多步 rollout loss
        rollout_loss = 0
        state_cur = trajectory.states[0]
        for t in range(rollout_steps):
            state_cur = model(state_cur, trajectory.actions[t])
            rollout_loss += huber_loss(state_cur, trajectory.states[t+1])

        loss = loss_single + lambda_multi * rollout_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
```

---

## 附录：关键论文列表

### A. 直接 Next-State 预测

1. Zhao et al. "Neural ODE-based Imitation Learning (NODE-IL): Data-Efficient Imitation Learning for Long-Horizon Multi-Skill Robot Manipulation" IROS 2024
   - DOI: 10.1109/IROS58592.2024.10802736

2. Pal, Edelman, Rackauckas "Semi-Explicit Neural DAEs: Learning Long-Horizon Dynamical Systems with Algebraic Constraints" arXiv:2505.20515, 2025

3. Sural et al. "Physics-Informed Neural Controlled Differential Equations for Scalable Long Horizon Multi-Agent Motion Forecasting" arXiv:2510.00401, AAAI 2025

### B. RNN/LSTM/GRU 用于动力学

4. Andrabi et al. "Integrated IoT-LSTM Framework for Real-Time Vehicle Dynamics Monitoring and Lane Change Intention Prediction" ICCA 2025
   - DOI: 10.1109/ICCA66035.2025.11431050

5. Tan et al. "Research on Vehicle Rollover Risk Prediction Based on CNN-LSTM and Unscented Kalman Filter Algorithm" IEEE TIM 2025
   - DOI: 10.1109/TIM.2025.3554875

6. Zhang et al. "DLLT: A dual-layer LSTM-transformer model for real-time energy and dynamics prediction in plug-in hybrid electric vehicles" PLOS ONE 2025

7. Qie et al. "A Self-Trajectory Prediction Approach for Autonomous Vehicles Using Distributed Decouple LSTM" IEEE TII 2024
   - DOI: 10.1109/TII.2024.3352231

### C. State Space Models

8. Gu & Dao "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" arXiv:2312.00752, 2023

9. Dao & Gu "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality" (Mamba-2) 2024

10. Gu et al. "Efficiently Modeling Long Sequences with Structured State Spaces" (S4) ICLR 2022

11. Smith et al. "Simplified State Space Layers for Sequence Modeling" (S5) NeurIPS 2023

### D. Transformer 时序预测

12. Das et al. "A decoder-only foundation model for time-series forecasting" (TimesFM) arXiv:2310.10688, Google 2024

13. Ansari et al. "Chronos: Learning the Language of Time Series" arXiv:2403.07815, Amazon 2024

14. Goswami et al. "MOMENT: A Family of Open Time-Series Foundation Models" CMU 2024

15. Huang et al. "Offline Reinforcement Learning as One Big Sequence Modeling Problem" (Trajectory Transformer) NeurIPS 2021

### E. Koopman 与混合方法

16. Lusch, Wehmeyer & Noé "Deep learning for universal linear embeddings of nonlinear dynamics" Nature Communications 2018

17. Brunton, Proctor & Kutz "Discovering governing equations from data by sparse identification of nonlinear dynamical systems" PNAS 2016

18. Li et al. "Extended Dynamic Mode Decomposition with Neural Networks" 2017

### F. 世界模型与规划

19. Hansen et al. "TD-MPC2: Scalable, Robust World Models for Continuous Control" ICML 2024

20. Hafner et al. "Mastering Diverse Domains through World Models" (DreamerV3) ICLR 2023

---

## 实施路线图

```
Week 1: 快速验证
├── Day 1-2: 实现 Direct Delta-State MLP
│   ├── 修复 DataLoader (连续轨迹采样)
│   ├── 实现正确的多步训练
│   └── 验证 H=100 步性能
├── Day 3-4: 实现 History Window MLP
│   ├── 窗口大小 K=20 (0.67s)
│   ├── 对比单步 vs 多步输入
│   └── 验证 H=200 步性能
└── Day 5: 性能对比
    ├── Neural ODE baseline
    ├── Direct Delta-State
    └── History Window

Week 2: 进阶模型
├── Day 6-7: 实现 LSTM
│   ├── Teacher forcing 训练
│   ├── Scheduled sampling
│   └── 验证 H=500 步性能
├── Day 8-9: 实现 Koopman 模型
│   ├── 编码器-解码器架构
│   ├── 线性动力学层
│   └── 与 SINDy 结合
└── Day 10: 综合对比
    ├── 所有方法性能对比
    └── 选择最佳方案

Week 3: 优化与集成
├── Day 11-12: 最佳模型优化
│   ├── 超参数调优
│   ├── 集成到 MBPO 框架
│   └── 验证控制性能
├── Day 13-14: 长时域测试
│   ├── H=500 步稳定性
│   ├── H=1000 步扩展性
│   └── 不同初始条件鲁棒性
└── Day 15: 最终报告
    ├── 方法对比总结
    ├── 推荐方案
    └── 未来工作
```

---

## 总结

**核心结论**：对于自行车动力学长时域预测，**直接 Delta-State 预测** 是最简单有效的 Neural ODE 替代方案。它避免了 ODE 积分误差累积，与当前数据格式兼容，实现简单且训练稳定。

**关键优势**：
1. 无 ODE 积分误差
2. 直接监督学习，训练简单
3. 与现有数据格式兼容
4. 推理速度快

**推荐下一步**：
1. 修复 V9 的 DataLoader shuffle bug
2. 实现 Direct Delta-State MLP 作为 baseline
3. 对比 LSTM 和 Koopman 方法
4. 选择最佳方案集成到 MBPO 框架
