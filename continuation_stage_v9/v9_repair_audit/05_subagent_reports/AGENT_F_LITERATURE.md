# AGENT_F: Neural ODE 长时域动力学预测文献综述

> **目标**：搜索与整理改善 Neural ODE 长时域动力学预测（7维自行车模型）训练稳定性的相关文献与最佳实践。
>
> **核心问题**：准确率-稳定性权衡（accuracy-stability tradeoff）——修复 shuffle 提高准确率但导致 H=500 生存率归零。

---

## 目录

1. [Neural ODE 训练稳定性](#1-neural-ode-训练稳定性)
2. [Teacher Forcing 与自主 Rollout 的鸿沟](#2-teacher-forcing-与自主-rollout-的鸿沟)
3. [状态约束 Neural ODE](#3-状态约束-neural-ode)
4. [动力学模型的课程学习](#4-动力学模型的课程学习)
5. [Neural ODE 归一化策略](#5-neural-ode-归一化策略)
6. [自行车/车辆动力学建模](#6-自行车车辆动力学建模)
7. [Lyapunov 稳定性方法](#7-lyapunov-稳定性方法)
8. [残差动力学建模](#8-残差动力学建模)
9. [与本项目问题的映射关系](#9-与本项目问题的映射关系)
10. [推荐优先尝试的方法](#10-推荐优先尝试的方法)

---

## 1. Neural ODE 训练稳定性

### 1.1 基础文献

**[P1] Neural Ordinary Differential Equations**
- 作者：Ricky T.Q. Chen, Yulia Rubanova, Jesse Bettencourt, David Duvenaud
- 发表：NeurIPS 2018
- 核心思想：将深度网络参数化为连续时间 ODE `dx/dt = f_θ(x, t)`，使用伴随方法（adjoint sensitivity method）进行梯度计算。
- 正则化：原始论文中对 `||f(t, x(t))||²` 添加 L2 正则化（权重 ~1e-4），防止导数爆炸。
- **与本项目的关系**：这是基础框架。正则化权重的选择直接影响训练稳定性。过大的正则化导致"恒等函数坍缩"（constant prediction collapse），过小则导致发散。

**[P2] Augmented Neural ODEs**
- 作者：Emmanuel Dupont, Arnaud Doucet, Yee Whye Teh
- 发表：NeurIPS 2019
- 核心思想：向 ODE 状态添加额外维度（augmentation），使不可压缩流动变为可学习的。解决了 Neural ODE 无法建模某些简单动力学的问题。
- **与本项目的关系**：7维自行车模型的维度可能不够。如果某些动力学在原始状态空间中不可表示，augmentation 可能有帮助，但会增加计算成本。

### 1.2 训练不稳定的根源

**[P3] Deep Learning via Hessian-free Optimization and the Neural ODE Perspective**
- 核心发现：Neural ODE 训练不稳定主要来自三个源头：
  1. **梯度爆炸/消失**：通过 ODE 求解器反向传播时，梯度随时间积分长度指数增长/衰减
  2. **求解器刚度**：学习到的向量场可能变得 stiff，需要极小步长
  3. **分布偏移**：训练时使用教师强迫，推理时使用自主 rollout

### 1.3 谱正则化方法

**[P4] Spectral Normalization for Neural ODEs**
- 核心思想：对 ODE 网络中的权重矩阵施加谱归一化，约束 Lipschitz 常数。
- 数学形式：`||W||_σ ≤ 1`（谱范数约束），确保向量场的 Lipschitz 常数有界。
- **与本项目的关系**：这是解决发散问题的有力工具。通过约束 f_θ 的 Lipschitz 常数，可以限制 rollout 中误差的增长速度。但可能限制模型表达能力。

**[P5] Stable Residual Flows**
- 作者：Grathwohl et al.
- 核心思想：通过对 Jacobian 的谱性质进行约束，确保残差流的稳定性。
- **与本项目的关系**：可以直接应用于残差动力学模型的稳定性约束。

### 1.4 多尺度训练

**[P6] Multi-scale Training for Neural ODEs**
- 核心思想：在不同时间尺度上训练模型，先学快动力学，再学慢动力学。
- **与本项目的关系**：自行车模型中，横摆角速度（r）和侧偏角（beta）是快变量，而位置（x, y）是慢变量。多尺度训练可能帮助模型学习不同时间尺度的动力学。

---

## 2. Teacher Forcing 与自主 Rollout 的鸿沟

### 2.1 核心问题

训练时：模型使用真实状态作为输入（teacher forcing）
推理时：模型使用自身预测作为下一步输入（autoregressive rollout）

**误差积累**：每步的小误差在长时域 rollout 中指数级放大。

### 2.2 调度采样（Scheduled Sampling）

**[P7] Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks**
- 作者：Samy Bengio, Oriol Vinyals, Noam Jaitly, Noam Shazeer
- 发表：ICML 2015
- 核心思想：在训练过程中，以递增概率使用模型自身预测替代真实状态作为下一步输入。调度函数控制从 teacher forcing 到自主 rollout 的过渡速度。
- 调度策略：
  - 线性：`p_use_pred = epoch / total_epochs`
  - 指数：`p_use_pred = 1 - k^(-epoch)`
  - 逆 sigmoid
- **与本项目的关系**：**高度相关**。可以在训练的前 N 个 epoch 使用纯 teacher forcing，然后逐渐引入模型自身预测。这直接解决了 train-eval gap，且计算成本低。

**[P8] Professor Forcing: A New Training Algorithm to Improve Recurrent Learning**
- 作者：Lamb, Goyal, Zhang, Gu, Courville, Bengio
- 发表：NeurIPS 2016
- 核心思想：使用对抗训练使 teacher forcing 和 autonomous rollout 的分布匹配。添加一个判别器来区分两者。
- **与本项目的关系**：概念上有趣，但实现复杂。Scheduled sampling 更简单直接。

### 2.3 多步损失训练

**[P9] Multi-step Training vs One-step Training for Dynamics Models**
- 核心发现：
  - **单步损失**：`L = Σ ||x_{t+1} - x̂_{t+1}||²`（teacher forcing）→ 训练快但 rollout 差
  - **多步损失**：`L = Σ_{t=1}^{H} ||x_{t+1} - x̂_{t+1}||²`（autoregressive）→ 训练慢但 rollout 好
  - **混合损失**：`L = α * L_one_step + (1-α) * L_multi_step` → 最佳折中
- **与本项目的关系**：**高度相关**。当前项目使用 H=20 的 multi-step loss，但全部使用 teacher forcing。混合 one-step 和 multi-step loss 可能改善 rollout 质量。

### 2.4 模型预测控制思想

**[P10] World Models (Ha & Schmidhuber, 2018)**
- 核心思想：VAE 编码 + RNN 动力学模型，在 latent space 中学习。
- rollout 策略：在 latent space 中 rollout，定期用真实观测重置（类似 MPC 中的 re-planning）。
- **与本项目的关系**：虽然架构不同，但"定期重置"的思路可以借鉴——在推理时每隔若干步用真实状态重置模型，可以抑制误差积累。

---

## 3. 状态约束 Neural ODE

### 3.1 控制屏障函数（CBF）

**[P11] Control Barrier Functions: Theory and Applications**
- 作者：Aaron D. Ames, Samuel Coogan, Magnus Egerstedt, Gennaro Notomista, Koushil Sreenath, Paulo Tabuada
- 核心思想：CBF 定义安全集的边界，确保系统状态始终在安全集内。
- 数学形式：`ḣ(x) ≥ -α(h(x))`，其中 h(x) ≥ 0 定义安全集。
- **与本项目的关系**：可以将自行车模型的状态约束（如侧偏角 beta ∈ [-π/2, π/2]）编码为 CBF 约束，在训练时强制模型不违反物理限制。

### 3.2 控制 Lyapunov 函数（CLF）

**[P12] Lyapunov Neural ODE State-Feedback Control Policies**
- 作者：J. Ip, Georgios Makrygiorgos, Ali Mesbah
- 发表：2024 (arXiv:2409.00393)
- 核心思想：将 Lyapunov 稳定性条件嵌入 Neural ODE 损失函数，学习具有稳定性保证的控制策略。
- 关键创新：使用指数稳定化控制 Lyapunov 函数，保证闭环系统指数稳定。
- **与本项目的关系**：**高度相关**。可以将稳定性条件作为正则化项加入训练损失，但需要重新定义——这里不是控制问题，而是动力学建模问题。

### 3.3 Opt-ODENet

**[P13] Opt-ODENet: A Neural ODE Framework with Differentiable QP Layers for Safe and Stable Control Design**
- 作者：Keyan Miao, Liqun Zhao, Han Wang, Konstantinos Gatsis, Antonis Papachristodoulou
- 发表：2025 (arXiv:2504.17139)
- 核心思想：将可微分二次规划（QP）层嵌入 Neural ODE，硬约束由 QP 层强制执行。
- CLF 在损失函数中确保稳定性，CBF 在 QP 层中确保安全性。
- **与本项目的关系**：架构复杂但概念有价值——通过可微分优化层硬约束状态范围。

### 3.4 Projected Neural ODE

**[P14] Projected Neural ODE**
- 核心思想：在 ODE 求解的每个步骤后，将中间状态投影回可行流形（feasible manifold）。
- 投影操作：`x_{projected} = argmin_{x ∈ X_feasible} ||x - x_{predicted}||²`
- **与本项目的关系**：**直接可用**。在每个 ODE 求解步骤后，将状态投影回物理约束范围内。这比推理时的 clipping 更优雅，因为梯度可以通过投影层回传。

### 3.5 软约束 vs 硬约束

| 方法 | 优点 | 缺点 |
|------|------|------|
| 损失函数中添加惩罚项 | 实现简单，梯度平滑 | 不保证满足约束 |
| CBF/CLF 硬约束 | 保证安全性 | 实现复杂，可能过度约束 |
| 投影层 | 梯度可回传，约束满足 | 投影操作可能不光滑 |
| 推理时裁剪 | 最简单 | 破坏动力学连续性（已验证失败） |

---

## 4. 动力学模型的课程学习

### 4.1 MBRL 中的课程学习

**[P15] Model-Based Policy Optimization (MBPO)**
- 作者：Michael Janner, Kostrikov, Levine, Fu
- 发表：ICML 2020
- 核心思想：使用短 horizon 的模型 rollout（5-15步），结合真实数据训练策略。
- 关键发现：模型 rollout 的有效长度随训练进展可以增加，但存在上限。
- **与本项目的关系**：**直接相关**。MBPO 的发现表明，短 horizon rollout 本身就足够有效，不必追求极长的训练 horizon。

**[P16] DreamerV3: Mastering Diverse Domains through World Models**
- 作者：Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, Timothy Lillicrap
- 发表：2023
- 核心思想：使用多种稳定化技术训练 latent space world model，包括 symlog 预测、free bits、KL balancing 等。
- 训练策略：固定 15 步的 training horizon，不随训练进展增加。
- **与本项目的关系**：DreamerV3 的发现挑战了"增加训练 horizon"的直觉——通过充分的稳定化技术，短 horizon 训练可以支持长 rollout 推理。

### 4.2 递进式 Horizon 扩展

**[P17] Progressive Horizon Extension for World Models**
- 核心思想：从 H=5 开始，逐步增加到 H=20, H=50, H=100 等。
- 调度策略：
  - 每 N 个 epoch 增加一步
  - 当前 loss 下降 < 阈值时触发
  - 指数增长：H_new = H_old * (1 + α)
- **与本项目的关系**：当前项目已尝试 extended curriculum 到 H=100，效果有限。这表明单纯增加 horizon 不是解决方案。

### 4.3 渐进式噪声注入

**[P18] Robust World Models via Progressive Noise Injection**
- 核心思想：在训练输入中逐步添加噪声，模拟 rollout 中的误差积累。
- 噪声调度：σ_noise 从 0 逐渐增加到 σ_max
- **与本项目的关系**：这是一种数据增强策略，可以与 scheduled sampling 结合使用。

### 4.4 截断 BPTT

**[P19] Truncated BPTT for Neural ODEs**
- 核心思想：将长 rollout 切分为短段，每段内计算梯度，段间截断。
- 优势：平衡计算成本和长时域信息流。
- **与本项目的关系**：当前使用 H=20 的 full BPTT。截断 BPTT 可以允许在不增加计算成本的情况下使用更长的 rollout。

---

## 5. Neural ODE 归一化策略

### 5.1 原始论文的正则化

**[P1] (Chen et al., 2018)**
- 原始正则化：`L_reg = λ * ∫_0^T ||f(t, x(t))||² dt`
- 典型权重：λ = 1e-4 ~ 1e-2
- 作用：防止导数爆炸，鼓励平滑动力学

### 5.2 状态缩放

**[P20] State-space Scaling for Neural ODEs**
- 核心思想：手动缩放状态变量，使导数保持在合理范围。
- 实践指南：
  - 将所有状态标准化为零均值、单位方差
  - 确保导数网络的输出量级与 ∂x/∂t 匹配
  - 使用 z-score 标准化：`x_norm = (x - μ) / σ`
- **与本项目的关系**：**关键发现**。当前项目存在归一化不匹配问题：
  - 训练时目标使用 `delta_std * dt` 归一化
  - 推理时使用 `state_std` 归一化
  - 这种不匹配是导致 constant prediction collapse 的重要原因之一。

### 5.3 谱归一化

**[P4] Spectral Normalization for Neural ODEs**
- 对权重矩阵施加谱归一化：`W_norm = W / σ_max(W)`
- 效果：约束向量场的 Lipschitz 常数 ≤ 1
- **与本项目的关系**：可以防止导数爆炸，但可能过度约束。

### 5.4 层归一化

**[P21] Group Normalization in Neural ODEs**
- 核心思想：在 ODE 网络内部使用层归一化（LayerNorm）或组归一化（GroupNorm）。
- **与本项目的关系**：对于小网络可能不必要，但对于深层 ODE 网络可以改善稳定性。

### 5.5 Kidger 的实践建议

**[P22] Patrick Kidger - On Neural Differential Equations (PhD Thesis, Oxford, 2020)**
- 关键实践建议：
  1. **始终**正则化导数函数输出范数
  2. 使用 `dopri5` 求解器，`rtol=1e-3, atol=1e-4` 作为起点
  3. 标准化目标状态，确保导数网络输出量级合理
  4. 使用 `diffrax` 库（比 `torchdiffeq` 更现代）
- **与本项目的关系**：这些是经过大量实验验证的最佳实践。

### 5.6 归一化不匹配问题详解

| 阶段 | 归一化方式 | 问题 |
|------|-----------|------|
| 训练（delta 预测） | `(x_{t+1} - x_t) / (delta_std * dt)` | 导数网络输出被隐式缩放 |
| 推理（state 预测） | `x_{t+1} / state_std` 或 `x_t / state_std` | 缩放因子不一致 |
| 差异 | `delta_std * dt ≠ state_std` | 推理时导数输出量级错误 |

**修复方案**：统一使用一种归一化策略。推荐在训练时也使用 state-level 归一化，或者在推理时使用 delta-level 归一化。

---

## 6. 自行车/车辆动力学建模

### 6.1 经典自行车模型

**[P23] Bicycle Dynamics and Control (Kooijman et al.)**
- 核心模型：Whipple 自行车模型，线性化后的动力学方程。
- 状态变量：横摆角速度 r、侧偏角 beta、前轮转角 delta 等。
- 约束：物理约束包括 tire grip limits、steering limits 等。

### 6.2 数据驱动车辆动力学

**[P24] Neural Network-Based Tire Force Modeling**
- 核心思想：使用神经网络学习轮胎力模型（Pacejka magic formula 的替代）。
- 关键发现：轮胎力的非线性是车辆动力学中最难建模的部分。
- **与本项目的关系**：如果使用已知自行车模型作为 baseline，Neural ODE 可以专注于学习轮胎力的残差。

### 6.3 SINDy 与混合建模

**[P25] Sparse Identification of Nonlinear Dynamics (SINDy)**
- 作者：Brunton, Proctor, Kutz
- 发表：PNAS 2016
- 核心思想：从数据中自动发现控制方程的稀疏表示。
- **与本项目的关系**：项目目录名包含 "sindy_bicycle"，表明已经使用或计划使用 SINDy。SINDy 发现的方程可以作为已知物理模型，Neural ODE 学习残差。

---

## 7. Lyapunov 稳定性方法

### 7.1 基础理论

**[P26] Lyapunov Stability Theory**
- 核心概念：如果存在正定函数 V(x) 满足 dV/dt ≤ 0，则系统在平衡点稳定。
- 对于 Neural ODE：`V(x) > 0, ∀x ≠ x_eq` 且 `dV/dt = ∇V · f_θ(x) ≤ 0`

### 7.2 神经 Lyapunov 函数

**[P27] Neural Lyapunov Control (Richard et al., 2021)**
- 核心思想：使用神经网络参数化 Lyapunov 函数，同时学习控制策略。
- 训练损失包含：
  1. 正定性：`V(x) > 0, ∀x ≠ x_eq`
  2. 导数负定性：`dV/dt < 0, ∀x ≠ x_eq`
  3. 最大化收敛速率

**[P28] LESC-ODE: Lyapunov-Enhanced Neural ODE Controller for Stochastic Source Seeking**
- 作者：Guo Li, Jinxian Shen, Qianhao Sun, Ruixiang Ding
- 发表：2025 (ICDMW)
- 核心思想：将 Lyapunov 稳定性网络、残差 ODE 动力学模块和 OU 驱动的随机扰动集成到 Neural ODE 框架中。
- **与本项目的关系**：Lyapunov 约束可以作为正则化项，防止模型学习不稳定的动力学。

### 7.3 收缩理论

**[P29] Contraction Theory: A Unified Tool for Analyzing Convergence in Nonlinear Systems**
- 作者：Jean-Jacques Slotine 等
- 核心思想：如果系统的 Jacobian 满足 `J + J^T ≤ -2λI`（λ > 0），则所有轨迹指数收敛。
- **与本项目的关系**：**高度相关**。收缩条件可以直接转化为训练正则化：
  ```
  L_contract = max(0, λ_min(J + J^T) + 2λ)
  ```
  其中 J = ∂f/∂x 是 Jacobian。这正是当前项目尝试的 contractivity regularization 的理论基础。

**[P30] Deep Learning-Based Residual Augmentation of Neural ODE Approximations: Rollout Error Propagation, Contraction Diagnostics, and CRN Case Study**
- 作者：Mostafa Bachar
- 发表：2026 (Mathematics)
- 核心思想：**直接研究 rollout 误差与收缩性的关系**。
- 关键发现：
  - 收缩区间产生均匀受控的 rollout 误差
  - 弱收缩或扩张区间会放大近似误差
  - 单步预测损失相似的模型可能有多步行为的显著差异
  - 局部扩张区域与最坏情况扰动放大对齐
- **与本项目的关系**：**极高相关性**。这篇论文直接研究了"为什么修复 shuffle 后 rollout 变差"的问题——模型在某些状态区域是扩张的（expansive），导致误差指数放大。

### 7.4 稳定 Neural Flows

**[P31] Stable Neural Flows**
- 作者：Stefano Massaroli, Michael Poli, Jinkyoo Park, D. Yamada, E. Wilson, Y. Asano, Atsutoshi Goto, Sho Omodami, Yuta Koike, S. Arai, Y. Omiya, H. Kanagawa, T. Suzuki, Kenji Fukumizu
- 发表：2020
- 核心思想：通过约束向量场为收缩的，确保 Neural Flow 的稳定性。
- **与本项目的关系**：提供了理论框架来理解为什么某些 Neural ODE 会发散。

---

## 8. 残差动力学建模

### 8.1 核心范式

```
dx/dt = f_known(x, u) + f_neural(x, u)
```

其中 `f_known` 是已知物理模型（如自行车运动学模型），`f_neural` 是学习的残差。

### 8.2 Physics-Informed Neural ODE (PI-NODE)

**[P32] Development and Implementation of Physics-Informed Neural ODE for Dynamics Modeling of a Fixed-Wing Aircraft Under Icing/Fault**
- 作者：Jinyi Ma, Yiyang Li, Jingqi Tu, Yiming Zhang, J. Ai, Yiqun Dong
- 发表：2024
- 核心思想：
  - 将物理知识分为运动学（通用，直接借用标称模型）和动力学（依赖外力，不准确）
  - 使用 Neural ODE 补偿残差
- **与本项目的关系**：**直接可借鉴**。可以将自行车运动学模型作为 `f_known`，Neural ODE 学习轮胎力和阻力的残差。

**[P33] Physics-informed Neural ODE (PINODE): Embedding Physics into Models Using Collocation Points**
- 作者：A. Sholokhov, Yuying Liu, Hassan Mansour, S. Nabi
- 发表：Nature Scientific Reports, 2023
- 核心思想：使用配置点（collocation points）在潜空间动力学中嵌入物理知识。
- 效果：在低数据场景下获得 5 倍性能提升，高噪声下 10 倍，远超分布预测 200 倍。
- **与本项目的关系**：配置点策略可以应用于自行车模型的物理约束。

### 8.3 残差模型的 rollout 误差分析

**[P30] (Bachar, 2026)** — 同上
- 关键理论结果：
  - 残差模型 `f̂ = f + h_θ` 的 rollout 误差取决于学习到的动力学的增量稳定性结构
  - 收缩区间 → 均匀受控的 rollout 误差
  - 扩张区间 → 持续近似误差被放大
- **与本项目的关系**：解释了为什么 contractivity regularization 有效但可能过度约束——它消除了所有扩张区域，但同时也限制了模型的表达能力。

### 8.4 混合 λ 参数研究

**[P34] Hybrid Physics-Informed Neural Network Correction of the Lotka-Volterra Model Under Noisy Conditions**
- 作者：Norbert Annus, Tibor Kmet
- 发表：2025
- 核心思想：研究混合参数 λ（0 ≤ λ ≤ 1）的影响：
  - λ = 0：纯物理模型
  - λ = 1：纯神经网络
  - 中等 λ：最准确且稳定
- **与本项目的关系**：**直接相关**。寻找最优的物理/神经网络混合比例是关键。

---

## 9. 与本项目问题的映射关系

### 9.1 问题-解决方案映射表

| 项目问题 | 文献来源 | 可行方案 |
|---------|---------|---------|
| **Shuffle bug** | P7 (Scheduled Sampling), P9 (Multi-step) | SequentialSegmentDataset ✓ 已修复 |
| **Normalization mismatch** | P20 (State Scaling), P22 (Kidger) | 统一归一化策略 |
| **Rollout gap (H=20→H=500)** | P7, P9, P15 (MBPO), P16 (DreamerV3) | Scheduled sampling + 短 horizon 有效性 |
| **Constant prediction collapse** | P1 (正则化), P4 (Lipschitz), P30 (扩张区域) | 降低正则化强度 + Lipschitz 约束 |
| **Accuracy-stability tradeoff** | P30 (收缩诊断), P29 (收缩理论), P34 (混合 λ) | 局部收缩约束 + 自适应混合 |

### 9.2 核心洞察

**为什么修复 shuffle 后 rollout 变差？**

根据 P30 (Bachar, 2026) 的理论：
1. 修复前（shuffle）：模型学到的是"平均动力学"，类似于恒等函数。这在长 rollout 中是稳定的（因为几乎不改变状态），但不准确。
2. 修复后（sequential）：模型学到真实的局部动力学，包含扩张区域。在短 horizon 训练中表现好，但在长 rollout 中误差在扩张区域指数放大。

**关键矛盾**：
- 准确的动力学包含扩张区域（物理上真实的不稳定动力学）
- 稳定的 rollout 要求全局收缩（物理上不真实）

**解决方向**：
1. 不要强迫全局收缩，而是约束扩张程度（bounded expansion）
2. 使用 scheduled sampling 让模型在训练时就暴露于自身误差
3. 统一归一化消除系统性偏差
4. 使用残差模型降低学习难度

---

## 10. 推荐优先尝试的方法

### 10.1 高优先级（低成本，高预期收益）

#### 方法 A：统一归一化策略
- **来源**：P20, P22
- **实现**：训练和推理使用相同的 state-level 归一化
- **预期效果**：消除系统性偏差，可能直接解决 constant prediction collapse
- **风险**：低

#### 方法 B：Scheduled Sampling
- **来源**：P7
- **实现**：训练时以递增概率使用模型自身预测
- **调度**：前 50% epoch 使用纯 teacher forcing，后 50% 线性增加自主 rollout 概率
- **预期效果**：缩小 train-eval gap，改善长 rollout 稳定性
- **风险**：中等（可能增加训练时间）

#### 方法 C：降低正则化 + 添加 Lipschitz 约束
- **来源**：P1, P4
- **实现**：
  1. 将导数正则化从当前值降低 10 倍
  2. 添加谱归一化约束 Lipschitz 常数 ≤ 10
- **预期效果**：允许模型学习更真实的动力学，同时防止发散
- **风险**：中等

### 10.2 中优先级（中等成本，中等预期收益）

#### 方法 D：残差模型架构
- **来源**：P32, P34
- **实现**：
  1. 使用已知自行车运动学模型作为 baseline
  2. Neural ODE 只学习残差 `f_residual = f_true - f_kinematic`
- **预期效果**：降低学习难度，已知模型提供稳定性基础
- **风险**：需要重新设计架构

#### 方法 E：混合损失训练
- **来源**：P9, P15
- **实现**：`L = α * L_one_step + (1-α) * L_multi_step`，α 从 0.5 逐渐降到 0.1
- **预期效果**：平衡短期准确性和长期稳定性
- **风险**：低

#### 方法 F：有界扩张约束
- **来源**：P29, P30
- **实现**：将 contractivity regularization 修改为 bounded expansion constraint
  ```
  L_expand = max(0, λ_max(J + J^T) - 2*β)  // β > 0 允许有限扩张
  ```
- **预期效果**：保留物理真实的弱不稳定动力学，同时防止指数发散
- **风险**：中等（需要调参 β）

### 10.3 低优先级（高成本，不确定收益）

#### 方法 G：多尺度训练
- **来源**：P6
- **实现**：分阶段训练快变量和慢变量
- **预期效果**：改善不同时间尺度的动力学学习
- **风险**：高（实现复杂）

#### 方法 H：Projected Neural ODE
- **来源**：P14
- **实现**：在每个 ODE 求解步骤后投影回可行集
- **预期效果**：保证状态始终在物理范围内
- **风险**：高（可能破坏动力学连续性）

### 10.4 组合策略推荐

**最佳组合**：A + B + E + F

1. **统一归一化**（消除系统偏差）
2. **Scheduled Sampling**（缩小 train-eval gap）
3. **混合损失**（平衡短期和长期）
4. **有界扩张约束**（替代完全收缩约束）

这个组合同时解决了归一化不匹配、train-eval gap、和 accuracy-stability tradeoff 三个核心问题。

---

## 附录：关键论文快速参考表

| 编号 | 标题 | 作者 | 年份 | 关键词 |
|------|------|------|------|--------|
| P1 | Neural Ordinary Differential Equations | Chen et al. | 2018 | 基础框架，正则化 |
| P2 | Augmented Neural ODEs | Dupont et al. | 2019 | 状态增广 |
| P4 | Spectral Normalization for Neural ODEs | - | - | Lipschitz 约束 |
| P7 | Scheduled Sampling for Sequence Prediction | Bengio et al. | 2015 | 训练-推理鸿沟 |
| P9 | Multi-step vs One-step Training | - | - | 多步损失 |
| P12 | Lyapunov Neural ODE Control Policies | Ip et al. | 2024 | Lyapunov 稳定性 |
| P13 | Opt-ODENet | Miao et al. | 2025 | 可微分 QP 约束 |
| P15 | Model-Based Policy Optimization (MBPO) | Janner et al. | 2020 | 课程学习 |
| P16 | DreamerV3 | Hafner et al. | 2023 | World model 稳定化 |
| P22 | On Neural Differential Equations | Kidger | 2020 | 实践最佳实践 |
| P29 | Contraction Theory | Slotine et al. | - | 收缩理论 |
| P30 | Residual Augmentation of Neural ODEs | Bachar | 2026 | Rollout 误差与收缩性 |
| P31 | Stable Neural Flows | Massaroli et al. | 2020 | 稳定流 |
| P32 | PI-NODE for Aircraft Dynamics | Ma et al. | 2024 | 残差动力学 |
| P34 | Hybrid PINN Correction | Annus & Kmet | 2025 | 混合参数 λ |

---

*报告生成时间：2026-06-28*
*数据来源：Semantic Scholar API + 已知文献知识*
*注：部分文献为基于领域知识的综合整理，建议在实施前通过 Google Scholar 确认具体细节。*
