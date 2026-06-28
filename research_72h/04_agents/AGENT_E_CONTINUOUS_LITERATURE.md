# Agent E: Continuous-Time Dynamics Literature Review

> 目标：搜索改善自行车动力学长时域预测 (H=100+) 的最新方法
> 当前问题：v9 Neural ODE 在 H=100+ 时 NMAE > 0.5，e_y 和 e_psi 误差最大
> 搜索日期：2026-06-28
> 来源：arXiv API 检索 (2023-2026)

---

## 1. 稳定 Neural ODE (Stable Neural ODE)

### 1.1 Locally Stable Neural ODEs with Characterized Region of Attraction
- **年份**: 2026 (arXiv:2606.19109)
- **来源**: arXiv, 2026-06-17
- **核心方法**: 提出一类神经 ODE，通过联合学习的最大 Lyapunov 函数的梯度场约束动力学，使其在吸引域内逼近局部指数稳定动力学。模型的吸引域恰好由学习到的 Lyapunov 函数的 1-子水平集刻画。
- **长步程机制**: Lyapunov 函数约束保证了轨迹不会发散，吸引域提供了理论保证的安全区域。
- **相关性**: **极高** - 直接解决 Neural ODE 长时域发散问题。对于自行车模型，可以约束 e_y 和 e_psi 不发散。
- **实现难度**: 中高 - 需要实现 Lyapunov 网络和梯度场约束训练。

### 1.2 ISS-BKNO: Input-to-State Stable Bundle Koopman Neural ODEs
- **年份**: 2026 (arXiv:2606.04395)
- **来源**: arXiv, 2026-06-03
- **核心方法**: 统一框架集成 Koopman 算子辨识、Neural ODE、纤维束几何和 ISS 稳定性认证。三阶段提升管道：束感知编码器、环境条件化 Koopman 主干（谱约束在左半平面）、残差 Neural ODE 修正（Jacobian 满足二次扇区界）。Lyapunov-based ISS 正则化将稳定性要求转化为可微惩罚。
- **长步程机制**: Koopman 谱约束 + ISS Lyapunov 正则化 + Jacobian 扇区界，三重保证全局收敛。
- **相关性**: **极高** - 结合了 Koopman 线性化和 Neural ODE 的稳定性，特别适合自行车这种有控制输入的系统。
- **实现难度**: 高 - 架构复杂，需要实现纤维束编码器、Koopman 谱约束和 ISS 惩罚。

### 1.3 Stable Long-Horizon Neural ODE ROM via Learned Feedback
- **年份**: 2026 (arXiv:2604.13820)
- **来源**: arXiv, 2026-04-15
- **核心方法**: 提出闭环架构，将编码后的演化场特征反馈回动力学，解决自回归潜空间动力学模型的长时域误差累积问题。比较了三种反馈表示：标量、线性 POD 和非线性 CNN。CNN 反馈显著稳定了长时域 rollout。
- **长步程机制**: **闭环反馈** - 在每个时间步将编码后的状态特征反馈给动力学网络，防止误差累积。最佳模型在临床容差内捕获 90.3% 验证案例（开环基线仅 43.7%）。
- **相关性**: **极高** - 闭环反馈机制直接可应用于自行车 Neural ODE，通过反馈当前状态特征来修正预测。
- **实现难度**: 中 - 需要在 ODE 求解循环中加入反馈编码器。

### 1.4 Tracking Finite-Time Lyapunov Exponents to Robustify Neural ODEs
- **年份**: 2026 (arXiv:2602.09613)
- **来源**: arXiv, 2026-02-10
- **核心方法**: 研究有限时间 Lyapunov 指数 (FTLE) 作为 Neural ODE 输入扰动的指数分离度量。提出 FTLE 正则化训练算法，通过在输入动力学早期阶段抑制远离零的指数来提高鲁棒性。
- **长步程机制**: FTLE 正则化防止轨迹指数分离，减少对初始条件的敏感性。
- **相关性**: **高** - FTLE 正则化可以作为额外的训练损失项，防止 e_y 和 e_psi 指数增长。
- **实现难度**: 中 - 需要计算 Jacobian 和 FTLE，但有现成的 autograd 支持。

### 1.5 FxTS-Net: Fixed-Time Stable Learning Framework for Neural ODEs
- **年份**: 2024 (arXiv:2411.09118)
- **来源**: arXiv, 2024-11-14
- **核心方法**: 基于固定时间稳定性 (FxTS) Lyapunov 条件训练 Neural ODE。FxTS-Loss 鼓励动力学在用户定义的固定时间内收敛到准确预测。还提供了更精确的有界非消失扰动系统时间上界估计。
- **长步程机制**: 固定时间稳定性保证 - 动力学在有界时间内收敛，即使存在输入扰动。
- **相关性**: **高** - 固定时间收敛保证对长时域预测很有价值，可以设置合理的收敛时间界。
- **实现难度**: 中 - 需要实现 FxTS-Loss 和扰动采样。

### 1.6 Robust Convolution Neural ODEs via Contractivity-promoting Regularization
- **年份**: 2025 (arXiv:2508.11432)
- **来源**: arXiv, 2025-08-15
- **核心方法**: 通过收缩性促进正则化使 Convolution Neural ODE 对输入噪声和对抗攻击鲁棒。
- **长步程机制**: 收缩性正则化保证轨迹不会指数发散。
- **相关性**: **高** - 收缩性是解决长时域预测不稳定的关键性质。
- **实现难度**: 中 - 正则化项相对容易实现。

### 1.7 How to Train Your Neural ODE: Jacobian and Kinetic Regularization
- **年份**: 2020 (arXiv:2002.02798) - 经典
- **来源**: arXiv, 2020-02-07
- **核心方法**: 引入最优传输和稳定性正则化的理论组合，鼓励 Neural ODE 偏好更简单的动力学。更简单的动力学导致更快收敛和更少的求解器离散化。
- **长步程机制**: Jacobian 正则化 + 动能正则化限制动力学复杂度。
- **相关性**: **高** - 经典方法，可作为基线正则化策略。
- **实现难度**: 低 - 正则化项容易添加到现有训练循环。

---

## 2. 多步/轨迹级训练 (Multi-step / Trajectory-level Training)

### 2.1 MPINeuralODE: Multiple-Initial-Condition Physics-Informed Neural ODEs
- **年份**: 2026 (arXiv:2605.13305)
- **来源**: arXiv, 2026-05-13
- **核心方法**: 结合软物理信息残差和多初始条件 (MIC) 多重射击课程。物理项锚定向量场幅度，MIC 扩大支持集。沿三个轴评估：样本外误差、长时域稳定性、Hamilton 漂移。
- **长步程机制**: **多重射击 + 物理约束** - 多重射击防止长轨迹误差累积，物理残差约束向量场。在 Lotka-Volterra 上比基线 Neural ODE 减少 26% MSE。
- **相关性**: **极高** - 直接解决长时域预测问题，多重射击策略可直接应用于自行车模型。
- **实现难度**: 中 - 需要实现多初始条件采样和物理残差损失。

### 2.2 Latent Neural ODEs with Sparse Bayesian Multiple Shooting
- **年份**: 2022 (arXiv:2210.03466)
- **来源**: arXiv, 2022-10-07
- **核心方法**: 原则性的多重射击技术，将轨迹分成可管理的短段并行优化，同时确保连续段上的概率控制。使用 Transformer 识别网络进行不规则采样轨迹的 amortized 编码。
- **长步程机制**: **贝叶斯多重射击** - 轨迹分段并行优化 + 概率连续性约束。在多个大规模基准上达到 SOTA。
- **相关性**: **极高** - 多重射击是解决长轨迹训练的标准方法，贝叶斯版本提供了更好的不确定性量化。
- **实现难度**: 中高 - 需要实现变分推断和 Transformer 识别网络。

### 2.3 Learning Long-Horizon Predictions for Quadrotor Dynamics
- **年份**: 2024 (arXiv:2407.12964)
- **来源**: arXiv, 2024-07-17
- **核心方法**: 系统研究了长时域预测的关键设计选择：多种架构、历史数据、多步损失公式。提出了解耦动力学学习方法，简化学习过程并增强模块性。**序列建模技术在减少累积误差方面优于其他方案**。
- **长步程机制**: 多步损失 + 解耦动力学 + 序列建模。在真实四旋翼数据上验证。
- **相关性**: **极高** - 四旋翼和自行车都是非线性动力学系统，设计选择直接可迁移。特别是多步损失和解耦动力学学习。
- **实现难度**: 中 - 多步损失容易实现，解耦学习需要架构调整。

### 2.4 Autoregressive Long-Horizon Prediction of Plasma Edge Dynamics
- **年份**: 2025 (arXiv:2512.23884)
- **来源**: arXiv, 2025-12-29
- **核心方法**: Transformer 自回归代理模型，训练时使用增加的自回归视野 (1-100 步)。**更长视野的训练系统地改善了 rollout 稳定性并缓解了误差累积**，实现了数百到数千步的稳定预测。
- **长步程机制**: **课程视野训练** - 逐步增加训练时的 rollout 长度。这是最简单有效的长时域训练策略之一。
- **相关性**: **高** - 课程视野训练策略可直接应用于 Neural ODE 训练。
- **实现难度**: 低 - 仅需修改训练循环逐步增加 rollout 长度。

### 2.5 Neural CDEs as Correctors for Learned Time Series Models
- **年份**: 2025 (arXiv:2512.12116)
- **来源**: arXiv, 2025-12-13
- **核心方法**: 预测-修正框架，其中预测器生成多步预测，Neural CDE 修正器修正预测误差。修正器与不规则采样时间序列兼容，兼容连续和离散时间预测器。引入两种正则化策略改善外推性能。
- **长步程机制**: **预测-修正** - Neural CDE 作为修正器可以叠加在任何基础预测器上，逐步修正累积误差。
- **相关性**: **高** - 可以在现有 Neural ODE 基础上叠加 CDE 修正器。
- **实现难度**: 中 - 需要实现 CDE 修正器和两阶段训练。

---

## 3. 物理-数据混合 (Physics-Data Hybrid)

### 3.1 PI-NODE-SR: Physics-Informed Neural ODEs with Scale-Aware Residuals
- **年份**: 2025 (arXiv:2511.11734)
- **来源**: arXiv, 2025-11-13
- **核心方法**: 结合低阶显式求解器 (Heun 方法) 和残差归一化，平衡不同时间尺度上演化的状态变量贡献。在 Hodgkin-Huxley 方程上，从单个振荡学习并外推超过 100ms。
- **长步程机制**: 残差归一化 + 物理信息约束。端到端学习向量场使神经修正可以补偿数值扩散。
- **相关性**: **极高** - 自行车动力学中 e_y 和 e_psi 可能有不同的时间尺度，残差归一化可以平衡它们的贡献。
- **实现难度**: 中 - 需要实现残差归一化和 Heun 求解器。

### 3.2 Residual-Corrected ECM with Universal Differential Equations
- **年份**: 2026 (arXiv:2605.06419)
- **来源**: arXiv, 2026-05-07
- **核心方法**: 残差修正混合公式，其中一阶 Thevenin 等效电路提供主导电压结构，嵌入为 UDE 的紧凑神经网络仅修正潜在极化失配。物理模型锚定了纯学习模型最脆弱的预测。
- **长步程机制**: **物理锚定 + 神经残差** - 物理模型提供主导结构，神经网络仅学习残差。在所有条件下都达到最低电压误差，比 LSTM 减少 48% MAE。
- **相关性**: **极高** - 这正是自行车模型需要的方法：用已知的自行车动力学方程作为物理锚定，用 Neural ODE 学习残差。
- **实现难度**: 中 - 需要将已知自行车动力学方程嵌入 ODE 右端。

### 3.3 Continual Robot Policy Learning via Variational Neural Dynamics
- **年份**: 2026 (arXiv:2606.27353)
- **来源**: arXiv, 2026-06-25
- **核心方法**: 学习条件感知动力学模型，结合解析物理先验和神经残差。循环编码器从最近交互推断当前隐藏条件，条件化残差模型和策略。在真实四旋翼轨迹跟踪中，策略在约 1 秒内从重复干扰中恢复。
- **长步程机制**: **物理先验 + 神经残差 + 条件编码** - 物理先验提供基础动力学，神经残差捕获未建模效应，条件编码处理时变参数。
- **相关性**: **高** - 框架可迁移，自行车也有已知的物理先验和未建模的残差动力学。
- **实现难度**: 中高 - 需要实现变分编码器和条件化残差模型。

### 3.4 Hard-Constrained Neural Networks for Residual Dynamics Learning
- **年份**: 2025 (arXiv:2511.23307)
- **来源**: arXiv, 2025-11-28
- **核心方法**: Hybrid Recurrent Physics-Informed Neural Network (HRPINN)，将已知物理作为硬结构约束嵌入循环积分器中，仅学习残差动力学。Projected HRPINN 集成预测-投影机制严格强制代数不变量。
- **长步程机制**: **硬物理约束 + 残差学习** - 物理约束由架构保证，不是软惩罚。
- **相关性**: **高** - 硬约束比软惩罚更可靠，可以保证自行车动力学的基本物理性质。
- **实现难度**: 中 - 需要将物理约束编码为网络架构。

### 3.5 Forecasting N-Body Dynamics: Neural ODE vs Universal Differential Equations
- **年份**: 2025 (arXiv:2512.20643)
- **来源**: arXiv, 2025-12-12
- **核心方法**: 比较 Neural ODE 和 UDE 在 N 体问题上的性能。UDE 模型更数据高效，仅需 20% 数据即可正确预测，而 Neural ODE 需要 90%。
- **长步程机制**: UDE 嵌入已知物理定律，减少对数据量的需求。
- **相关性**: **高** - 直接证明了 UDE 相比纯 Neural ODE 的优势。
- **实现难度**: 中 - 需要在 Julia 中实现（DiffEqFlux.jl）。

### 3.6 Forecasting N-Body Dynamics (Comparative Study)
- **年份**: 2026 (arXiv:2603.26921)
- **来源**: arXiv, 2026-03-27
- **核心方法**: 系统比较 PINNs 和 Neural ODEs 在非线性生物系统 (Morris-Lecar 模型) 上的性能。
- **长步程机制**: 对比不同方法在长时域预测上的优劣。
- **相关性**: **中** - 提供方法选择的参考。
- **实现难度**: 低 - 纯参考价值。

---

## 4. Neural CDE/SDE

### 4.1 PINCoDE: Physics-Informed Neural CDEs for Long Horizon Multi-Agent Motion
- **年份**: 2025 (arXiv:2510.00401)
- **来源**: arXiv, 2025-10-01
- **核心方法**: 基于 Neural CDE 的长时域运动预测模型，结合物理信息约束。条件化于未来目标，强制机器人运动的物理约束。通过课程学习逐步训练，从 10 个机器人扩展到 100 个。
- **长步程机制**: **物理约束 + 课程学习 + Neural CDE** - 连续时间框架避免离散化误差，物理约束防止不合理轨迹。课程学习使 4 分钟视野的预测误差减少 2.7 倍。
- **相关性**: **极高** - Neural CDE 的连续时间特性适合自行车动力学，物理约束和课程学习策略可直接迁移。
- **实现难度**: 中 - 需要实现 Neural CDE 和物理约束。

### 4.2 ImProNCDE: Impulse-Corrected Neural CDEs with Prototype Learning
- **年份**: 2026 (arXiv:2606.19680)
- **来源**: arXiv, 2026-06-18
- **核心方法**: 残差脉冲校准 (RIC) 在访问时间注入残差脉冲校正，当观测偏离连续预测时重新校准潜状态。原型引导轨迹稳定器 (PTS) 将潜轨迹吸引向可学习的诊断原型。
- **长步程机制**: **脉冲校准 + 原型吸引** - RIC 修正累积误差，PTS 防止轨迹漂移。专门解决长期数值积分的误差累积。
- **相关性**: **高** - 脉冲校准思想可应用于 Neural ODE，在观测点修正累积误差。
- **实现难度**: 中 - 需要实现脉冲注入和原型学习。

### 4.3 Efficient Neural CDEs via Attentive Kernel Smoothing
- **年份**: 2026 (arXiv:2602.02157)
- **来源**: arXiv, 2026-02-02
- **核心方法**: 用核和高斯过程平滑替代精确插值，控制轨迹正则性。注意力多视图 CDE (MV-CDE) 使用可学习查询进行路径重建。显著减少函数评估次数 (NFE) 和推理时间。
- **长步程机制**: 核平滑减少控制路径的高频振荡，使自适应求解器不需要过小步长。
- **相关性**: **中高** - 减少 NFE 可以加速长时域预测，但不直接解决精度问题。
- **实现难度**: 中 - 需要实现核平滑和注意力机制。

### 4.4 Neural ODE and SDE Models for Adaptation and Planning in MBRL
- **年份**: 2026 (arXiv:2603.23245)
- **来源**: arXiv, 2026-03-24
- **核心方法**: 比较 Neural ODE 和 SDE 在模型基强化学习中的性能。Neural SDE 更有效地捕获转移动力学的内在随机性。引入潜 SDE 模型，结合 ODE 和 GAN 训练的随机组件。
- **长步程机制**: Neural SDE 的随机组件可以建模不确定性，防止过拟合到确定性轨迹。
- **相关性**: **中高** - 如果自行车数据有随机性（如风扰动），SDE 可能比 ODE 更合适。
- **实现难度**: 中 - 需要实现 Neural SDE 和 GAN 训练。

### 4.5 G-SLiCEs: Universal Time Series Generation with Neural CDEs
- **年份**: 2026 (arXiv:2605.28507)
- **来源**: arXiv, 2026-05-27
- **核心方法**: 证明最大表达力的结构化线性控制微分方程 (SLiCEs) 是通用时间序列生成器。提出 G-SLiCEs，基于路径空间的流匹配连续时间模型。
- **长步程机制**: 连续时间框架自然支持任意观测网格，表达力改善概率预测。
- **相关性**: **中** - 理论价值高，但更偏向生成任务。
- **实现难度**: 高 - 需要实现路径空间流匹配。

---

## 5. Koopman 和线性潜空间

### 5.1 Adaptive Deep Koopman Operator for Vehicle Dynamics Modeling
- **年份**: 2026 (arXiv:2606.15094)
- **来源**: arXiv, 2026-06-13
- **核心方法**: **直接应用于车辆动力学！** 将 7DOF 动态平衡约束嵌入学习目标，确保提升流形的结构保真度和物理可解释性。提出 Physics-Informed Variable Step-Size NLMS 算法，利用 NLMS 投影性质作为稳定伪逆求解器。
- **长步程机制**: Koopman 线性化 + 物理约束 + 在线自适应。在线更新保证实时可行性（0.421ms 执行时间）。
- **相关性**: **极高** - 直接应用于车辆动力学，7DOF 模型与自行车模型高度相关。物理约束的 Koopman 提升可以线性化自行车动力学。
- **实现难度**: 中 - 需要实现 Deep Koopman 编码器和在线更新算法。

### 5.2 Koopman Autoencoders with Continuous-Time Latent Dynamics for Fluid Dynamics
- **年份**: 2026 (arXiv:2602.02832)
- **来源**: arXiv, 2026-02-02
- **核心方法**: 连续时间 Koopman 自编码器，潜动力学服从 dz/dt = K_cont * z，通过矩阵指数 z(tau) = exp(K_cont * tau) * z(0) 在任意视野单步闭式推断。识别了一组有效的结构约束：**rollout 训练、前向-后向一致性、潜正则化、物理条件化 LoRA**。
- **长步程机制**: **Koopman 线性化 + 结构约束** - 线性潜动力学从根本上消除了非线性误差累积。rollout 训练确保长时域稳定性。110x 推理加速。
- **相关性**: **极高** - 如果能将自行车动力学提升到线性潜空间，长时域预测将根本性改善。rollout 训练策略可直接应用。
- **实现难度**: 中高 - 需要实现连续时间 Koopman 自编码器和矩阵指数。

### 5.3 DeepMDMD: Deep Embedded Multiplicative DMD
- **年份**: 2026 (arXiv:2606.05131)
- **来源**: arXiv, 2026-06-03
- **核心方法**: 学习潜空间和分区，同时强制 Koopman 乘积规则作为精确代数约束。训练交替进行精确乘法算子更新和可微潜聚类步骤。非零谱位于单位圆上，字典由动力学塑造。
- **长步程机制**: 代数约束保证 Koopman 闭合，减少谱污染。在 158,624 维系统上保持相干结构和长期谱统计。
- **相关性**: **高** - 代数约束的 Koopman 学习可以改善长期预测稳定性。
- **实现难度**: 高 - 需要实现乘法算子更新和潜聚类。

### 5.4 Deep-Koopman-KANDy: Dictionary Discovery with KANs
- **年份**: 2026 (arXiv:2605.06000)
- **来源**: arXiv, 2026-05-07
- **核心方法**: Deep-Koopman 编码器和解码器替换为两层 KAN，层集构造和链式梯度恒等式暴露学习可观测量的组合结构。后训练符号字典读出。
- **长步程机制**: KAN 提供可解释的字典，后训练读出避免了训练时的字典选择问题。
- **相关性**: **中高** - 可解释性好，但更偏向符号发现。
- **实现难度**: 中高 - 需要实现 KAN 和字典读出。

### 5.5 Learning Predictive Control with Deep Koopman for Autonomous Vehicles
- **年份**: 2026 (arXiv:2606.08136)
- **来源**: arXiv, 2026-06-06
- **核心方法**: Deep Koopman 提升到可解释线性可观空间，学习预测控制 (LPC) 框架在每个预测区间内通过 receding-horizon actor-critic 学习闭环状态反馈策略。
- **长步程机制**: Koopman 线性化 + 闭环策略学习。在红旗 EHS3 平台上实际验证。
- **相关性**: **高** - 自动驾驶车辆的 Koopman 方法，与自行车模型相关。
- **实现难度**: 中高 - 需要实现 actor-critic 和 Koopman 预测器。

### 5.6 Physics-informed Deep Mixture-of-Koopmans for Vehicle Dynamics
- **年份**: 2026 (arXiv:2603.17416)
- **来源**: arXiv, 2026-03-18
- **核心方法**: 双分支编码器 + 混合 Koopman 算子框架，适应分布式电动卡车的多样驾驶模式。物理信息监督机制基于时间车辆运动的几何一致性。
- **长步程机制**: 混合 Koopman 适应不同驾驶模式，物理监督保证几何一致性。
- **相关性**: **高** - 车辆动力学的 Koopman 方法，混合策略可处理不同工况。
- **实现难度**: 中高 - 需要实现混合 Koopman 和物理监督。

---

## 6. 长时域预测最新方法 (2024-2026)

### 6.1 MODE: Mamba Enhanced by Low-Rank Neural ODEs
- **年份**: 2026 (arXiv:2601.00920)
- **来源**: arXiv, 2026-01-01
- **核心方法**: 整合低秩 Neural ODE 与增强 Mamba 架构。分段选择性扫描机制，受伪 ODE 动力学启发，自适应聚焦于显著子序列。低秩公式减少计算开销同时保持表达力。
- **长步程机制**: Mamba 选择性扫描 + 低秩 Neural ODE。分段扫描改善长程依赖建模。
- **相关性**: **中高** - Mamba 架构在长序列建模上表现优异，可考虑替代或增强现有 Neural ODE。
- **实现难度**: 中 - 需要实现 Mamba 块和低秩 ODE。

### 6.2 Error-Conditioned Neural Solvers (ENS)
- **年份**: 2026 (arXiv:2606.27354)
- **来源**: arXiv, 2026-06-25
- **核心方法**: PDE 残差场作为网络每次迭代的直接输入，使网络能读取自身错误的空间结构并学习更新策略来迭代修正预测。在湍流 Kolmogorov 流上达到 10 倍精度提升。
- **长步程机制**: **误差条件化** - 网络学习从自身错误中修正，而不是优化残差。
- **相关性**: **中高** - 误差条件化思想可应用于 Neural ODE，在每个时间步读取并修正累积误差。
- **实现难度**: 中 - 需要实现误差编码和迭代修正。

### 6.3 Function-Space Priors for Bayesian Neural ODEs (Vessel Trajectory)
- **年份**: 2026 (arXiv:2606.06351)
- **来源**: arXiv, 2026-06-04
- **核心方法**: 在向量场的有限测量点集上直接施加 GP 核先验，结合概率多重射击在时间段间解耦推断同时保持全局一致性。
- **长步程机制**: **GP 先验 + 概率多重射击** - GP 先验编码向量场的平滑性和局部性，多重射击防止长轨迹误差累积。
- **相关性**: **高** - 贝叶斯方法提供不确定性量化，多重射击解决长时域问题。
- **实现难度**: 高 - 需要实现变分推断和 GP 核正则化。

### 6.4 Efficient, Accurate and Stable Gradients for Neural ODEs
- **年份**: 2024 (arXiv:2410.11648)
- **来源**: arXiv, 2024-10-15
- **核心方法**: 代数可逆 ODE 求解器，显著改善递归检查点的时间和内存成本。精确梯度、高阶且数值稳定。
- **长步程机制**: 可逆求解器允许更高效的长轨迹训练，减少内存瓶颈。
- **相关性**: **中** - 工程优化，改善训练效率但不直接改善预测精度。
- **实现难度**: 中 - 需要实现可逆求解器。

---

## 7. 综合推荐：按优先级排序的改进策略

### Tier 1: 立即尝试 (实现难度低，预期收益高)

| 方法 | 来源 | 预期改善 | 实现要点 |
|------|------|----------|----------|
| **课程视野训练** | 2.4 Plasma Edge | H=100+ 稳定性 | 逐步增加 rollout 长度: 10→50→100→200 |
| **多步 rollout 损失** | 2.3 Quadrotor | e_y/e_psi 累积误差 | 在 loss 中包含 [H=10, 50, 100] 的预测误差 |
| **Jacobian + 动能正则化** | 1.7 How to Train | 动力学平滑性 | 添加 \|\|df/dx\|\|_F 和 \|\|f\|\|^2 正则项 |
| **物理锚定 + 神经残差** | 3.2 UDE Battery | 数据效率 + 物理一致性 | dX/dt = f_bicycle(X,u) + NN(X,u) |

### Tier 2: 中期改进 (实现难度中，预期收益高)

| 方法 | 来源 | 预期改善 | 实现要点 |
|------|------|----------|----------|
| **闭环反馈 Neural ODE** | 1.3 Learned Feedback | 长时域稳定性 | 每步反馈编码特征给 ODE 网络 |
| **多重射击训练** | 2.1 MPINeuralODE | 长轨迹训练稳定性 | 轨迹分段 + 连续性约束 |
| **Koopman 线性化** | 5.2 Continuous Koopman | 根本性消除非线性误差累积 | 学习线性潜空间 dz/dt = Kz |
| **FTLE 正则化** | 1.4 FTLE Robustify | 防止指数分离 | 正则化 Lyapunov 指数 |
| **残差归一化** | 3.1 PI-NODE-SR | 平衡不同状态尺度 | 对 e_y, e_psi 分别归一化 |

### Tier 3: 长期探索 (实现难度高，理论价值高)

| 方法 | 来源 | 预期改善 | 实现要点 |
|------|------|----------|----------|
| **Lyapunov 约束 Neural ODE** | 1.1 Locally Stable | 稳定性理论保证 | Lyapunov 网络 + 梯度场约束 |
| **ISS-BKNO** | 1.2 ISS-BKNO | 全局收敛 + ISS 增益 | 纤维束编码器 + Koopman + ISS |
| **Neural CDE 修正器** | 2.5 CDE Corrector | 累积误差修正 | CDE 叠加在 Neural ODE 上 |
| **连续时间 Koopman** | 5.2 Koopman AE | 线性潜空间 + 110x 加速 | 矩阵指数 + rollout 训练 |

---

## 8. 与当前项目的具体关联分析

### 当前问题诊断
- **e_y 和 e_psi 误差最大**: 这两个状态代表横向位置偏航角误差，是自行车动力学中最敏感的状态
- **H=100+ 时 NMAE > 0.5**: 表明误差在长 rollout 中指数累积

### 根本原因
1. **开环 rollout 误差累积**: 没有任何修正机制
2. **纯数据驱动**: 没有利用已知的自行车动力学方程
3. **单步训练**: 只优化一步预测，不考虑多步误差传播
4. **无稳定性约束**: 动力学可能不稳定

### 推荐改进路径

```
Phase 1 (立即):
  ├─ 课程视野训练: 10→50→100→200
  ├─ 多步 rollout 损失: L = sum(L_h for h in [10, 50, 100])
  └─ 物理残差: dX/dt = f_bicycle(X,u) + NN(X,u)

Phase 2 (中期):
  ├─ 闭环反馈: 每步反馈编码特征
  ├─ 多重射击: 轨迹分段训练
  └─ Jacobian 正则化: ||df/dx||_F^2

Phase 3 (长期):
  ├─ Koopman 线性化: 学习线性潜空间
  └─ Lyapunov 约束: 稳定性理论保证
```

---

## 9. 参考文献索引

### 稳定 Neural ODE
1. arXiv:2606.19109 - Locally Stable Neural ODEs (2026)
2. arXiv:2606.04395 - ISS-BKNO (2026)
3. arXiv:2604.13820 - Stable Long-Horizon ROM via Feedback (2026)
4. arXiv:2602.09613 - FTLE Robustify (2026)
5. arXiv:2411.09118 - FxTS-Net (2024)
6. arXiv:2508.11432 - Contractivity Regularization (2025)
7. arXiv:2002.02798 - Jacobian & Kinetic Regularization (2020)

### 多步/轨迹级训练
8. arXiv:2605.13305 - MPINeuralODE (2026)
9. arXiv:2210.03466 - Bayesian Multiple Shooting (2022)
10. arXiv:2407.12964 - Quadrotor Long-Horizon (2024)
11. arXiv:2512.23884 - Plasma Edge Curriculum (2025)
12. arXiv:2512.12116 - CDE Corrector (2025)

### 物理-数据混合
13. arXiv:2511.11734 - PI-NODE-SR (2025)
14. arXiv:2605.06419 - UDE Battery (2026)
15. arXiv:2606.27353 - Variational Neural Dynamics (2026)
16. arXiv:2511.23307 - Hard-Constrained HRPINN (2025)
17. arXiv:2512.20643 - N-Body UDE vs NODE (2025)

### Neural CDE/SDE
18. arXiv:2510.00401 - PINCoDE (2025)
19. arXiv:2606.19680 - ImProNCDE (2026)
20. arXiv:2602.02157 - Attentive CDE (2026)
21. arXiv:2603.2325 - Neural ODE/SDE for MBRL (2026)
22. arXiv:2605.28507 - G-SLiCEs (2026)

### Koopman 和线性潜空间
23. arXiv:2606.15094 - Adaptive Koopman Vehicle (2026)
24. arXiv:2602.02832 - Continuous Koopman AE (2026)
25. arXiv:2606.05131 - DeepMDMD (2026)
26. arXiv:2605.06000 - Deep-Koopman-KANDy (2026)
27. arXiv:2606.08136 - Koopman MPC Vehicle (2026)
28. arXiv:2603.17416 - Mixture-of-Koopmans Vehicle (2026)

### 长时域预测
29. arXiv:2601.00920 - MODE Mamba+NeuralODE (2026)
30. arXiv:2606.27354 - ENS Error-Conditioned (2026)
31. arXiv:2606.06351 - Bayesian Neural ODE Vessel (2026)
32. arXiv:2410.11648 - Reversible ODE Solvers (2024)
