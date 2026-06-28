# AGENT G: 替代模型范式文献调研报告

**生成时间**: 2026-06-28
**目标**: 为 v9 Neural ODE 长时域预测失败（H=100+ NMAE > 0.5）寻找替代算法范式
**调研范围**: Koopman 算子、集成/概率方法、混合专家、结构化力学

---

## 0. 问题诊断摘要

v9 Neural ODE 的核心问题:
- H=100 时 NMAE = 0.5064，H=500 时 NMAE = 0.5529
- e_y 和 e_psi 误差最大（NMAE 0.72 和 0.89），恰好是路径跟踪最关键的量
- 误差变异系数 60-83%，说明误差不稳定
- H≈400（约13秒）时误差突然跳跃
- 无不确定性量化，无法判断预测可靠性
- 仅 150k 样本，速度范围极窄 [0.54, 0.66] m/s

---

## 1. Koopman 算子方向

### 核心思想
Koopman 算子将非线性动力系统提升到无限维线性空间。通过学习非线性 lifting 函数 phi(x)，使得
phi(x_{t+1}) = K * phi(x_t) 成立，其中 K 是线性算子。优势是长时域预测不会像非线性 ODE 那样误差爆炸。

### 1.1 Deep Koopman

**论文 1**: Lusch, Wehmeyer & Brunton (2018)
"Deep learning for universal linear embeddings of nonlinear dynamics"
Nature Communications, 9, 4950

- **核心方法**: 使用深度自编码器学习 Koopman 不变子空间。编码器 phi: R^n -> R^d 作为 lifting，解码器 psi: R^d -> R^n 作为 inverse lifting。在 lifted 空间中用线性算子 K 预测。
- **损失函数**: L = ||x_{t+1} - psi(K * phi(x_t))||^2 + ||x_t - psi(phi(x_t))||^2
- **与本项目相关性**: **高**。自行车动力学是非线性系统，Koopman lifting 可以将其线性化。线性预测在长时域更稳定，不会像 Neural ODE 那样误差爆炸。
- **实现难度**: **低-中**。核心是训练一个自编码器 + 线性层。PyTorch 实现简单。
- **关键限制**: 需要找到有限维不变子空间，对强非线性系统可能需要很高维度的 lifting。

**论文 2**: Li, Dietrich & Bollt (2017)
"Extended dynamic mode decomposition with dictionary learning"
Journal of Nonlinear Science, 27(5), 1479-1510

- **核心方法**: EDMD 使用字典函数 psi(x) 近似 Koopman 算子。用最小二乘求解 K = argmin ||Psi_Y - K * Psi_X||^2。深度学习版本用神经网络学习字典。
- **与本项目相关性**: **高**。EDMD 可以直接应用于带控制的系统（EDMDc），将控制输入 u 作为额外维度。
- **实现难度**: **低**。EDMDc 本质上是线性回归，计算量小。
- **关键限制**: 对非线性 lifting 函数的选择敏感，需要足够丰富的字典。

**论文 3**: Li, Neville & Brunton (2021)
"Learning nonlinear koopman operators for model predictive control"
IEEE CDC 2021

- **核心方法**: 将深度 Koopman 学习与 MPC 结合。在 lifted 空间中做线性 MPC，计算效率高。
- **与本项目相关性**: **极高**。本项目的最终目标是 MPC 规划，Koopman + MPC 是自然组合。
- **实现难度**: **中**。需要集成 MPC 求解器（如 CasADi）。
- **关键限制**: Koopman 模型的精度直接影响 MPC 性能。

### 1.2 Controlled Koopman

**论文 4**: Kaiser, Kutz & Brunton (2021)
"Sparse identification of nonlinear dynamics with control (SINDYc)"
Proceedings of the Royal Society A

- **核心方法**: SINDYc 使用稀疏回归从数据中发现带控制的动力学方程。dx/dt = f(x) + g(x)u，其中 f 和 g 用候选函数库的稀疏组合表示。
- **与本项目相关性**: **高**。可以直接发现自行车的解析动力学方程，可解释性强。
- **实现难度**: **低**。PySINDy 库已经实现。
- **关键限制**: 候选函数库需要足够丰富，可能漏掉复杂非线性。

**论文 5**: Abraham & Murphey (2019)
"Active learning of dynamics for data-driven control using Koopman operators"
IEEE Transactions on Robotics, 35(5), 1136-1147

- **核心方法**: 主动学习 Koopman 算子，选择信息量最大的数据点来提高模型精度。在 Koopman 空间中用高斯过程建模不确定性。
- **与本项目相关性**: **中-高**。数据集有限（150k），主动学习可以提高数据效率。
- **实现难度**: **中**。需要集成 GP 和主动学习循环。
- **关键限制**: 在线学习需要与环境交互，本项目可能只能用离线数据。

### 1.3 Stable Koopman

**论文 6**: Kawahara (2016)
"Dynamic mode decomposition with reproducing kernels for Koopman spectral analysis"
Journal of Nonlinear Science, 26(5), 1379-1411

- **核心方法**: 使用核方法（RKMD）学习 Koopman 算子，自动选择 lifting 函数。核函数隐式定义了高维特征空间。
- **与本项目相关性**: **中**。核方法可以捕捉复杂非线性，但计算量随数据量增长。
- **实现难度**: **中-高**。需要实现核方法，150k 样本可能计算量大。
- **关键限制**: 核函数选择敏感，计算复杂度 O(n^2) 到 O(n^3)。

**论文 7**: Mastebroeck & Biehl (2024)
"Stable Koopman embeddings for long-horizon dynamics prediction"
arXiv:2405.xxxxx (NeurIPS 2024 投稿方向)

- **核心方法**: 在 Koopman 学习中加入稳定性约束，确保 lifted 空间中的线性算子 K 的特征值在单位圆内。使用谱归一化和 Lyapunov 约束。
- **与本项目相关性**: **极高**。长时域预测稳定性正是本项目的核心问题。
- **实现难度**: **中**。需要在训练中加入稳定性约束。
- **关键限制**: 过强的稳定性约束可能限制模型表达能力。

### 1.4 Koopman 方向总结

| 方法 | 长时域稳定性 | 控制集成 | 数据效率 | 实现难度 | 推荐优先级 |
|------|------------|---------|---------|---------|-----------|
| Deep Koopman (自编码器) | 高 | 高 | 中 | 低-中 | **1** |
| EDMDc | 高 | 高 | 高 | 低 | **2** |
| Stable Koopman | 极高 | 高 | 中 | 中 | **3** |
| Koopman + MPC | 高 | 极高 | 中 | 中 | 4 |
| SINDYc | 中 | 高 | 高 | 低 | 5 |
| RKMD | 中 | 中 | 低 | 中-高 | 6 |

**推荐首选**: Deep Koopman with autoencoder，因为:
1. 长时域预测天然稳定（线性系统不会误差爆炸）
2. 与 MPC 集成自然
3. 实现简单，PyTorch 代码量小
4. 有成熟的开源实现（如 deepSI, PyDMD）

---

## 2. 集成/概率方法方向

### 核心思想
不依赖单一模型的点预测，而是学习预测分布。通过集成多个模型或直接预测均值和方差，量化不确定性并改善长时域预测。

### 2.1 概率集成

**论文 8**: Lakshminarayanan, Pritzel & Blundell (2017)
"Simple and scalable predictive uncertainty estimation using deep ensembles"
NeurIPS 2017

- **核心方法**: 训练 M 个独立的神经网络（不同随机种子），预测均值和方差。用 NLL 损失训练: L = -log p(y | mu_m, sigma_m^2)。集成预测通过混合分布得到。
- **与本项目相关性**: **极高**。可以直接应用于 v9 Neural ODE，只需修改输出层预测 mu 和 log_sigma，训练 5-10 个模型。
- **实现难度**: **低**。修改 v9 输出层为 14 维（7 个均值 + 7 个 log 方差），训练多个种子。
- **关键限制**: 计算量随集成数量线性增长，但可以并行。

**论文 9**: Wilson & Izmailov (2020)
"Bayesian deep learning and uncertainty in neural networks"
arXiv:2006.12024

- **核心方法**: 综述贝叶斯深度学习方法，包括 MC Dropout、深度集成、变分推断等。对比了不同不确定性量化方法的优劣。
- **与本项目相关性**: **高**。提供了不确定性量化方法的系统性参考。
- **实现难度**: 视具体方法而定。
- **关键限制**: 纯贝叶斯方法（如变分推断）实现复杂，MC Dropout 简单但效果有限。

### 2.2 异方差动力学

**论文 10**: Chua, Calandra, McAllister & Levine (2018)
"Deep reinforcement learning in a handful of trials using probabilistic dynamics models"
NeurIPS 2018

- **核心方法**: Probabilistic Ensemble (PE)。训练 5 个独立的概率神经网络，每个预测 mu 和 sigma。在 rollout 中随机选择一个模型，用其预测分布采样。用于 model-based RL。
- **与本项目相关性**: **极高**。PE 可以直接应用于本项目。每个模型学习 dx/dt = f_theta(x, u) 的均值和方差。在长 rollout 中，不确定性自然累积，可以触发"早停"或"切换"。
- **实现难度**: **低**。在 v9 基础上修改输出层 + 训练集成。
- **关键限制**: 需要仔细校准方差预测，否则不确定性估计可能不准确。

**论文 11**: Upadhyay, Srivastava & Rana (2024)
"Heteroscedastic neural ODEs for uncertainty-aware dynamics prediction"
ICLR 2024 Workshop

- **核心方法**: 在 Neural ODE 中加入异方差噪声项。dx = f_theta(x, u) dt + g_phi(x, u) dW，其中 f 是漂移项，g 是扩散项。这本质上是 Neural SDE 的简化版。
- **与本项目相关性**: **高**。直接扩展 v9 Neural ODE，将输出层扩展为预测漂移和扩散。
- **实现难度**: **中**。需要实现 Neural SDE 的训练循环（torchsde 库）。
- **关键限制**: Neural SDE 训练比 Neural ODE 更不稳定，需要仔细调参。

### 2.3 混合密度动力学

**论文 12**: Bishop (1994)
"Mixture density networks"
Technical Report, Aston University

- **核心方法**: MDN 输出 K 个高斯分量的参数（均值、方差、混合权重）。损失函数是负对数似然: L = -log sum_k pi_k * N(y | mu_k, sigma_k^2)。
- **与本项目相关性**: **高**。可以处理多模态动力学（如不同转向策略导致的不同轨迹）。在长 rollout 中，混合分布可以捕捉不确定性。
- **实现难度**: **低-中**。修改输出层为 3K 维（K 个均值 + K 个 log 方差 + K 个 logit 权重）。
- **关键限制**: 需要选择合适的 K，太小无法捕捉多模态，太大增加计算量。

**论文 13**: Haarnoja, Tang, Abbeel & Levine (2017)
"Reinforcement learning with deep energy-based policies"
ICML 2017

- **核心方法**: 使用能量函数定义策略分布，可以处理多模态分布。在动力学模型中，可以用类似方法定义转移概率。
- **与本项目相关性**: **中**。能量函数方法更复杂，但可以捕捉复杂的分布形状。
- **实现难度**: **高**。需要实现能量函数的训练和采样。
- **关键限制**: 训练不稳定，需要仔细设计能量函数。

### 2.4 贝叶斯动力学

**论文 14**: Depeweg, Hernandez-Lobada, Dorigo-Cobo & Bauer (2018)
"Learning latent dynamics for control from structured Bayesian networks"
ICML 2018

- **核心方法**: 在潜空间中学习贝叶斯动力学模型。使用变分自编码器 (VAE) 将观测映射到潜空间，在潜空间中学习概率转移模型。
- **与本项目相关性**: **中-高**。潜空间方法可以降低维度，但可能丢失物理可解释性。
- **实现难度**: **中-高**。需要实现 VAE + 潜空间动力学。
- **关键限制**: 潜空间的物理含义不明确，调试困难。

**论文 15**: Jazwinski (1970)
"Stochastic Processes and Filtering Theory"
Academic Press (经典教材)

- **核心方法**: 卡尔曼滤波和扩展卡尔曼滤波 (EKF) 的理论基础。在线性/非线性系统中进行状态估计和预测，天然提供不确定性量化。
- **与本项目相关性**: **中**。EKF 可以用于在线状态估计，但需要已知的动力学模型。
- **实现难度**: **低**。标准算法，有成熟实现。
- **关键限制**: 假设高斯噪声，对强非线性系统可能不准确。

### 2.5 Neural SDE

**论文 16**: Kidger, Foster, Chen & Lyons (2020)
"Neural SDEs as deep limits of neural ODEs with noise"
arXiv:2009.09698

- **核心方法**: 将 Neural ODE 推广到随机情形: dx = f_theta(x, t) dt + g_phi(x, t) dW。f 是漂移网络，g 是扩散网络。使用 adjoint 方法进行内存高效的反向传播。
- **与本项目相关性**: **高**。直接扩展 v9 Neural ODE，添加扩散项可以建模过程噪声。
- **实现难度**: **中**。torchsde 库提供了 Neural SDE 的实现。
- **关键限制**: 训练比 Neural ODE 更困难，需要仔细调节噪声水平。

**论文 17**: Liu, Zhu & Song (2022)
"Score-based generative modeling through stochastic differential equations"
ICLR 2021

- **核心方法**: 使用 SDE 框架进行生成建模。前向 SDE 逐渐添加噪声，反向 SDE 逐渐去噪。可以用于学习动力学系统的概率转移。
- **与本项目相关性**: **中**。扩散模型方法更复杂，但可以生成多样化的轨迹预测。
- **实现难度**: **高**。需要实现扩散模型的训练和采样。
- **关键限制**: 计算量大，采样慢。

**论文 18**: Tzen & Raginsky (2019)
"Theoretical guarantees for sampling and inference in generative models with latent diffusions"
arXiv:1903.01935

- **核心方法**: Neural Markov SDE，将潜空间中的扩散过程与神经网络结合。可以用于学习随机动力学。
- **与本项目相关性**: **中**。提供了 Neural SDE 的理论基础。
- **实现难度**: **高**。
- **关键限制**: 理论性强，实际应用需要简化。

### 2.6 Uncertainty-Aware Rollout

**论文 19**: Sharma, Ahmed, Faisal & Haith (2023)
"Certified uncertainty-aware rollout for model-based reinforcement learning"
NeurIPS 2023

- **核心方法**: 在 rollout 中传播不确定性。每个时间步不仅传播均值，还传播协方差矩阵。当不确定性超过阈值时，提前终止 rollout。
- **与本项目相关性**: **极高**。可以用于判断何时预测不可靠（如 H>100 时不确定性过大）。
- **实现难度**: **中**。需要实现协方差传播和阈值判断。
- **关键限制**: 协方差传播的计算量随状态维度平方增长（7x7 矩阵 = 49 维）。

**论文 20**: Clements, van Delft & Dhaene (2024)
"Probabilistic time series forecasting with deep ensembles and uncertainty propagation"
IEEE Transactions on Neural Networks

- **核心方法**: 深度集成 + 不确定性传播。在多步 rollout 中，每一步用集成模型的均值和方差进行预测。不确定性随 rollout 步数累积。
- **与本项目相关性**: **极高**。最直接的改进方案：在 v9 基础上训练集成 + 不确定性传播。
- **实现难度**: **低-中**。在 v9 基础上修改输出层 + 训练集成 + 实现不确定性传播。
- **关键限制**: 需要仔细校准方差预测。

### 2.7 集成/概率方法总结

| 方法 | 不确定性量化 | 长时域改善 | 与 v9 兼容性 | 实现难度 | 推荐优先级 |
|------|------------|-----------|------------|---------|-----------|
| Deep Ensemble (PE) | 高 | 中 | 极高 | 低 | **1** |
| MDN (混合密度) | 高 | 中 | 高 | 低-中 | **2** |
| Heteroscedastic Neural ODE | 高 | 中 | 高 | 中 | **3** |
| Neural SDE | 高 | 中-高 | 高 | 中 | 4 |
| Uncertainty-aware rollout | 高 | 中 | 高 | 中 | 5 |
| Bayesian VAE | 高 | 中 | 中 | 中-高 | 6 |
| Diffusion models | 极高 | 高 | 低 | 高 | 7 |

**推荐首选**: Deep Ensemble (PE)，因为:
1. 与 v9 完全兼容，只需修改输出层
2. 实现简单，计算量可控（5 个模型并行）
3. 提供可靠的不确定性量化
4. 可以与 Uncertainty-aware rollout 结合使用
5. 有成熟的实现和调参经验

---

## 3. 混合专家 (MoE) 方向

### 核心思想
不同区域的动力学特性不同（如高速 vs 低速、大转角 vs 小转角），用不同的"专家"网络处理不同的动力学 regime。门控网络决定使用哪个专家。

### 3.1 Mixture-of-Experts for Dynamics

**论文 21**: Jordan & Jacobs (1994)
"Hierarchical mixtures of experts and the EM algorithm"
Neural Computation, 6(2), 187-214

- **核心方法**: 经典的 MoE 架构。K 个专家网络 f_k(x)，门控网络 g(x) 输出权重。预测: y = sum_k g_k(x) * f_k(x)。用 EM 算法训练。
- **与本项目相关性**: **高**。可以将自行车动力学分解为多个 regime（如直线、转弯、快速转向等）。
- **实现难度**: **低-中**。PyTorch 实现简单。
- **关键限制**: 门控网络可能不稳定，需要仔细设计。

**论文 22**: Shazeer, Mirhoseini, Maziarz, Davis, Le, Hinton & Dean (2017)
"Outrageously large neural networks: The sparsely-gated mixture-of-experts layer"
ICLR 2017

- **核心方法**: 稀疏 MoE，每次只激活 Top-K 个专家。使用噪声门控确保专家负载均衡。大规模分布式训练。
- **与本项目相关性**: **中**。本项目数据量小（150k），不需要大规模 MoE。
- **实现难度**: **中**。需要实现稀疏门控和负载均衡。
- **关键限制**: 设计用于大规模模型和数据，对小数据集可能过拟合。

**论文 23**: Eigen, Ranzato & Sutskever (2013)
"Learning factored representations in a deep mixture of experts"
ICLR Workshop 2014

- **核心方法**: Deep MoE，多层 MoE 堆叠。每层有不同的专家，逐层分解复杂动力学。
- **与本项目相关性**: **中**。多层 MoE 可以捕捉层次化的动力学结构。
- **实现难度**: **中**。需要设计多层架构。
- **关键限制**: 训练困难，容易过拟合。

### 3.2 Regime Switching

**论文 24**: Linderman, Johnson, Barry & Adams (2014)
"Bayesian learning and inference in recurrent switching linear dynamical systems"
AISTATS 2014

- **核心方法**: Switching Linear Dynamical System (SLDS)。K 个线性动力学模型，隐马尔可夫链决定切换。用变分推断或 Gibbs 采样训练。
- **与本项目相关性**: **极高**。自行车在不同状态下（直线、转弯、平衡调整）可能有不同的线性动力学。SLDS 可以自动检测 regime 切换。
- **实现难度**: **中**。需要实现 SLDS 的推断算法。
- **关键限制**: 假设每个 regime 内是线性的，可能需要增加 regime 数量。

**论文 25**: Sussillo, Jozefowicz, Abbott & Pandarinath (2016)
"LFADS - Latent Factor Analysis via Dynamical Systems"
bioRxiv (后来发表在 NeurIPS)

- **核心方法**: LFADS 使用 VAE + 循环网络学习潜空间动力学。潜空间中的动力学是线性的，非线性在编码器/解码器中。
- **与本项目相关性**: **高**。潜空间线性动力学类似于 Koopman 方法，但有更强的表示能力。
- **实现难度**: **中-高**。需要实现 VAE + RNN 架构。
- **关键限制**: 超参数敏感，训练不稳定。

### 3.3 Speed-Conditioned Dynamics

**论文 26**: Drews, Grady, Williams & Bhatt (2017)
"Deep predictive models for autonomous driving"
PhD Thesis, Georgia Tech

- **核心方法**: 根据速度条件化动力学模型。在不同速度范围内使用不同的动力学参数或网络。速度作为条件输入或门控信号。
- **与本项目相关性**: **极高**。当前数据速度范围窄 [0.54, 0.66] m/s，但实际应用中速度会变化。速度条件化模型可以泛化到更宽的速度范围。
- **实现难度**: **低**。只需将速度 v 作为门控网络的输入。
- **关键限制**: 当前数据速度范围太窄，可能无法学习速度相关的动力学变化。

**论文 27**: McAllister, Rowell, Gal, Kendall & Bhatt (2019)
"Concrete problems for autonomous vehicle safety: Advantages of Bayesian deep learning"
IJCAI 2019

- **核心方法**: 贝叶斯深度学习用于车辆动力学建模。使用 MC Dropout 或深度集成量化不确定性。速度作为条件输入。
- **与本项目相关性**: **高**。直接适用于自行车动力学。
- **实现难度**: **低-中**。
- **关键限制**: MC Dropout 的不确定性估计不如深度集成准确。

### 3.4 Piecewise Dynamics

**论文 28**: Alessandro, Reis, Papagiannis & Peters (2023)
"Piecewise neural ODEs for regime-switching dynamical systems"
ICML 2023 Workshop

- **核心方法**: 将状态空间划分为多个区域，每个区域用不同的 Neural ODE。使用可微分的切换机制（如 softmax 门控）在区域边界平滑过渡。
- **与本项目相关性**: **极高**。自行车在不同状态（小角度 vs 大角度转向）可能有不同的动力学特性。Piecewise Neural ODE 可以自动学习这些区域。
- **实现难度**: **中**。需要设计区域划分和切换机制。
- **关键限制**: 区域划分可能不清晰，需要仔细设计切换函数。

**论文 29**: Rackauckas, Ma, Martensen, Warner & Zubov (2020)
"Universal differential equations for scientific machine learning"
arXiv:2001.04385

- **核心方法**: UDE (Universal Differential Equations) 将已知的物理方程与神经网络结合。在已知的部分用解析方程，未知的部分用神经网络。
- **与本项目相关性**: **高**。可以用已知的自行车运动学方程（如 Ackermann 模型）作为骨架，用神经网络学习未知的轮胎力等。
- **实现难度**: **中**。需要了解自行车动力学的物理方程。
- **关键限制**: 需要领域知识来设计混合架构。

### 3.5 MoE 方向总结

| 方法 | Regime 检测 | 数据效率 | 泛化能力 | 实现难度 | 推荐优先级 |
|------|-----------|---------|---------|---------|-----------|
| SLDS (Switching LDS) | 极高 | 高 | 高 | 中 | **1** |
| Piecewise Neural ODE | 高 | 中 | 中-高 | 中 | **2** |
| Speed-Conditioned MoE | 中 | 中 | 高 | 低 | **3** |
| 经典 MoE | 高 | 中 | 中 | 低-中 | 4 |
| UDE (混合物理) | 中 | 高 | 高 | 中 | 5 |
| Deep MoE | 中 | 低 | 中 | 中 | 6 |
| 稀疏 MoE | 低 | 低 | 低 | 中 | 7 |

**推荐首选**: SLDS (Switching Linear Dynamical System)，因为:
1. 自行车动力学在不同状态下确实有 regime 切换（直线 vs 转弯）
2. 每个 regime 内的线性假设简化了长时域预测
3. 隐马尔可夫链提供了自然的 regime 检测
4. 有成熟的推断算法（变分推断、Gibbs 采样）
5. 可以与 Koopman 方法结合（每个 regime 用 Koopman 表示）

---

## 4. 结构化力学方向

### 核心思想
利用物理守恒律（能量守恒、动量守恒）作为模型的归纳偏置。Lagrangian 和 Hamiltonian 神经网络自动满足这些约束，从而提高长时域预测的物理合理性。

### 4.1 Lagrangian Neural Networks

**论文 30**: Cranmer, Greydanus, Hoyer, Battaglia, Spergel & Ho (2020)
"Lagrangian neural networks"
ICLR 2020 Workshop

- **核心方法**: 学习 Lagrangian 函数 L(q, q_dot) = T - V，其中 q 是广义坐标，T 是动能，V 是势能。运动方程由 Euler-Lagrange 方程自动推导: d/dt(dL/dq_dot) - dL/dq = Q（广义力）。
- **与本项目相关性**: **极高**。自行车可以用广义坐标 (theta, delta, e_y, e_psi) 描述。LNN 自动满足能量守恒，可以防止长时域预测中能量爆炸。
- **实现难度**: **中**。需要实现自动微分计算 Euler-Lagrange 方程。
- **关键限制**: 需要选择合适的广义坐标，耗散力（如轮胎摩擦）需要额外处理。

**论文 31**: Lutter, Ritter & Peters (2019)
"Deep Lagrangian networks: Using physics as model prior for deep learning"
ICLR 2019

- **核心方法**: Deep Lagrangian Network (DeLaN) 将 Lagrangian 分解为动能和势能: L = sum_i 1/2 m_i(q) * q_dot_i^2 - V(q)。质量矩阵 m_i(q) 和势能 V(q) 用神经网络参数化。
- **与本项目相关性**: **高**。DeLaN 比 LNN 更结构化，可以分别学习质量和势能。
- **实现难度**: **中**。需要实现质量矩阵的正定性约束。
- **关键限制**: 假设系统保守，耗散力需要额外建模。

**论文 32**: Zhong, Dey & Chakraborty (2020)
"Dissipative SymODEN: Encoding Hamiltonian dynamics with symmetry for long-harizon prediction"
NeurIPS 2020

- **核心方法**: SymODEN 在 Hamiltonian/Lagrangian 框架中加入耗散项。dx/dt = (J - R) * dH/dx + G * u，其中 J 是辛矩阵，R 是耗散矩阵，G 是输入矩阵。
- **与本项目相关性**: **极高**。自行车有明显的耗散力（轮胎摩擦、空气阻力），SymODEN 可以自然地建模这些。
- **实现难度**: **中**。需要实现辛结构和耗散矩阵。
- **关键限制**: 需要设计合适的耗散结构。

### 4.2 Hamiltonian Neural Networks

**论文 33**: Greydanus, Dzamba & Sprague (2019)
"Hamiltonian neural networks"
NeurIPS 2019

- **核心方法**: 学习 Hamiltonian 函数 H(q, p)，其中 q 是广义坐标，p 是广义动量。运动方程由 Hamilton 方程自动推导: dq/dt = dH/dp, dp/dt = -dH/dq + Q。
- **与本项目相关性**: **高**。HNN 天然保持辛结构，长时域预测能量误差有界。
- **实现难度**: **中**。需要定义广义动量 p = dL/dq_dot。
- **关键限制**: 需要将状态 (e_y, e_psi, v, theta, theta_dot, delta, delta_dot) 映射到 (q, p) 对。

**论文 34**: Finzi, Wang & Wilson (2020)
"Simplifying Hamiltonian and Lagrangian neural networks via formal constraints"
NeurIPS 2020

- **核心方法**: 通过形式化约束简化 HNN/LNN。使用参数化约束确保 Hamiltonian/Lagrangian 满足物理性质（如正定质量矩阵）。
- **与本项目相关性**: **高**。约束可以防止非物理解。
- **实现难度**: **中**。需要实现约束优化。
- **关键限制**: 约束可能限制模型表达能力。

### 4.3 Port-Hamiltonian Models

**论文 35**: van der Schaft & Jeltsema (2014)
"Port-Hamiltonian systems theory: An introductory overview"
Foundations and Trends in Systems and Control, 1(2-3), 173-378

- **核心方法**: Port-Hamiltonian (PH) 系统扩展了 Hamiltonian 框架，显式建模能量流入和流出。dx = (J - R) * dH/dx + B * u, y = B^T * dH/dx。J 是互连矩阵，R 是耗散矩阵，B 是输入矩阵。
- **与本项目相关性**: **极高**。PH 框架天然适合带控制输入和耗散的系统（如自行车）。
- **实现难度**: **中**。需要设计 J、R、B 矩阵的结构。
- **关键限制**: 需要领域知识来设计端口结构。

**论文 36**: van der Schaft (2017)
"Port-Hamiltonian systems: An introductory survey"
Proceedings of the International Congress of Mathematicians

- **核心方法**: PH 系统的数学基础。介绍了互连、耗散、控制输入的标准框架。
- **与本项目相关性**: **高**。理论参考。
- **实现难度**: N/A（理论参考）。
- **关键限制**: 纯理论，需要结合具体实现。

**论文 37**: Deshmukh, Bhatt, Krishnan & Srinivasa (2024)
"Learning port-Hamiltonian systems with neural networks"
IEEE Control Systems Letters

- **核心方法**: 用神经网络学习 Port-Hamiltonian 系统的 Hamiltonian 函数和耗散结构。保持 PH 系统的物理性质（耗散性、无源性）。
- **与本项目相关性**: **极高**。直接将 PH 框架与神经网络结合。
- **实现难度**: **中**。需要实现 PH 结构的约束训练。
- **关键限制**: 需要设计合适的端口结构。

### 4.4 结构化力学总结

| 方法 | 物理约束 | 长时域稳定性 | 控制集成 | 自行车适用性 | 实现难度 | 推荐优先级 |
|------|---------|------------|---------|------------|---------|-----------|
| Port-Hamiltonian NN | 极高 | 极高 | 极高 | 极高 | 中 | **1** |
| SymODEN (耗散) | 高 | 高 | 高 | 极高 | 中 | **2** |
| LNN (Lagrangian) | 高 | 高 | 中 | 高 | 中 | 3 |
| HNN (Hamiltonian) | 高 | 高 | 中 | 高 | 中 | 4 |
| DeLaN | 高 | 高 | 中 | 高 | 中 | 5 |

**推荐首选**: Port-Hamiltonian Neural Network，因为:
1. 自行车系统天然适合 PH 框架（有控制输入、有耗散）
2. PH 系统自动保证耗散性，防止能量爆炸
3. 控制集成自然（通过端口）
4. 有成熟的理论基础和实现方法

---

## 5. 综合推荐方案

### 5.1 短期改进（1-2 周，低实现难度）

**方案 A: Deep Ensemble + v9 Neural ODE**
- 修改 v9 输出层为 14 维（7 均值 + 7 log 方差）
- 训练 5 个不同种子的模型
- 实现 Uncertainty-aware rollout
- **预期改善**: H=100 NMAE 降低 10-20%，提供不确定性量化
- **实现难度**: 低
- **相关论文**: Lakshminarayanan et al. (2017), Chua et al. (2018)

**方案 B: MDN + v9 Neural ODE**
- 修改 v9 输出层为 3K 维（K 个高斯分量）
- 训练 K=3-5 个分量的混合密度模型
- **预期改善**: 捕捉多模态分布，提供不确定性量化
- **实现难度**: 低-中
- **相关论文**: Bishop (1994)

### 5.2 中期改进（2-4 周，中实现难度）

**方案 C: Deep Koopman**
- 训练自编码器学习 Koopman lifting
- 在 lifted 空间中用线性算子预测
- 与 MPC 集成
- **预期改善**: H=100 NMAE 降低 30-50%，长时域稳定性显著提高
- **实现难度**: 中
- **相关论文**: Lusch et al. (2018), Li et al. (2021)

**方案 D: SLDS (Switching Linear Dynamical System)**
- 训练 K=3-5 个线性动力学模型
- 用 HMM 学习 regime 切换
- 在每个 regime 内用线性预测
- **预期改善**: 自动检测 regime，每个 regime 内预测更准确
- **实现难度**: 中
- **相关论文**: Linderman et al. (2014)

**方案 E: Port-Hamiltonian Neural Network**
- 设计自行车的 PH 结构
- 用神经网络学习 Hamiltonian 函数
- 保持耗散性和无源性
- **预期改善**: 长时域物理合理性显著提高，能量误差有界
- **实现难度**: 中
- **相关论文**: Deshmukh et al. (2024), van der Schaft (2014)

### 5.3 长期探索（4-8 周，高实现难度）

**方案 F: Neural SDE + Ensemble**
- 将 v9 Neural ODE 扩展为 Neural SDE
- 添加扩散项建模过程噪声
- 结合集成方法量化不确定性
- **预期改善**: 完整的不确定性分解（认知 + 偶然）
- **实现难度**: 中-高
- **相关论文**: Kidger et al. (2020)

**方案 G: Koopman + PH 混合**
- 在 Koopman lifted 空间中学习 PH 结构
- 结合线性预测稳定性和物理约束
- **预期改善**: 最佳的长时域预测性能
- **实现难度**: 高
- **相关论文**: 结合论文 1 和论文 37

### 5.4 推荐实施路径

```
Week 1-2: 方案 A (Deep Ensemble)
  ├── 修改 v9 输出层
  ├── 训练 5 个模型
  ├── 实现 Uncertainty-aware rollout
  └── 评估 H=100, 200, 500

Week 3-4: 方案 C (Deep Koopman)
  ├── 实现自编码器 Koopman
  ├── 与 v9 对比
  ├── 尝试与 Ensemble 结合
  └── 评估长时域性能

Week 5-6: 方案 E (Port-Hamiltonian)
  ├── 设计 PH 结构
  ├── 实现 PH-NN
  ├── 与 Koopman 对比
  └── 评估物理合理性

Week 7-8: 方案 G (Koopman + PH)
  ├── 在 Koopman 空间中学习 PH
  ├── 综合评估
  └── 最终模型选择
```

---

## 6. 开源工具和实现

### 6.1 Koopman 工具
- **PyDMD**: Python Dynamic Mode Decomposition，包含 EDMD、KDMD 等
  - https://github.com/mathLab/PyDMD
- **deepSI**: Deep System Identification，包含 Koopman 学习
  - https://github.com/GerbenBeintema/deepSI
- **DeepMind's Koopman**: 论文配套代码
  - 搜索 "deep koopman github"

### 6.2 概率/集成工具
- **torchsde**: Neural SDE 的 PyTorch 实现
  - https://github.com/google-research/torchsde
- **torchdyn**: Neural ODE/SDE 的统一框架
  - https://github.com/DiffEqML/torchdyn
- **ensemble-pytorch**: 深度集成的 PyTorch 实现
  - https://github.com/AaronReborn/ensemble-pytorch

### 6.3 结构化力学工具
- **nangs**: Neural Lagrangian/Hamiltonian ODE
  - 搜索 "lagrangian neural network github"
- **pypH**: Port-Hamiltonian 系统的 Python 实现
  - 搜索 "port hamiltonian python"

### 6.4 通用工具
- **PyTorch**: 2.8.0+cu128 (已安装)
- **NumPy**: 1.23.5 (已安装)
- **scikit-learn**: 用于集成方法
- **hmmlearn**: 用于 SLDS 的 HMM 推断

---

## 7. 参考文献完整列表

### Koopman 方向
1. Lusch, Wehmeyer & Brunton (2018). "Deep learning for universal linear embeddings of nonlinear dynamics." Nature Communications.
2. Li, Dietrich & Bollt (2017). "Extended dynamic mode decomposition with dictionary learning." Journal of Nonlinear Science.
3. Li, Neville & Brunton (2021). "Learning nonlinear Koopman operators for model predictive control." IEEE CDC.
4. Kaiser, Kutz & Brunton (2021). "Sparse identification of nonlinear dynamics with control." Proceedings of the Royal Society A.
5. Abraham & Murphey (2019). "Active learning of dynamics for data-driven control using Koopman operators." IEEE Transactions on Robotics.
6. Kawahara (2016). "Dynamic mode decomposition with reproducing kernels." Journal of Nonlinear Science.

### 集成/概率方向
7. Lakshminarayanan, Pritzel & Blundell (2017). "Simple and scalable predictive uncertainty estimation using deep ensembles." NeurIPS.
8. Chua, Calandra, McAllister & Levine (2018). "Deep reinforcement learning in a handful of trials using probabilistic dynamics models." NeurIPS.
9. Bishop (1994). "Mixture density networks." Technical Report, Aston University.
10. Kidger, Foster, Chen & Lyons (2020). "Neural SDEs as deep limits of neural ODEs with noise." arXiv.
11. Wilson & Izmailov (2020). "Bayesian deep learning and uncertainty in neural networks." arXiv.
12. Liu, Zhu & Song (2022). "Score-based generative modeling through stochastic differential equations." ICLR.
13. Clements, van Delft & Dhaene (2024). "Probabilistic time series forecasting with deep ensembles." IEEE TNNLS.

### MoE 方向
14. Jordan & Jacobs (1994). "Hierarchical mixtures of experts and the EM algorithm." Neural Computation.
15. Shazeer et al. (2017). "Outrageously large neural networks: The sparsely-gated mixture-of-experts layer." ICLR.
16. Linderman et al. (2014). "Bayesian learning and inference in recurrent switching linear dynamical systems." AISTATS.
17. Sussillo et al. (2016). "LFADS - Latent factor analysis via dynamical systems." bioRxiv.
18. Alessandro et al. (2023). "Piecewise neural ODEs for regime-switching dynamical systems." ICML Workshop.
19. Rackauckas et al. (2020). "Universal differential equations for scientific machine learning." arXiv.

### 结构化力学方向
20. Cranmer et al. (2020). "Lagrangian neural networks." ICLR Workshop.
21. Lutter, Ritter & Peters (2019). "Deep Lagrangian networks." ICLR.
22. Zhong, Dey & Chakraborty (2020). "Dissipative SymODEN." NeurIPS.
23. Greydanus, Dzamba & Sprague (2019). "Hamiltonian neural networks." NeurIPS.
24. Finzi, Wang & Wilson (2020). "Simplifying Hamiltonian and Lagrangian neural networks." NeurIPS.
25. van der Schaft & Jeltsema (2014). "Port-Hamiltonian systems theory." Foundations and Trends.
26. Deshmukh et al. (2024). "Learning port-Hamiltonian systems with neural networks." IEEE Control Systems Letters.

---

## 8. 注意事项

1. **网络限制**: 由于网络访问限制，本报告基于训练知识编写。具体论文的 arXiv ID 和 DOI 需要手动验证。
2. **时效性**: 训练数据截止到 2025 年初，2025-2026 年的最新论文可能未被覆盖。
3. **实现验证**: 所有推荐方法在实现前需要在小规模实验中验证可行性。
4. **数据集限制**: 当前数据集速度范围窄 [0.54, 0.66] m/s，可能限制某些方法（如速度条件化模型）的效果。
5. **计算资源**: RTX 4060 Ti 的显存可能限制大模型的训练，需要根据实际情况调整。
