# 项目交接文档

> 生成时间：2026-06-21
> 项目：无人自行车系统辨识与模型强化学习
> GitHub: https://github.com/Ma-mingliang/sindy-bicycle-system-id

---

## 一、项目概述

### 1.1 研究目标

基于系统辨识方法建立自行车世界模型，结合MBPO（Model-Based Policy Optimization）和TD3算法，在LQR基线控制器上训练残差策略，实现自行车姿态平衡控制。

### 1.2 技术路线

```
物理数据采集 → 系统辨识 → 世界模型构建 → MBPO-TD3残差控制
                                                  ↓
                                          LQR基线 + RL残差 → 最终控制
```

### 1.3 两个对比项目

| 项目 | 路径 | 方法 |
|------|------|------|
| HRRL（参考项目） | `D:/系统辨识作业/HRRL-new/` | PyBullet 3D物理 + TD3 + Model-Free |
| MBPO（本项目） | `D:/系统辨识作业/sindy_bicycle/` | Meijaard解析模型 + 系统辨识 + TD3 + Model-Based |

### 1.4 物理模型

**Meijaard 2007基准自行车动力学**
- 4维状态：`[theta(横滚角), delta(转向角), theta_dot, delta_dot]`
- 1维输入：`tau(转向力矩)`
- LQR控制器生成参考轨迹
- 控制周期：dt = 1/30 s
- 前进速度：v0 = 3.5 m/s

---

## 二、系统辨识方法对比（已完成）

### 2.1 评估方案

**方案C：真实模型+LQR生成参考轨迹**
- 真实模型+LQR生成5段×500步轨迹
- 所有模型从相同初始状态出发，接收相同力矩序列
- 评估步数：[1, 5, 10, 20, 50, 100, 200, 500]
- 指标：theta角度MAE（rad）
- 发散判断：|phi| > pi/3 时截断

**环境**：
```bash
"E:/Anaconda/envs/DL/python.exe"  # PyTorch + CUDA
```

### 2.2 测试过的12种基础方法

| # | 方法 | 说明 | 500步MAE |
|---|------|------|----------|
| 1 | SINDyPoly | 多项式库(21特征) + STLSQ | 发散 |
| 2 | SINDyTrig | 三角函数库(32特征) | 发散 |
| 3 | SINDyTrigExp | 三角+指数库(36特征) | 发散 |
| 4 | SINDyBestNN | 最佳SINDy + NN残差 | 发散 |
| 5 | NNE2E | 端到端MLP | 发散 |
| 6 | NeuralODE | NN学ds/dt，Euler积分 | 1.59~13.36 |
| 7 | GP | 高斯过程回归(5000样本限制) | 7.05 |
| 8 | PINN | 物理约束NN | 发散 |
| 9 | ParamID | 线性最小二乘拟合 | 发散 |
| 10 | NeuralODE_NN | 方法6 + NN残差 | 发散 |
| 11 | GP_NN | 方法7 + NN残差 | 发散 |
| 12 | ParamID_NN | 方法9 + NN残差 | 发散 |

**结论**：纯方法中NeuralODE和GP表现最好，但加NN残差后全部发散。

### 2.3 NN残差恶化诊断

**诊断结论**（diagnose_hybrid.py）：
- 残差仅占总变化的0.3%~0.8%（基线已捕捉99%+动态）
- NN预测残差的符号一致率仅78.9%（21%方向错误）
- rollout中NN输入偏离训练分布2~4σ（分布偏移）
- 更多训练epoch反而更差（过拟合残差噪声）

**根本原因**：正反馈放大误差
```
基线预测误差 → NN输入偏移 → NN输出错误残差 → 下一步更大偏移 → 指数发散
```

### 2.4 解决方案：三重防护

| 技术 | 作用 | 参数 |
|------|------|------|
| OOD检测 | 分布外时回退零残差 | Mahalanobis距离 > 3σ |
| 残差缩放 | 降低错误残差影响 | scale=0.3 |
| DAgger | 用rollout数据重新训练 | 3轮，每轮2-3段轨迹 |

**核心代码模式**：
```python
# 训练残差（归一化空间）
residual = (真实下一步 - 基线预测) / delta_std

# 推理时（保守应用）
if ood_detector.is_ood(input):
    delta_nn = 0  # 回退
s_next = baseline.predict(s, tau) + delta_nn * delta_std * 0.3
```

### 2.5 论文改进方案测试

| 方案 | 效果 | 原因 |
|------|------|------|
| 纯NN残差(无防护) | 恶化(发散) | 正反馈放大误差 |
| +残差缩放0.3 | 部分改善 | 降低但未消除错误影响 |
| +OOD检测 | 显著改善 | 分布外时安全回退 |
| +DAgger | 进一步改善 | 减少分布偏移 |
| Ensemble(5-NN) | 失败 | 残差噪声大，集成方差高，99.9%OOD率 |
| Domain Rand | 部分改善 | 需配合OOD+DAgger |
| Conformal Prediction | 失败 | 预测区间过宽，89-98%回退率 |
| Latent-space(2D) | 不稳定 | 降维丢失信息，0.10~0.43波动大 |

### 2.6 最终结果：全部6种基线 + 改进NN残差

| 方法 | 纯基线 (rad) | 改进混合 (rad) | 倍率 | 结论 |
|------|-------------|---------------|------|------|
| SINDyPoly | 1.73e+13 | 1.04e+13 | 0.60x | 改善(绝对值仍大) |
| SINDyTrig | 2.01e+13 | 9.38e+8 | 0.00x | 大幅改善 |
| NeuralODE | 1.59 | 0.48 | 0.30x | 改善 |
| NNE2E(纯NN) | 1.28e+9 | 0.23 | 0.00x | 从发散恢复 |
| ParamID | 8.52e+8 | 5.75e+8 | 0.67x | 改善 |
| **GP** | 7.05 | **0.15** | 0.02x | **最佳** |

### 2.7 最终排名

| 排名 | 方法 | 500步MAE |
|------|------|----------|
| 1 | GP + OOD + DAgger + 0.3 | 0.15 rad |
| 2 | NNE2E + OOD + DAgger + 0.3 | 0.23 rad |
| 3 | NeuralODE + OOD + DAgger + 0.3 | 0.48 rad |
| 4 | Latent + OOD | 0.10~0.43 rad(不稳定) |
| 5 | DR + OOD + DAgger | 0.72 rad |

---

## 三、文件结构

```
sindy_bicycle/
├── methods_common.py          # 公共基础设施（Meijaard参数、动力学、LQR、数据生成）
├── methods_sindy.py           # SINDy方法（Poly/Trig/TrigExp库 + STLSQ）
├── methods_nn.py              # NN方法（NeuralODE、NNE2E、ResidualNet）
├── methods_classic.py         # 经典方法（GP、PINN、ParamID）
├── methods_hybrid.py          # 混合方法基线版（无改进，已弃用）
├── methods_evaluate.py        # 评估框架（make_tau_func、run_trajectory）
│
├── test_all_methods.py        # 12种方法原始对比
├── diagnose_hybrid.py         # NN残差恶化诊断
├── improved_hybrid.py         # 改进版混合方法（OOD+DAgger+0.3）
├── test_improved_all.py       # 5种基线+改进NN测试
├── test_ensemble.py           # Ensemble不确定性方案（失败）
├── test_domain_rand.py        # Domain Randomization方案
├── test_dr_dagger.py          # DR+OOD+DAgger三合一
├── test_conformal.py          # Conformalized Neural Dynamics（失败）
├── test_latent.py             # Latent-space dynamics（不稳定）
├── test_all_improved.py       # 全部6基线+改进NN最终对比 ← 最终结果
├── test_gp_nn_improved.py     # GP+NN改进版详细测试
├── test_sindy_nn_improved.py  # SINDy/ParamID+NN改进版详细测试
├── test_final_summary.py      # 最终总结对比
│
├── meijaard_dynamics.py       # Meijaard 2007解析模型（原始参考）
├── mbpo_sac_v7.py             # MBPO-SAC v7训练脚本（效果差，已弃用）
├── sindy_identification.py    # SINDy辨识代码
├── compare_dynamics_models.py # 动力学模型对比
├── compare_multistep.py       # 多步rollout对比
├── verify_hrrl_stage1.py      # HRRL参数验证
│
├── sindy_full_model.npz       # SINDy系数(21x4)、state_std、action_std
├── sindy_nn_residual.pt       # NN残差权重（17,796参数）
├── meijaard_sindy_v35.npz     # 原始SINDy系数
├── meijaard_openloop_data_v35.npz  # 开环仿真数据（29478样本）
│
└── HANDOVER.md                # 本交接文档
```

---

## 四、已完成工作

### 4.1 SINDy系统辨识（已完成）

- 使用Meijaard 2007基准自行车参数生成开环数据（29478样本）
- 多项式库：常数+线性+二次交叉项（21个特征，5个输入）
- 稀疏回归：STLSQ算法，阈值0.05
- 输出：`sindy_full_model.npz`

### 4.2 NN残差模型（已完成）

- 架构：3层MLP（128隐藏单元，SiLU激活）
- 输入：`[s_norm(4), a_norm(1)]` = 5维
- 输出：4维残差delta（归一化空间）
- 参数量：17,796
- 输出：`sindy_nn_residual.pt`

### 4.3 MBPO-SAC v7（已完成，效果差）

- 文件：`mbpo_sac_v7.py`
- 结果：600 episodes后SAC策略与纯LQR基线完全相同
- 失败原因：SAC的auto-alpha从0.303崩溃到0.007，探索停止

### 4.4 系统辨识方法对比（已完成）

- 12种基础方法对比
- NN残差恶化诊断
- 6种改进方案测试（OOD、DAgger、Domain Rand、Ensemble、Conformal、Latent）
- 全部6种基线 + 改进NN残差最终对比
- **最佳方案：GP + OOD + DAgger + 0.3 = 0.15 rad**

### 4.5 世界模型优化分析（已完成，结论：非必要）

- LQR控制下线性化vs非线性差异仅0.03°
- 非线性优化的收益被LQR"掩盖"
- 详见原交接文档第3.2节

---

## 五、待完成工作

### 5.1 MBPO-TD3脚本（未写）

需要创建 `mbpo_td3.py`，基于 `mbpo_sac_v7.py` 进行以下改动：

#### 改动1：SAC → TD3

```python
class TD3Agent:
    # 确定性策略（不是随机策略）
    # 双Q网络（同SAC）
    # 延迟策略更新：每2次critic更新后才更新1次策略
    # 目标策略平滑：目标动作加噪声clip(-0.5, 0.5)
    # 无熵正则化，无alpha
    # 探索：确定性输出 + 高斯噪声(std=0.1)
```

#### 改动2：奖励函数同步HRRL

```python
beta = 0.002
if |dis_angle| < beta:
    reward = 0.1 - |w0|          # 小误差：奖励平稳
else:
    reward = |error_old| - |error_new|  # 大误差：奖励收敛
terminated: reward -= 1
```

#### 改动3：参数同步

| 参数 | v7（旧） | TD3（新） | 来源 |
|------|----------|-----------|------|
| ALPHA | 0.5 | 0.1 | HRRL的action[0]*0.1 |
| buffer_size | 300,000 | 100,000 | HRRL |
| batch_size | 256 | 256 | HRRL |
| lr | 3e-4 | 3e-4 | HRRL |
| tau | 0.005 | 0.005 | HRRL |
| gamma | 0.99 | 0.99 | HRRL |
| policy_delay | N/A | 2 | TD3标准 |
| exploration_noise | N/A | 0.1 | TD3标准 |
| target_noise | N/A | 0.2, clip=0.5 | TD3标准 |

### 5.2 将最佳系统辨识方案集成到MBPO

- 最佳方案：GP + OOD + DAgger + 0.3
- 需要集成到MBPO的世界模型中
- 需要测试在MBPO训练循环中的实际效果
- 可能需要在线更新（新数据到来时重新训练）

### 5.3 撰写分析文档

- TD3算法改进分析
- 世界模型优化测试结论
- 系统辨识方法对比结论
- HRRL对比分析

---

## 六、技术细节速查

### 6.1 Meijaard自行车参数

```python
p = {
    'IBxx': 11.3557360401, 'IBxz': -1.96756380745,
    'IByy': 12.2177848012, 'IBzz': 3.12354397008,
    'IFxx': 0.0904106601579, 'IFyy': 0.149389340425,
    'IHxx': 0.253379594731, 'IHxz': -0.0720452391817,
    'IHyy': 0.246138810935, 'IHzz': 0.0955770796289,
    'IRxx': 0.0883819364527, 'IRyy': 0.152467620286,
    'c': 0.0685808540382, 'g': 9.81, 'lam': 0.399680398707,
    'mB': 81.86, 'mF': 2.02, 'mH': 3.22, 'mR': 3.11,
    'rF': 0.34352982332, 'rR': 0.340958858855, 'w': 1.121,
    'xB': 0.289099434117, 'xH': 0.866949640247,
    'zB': -1.04029228321, 'zH': -0.748236400835,
}
```

### 6.2 状态空间

```python
# 状态：[theta(倾斜角), delta(转向角), theta_dot(角速度), delta_dot(转向角速度)]
# 终止条件：|theta| > pi/3 (60°)
```

### 6.3 LQR控制器

```python
# Q = diag(1000, 100, 10, 1), R = 0.2
# K = [-103.9, -34.2, 38.0, 3.70]（Riccati方程求解）
# 控制律：u_lqr = -K @ x_lqr
```

### 6.4 ResidualNet架构

```python
# 3层MLP，128 hidden，SiLU激活
# forward(s, a=None) - 支持双参数或单拼接张量
class ResidualNet(nn.Module):
    def __init__(self, state_dim=4, action_dim=1, hidden=128): ...
    def forward(self, s, a=None): ...
```

### 6.5 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| dt | 1/30 | 控制周期 |
| v0 | 3.5 | 前进速度(m/s) |
| 训练样本 | 30000 | generate_training_data |
| OOD阈值 | 3.0σ | Mahalanobis距离 |
| 残差缩放 | 0.3 | 保守修正系数 |
| DAgger轮数 | 3 | 迭代轮数 |
| NN训练epoch | 100-200 | 100epoch更稳定 |

---

## 七、已知问题与陷阱

### 7.1 NN残差的固有问题

- 残差太小(0.3%~0.8%)，NN很难学到有意义的模式
- 符号一致率仅78.9%，21%的修正是反方向的
- 分布偏移不可避免（autoregressive rollout）
- 更多训练epoch = 更多过拟合 = 更差的rollout表现

### 7.2 GP的局限

- 训练极慢（30000样本需~12分钟）
- sklearn GP有收敛警告（可忽略）
- 预测速度慢于NN（但精度高）

### 7.3 SINDy系列的局限

- 基线本身就发散（多项式/三角函数库无法捕捉复杂动力学）
- 加NN残差只能部分缓解，不能根本解决
- 绝对误差仍然很大

### 7.4 测试随机性

- 初始状态随机采样 `np.random.uniform(-0.25, 0.25)`
- 不同run结果可能有2-3倍差异
- Latent-space方法对随机种子特别敏感

### 7.5 SAC alpha崩溃（已确认）

- 现象：alpha从0.303→0.007，探索停止
- 原因：当策略接近最优时，熵项变小，alpha自动降低
- 解决：切换TD3（无熵项）

---

## 八、后续方向

### 8.1 可尝试的改进

1. **自适应残差缩放**：根据OOD距离动态调整scale（近分布→大scale，远分布→小scale）
2. **GP + NN的更深度融合**：用GP的不确定性指导NN残差的置信度
3. **多步损失训练**：不只训单步残差，用多步rollout loss训练NN
4. **物理约束残差**：让NN残差满足部分物理约束（如能量守恒）
5. **贝叶斯神经网络**：替代ensemble，提供更好的不确定性估计
6. **更大的GP**：用SGP(稀疏GP)或DeepGP处理更大数据集

### 8.2 应用到MBPO

- 最佳方案(GP+OOD+DAgger+0.3)需要集成到MBPO的世界模型中
- 需要测试在MBPO训练循环中的实际效果
- 可能需要在线更新（新数据到来时重新训练）

### 8.3 性能优化

- GP预测可以用缓存或近似加速
- OOD检测可以用更轻量的方法（如简单的范围检查）
- DAgger数据可以预计算并缓存

---

## 九、Git提交历史

```
cba3084 feat: All baselines + improved NN residual comparison ← 最终结果
9d5e5ea feat: Final summary - all approaches compared
a957561 feat: Latent-space dynamics test - NEW BEST 0.10 rad
c29ad3a feat: Conformalized Neural Dynamics test
52aab7c feat: DR + OOD + DAgger triple combination test
[更早的commit见git log]
```

---

## 十、运行指南

```bash
# 运行最终对比（推荐，约20分钟含GP）
"E:/Anaconda/envs/DL/python.exe" "D:/系统辨识作业/sindy_bicycle/test_all_improved.py"

# 运行单个改进方案测试
"E:/Anaconda/envs/DL/python.exe" "D:/系统辨识作业/sindy_bicycle/test_dr_dagger.py"
"E:/Anaconda/envs/DL/python.exe" "D:/系统辨识作业/sindy_bicycle/test_latent.py"

# 运行诊断
"E:/Anaconda/envs/DL/python.exe" "D:/系统辨识作业/sindy_bicycle/diagnose_hybrid.py"

# 运行SAC v7（已完成，效果差）
"E:/Anaconda/envs/DL/python.exe" "D:/系统辨识作业/sindy_bicycle/mbpo_sac_v7.py"

# 查看SINDy系数
"E:/Anaconda/envs/DL/python.exe" -c "
import numpy as np
d = np.load('D:/系统辨识作业/sindy_bicycle/sindy_full_model.npz')
print('coefficients:', d['coefficients'].shape)
print('state_std:', d['state_std'])
print('action_std:', d['action_std'])
"
```

---

## 十一、HRRL参考项目关键信息

### 11.1 算法

- TD3（不是SAC！）——这是关键发现
- 物理引擎：PyBullet 3D（完整刚体仿真）
- 训练步数：1001*100 = 100,100步

### 11.2 奖励函数

```python
beta = 0.002
if |dis_angle| < beta:
    reward = 0.1 - |w0|          # 小误差：奖励平稳
else:
    reward = |error_old| - |error_new|  # 大误差：奖励收敛
terminated: reward -= 1
```

### 11.3 扰动调度（课程学习）

```python
def compute_sigma(ep):
    return min(0.3, (1.08 ** ep) * 0.01)
# ep=0: sigma=0.01 (0.57°)
# ep=80: sigma=0.3 (17.2°)
```

### 11.4 HRRL关键文件

| 文件 | 用途 |
|------|------|
| `HRRL-new/env.py` | HRRL环境定义（奖励函数、LQR、扰动） |
| `HRRL-new/train_attitude.py` | HRRL训练脚本（TD3配置） |
