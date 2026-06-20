# 项目交接文档

> 生成时间：2026-06-20
> 项目：无人自行车系统辨识与模型强化学习

---

## 一、项目概述

### 1.1 研究目标

基于SINDy（稀疏动力学辨识）建立自行车世界模型，结合MBPO（Model-Based Policy Optimization）和TD3算法，在LQR基线控制器上训练残差策略，实现自行车姿态平衡控制。

### 1.2 技术路线

```
物理数据采集 → SINDy系统辨识 → 世界模型构建 → MBPO-TD3残差控制
                                                      ↓
                                              LQR基线 + RL残差 → 最终控制
```

### 1.3 两个对比项目

| 项目 | 路径 | 方法 |
|------|------|------|
| HRRL（参考项目） | `D:/系统辨识作业/HRRL-new/` | PyBullet 3D物理 + TD3 + Model-Free |
| MBPO（本项目） | `D:/系统辨识作业/sindy_bicycle/` | Meijaard解析模型 + SINDy+NN + TD3 + Model-Based |

---

## 二、已完成工作

### 2.1 SINDy系统辨识（已完成）

- 使用Meijaard 2007基准自行车参数生成开环数据（29478样本）
- 状态空间：`[theta, delta, theta_dot, delta_dot]`（4维）
- 动作空间：`tau`（转向力矩，标量）
- 多项式库：常数+线性+二次交叉项（21个特征，5个输入）
- 稀疏回归：STLSQ算法，阈值0.05
- 输出：`sindy_full_model.npz`（系数矩阵21x4，8个非零项）

### 2.2 NN残差模型（已完成）

- 架构：3层MLP（128隐藏单元，SiLU激活）
- 输入：`[s_norm(4), a_norm(1)]` = 5维
- 输出：4维残差delta（归一化空间）
- 参数量：17,796
- 精度：单步RMSE比纯线性模型好35倍
- 输出：`sindy_nn_residual.pt`

### 2.3 多步rollout对比（已完成）

| 模型 | 单步RMSE | 10步RMSE | 20步RMSE |
|------|----------|----------|----------|
| 线性(A_d@s+B_d*tau) | 0.003736 | 0.045 | 发散 |
| SINDy | 0.000892 | 0.012 | 0.08 |
| SINDy+NN | 0.000103 | 0.003 | 0.02 |
| 纯NN | 0.000156 | 0.004 | 0.03 |

**结论**：SINDy+NN在10步内最准确，超过20步后所有模型因混沌发散。

### 2.4 MBPO-SAC v7（已完成，效果差）

- 文件：`mbpo_sac_v7.py`
- 结果：600 episodes后SAC策略与纯LQR基线完全相同（theta_rms: 14.99°）
- 失败原因：SAC的auto-alpha从0.303崩溃到0.007，探索停止，策略收敛为"什么都不做"
- 详细输出：`model_stage1_20260620_094933/`

### 2.5 HRRL参考项目分析（已完成）

- 算法：TD3（不是SAC！）——这是关键发现
- 物理引擎：PyBullet 3D（完整刚体仿真）
- 训练步数：1001*100 = 100,100步
- 动作映射：`targetPosition = action[0]*0.1 + pid_control`
- 奖励函数：误差减小量模式 + 平稳性模式（beta=0.002切换）
- 扰动调度：`sigma = min(0.3, 1.08^ep * 0.01)`
- LQR参数：kp=15.249, kd=2.96, k=12.3（手工调参）

---

## 三、当前状态（待完成）

### 3.1 MBPO-TD3脚本（未写）

需要创建 `mbpo_td3.py`，基于 `mbpo_sac_v7.py` 进行以下改动：

#### 改动1：SAC → TD3

```python
# SAC（失败）→ TD3（待实现）
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
# 旧（v7，效果差）:
dis_reward + angle_penalty(-2*|theta|) + vel_penalty(-0.05*|w|) + upright_bonus
terminated: -10 - |w|

# 新（HRRL原版，待实现）:
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

### 3.2 世界模型优化分析（已测试，结论：非必要）

#### 问题假设

训练过程中倾斜角达到17°，线性化模型在此角度下误差显著，需要非线性动力学优化。

#### 实际测试结果（test_nonlinear_dynamics.py）

**测试方法**：对比线性化Meijaard vs 非线性Meijaard+RK4+sin(phi)/phi修正

**测试1：单步误差**

| 角度 | 线性化phi_next | 非线性phi_next | 差异(°) | 相对误差 |
|------|---------------|---------------|---------|---------|
| 5° | 0.087387 | 0.087372 | -0.001° | 0.02% |
| 10° | 0.174970 | 0.175027 | +0.003° | 0.03% |
| 17° | 0.297587 | 0.297731 | +0.008° | 0.05% |
| 30° | 0.525303 | 0.525540 | +0.014° | 0.04% |

单步误差在17°时仅0.008°（0.05%），远小于预期。

**测试2：开环长时间rollout（无控制，tau=5.0 Nm）**

| 步数 | 线性化最终 | 非线性最终 | 差异 |
|------|-----------|-----------|------|
| 10步 | +16.19° | +15.61° | 0.58° |
| 20步 | -21.07° | -23.97° | 2.90° |
| 50步 | -726.40° | +254.78° | **981°** |

开环时两种模型在20步后完全发散（混沌系统），但这不代表实际场景。

**测试3：带LQR控制的rollout（最关键！）**

| 步数 | 线性化最终 | 非线性最终 | **差异** |
|------|-----------|-----------|---------|
| 10步 | +10.58° | +10.37° | **0.21°** |
| 20步 | +3.48° | +3.48° | **0.00°** |
| 50步 | +0.14° | +0.14° | **0.00°** |
| 100步 | +0.00° | +0.00° | **0.00°** |

**关键发现：LQR控制使两种模型的轨迹差异几乎为零。**

**测试4：带LQR+随机残差（模拟RL虚拟rollout）**

| Episode | 初始角度 | 线性化 | 非线性 | 差异 |
|---------|---------|--------|--------|------|
| 1 | -4.3° | -9.26° | -9.23° | 0.03° |
| 2 | +14.5° | +2.77° | +2.77° | 0.00° |
| 3 | -3.1° | -6.57° | -6.56° | 0.01° |
| 4 | +7.8° | +6.23° | +6.22° | 0.01° |
| 5 | +1.7° | +8.71° | +8.69° | 0.02° |

**50步rollout中，最大差异仅0.03°。**

#### 结论

**世界模型优化非必要。** 原因：

1. 实际系统使用 `tau = u_lqr + action * 0.1`，LQR主导控制
2. LQR的强纠正能力使线性化误差被持续补偿
3. 在LQR控制下，10步rollout的两种模型差异仅0.21°，50步后差异为0°
4. 非线性优化的收益被LQR"掩盖"了
5. 真正的瓶颈是算法（SAC→TD3），不是世界模型精度

**建议：直接使用当前SINDy+NN世界模型+TD3，不做世界模型优化。**

### 3.3 分析文档（未写）

需要创建 `.md` 文件，记录：
- TD3算法改进分析
- 世界模型优化测试结论（已记录在3.2）
- 当前条件如实记录
- HRRL对比分析

---

## 四、关键文件清单

### 4.1 核心代码文件

| 文件 | 用途 | 状态 |
|------|------|------|
| `sindy_bicycle/mbpo_sac_v7.py` | MBPO-SAC v7训练脚本 | 已完成（效果差） |
| `sindy_bicycle/mbpo_td3.py` | MBPO-TD3训练脚本 | **待创建** |
| `sindy_bicycle/meijaard_dynamics.py` | Meijaard动力学函数 | 已完成 |
| `sindy_bicycle/sindy_identification.py` | SINDy辨识代码 | 已完成 |
| `sindy_bicycle/compare_dynamics_models.py` | 动力学模型对比 | 已完成 |
| `sindy_bicycle/compare_multistep.py` | 多步rollout对比 | 已完成 |
| `sindy_bicycle/verify_hrrl_stage1.py` | HRRL参数验证 | 已完成 |

### 4.2 预训练模型文件

| 文件 | 内容 | 大小 |
|------|------|------|
| `sindy_bicycle/sindy_full_model.npz` | SINDy系数(21x4)、state_std(4)、action_std | ~10KB |
| `sindy_bicycle/sindy_nn_residual.pt` | NN残差权重（17,796参数） | ~75KB |
| `sindy_bicycle/meijaard_sindy_v35.npz` | 原始SINDy系数（用于A_d/B_d提取） | ~10KB |

### 4.3 数据文件

| 文件 | 内容 |
|------|------|
| `sindy_bicycle/meijaard_openloop_data_v35.npz` | 开环仿真数据（29478样本） |
| `sindy_bicycle/model_stage1_20260620_094933/` | SAC v7训练输出 |

### 4.4 HRRL参考文件

| 文件 | 用途 |
|------|------|
| `HRRL-new/env.py` | HRRL环境定义（奖励函数、LQR、扰动） |
| `HRRL-new/train_attitude.py` | HRRL训练脚本（TD3配置） |

---

## 五、技术细节速查

### 5.1 Meijaard自行车参数

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
# 总质量：89.21 kg，轴距：1.121 m，trail：0.0686 m
# 自稳定速度范围：约4-6 m/s（实验验证）
```

### 5.2 状态空间

```python
# 物理状态：[theta(倾斜角), delta(转向角), theta_dot(角速度), delta_dot(转向角速度)]
# 归一化：[theta/1.57, delta/1.57, theta_dot/10, v/5]
# 终止条件：|theta| > pi/3 (60°)
# Episode长度：1000步
```

### 5.3 LQR控制器

```python
# Q = diag(1000, 100, 10, 1), R = 0.2
# K = [-103.9, -34.2, 38.0, 3.70]（Riccati方程求解）
# 控制律：u_lqr = -K @ x_lqr
# x_lqr = [theta - target_theta, theta_dot, delta, delta_dot]
```

### 5.4 扰动调度（课程学习）

```python
def compute_sigma(ep):
    return min(0.3, (1.08 ** ep) * 0.01)
# ep=0: sigma=0.01 (0.57°)
# ep=80: sigma=0.3 (17.2°)
# target_theta ~ N(0, sigma), clip to ±pi/12 (±15°)
# 每100步重新采样
# 前100步：target_theta = 0
```

### 5.5 SINDy多项式库

```python
# 输入：[s_norm(4), a_norm(1)] = 5维
# 特征：常数(1) + 线性(5) + 二次交叉(15) = 21个
# 系数矩阵Xi：(21, 4)
# 非零项：8个
```

### 5.6 NN残差架构

```python
ResidualNet:
    Linear(5, 128) → SiLU → Linear(128, 128) → SiLU → Linear(128, 4)
    参数量：17,796
    输入：[s_norm(4), a_norm(1)]
    输出：4维残差delta（归一化空间）
```

---

## 六、下一步执行计划

### 步骤1：创建 `mbpo_td3.py`

基于 `mbpo_sac_v7.py`，需要改动的部分：

1. **替换SACAgent为TD3Agent**
   - 确定性策略网络（无log_std层）
   - 双Q网络（同SAC）
   - 延迟策略更新（policy_delay=2）
   - 目标策略平滑（noise_std=0.2, noise_clip=0.5）
   - 探索噪声（std=0.1）

2. **替换compute_reward为HRRL版本**
   - beta=0.002
   - 小误差：0.1-|w0|
   - 大误差：|error_old|-|error_new|
   - 终止：reward-=1

3. **修改参数**
   - ALPHA: 0.5 → 0.1
   - buffer_size: 300000 → 100000

4. **更新所有SAC相关引用**
   - sac.select_action → td3.select_action
   - sac.update → td3.update
   - sac.alpha_val → 移除
   - sac_weight → td3_weight

### 步骤2：优化世界模型（可选，建议在TD3效果不佳时再做）

在虚拟rollout中用非线性动力学替换线性化模型：

```python
def nonlinear_step(s, tau, dt, M, C1, K0, K2, v, g):
    """RK4积分的非线性Meijaard动力学一步。"""
    # M @ q_dd + C1*v @ q_dot + (g*K0 + v²*K2) @ q = [[0], [tau]]
    # q_dd = invM @ (-C1*v*q_dot - (g*K0 + v²*K2)*q + [[0],[tau]])
    # 使用sin(phi)/phi修正（大角度更准）
```

### 步骤3：执行训练

```bash
"E:/Anaconda/python.exe" "D:/系统辨识作业/sindy_bicycle/mbpo_td3.py"
```

预期时间：30-60分钟（GPU）

### 步骤4：撰写分析文档

将所有分析结果如实记录到 `.md` 文件。

---

## 七、已知问题与风险

### 7.1 SAC alpha崩溃（已确认）

- 现象：alpha从0.303→0.007，探索停止
- 原因：当策略接近最优时，熵项变小，alpha自动降低以"鼓励"利用
- 解决：切换TD3（无熵项）

### 7.2 线性化模型精度（已确认）

- 现象：17°时线性化误差~4%
- 影响：虚拟rollout中的"想象"经验有偏差
- 解决：非线性动力学积分（待实现）

### 7.3 手写TD3 vs SB3（潜在风险）

- HRRL使用stable-baselines3的TD3（成熟库）
- 本项目手写TD3（可能有实现细节差异）
- 关键细节：target policy smoothing的实现、exploration noise的衰减

### 7.4 虚拟rollout误差累积（已知限制）

- SINDy+NN在10步内准确，超过20步发散
- 当前rollout horizon=10（在安全范围内）
- 但如果世界模型有系统性偏差，10步累积仍可能导致虚假经验

---

## 八、运行环境

```bash
# Python环境
E:/Anaconda/python.exe

# 依赖
numpy, torch(CUDA), scipy, sklearn

# GPU
CUDA可用时自动使用GPU

# 工作目录
D:/系统辨识作业/sindy_bicycle/
```

---

## 九、关键命令

```bash
# 运行SAC v7（已完成，效果差）
"E:/Anaconda/python.exe" "D:/系统辨识作业/sindy_bicycle/mbpo_sac_v7.py"

# 运行TD3（待创建）
"E:/Anaconda/python.exe" "D:/系统辨识作业/sindy_bicycle/mbpo_td3.py"

# 运行动力学模型对比
"E:/Anaconda/python.exe" "D:/系统辨识作业/sindy_bicycle/compare_dynamics_models.py"

# 运行多步rollout对比
"E:/Anaconda/python.exe" "D:/系统辨识作业/sindy_bicycle/compare_multistep.py"

# 查看SINDy系数
"E:/Anaconda/python.exe" -c "
import numpy as np
d = np.load('D:/系统辨识作业/sindy_bicycle/sindy_full_model.npz')
print('coefficients:', d['coefficients'].shape)
print('state_std:', d['state_std'])
print('action_std:', d['action_std'])
"
```
