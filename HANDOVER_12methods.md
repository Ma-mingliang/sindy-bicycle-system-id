# 交接文档：12种系统辨识方法对比测试

> 记录时间：2026-06-20
> 会话ID：`4620965f-97a0-4ec3-964d-0419836f1c8f`
> 计划文件：`C:\Users\lenovo_mml\.claude\plans\delightful-crunching-hinton.md`

---

## 一、任务目标

基于Meijaard 2007解析模型作为真实动力学，测试12种系统辨识方法，找出**多步rollout精度最高**的方法。为用户的**双轮可转向自行车**选择最佳辨识方案，用于MBPO世界模型。

---

## 二、环境（必须用DL环境）

```
Python: E:/Anaconda/envs/DL/python.exe（DL conda环境，不要用base）
Python 3.9.23, numpy 1.23.5, torch 2.7.1+cu128, CUDA: True
GPU: NVIDIA GeForce RTX 5060 Laptop GPU
scipy 1.10.1, scikit-learn 1.6.1, torchdiffeq 0.2.5
```

---

## 三、评估方案（方案C：真实模型LQR + 同一力矩回放）

**核心思路**：验证拟合模型 vs 真实模型的差异，所有模型必须接收**完全相同的输入**。

1. 用**真实模型 + LQR** 生成5段×500步参考轨迹，记录动作序列 `[tau_0, tau_1, ..., tau_499]`
2. 所有模型从**相同初始状态**出发，接收**相同的力矩序列**
3. 比较各模型预测 vs 真实轨迹
4. 评估步数：[1, 5, 10, 20, 50, 100, 200, 500]
5. 指标：theta角度RMSE（rad）

**不使用开环随机力矩**（系统易发散），**不使用各自LQR**（输入不公平）。

---

## 四、12种方法实现蓝图

### 基础方法（9种）

| # | 方法 | 实现要点 |
|---|------|---------|
| 1 | SINDy多项式（21特征） | `build_poly_library`: 常数+线性+二次交叉，STLSQ threshold=0.05 |
| 2 | SINDy+三角函数（32特征） | 加 sin(phi), cos(phi), sin(phi)*x_i, cos(phi)*x_i |
| 3 | SINDy+三角+指数（36特征） | 加 exp(-phi^2), exp(-delta^2), exp(-phi^2)*x_i |
| 4 | SINDy最佳+NN残差 | 从1-3选RMSE最低的，NN学残差（3层MLP, 128 hidden, SiLU） |
| 5 | NN端到端 | 直接学 (s_norm, a_norm) → delta_s_norm |
| 6 | Neural ODE | NN学 ds/dt=f(s,a)，用torchdiffeq的odeint积分 |
| 7 | 高斯过程 | sklearn GP，下采样5000样本，4输出分别训练 |
| 8 | PINN | NN + 物理约束损失（M·q_dd结构） |
| 9 | 参数辨识 | 假设Meijaard结构，拟合M,C1,K0,K2的2×2矩阵 |

### 混合方法（3种）

| # | 方法 | 实现要点 |
|---|------|---------|
| 10 | Neural ODE+NN残差 | 方法6做基线 + NN学残差 |
| 11 | GP+NN残差 | 方法7做基线 + NN学残差 |
| 12 | 参数辨识+NN残差 | 方法9做基线 + NN学残差 |

---

## 五、已有代码参考（必须读取）

| 文件 | 可复用内容 |
|------|-----------|
| `meijaard_dynamics.py` | `benchmark_par_to_canonical(p)` → M,C1,K0,K2；`ab_matrix()` → A,B |
| `test_sindy_trig_library.py` | `build_poly_library()`, `build_trig_library()`, `stlsq()` |
| `test_worldmodel_vs_real.py` | 完整的多步RMSE评估框架 |
| `train_nonlinear_residual_nn.py` | `ResidualNet` 架构（NN残差） |

---

## 六、待创建文件

| 文件 | 说明 |
|------|------|
| `test_all_methods.py` | 12种方法统一测试脚本（单文件，~800行） |
| `方向探索.md`（更新） | 添加测试结果和分析 |

---

## 七、运行命令

```bash
"E:/Anaconda/envs/DL/python.exe" "D:/系统辨识作业/sindy_bicycle/test_all_methods.py"
```

---

## 八、交接词

新会话请使用以下交接词：

```
请阅读 D:\系统辨识作业\sindy_bicycle\HANDOVER_12methods.md 了解任务背景。

任务：创建 D:\系统辨识作业\sindy_bicycle\test_all_methods.py，实现12种系统辨识方法的对比测试。

环境必须用DL环境：E:/Anaconda/envs/DL/python.exe

评估方案C：用真实模型+LQR生成参考轨迹和动作序列，所有模型接收相同的动作序列，比较预测精度。

请先阅读以下文件获取代码参考：
- meijaard_dynamics.py（Meijaard参数和动力学函数）
- test_sindy_trig_library.py（SINDy库函数和STLSQ）
- test_worldmodel_vs_real.py（多步RMSE评估框架）
- train_nonlinear_residual_nn.py（NN残差架构）

然后创建 test_all_methods.py，实现12种方法并运行测试。
```
