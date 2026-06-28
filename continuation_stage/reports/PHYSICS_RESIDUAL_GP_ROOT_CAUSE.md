# 物理残差GP失败根因诊断

> **生成时间**: 2026-06-25
> **诊断对象**: gp_b3_physics_residual.py

---

## 1. 问题描述

物理残差GP (GP-B3) 在500步评估中MAE = 2709 ± 925 rad，比标准GP (7.05 rad) 差384倍。

---

## 2. 根因分析

### 2.1 物理模型不匹配 (主要原因)

`gp_b3_physics_residual.py` 中的 `physics_predict` 函数使用了**简化线性化模型**:

```python
# 实际使用的 (错误):
phi_ddot = (g/h) * phi + (v/h) * delta_dot  # 简化
delta_ddot = tau / I_steer                    # 简化, I_steer=0.1 是近似值
s_next = s + dsdt * dt                        # Euler积分, dt=0.01
```

而真实动力学使用:
- 完整 Meijaard 2007 非线性动力学 (25个物理参数)
- RK4 积分器 (5个子步, dt_sub = 1/150s)
- 正确的状态顺序 [phi, delta, phi_dot, delta_dot]

### 2.2 积分方法不匹配

| 方面 | physics_predict | real_step |
|------|----------------|-----------|
| 积分方法 | Euler | RK4 (4阶) |
| 时间步长 | dt=0.01 | dt=0.0333, 5子步 |
| 子步数 | 1 | 5 |
| 精度 | O(dt) | O(dt^4) |

### 2.3 参数不匹配

| 参数 | physics_predict | Meijaard 2007 |
|------|----------------|---------------|
| I_steer | 0.1 (近似) | 0.0589 (精确) |
| h (重心高度) | 0.97 (近似) | 0.97 (巧合正确) |
| g | 9.81 | 9.81 |
| v | 3.5 | 3.5 (v0) |

### 2.4 残差结构问题

由于物理模型与真实动力学差异巨大:
- 残差 = 真实增量 - 物理增量 ≈ O(1) (很大)
- GP需要学习的不是"小修正"，而是"主要动力学"
- 这违背了"物理先验 + 小残差"的设计初衷

---

## 3. 诊断结论

| 分类 | 是否成立 | 说明 |
|------|----------|------|
| 实现错误 | **是** | physics_predict 使用了错误的物理模型 |
| 物理模型不匹配 | **是** | 简化模型 ≠ Meijaard 2007 |
| 归一化错误 | 否 | 归一化方式正确 |
| 超参数问题 | 否 | 核函数和优化参数合理 |
| 数据问题 | 否 | 训练数据正确 |
| 方法局限 | 部分 | 方法本身可行，但实现有误 |

---

## 4. 正确实现方案

要正确实现物理残差GP，应:

1. **使用完整 Meijaard 动力学**: 调用 `methods_common.real_step` 或 `meijaard_dynamics.ab_matrix`
2. **使用正确的积分方法**: RK4 with 5 sub-steps
3. **使用正确的参数**: 从 Meijaard 2007 论文获取全部25个参数
4. **残差标签**: `residual = true_delta - physics_delta`
5. **GP学习残差**: 输入 [s, tau]，输出 residual

### 4.1 正确的 physics_predict 实现

```python
def physics_predict_correct(s, tau):
    """使用线性化 Meijaard 动力学 + RK4 积分。"""
    from meijaard_dynamics import ab_matrix
    from methods_common import M, C1, K0, K2, v0, dt

    A, B = ab_matrix(M, C1, K0, K2, v0, 9.81)

    # RK4 积分
    def f(s, tau):
        return A @ s + B[:, 0] * tau

    k1 = f(s, tau)
    k2 = f(s + 0.5 * dt * k1, tau)
    k3 = f(s + 0.5 * dt * k2, tau)
    k4 = f(s + dt * k3, tau)

    return s + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
```

---

## 5. 验证方法

1. 比较 `physics_predict_correct` 与 `real_step` 的单步预测差异
2. 差异应 < 0.01 rad (线性化近似误差)
3. 残差标准差应 < 0.1 * delta_std

---

*诊断时间: 2026-06-25*
*代码: gp_improvement/gp_b3_physics_residual.py*
