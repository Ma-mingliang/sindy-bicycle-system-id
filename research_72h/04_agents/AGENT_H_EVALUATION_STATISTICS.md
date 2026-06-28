# Agent H: 评估与统计协议设计

**创建时间**: 2026-06-28
**职责**: 为自行车动力学 Neural ODE 预测模型设计公平、严格、可复现的评估协议
**基线版本**: V9 Neural ODE (hidden=64, depth=3, tanh, seed=43)

---

## 0. 项目上下文摘要

| 属性 | 值 |
|------|-----|
| 数据集 | `data/stage2_dataset_150k.npz` |
| 样本数 | 150,000 |
| Episode 数 | 148 |
| 状态维度 | 7D (e_y, e_psi, v, theta, theta_dot, delta, delta_dot) |
| 动作维度 | 1D (转向指令) |
| 采样频率 | 30 Hz (dt = 1/30 s) |
| 当前划分 | 按 episode 75/25, seed=42 |
| 评估 segments | 5 段, 每段 1100 步 |
| 存活模式 | physical (各状态物理限制) |
| 主指标 | PrimaryLongHorizonScore = mean(NMAE_H100, NMAE_H200, NMAE_H500) |
| 基线分数 | 0.5110 |

### V9 基线性能

| Horizon | NMAE | 存活率 |
|---------|------|--------|
| H=1 | 0.0051 | 100% |
| H=10 | 0.0628 | 100% |
| H=50 | 0.4557 | 100% |
| H=100 | 0.5064 | 100% |
| H=200 | 0.4737 | 100% |
| H=500 | 0.5529 | 100% |

### V9 跨 Seed 变异 (已有数据)

| Seed | H=10 | H=50 | H=100 | H=200 | H=500 |
|------|------|------|-------|-------|-------|
| 42 | 0.0606 | 0.5901 | 0.6713 | 0.6669 | 0.8559 |
| 43 | 0.0534 | 0.5354 | 0.5777 | 0.5467 | 0.7145 |
| 44 | 0.0700 | 0.6011 | 0.6685 | 0.6804 | 0.7802 |

**关键观察**: 跨 seed 标准差在 H=100 时约 0.05, H=500 时约 0.07, 说明模型对初始化敏感, 多 seed 评估是必须的。

---

## 1. 数据划分协议

### 1.1 当前方案分析

当前方案: 按 episode 随机 75/25 划分 (seed=42)

**优点**:
- 保证训练/测试集来自不同 episode, 无时间序列泄漏
- 实现简单, 已在 `data_loader.py` 中正确实现

**风险与不足**:

| 风险 | 严重程度 | 说明 |
|------|----------|------|
| 单 seed 划分 | 高 | 划分结果依赖 seed=42, 可能偶然偏向某类工况 |
| 无工况分层 | 中 | 148 个 episode 可能包含不同速度/转向工况, 随机划分无法保证各工况均匀分布 |
| 无验证集隔离 | 中 | 当前从训练集临时划分验证集, 但未固定验证 episode |
| 评估 segment 选择 | 中 | 仅 5 个 segment, seed=42, 可能不代表整体分布 |

### 1.2 推荐划分协议

#### 1.2.1 三层划分 (Train / Validation / Test)

```
总 Episode: 148
├── 训练集: 70% = 103 episodes (用于模型训练)
├── 验证集: 10% = 15 episodes (用于早停和模型选择)
└── 测试集: 20% = 30 episodes (仅用于最终评估)
```

**划分方法**: 分层随机划分 (Stratified Split)

```python
def stratified_episode_split(episodes_meta, train_ratio=0.70, val_ratio=0.10,
                              test_ratio=0.20, seed=42):
    """按工况分层划分 episode.

    分层依据:
    1. 速度工况 (低/中/高, 基于 episode 内 v 的均值)
    2. 转向幅度 (小/中/大, 基于 episode 内 delta 的 std)

    如果工况信息不可用, 回退到随机划分但固定 seed.
    """
    rng = np.random.RandomState(seed)
    n = len(episodes_meta)
    perm = rng.permutation(n)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_eps = set(perm[:n_train])
    val_eps = set(perm[n_train:n_train + n_val])
    test_eps = set(perm[n_train + n_val:])

    return train_eps, val_eps, test_eps
```

#### 1.2.2 划分规则

| 规则 | 说明 |
|------|------|
| R1: Episode 级隔离 | 同一 episode 的所有样本必须在同一集合中 |
| R2: 归一化统计量 | `state_std`, `action_std`, `delta_std` 仅从训练集计算 |
| R3: 验证集固定 | 验证集 episode 在实验开始时固定, 不可更改 |
| R4: 测试集隔离 | 测试集在整个研究过程中不可见, 仅在最终评估时使用一次 |
| R5: 多划分验证 | 主实验使用 seed=42 划分, 稳健性检查使用 seed=42,43,44,45,46 |

#### 1.2.3 评估 Segment 选择

当前仅 5 个 segment, seed=42, 不够充分。推荐:

```python
EVAL_CONFIG = {
    'n_segments': 20,           # 从 5 增加到 20
    'segment_length': 1100,     # 保持不变
    'segment_seeds': [42, 43, 44, 45, 46],  # 5 个 seed 各生成 20 个 segment
    'total_evaluations': 100,   # 5 seeds x 20 segments
}
```

**理由**:
- 当前 5 个 segment 的 NMAE 方差可能较大
- 20 个 segment 可提供更稳定的估计
- 多 seed 评估可量化模型初始化的变异性

### 1.3 数据泄漏检查清单

| 检查项 | 通过条件 |
|--------|----------|
| 训练集和测试集无 episode 重叠 | `len(train_eps & test_eps) == 0` |
| 归一化统计量仅来自训练集 | `state_std` 从 `train_states` 计算 |
| 验证集不参与最终报告 | 最终结果仅使用测试集 |
| 评估 segment 不包含训练数据 | `test_mask` 正确应用 |
| 超参调优仅使用验证集 | 测试集不用于任何决策 |

---

## 2. 主指标预注册

### 2.1 PrimaryLongHorizonScore 评估

**当前定义**:
```
PrimaryLongHorizonScore = mean(NMAE_H100, NMAE_H200, NMAE_H500)
```

**合理性分析**:

| 维度 | 评估 | 说明 |
|------|------|------|
| 物理意义 | 合理 | H=100 (3.3s), H=200 (6.7s), H=500 (16.7s) 覆盖 MPC 规划窗口 |
| 问题相关性 | 合理 | 长时域预测是核心挑战, 短时域 (H=1,10) 已经较好 |
| 敏感性 | 合理 | 这三个 horizon 的 NMAE 变异大, 能区分不同方法 |
| 临床意义 | 合理 | H=500 约 17 秒, 对自动驾驶决策有意义 |

**建议调整**: 保持当前定义不变, 但添加以下补充指标。

### 2.2 推荐指标体系

#### 2.2.1 主指标 (唯一决策依据)

```
PrimaryLongHorizonScore = mean(NMAE_H100, NMAE_H200, NMAE_H500)
```

**不变**。这是唯一用于判断"巨大提升"的指标。

#### 2.2.2 辅助指标 (报告但不用于决策)

| 指标 | 定义 | 用途 |
|------|------|------|
| ShortHorizonScore | mean(NMAE_H1, NMAE_H10) | 确保短时域不退化 |
| MidHorizonScore | NMAE_H50 | 中间参考点 |
| WorstStateNMAE_H100 | max(per_state_nmae) at H=100 | 识别最弱状态 |
| SurvivalRate_H500 | 存活率 at H=500 | 数值稳定性 |
| DivergenceRate | 1 - survival_rate at H=500 | 发散比例 |

#### 2.2.3 Per-State 指标 (诊断用)

必须报告但不用于主决策:

```python
PER_STATE_METRICS = {
    'e_y':       {'critical': True,  'reason': '路径跟踪核心量'},
    'e_psi':     {'critical': True,  'reason': '航向控制核心量'},
    'v':         {'critical': False, 'reason': '变化极小, 参考'},
    'theta':     {'critical': False, 'reason': '横滚角, 物理约束强'},
    'theta_dot': {'critical': False, 'reason': '横滚角速度'},
    'delta':     {'critical': False, 'reason': '转向角'},
    'delta_dot': {'critical': False, 'reason': '转角速度'},
}
```

**关键状态不恶化条件** (用于"巨大提升"判断):
- e_y NMAE at H=100 不退化 > 10%
- e_psi NMAE at H=100 不退化 > 10%

### 2.3 权重分析

当前三个 horizon 等权重:

```
score = (NMAE_100 + NMAE_200 + NMAE_500) / 3
```

**是否需要调整权重?**

| 方案 | 权重 | 优点 | 缺点 |
|------|------|------|------|
| 等权重 (当前) | 1:1:1 | 简单, 无偏 | H=500 变异性大可能主导 |
| 递减权重 | 3:2:1 | 强调近期精度 | 可能低估长时域改进 |
| 递增权重 | 1:2:3 | 强调长时域 | H=500 噪声大 |
| 几何平均 | geo(NMAE) | 对异常值鲁棒 | 解释性差 |

**建议**: 保持等权重 (1:1:1), 理由:
1. 当前三个 horizon 的 NMAE 量级相近 (0.47-0.55), 权重调整影响有限
2. 等权重最简单, 不容易被质疑为 cherry-picking
3. 如果需要强调特定 horizon, 通过辅助指标单独报告

### 2.4 指标预注册模板

```json
{
  "pre_registered_metrics": {
    "primary": {
      "name": "PrimaryLongHorizonScore",
      "formula": "mean(NMAE_H100, NMAE_H200, NMAE_H500)",
      "direction": "lower_is_better",
      "baseline_value": 0.5110,
      "improvement_threshold": 0.3322,
      "used_for_decision": true
    },
    "guardrail": {
      "name": "ShortHorizonNonDegradation",
      "condition": "NMAE_H1 <= 0.0054 AND NMAE_H10 <= 0.0660",
      "max_degradation_pct": 5,
      "used_for_decision": true
    },
    "auxiliary": [
      "NMAE_H1",
      "NMAE_H10",
      "NMAE_H50",
      "per_state_nmae_at_H100",
      "survival_rate_at_H500"
    ]
  }
}
```

---

## 3. 多 Seed 评估协议

### 3.1 配对 Seed 设计

**核心原则**: 每个实验条件使用相同的 5 个 seed, 使得配对比较成为可能。

```python
PAIRING_SEEDS = [42, 43, 44, 45, 46]

# 每个 seed 对应:
# 1. 数据划分 seed (哪个 episode 分到哪个集合)
# 2. 模型初始化 seed (权重初始化)
# 3. 训练 shuffle seed (数据顺序)
# 4. 评估 segment seed (选择哪些 segment)

# 所有方法使用相同的 5 个 seed, 确保配对比较有效
```

### 3.2 评估流程

```
对于每个方法 M (包括基线):
    对于每个 seed S in [42, 43, 44, 45, 46]:
        1. 使用 seed=S 划分数据
        2. 使用 seed=S 初始化模型
        3. 使用 seed=S 训练模型
        4. 使用 seed=S 选择评估 segments
        5. 记录所有 horizon 的 NMAE
        6. 记录 per-state NMAE
        7. 记录存活率

结果矩阵: R[method][seed][horizon] -> NMAE
```

### 3.3 统计检验

#### 3.3.1 配对 t 检验

```python
from scipy import stats

def paired_t_test(baseline_scores, candidate_scores, alpha=0.05):
    """配对 t 检验: 候选方法是否显著优于基线.

    Args:
        baseline_scores: shape (n_seeds,) - 基线在各 seed 的 PrimaryLongHorizonScore
        candidate_scores: shape (n_seeds,) - 候选在各 seed 的 PrimaryLongHorizonScore
        alpha: 显著性水平

    Returns:
        dict with t_stat, p_value, significant, ci_95
    """
    diff = baseline_scores - candidate_scores  # 正值表示候选更优
    n = len(diff)
    mean_diff = np.mean(diff)
    se_diff = stats.sem(diff)

    # 配对 t 检验
    t_stat, p_value = stats.ttest_rel(baseline_scores, candidate_scores)

    # 95% 置信区间
    ci = stats.t.interval(1 - alpha, df=n-1, loc=mean_diff, scale=se_diff)

    return {
        'mean_improvement': float(mean_diff),
        'std_improvement': float(np.std(diff)),
        't_stat': float(t_stat),
        'p_value': float(p_value),
        'significant': p_value < alpha,
        'ci_95_lower': float(ci[0]),
        'ci_95_upper': float(ci[1]),
        'n_seeds': n,
    }
```

#### 3.3.2 Wilcoxon 符号秩检验

当样本量小 (n=5) 且分布可能非正态时, Wilcoxon 检验更稳健:

```python
def wilcoxon_test(baseline_scores, candidate_scores, alpha=0.05):
    """Wilcoxon 符号秩检验.

    适用于:
    - 样本量小 (n < 20)
    - 差值分布可能非正态
    - 配对设计
    """
    diff = baseline_scores - candidate_scores

    # 移除零差值
    diff_nonzero = diff[diff != 0]
    if len(diff_nonzero) < 3:
        return {'significant': False, 'reason': 'insufficient_nonzero_diffs'}

    stat, p_value = stats.wilcoxon(diff_nonzero, alternative='greater')

    return {
        'statistic': float(stat),
        'p_value': float(p_value),
        'significant': p_value < alpha,
        'n_nonzero': len(diff_nonzero),
    }
```

#### 3.3.3 Bootstrap 置信区间

Bootstrap 适合小样本, 不依赖正态假设:

```python
def bootstrap_ci(baseline_scores, candidate_scores, n_bootstrap=10000,
                 alpha=0.05, seed=42):
    """Bootstrap 置信区间估计.

    使用配对 bootstrap: 对 (baseline_i, candidate_i) 对进行重采样.
    """
    rng = np.random.RandomState(seed)
    n = len(baseline_scores)
    improvements = []

    for _ in range(n_bootstrap):
        # 配对重采样
        idx = rng.choice(n, size=n, replace=True)
        b_mean = np.mean(baseline_scores[idx])
        c_mean = np.mean(candidate_scores[idx])
        improvements.append(b_mean - c_mean)

    improvements = np.array(improvements)

    return {
        'mean_improvement': float(np.mean(improvements)),
        'ci_lower': float(np.percentile(improvements, 100 * alpha / 2)),
        'ci_upper': float(np.percentile(improvements, 100 * (1 - alpha / 2))),
        'prob_improvement': float(np.mean(improvements > 0)),
        'n_bootstrap': n_bootstrap,
    }
```

### 3.4 效应量计算

#### 3.4.1 Cohen's d (配对)

```python
def cohens_d_paired(baseline_scores, candidate_scores):
    """配对 Cohen's d 效应量.

    d = mean(diff) / std(diff)

    解释:
    - |d| < 0.2: 可忽略
    - 0.2 <= |d| < 0.5: 小效应
    - 0.5 <= |d| < 0.8: 中等效应
    - |d| >= 0.8: 大效应
    """
    diff = baseline_scores - candidate_scores
    d = np.mean(diff) / np.std(diff, ddof=1)

    return {
        'cohens_d': float(d),
        'magnitude': 'large' if abs(d) >= 0.8 else
                     'medium' if abs(d) >= 0.5 else
                     'small' if abs(d) >= 0.2 else 'negligible',
    }
```

#### 3.4.2 相对改进百分比

```python
def relative_improvement(baseline_scores, candidate_scores):
    """相对改进百分比."""
    b_mean = np.mean(baseline_scores)
    c_mean = np.mean(candidate_scores)
    pct = (b_mean - c_mean) / b_mean * 100

    return {
        'baseline_mean': float(b_mean),
        'candidate_mean': float(c_mean),
        'improvement_pct': float(pct),
    }
```

### 3.5 多 Seed 报告模板

```
=== 多 Seed 评估报告 ===

方法: [方法名]
评估 Seeds: [42, 43, 44, 45, 46]

Per-Seed 结果:
| Seed | NMAE_H100 | NMAE_H200 | NMAE_H500 | PrimaryScore |
|------|-----------|-----------|-----------|--------------|
| 42   | 0.xxxx    | 0.xxxx    | 0.xxxx    | 0.xxxx       |
| 43   | 0.xxxx    | 0.xxxx    | 0.xxxx    | 0.xxxx       |
| 44   | 0.xxxx    | 0.xxxx    | 0.xxxx    | 0.xxxx       |
| 45   | 0.xxxx    | 0.xxxx    | 0.xxxx    | 0.xxxx       |
| 46   | 0.xxxx    | 0.xxxx    | 0.xxxx    | 0.xxxx       |

汇总:
- Mean ± Std: 0.xxxx ± 0.xxxx
- Median: 0.xxxx
- Min / Max: 0.xxxx / 0.xxxx

vs 基线 (配对比较):
- Mean improvement: 0.xxxx (xx.x%)
- 95% CI: [0.xxxx, 0.xxxx]
- Paired t-test p-value: 0.xxxx
- Wilcoxon p-value: 0.xxxx
- Bootstrap P(improvement): 0.xxxx
- Cohen's d: 0.xxxx (magnitude)
- 方向一致性: x/5 seeds 改善
```

---

## 4. 消融实验设计

### 4.1 消融原则

| 原则 | 说明 |
|------|------|
| 单变量变化 | 每次只改变一个组件, 其他完全固定 |
| 相同种子 | 所有消融使用相同的 5 个 seed |
| 相同数据 | 所有消融使用相同的训练/测试划分 |
| 全 horizon 报告 | 每个消融报告所有 horizon 的 NMAE |
| 完整统计 | 每个消融进行多 seed 配对比较 |

### 4.2 核心组件消融矩阵

#### 4.2.1 训练策略消融

| 编号 | 消融项 | 变化 | 固定条件 | 对应假设 |
|------|--------|------|----------|----------|
| ABL-01 | 无多步课程 | rollout_curriculum='1' | 其他全部相同 | H2: 分布偏移 |
| ABL-02 | 短课程 | rollout_curriculum='1,5' | 其他全部相同 | H2: 分布偏移 |
| ABL-03 | 长课程 | rollout_curriculum='1,5,10,20,50' | 其他全部相同 | H2: 分布偏移 |
| ABL-04 | 无 Jacobian 正则 | lambda_jacobian=0 | 其他全部相同 | 模型平滑性 |
| ABL-05 | 无一致性损失 | lambda_consistency=0 | 其他全部相同 | 物理一致性 |
| ABL-06 | 无多步损失 | lambda_multi=0 | 其他全部相同 | 多步训练 |

#### 4.2.2 架构消融

| 编号 | 消融项 | 变化 | 固定条件 | 对应假设 |
|------|--------|------|----------|----------|
| ABL-07 | 浅网络 | depth=2 | 其他全部相同 | H5: 模型容量 |
| ABL-08 | 深网络 | depth=4 | 其他全部相同 | H5: 模型容量 |
| ABL-09 | 窄网络 | hidden=32 | 其他全部相同 | H5: 模型容量 |
| ABL-10 | 宽网络 | hidden=128 | 其他全部相同 | H5: 模型容量 |
| ABL-11 | SiLU 激活 | activation='silu' | 其他全部相同 | 激活函数 |

#### 4.2.3 积分方法消融

| 编号 | 消融项 | 变化 | 固定条件 | 对应假设 |
|------|--------|------|----------|----------|
| ABL-12 | RK4 积分 | 使用 RK4 代替 Euler | 网络不变 | H1: 数值积分 |
| ABL-13 | 小步长 | dt=1/60 | 网络不变 | H1: 数值积分 |

#### 4.2.4 数据消融

| 编号 | 消融项 | 变化 | 固定条件 | 对应假设 |
|------|--------|------|----------|----------|
| ABL-14 | 更多数据 | 使用 stanley_ref_450k | 网络不变 | H3: 数据覆盖 |
| ABL-15 | 更少数据 | 使用 50% 训练数据 | 网络不变 | H3: 数据覆盖 |

### 4.3 恢复旧逻辑测试

目的是验证"修复"某个组件是否真正带来改进, 而不是其他附带变化:

| 编号 | 测试项 | 说明 |
|------|--------|------|
| ROL-01 | V9 原始配置完全复现 | 使用原始 checkpoint 和评估代码, 验证基线数值 |
| ROL-02 | V9 + 新评估代码 | 使用原始模型, 新评估协议, 检查评估代码变化的影响 |
| ROL-03 | V9 + 新数据划分 | 使用原始模型, 新数据划分, 检查划分变化的影响 |
| ROL-04 | V8 模型在 V9 评估下 | 检查 V8 模型是否真的更差 |

### 4.4 消融实验运行模板

```python
ABLATION_CONFIGS = {
    'abl_01_no_multi': {
        'name': '无多步课程',
        'changes': {'rollout_curriculum': '1'},
        'hypothesis': 'H2',
    },
    'abl_02_short_curriculum': {
        'name': '短课程',
        'changes': {'rollout_curriculum': '1,5'},
        'hypothesis': 'H2',
    },
    # ... 其他消融
}

def run_ablation(ablation_id, seeds=PAIRING_SEEDS):
    """运行单个消融实验."""
    cfg = ABLATION_CONFIGS[ablation_id]
    results = {}

    for seed in seeds:
        # 1. 加载数据 (使用 seed 划分)
        data = load_7d_data(seed=seed)

        # 2. 创建模型 (基线配置 + 单变量变化)
        config = NeuralODEConfig(seed=seed)
        for key, value in cfg['changes'].items():
            setattr(config, key, value)

        # 3. 训练
        model = NeuralODEV9(config)
        model.train(data['train_states'], data['train_actions'],
                    data['train_deltas'], data['state_std'],
                    data['action_std'], data['delta_std'])

        # 4. 评估
        segments = get_test_segments(data, n_segments=20, seed=seed)
        eval_results = multi_step_evaluate(
            model, segments, data['state_std'],
            horizons=[1, 10, 50, 100, 200, 500]
        )

        results[seed] = eval_results

    return results
```

### 4.5 消融结果报告模板

```
=== 消融实验报告 ===

| 消融项 | H=1 | H=10 | H=50 | H=100 | H=200 | H=500 | PrimaryScore | vs 基线 |
|--------|-----|------|------|-------|-------|-------|--------------|---------|
| V9 基线 | 0.0051 | 0.0628 | 0.4557 | 0.5064 | 0.4737 | 0.5529 | 0.5110 | - |
| ABL-01 无多步课程 | ... | ... | ... | ... | ... | ... | ... | +xx.x% |
| ABL-02 短课程 | ... | ... | ... | ... | ... | ... | ... | +xx.x% |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |

关键发现:
1. [消融项] 对 [指标] 影响最大 (xx.x% 变化)
2. ...
```

---

## 5. OOD (分布外) 测试协议

### 5.1 OOD 测试的必要性

当前数据的已知限制:
- 速度范围极窄: [0.54, 0.66] m/s (仅 0.12 m/s 范围)
- 工况可能单一: 可能都是低速直线或缓弯
- 模型可能过拟合到特定工况

OOD 测试可评估模型在未见工况下的鲁棒性。

### 5.2 OOD 测试维度

#### 5.2.1 速度工况 OOD

```python
VELOCITY_OOD_TESTS = {
    'v_low': {
        'description': '极低速 (v < 0.54 m/s)',
        'condition': lambda ep: np.mean(ep['v']) < 0.54,
        'availability': '需要新数据收集',
    },
    'v_high': {
        'description': '高速 (v > 0.66 m/s)',
        'condition': lambda ep: np.mean(ep['v']) > 0.66,
        'availability': '需要新数据收集',
    },
    'v_variable': {
        'description': '变速工况',
        'condition': lambda ep: np.std(ep['v']) > 0.05,
        'availability': '可能在现有数据中',
    },
}
```

#### 5.2.2 转向工况 OOD

```python
STEERING_OOD_TESTS = {
    'delta_large': {
        'description': '大转向角 (|delta| > 0.5 rad)',
        'condition': lambda ep: np.max(np.abs(ep['delta'])) > 0.5,
        'availability': '可能在现有数据中',
    },
    'delta_rapid': {
        'description': '快速转向 (|delta_dot| > 1.0 rad/s)',
        'condition': lambda ep: np.max(np.abs(ep['delta_dot'])) > 1.0,
        'availability': '可能在现有数据中',
    },
    'delta_oscillating': {
        'description': '振荡转向',
        'condition': lambda ep: count_zero_crossings(ep['delta_dot']) > 10,
        'availability': '需要分析',
    },
}
```

#### 5.2.3 初始条件 OOD

```python
IC_OOD_TESTS = {
    'ey_large': {
        'description': '大横向偏差 (|e_y| > 0.3 m)',
        'condition': lambda seg: abs(seg['states'][0][0]) > 0.3,
        'availability': '可能在现有数据中',
    },
    'epsi_large': {
        'description': '大航向偏差 (|e_psi| > 0.5 rad)',
        'condition': lambda seg: abs(seg['states'][0][1]) > 0.5,
        'availability': '可能在现有数据中',
    },
    'theta_large': {
        'description': '大横滚角 (|theta| > 0.3 rad)',
        'condition': lambda seg: abs(seg['states'][0][3]) > 0.3,
        'availability': '需要分析',
    },
}
```

### 5.3 OOD 测试实施

#### 5.3.1 从现有数据中提取 OOD 子集

```python
def extract_ood_subsets(data, test_mask):
    """从测试集中提取 OOD 子集.

    基于测试集中各 episode 的工况特征进行分组.
    """
    ood_subsets = {}

    # 按速度分组
    test_indices = np.where(test_mask)[0]
    # ... 提取各工况子集

    return ood_subsets
```

#### 5.3.2 OOD 评估报告

```
=== OOD 测试报告 ===

| OOD 子集 | N_segments | NMAE_H100 | NMAE_H500 | vs ID 性能 | 结论 |
|----------|------------|-----------|-----------|------------|------|
| ID (分布内) | 20 | 0.xxxx | 0.xxxx | 基线 | - |
| v_low | 5 | 0.xxxx | 0.xxxx | +xx.x% | 可接受/不可接受 |
| delta_large | 8 | 0.xxxx | 0.xxxx | +xx.x% | 可接受/不可接受 |
| ey_large | 6 | 0.xxxx | 0.xxxx | +xx.x% | 可接受/不可接受 |
| ... | ... | ... | ... | ... | ... |

OOD 鲁棒性判定:
- 所有 OOD 子集 NMAE 退化 < 50%: PASS
- 任一 OOD 子集 NMAE 退化 > 100%: FAIL
- 中间: WARNING
```

### 5.4 OOD 数据收集建议

如果现有数据无法提供足够的 OOD 测试, 建议:

| 优先级 | OOD 类型 | 数据来源 | 成本 |
|--------|----------|----------|------|
| 高 | 不同速度 | 修改 env 参数收集 | 低 |
| 高 | 大初始偏差 | 从偏差状态开始 episode | 低 |
| 中 | 快速转向 | 增加控制增益 | 中 |
| 低 | 不同路面 | 需要物理模型扩展 | 高 |

---

## 6. 统计方法详细规范

### 6.1 统计检验选择指南

```
样本量 n=5 (5 个 seed)
├── 配对设计 (相同 seed 用于所有方法)
│
├── 检验选择:
│   ├── 首选: Wilcoxon 符号秩检验
│   │   理由: 不依赖正态假设, 适合小样本
│   │
│   ├── 备选: 配对 t 检验
│   │   理由: 如果差值近似正态 (Shapiro-Wilk p > 0.05)
│   │
│   └── 补充: Bootstrap 置信区间
│       理由: 提供效应量的不确定性估计
│
└── 效应量:
    ├── Cohen's d (配对)
    └── 相对改进百分比
```

### 6.2 完整统计分析流程

```python
def full_statistical_analysis(baseline_results, candidate_results,
                               horizons=[100, 200, 500]):
    """完整统计分析流程.

    Args:
        baseline_results: dict {seed: {horizon: nmae}}
        candidate_results: dict {seed: {horizon: nmae}}

    Returns:
        dict with all statistical results
    """
    seeds = sorted(baseline_results.keys())
    n_seeds = len(seeds)

    # 1. 计算 PrimaryLongHorizonScore
    baseline_scores = np.array([
        np.mean([baseline_results[s][h] for h in horizons])
        for s in seeds
    ])
    candidate_scores = np.array([
        np.mean([candidate_results[s][h] for h in horizons])
        for s in seeds
    ])

    # 2. 配对 t 检验
    t_result = paired_t_test(baseline_scores, candidate_scores)

    # 3. Wilcoxon 检验
    wilcox_result = wilcoxon_test(baseline_scores, candidate_scores)

    # 4. Bootstrap CI
    bootstrap_result = bootstrap_ci(baseline_scores, candidate_scores)

    # 5. 效应量
    effect_size = cohens_d_paired(baseline_scores, candidate_scores)
    rel_improvement = relative_improvement(baseline_scores, candidate_scores)

    # 6. 正态性检验
    diff = baseline_scores - candidate_scores
    if len(diff) >= 3:
        shapiro_stat, shapiro_p = stats.shapiro(diff)
    else:
        shapiro_stat, shapiro_p = None, None

    # 7. 方向一致性
    n_improved = np.sum(candidate_scores < baseline_scores)

    return {
        'primary': {
            'baseline_mean': float(np.mean(baseline_scores)),
            'candidate_mean': float(np.mean(candidate_scores)),
            'improvement_pct': float(rel_improvement['improvement_pct']),
        },
        'paired_t_test': t_result,
        'wilcoxon_test': wilcox_result,
        'bootstrap_ci': bootstrap_result,
        'effect_size': effect_size,
        'normality': {
            'shapiro_stat': float(shapiro_stat) if shapiro_stat else None,
            'shapiro_p': float(shapiro_p) if shapiro_p else None,
            'is_normal': shapiro_p > 0.05 if shapiro_p is not None else None,
        },
        'direction': {
            'n_improved': int(n_improved),
            'n_total': n_seeds,
            'consistency': int(n_improved) / n_seeds,
        },
        'per_seed': {
            s: {
                'baseline': float(baseline_scores[i]),
                'candidate': float(candidate_scores[i]),
                'improvement': float(baseline_scores[i] - candidate_scores[i]),
            }
            for i, s in enumerate(seeds)
        },
    }
```

### 6.3 多重比较校正

当同时比较多个 horizon 或多个消融项时, 需要校正:

```python
from statsmodels.stats.multitest import multipletests

def correct_multiple_comparisons(p_values, method='holm'):
    """多重比较校正.

    方法选择:
    - 'holm': Holm-Bonferroni, 逐步降低阈值, 推荐
    - 'bonferroni': 最保守, 适用于比较数少
    - 'fdr_bh': Benjamini-Hochberg, 控制错误发现率

    Args:
        p_values: list of raw p-values
        method: 校正方法

    Returns:
        dict with corrected p-values and significance
    """
    reject, p_corrected, _, _ = multipletests(p_values, method=method)

    return {
        'raw_p_values': p_values,
        'corrected_p_values': p_corrected.tolist(),
        'significant': reject.tolist(),
        'method': method,
    }
```

**何时使用**:
- 消融实验: 比较 15 个消融项 vs 基线 -> 需要校正
- 多 horizon: 同时检验 H=100, H=200, H=500 -> 可选校正
- 单一主指标: PrimaryLongHorizonScore 单一检验 -> 不需要校正

### 6.4 统计功效分析

```python
def power_analysis(effect_size_d, n_seeds=5, alpha=0.05):
    """统计功效分析.

    回答: 给定效应量和样本量, 检测到显著差异的概率是多少?

    Args:
        effect_size_d: Cohen's d
        n_seeds: 样本量
        alpha: 显著性水平

    Returns:
        dict with power estimate
    """
    from scipy.stats import nct

    # 非中心参数
    ncp = effect_size_d * np.sqrt(n_seeds)
    df = n_seeds - 1
    t_crit = stats.t.ppf(1 - alpha / 2, df)

    # 功效 = P(reject H0 | H1 true)
    power = 1 - nct.cdf(t_crit, df, ncp) + nct.cdf(-t_crit, df, ncp)

    return {
        'effect_size_d': float(effect_size_d),
        'n_seeds': n_seeds,
        'alpha': alpha,
        'power': float(power),
        'interpretation': 'adequate' if power >= 0.8 else
                         'moderate' if power >= 0.6 else 'low',
    }
```

**功效估计** (n=5, alpha=0.05):

| Cohen's d | 功效 | 含义 |
|-----------|------|------|
| 0.2 (小) | ~0.09 | 几乎无法检测 |
| 0.5 (中) | ~0.20 | 检测能力低 |
| 0.8 (大) | ~0.37 | 检测能力中等 |
| 1.5 | ~0.73 | 接近充分 |
| 2.0 | ~0.89 | 充分 |
| 3.0 | ~0.99 | 非常充分 |

**结论**: n=5 只能可靠检测大效应 (d >= 1.5)。如果预期效应量小, 需要增加 seed 数量。

### 6.5 推荐 Seed 数量

| 预期效应量 | 推荐 Seed 数 | 功效 |
|------------|--------------|------|
| d >= 2.0 | 5 | >= 0.89 |
| d >= 1.5 | 5-7 | >= 0.73 |
| d >= 1.0 | 8-10 | >= 0.80 |
| d >= 0.8 | 12-15 | >= 0.80 |
| d >= 0.5 | 30+ | >= 0.80 |

**建议**: 至少 5 个 seed (当前), 如果资源允许增加到 8-10 个。

---

## 7. "巨大提升" 综合判定标准

### 7.1 判定流程

```
候选方法 M 的评估流程:

Step 1: 多 Seed 训练和评估
    ├── 5 个 seed (42, 43, 44, 45, 46)
    ├── 每个 seed 完整训练 + 评估
    └── 记录所有 horizon 的 NMAE

Step 2: 主指标计算
    ├── PrimaryLongHorizonScore = mean(H100, H200, H500)
    └── 每个 seed 的 PrimaryScore

Step 3: 统计检验
    ├── 配对 t 检验 (或 Wilcoxon)
    ├── Bootstrap 95% CI
    ├── Cohen's d
    └── 方向一致性

Step 4: 护栏检查
    ├── H=1 NMAE <= 0.0054 (不退化 > 5%)
    ├── H=10 NMAE <= 0.0660 (不退化 > 5%)
    ├── e_y at H=100 不退化 > 10%
    ├── e_psi at H=100 不退化 > 10%
    └── 存活率不下降

Step 5: 综合判定
```

### 7.2 判定标准

#### 7.2.1 必须全部满足的条件 (PASS/FAIL)

| 编号 | 条件 | 阈值 | 判定 |
|------|------|------|------|
| C1 | PrimaryScore 下降 | >= 35% (score <= 0.3322) | 必须 |
| C2 | H=500 NMAE 下降 | >= 30% (NMAE <= 0.3870) | 必须 |
| C3 | 配对统计显著 | p < 0.05 或 bootstrap P(改善) >= 0.95 | 必须 |
| C4 | 方向一致性 | >= 4/5 seeds 改善 | 必须 |
| C5 | H=1 不退化 | NMAE <= 0.0054 | 必须 |
| C6 | H=10 不退化 | NMAE <= 0.0660 | 必须 |
| C7 | 关键状态不恶化 | e_y, e_psi at H=100 不退化 > 10% | 必须 |
| C8 | 存活率不下降 | H=500 存活率 >= 基线 | 必须 |
| C9 | 无数据泄漏 | 验证数据划分正确 | 必须 |
| C10 | 独立复现 | 至少 2 次独立训练一致 | 必须 |

#### 7.2.2 判定矩阵

```
全部 C1-C10 满足:
    └── 判定: "巨大提升" (HUGE_IMPROVEMENT)

C1-C4 满足, C5-C6 不满足:
    └── 判定: "有提升但短时域退化" (IMPROVED_WITH_REGRESSION)

C1-C2 满足, C3-C4 不满足:
    └── 判定: "点估计改善但统计不显著" (PROMISING_NOT_SIGNIFICANT)

C1 不满足:
    └── 判定: "无显著提升" (NO_IMPROVEMENT)

主指标变差:
    └── 判定: "退化" (REGRESSION)
```

### 7.3 灰色地带处理

如果结果落在灰色地带 (例如 PrimaryScore 下降 33%, 接近但未达到 35%):

1. **不降低阈值**: 阈值在实验开始前锁定
2. **报告完整结果**: 包括所有统计量和置信区间
3. **标记为 "边界案例"**: 供人工审查
4. **建议增加 seed**: 如果当前 seed 数量不足以得出结论

---

## 8. 报告格式规范

### 8.1 完整评估报告模板

```markdown
# 评估报告: [方法名]

## 1. 实验配置
- 方法: [描述]
- 与基线差异: [单变量变化]
- 训练配置: [完整配置]
- 评估配置: [seeds, segments, horizons]

## 2. 主结果

### 2.1 Per-Seed 结果
| Seed | H=1 | H=10 | H=50 | H=100 | H=200 | H=500 | PrimaryScore |
|------|-----|------|------|-------|-------|-------|--------------|
| 42   | ... | ...  | ...  | ...   | ...   | ...   | ...          |
| 43   | ... | ...  | ...  | ...   | ...   | ...   | ...          |
| 44   | ... | ...  | ...  | ...   | ...   | ...   | ...          |
| 45   | ... | ...  | ...  | ...   | ...   | ...   | ...          |
| 46   | ... | ...  | ...  | ...   | ...   | ...   | ...          |

### 2.2 汇总统计
- PrimaryScore: 0.xxxx +/- 0.xxxx (mean +/- std)
- vs 基线: -xx.x% (improvement)
- 95% CI: [0.xxxx, 0.xxxx]
- p-value: 0.xxxx (paired t-test) / 0.xxxx (Wilcoxon)
- Cohen's d: 0.xxxx (magnitude)
- 方向一致性: x/5

### 2.3 护栏检查
| 条件 | 阈值 | 实际值 | 通过? |
|------|------|--------|-------|
| H=1 NMAE | <= 0.0054 | 0.xxxx | PASS/FAIL |
| H=10 NMAE | <= 0.0660 | 0.xxxx | PASS/FAIL |
| e_y at H=100 | 不退化 > 10% | 0.xxxx | PASS/FAIL |
| e_psi at H=100 | 不退化 > 10% | 0.xxxx | PASS/FAIL |
| H=500 存活率 | >= 基线 | xx.x% | PASS/FAIL |

## 3. Per-State 分析 (H=100)
| 状态 | 基线 | 候选 | 变化 | 判定 |
|------|------|------|------|------|
| e_y  | 0.7214 | 0.xxxx | -xx.x% | 改善/退化/持平 |
| e_psi | 0.8883 | 0.xxxx | -xx.x% | 改善/退化/持平 |
| v    | 0.6041 | 0.xxxx | -xx.x% | 改善/退化/持平 |
| theta | 0.2536 | 0.xxxx | -xx.x% | 改善/退化/持平 |
| theta_dot | 0.3809 | 0.xxxx | -xx.x% | 改善/退化/持平 |
| delta | 0.2855 | 0.xxxx | -xx.x% | 改善/退化/持平 |
| delta_dot | 0.4113 | 0.xxxx | -xx.x% | 改善/退化/持平 |

## 4. 统计分析详情
[完整的统计检验结果]

## 5. OOD 测试
[如果有]

## 6. 判定
[PASS/FAIL/WARNING + 理由]
```

### 8.2 实验注册表更新格式

```csv
实验编号,候选算法,对应假设,唯一变化,PrimaryScore,vs基线,p值,Cohen's d,方向一致性,判定,时间戳
EXP0XX,[名称],[假设],[变化],0.xxxx,-xx.x%,0.xxxx,0.xxxx,x/5,PASS/FAIL,2026-XX-XX
```

---

## 9. 实现优先级

### Phase 1: 基础设施 (必须)

| 任务 | 优先级 | 工作量 |
|------|--------|--------|
| 实现三层数据划分 (train/val/test) | P0 | 2h |
| 实现多 seed 评估循环 | P0 | 3h |
| 实现统计分析函数 | P0 | 2h |
| 实现配对比较和 bootstrap | P0 | 2h |
| 更新评估报告生成 | P1 | 2h |

### Phase 2: 消融实验 (重要)

| 任务 | 优先级 | 工作量 |
|------|--------|--------|
| 实现消融配置系统 | P1 | 2h |
| 运行核心消融 (ABL-01 to ABL-06) | P1 | 12h |
| 运行架构消融 (ABL-07 to ABL-11) | P2 | 10h |
| 运行积分消融 (ABL-12 to ABL-13) | P2 | 4h |

### Phase 3: OOD 测试 (可选)

| 任务 | 优先级 | 工作量 |
|------|--------|--------|
| 分析现有数据的工况分布 | P2 | 2h |
| 提取 OOD 子集 | P2 | 2h |
| OOD 评估 | P2 | 4h |
| 必要时收集新数据 | P3 | 8h+ |

---

## 10. 附录

### A. 完整统计分析代码

```python
"""评估与统计分析工具库."""
import numpy as np
from scipy import stats
from typing import Dict, List, Optional


# ============================================================
# 配置
# ============================================================
PAIRING_SEEDS = [42, 43, 44, 45, 46]
PRIMARY_HORIZONS = [100, 200, 500]
ALL_HORIZONS = [1, 10, 50, 100, 200, 500]

# V9 基线值
V9_BASELINE = {
    1: 0.0051, 10: 0.0628, 50: 0.4557,
    100: 0.5064, 200: 0.4737, 500: 0.5529,
}
V9_PRIMARY_SCORE = 0.5110

# 护栏阈值
GUARDRAIL = {
    'h1_max': 0.0054,
    'h10_max': 0.0660,
    'critical_state_max_regression_pct': 10,
}

# "巨大提升" 阈值
HUGE_IMPROVEMENT = {
    'primary_score_max': 0.3322,
    'h500_max': 0.3870,
    'min_direction_consistency': 0.8,
    'min_prob_improvement': 0.95,
}


# ============================================================
# 核心指标
# ============================================================
def compute_primary_score(results: Dict[int, float],
                          horizons: List[int] = PRIMARY_HORIZONS) -> float:
    """计算 PrimaryLongHorizonScore."""
    return float(np.mean([results[h] for h in horizons]))


def compute_nmae_per_state(predicted, actual, state_std, state_names):
    """计算 per-state NMAE."""
    errors = np.abs(predicted - actual)
    nmae = np.mean(errors / state_std, axis=0)
    return {name: float(nmae[i]) for i, name in enumerate(state_names)}


# ============================================================
# 统计检验
# ============================================================
def paired_t_test(baseline: np.ndarray, candidate: np.ndarray,
                  alpha: float = 0.05) -> dict:
    """配对 t 检验."""
    diff = baseline - candidate
    t_stat, p_value = stats.ttest_rel(baseline, candidate)
    se = stats.sem(diff)
    ci = stats.t.interval(1 - alpha, df=len(diff)-1,
                          loc=np.mean(diff), scale=se)
    return {
        'mean_improvement': float(np.mean(diff)),
        'std_improvement': float(np.std(diff, ddof=1)),
        't_stat': float(t_stat),
        'p_value': float(p_value),
        'significant': bool(p_value < alpha),
        'ci_95': [float(ci[0]), float(ci[1])],
    }


def wilcoxon_test(baseline: np.ndarray, candidate: np.ndarray,
                  alpha: float = 0.05) -> dict:
    """Wilcoxon 符号秩检验."""
    diff = baseline - candidate
    diff_nz = diff[diff != 0]
    if len(diff_nz) < 3:
        return {'significant': False, 'reason': 'insufficient_nonzero_diffs'}
    stat, p_value = stats.wilcoxon(diff_nz, alternative='greater')
    return {
        'statistic': float(stat),
        'p_value': float(p_value),
        'significant': bool(p_value < alpha),
    }


def bootstrap_ci(baseline: np.ndarray, candidate: np.ndarray,
                 n_bootstrap: int = 10000, alpha: float = 0.05,
                 seed: int = 42) -> dict:
    """配对 Bootstrap 置信区间."""
    rng = np.random.RandomState(seed)
    n = len(baseline)
    improvements = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        improvements.append(np.mean(baseline[idx]) - np.mean(candidate[idx]))
    improvements = np.array(improvements)
    return {
        'mean_improvement': float(np.mean(improvements)),
        'ci_lower': float(np.percentile(improvements, 100 * alpha / 2)),
        'ci_upper': float(np.percentile(improvements, 100 * (1 - alpha / 2))),
        'prob_improvement': float(np.mean(improvements > 0)),
    }


def cohens_d_paired(baseline: np.ndarray, candidate: np.ndarray) -> dict:
    """配对 Cohen's d."""
    diff = baseline - candidate
    d = float(np.mean(diff) / np.std(diff, ddof=1))
    mag = ('large' if abs(d) >= 0.8 else
           'medium' if abs(d) >= 0.5 else
           'small' if abs(d) >= 0.2 else 'negligible')
    return {'cohens_d': d, 'magnitude': mag}


def full_analysis(baseline: np.ndarray, candidate: np.ndarray) -> dict:
    """完整统计分析."""
    n = len(baseline)
    direction_improved = int(np.sum(candidate < baseline))
    return {
        'paired_t_test': paired_t_test(baseline, candidate),
        'wilcoxon_test': wilcoxon_test(baseline, candidate),
        'bootstrap_ci': bootstrap_ci(baseline, candidate),
        'effect_size': cohens_d_paired(baseline, candidate),
        'relative_improvement_pct': float(
            (np.mean(baseline) - np.mean(candidate)) / np.mean(baseline) * 100
        ),
        'direction': {
            'n_improved': direction_improved,
            'n_total': n,
            'consistency': direction_improved / n,
        },
    }


# ============================================================
# 护栏检查
# ============================================================
def check_guardrails(candidate_results: Dict[int, dict],
                     seeds: List[int] = PAIRING_SEEDS) -> dict:
    """检查所有护栏条件."""
    checks = {}

    # H=1 护栏
    h1_vals = [candidate_results[s]['per_horizon'][1]['nmae_mean']
               for s in seeds]
    checks['h1'] = {
        'values': h1_vals,
        'mean': float(np.mean(h1_vals)),
        'threshold': GUARDRAIL['h1_max'],
        'pass': bool(np.mean(h1_vals) <= GUARDRAIL['h1_max']),
    }

    # H=10 护栏
    h10_vals = [candidate_results[s]['per_horizon'][10]['nmae_mean']
                for s in seeds]
    checks['h10'] = {
        'values': h10_vals,
        'mean': float(np.mean(h10_vals)),
        'threshold': GUARDRAIL['h10_max'],
        'pass': bool(np.mean(h10_vals) <= GUARDRAIL['h10_max']),
    }

    # 存活率护栏
    surv_vals = [candidate_results[s]['per_horizon'][500]['survival_rate']
                 for s in seeds]
    baseline_surv = 1.0  # V9 baseline 100% survival
    checks['survival_h500'] = {
        'values': surv_vals,
        'mean': float(np.mean(surv_vals)),
        'threshold': baseline_surv,
        'pass': bool(np.mean(surv_vals) >= baseline_surv),
    }

    # 综合判定
    checks['all_pass'] = all(c['pass'] for c in checks.values()
                            if isinstance(c, dict) and 'pass' in c)

    return checks


# ============================================================
# 功效分析
# ============================================================
def power_analysis(effect_size_d: float, n_seeds: int = 5,
                   alpha: float = 0.05) -> dict:
    """统计功效分析."""
    from scipy.stats import nct
    ncp = effect_size_d * np.sqrt(n_seeds)
    df = n_seeds - 1
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    power = 1 - nct.cdf(t_crit, df, ncp) + nct.cdf(-t_crit, df, ncp)
    return {
        'effect_size_d': effect_size_d,
        'n_seeds': n_seeds,
        'power': float(power),
        'adequate': power >= 0.8,
    }
```

### B. 参考文献

1. Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences.
2. Wilcoxon, F. (1945). Individual comparisons by ranking methods.
3. Efron, B., & Tibshirani, R. (1993). An Introduction to the Bootstrap.
4. Holm, S. (1979). A simple sequentially rejective multiple test procedure.

### C. 变更日志

| 日期 | 变更 | 原因 |
|------|------|------|
| 2026-06-28 | 初始版本 | 评估协议设计 |
