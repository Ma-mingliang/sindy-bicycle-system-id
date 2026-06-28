# AGENT E: 评估方法论审计报告

## 审计范围

审计V9 Neural ODE自行车动力学模型的评估方法论，涵盖NMAE计算、生存率指标、统计显著性、数据泄漏风险、评估horizon选择、以及验收标准的合理性。

**审计文件**:
- `canonical_node/evaluation_v9.py` (130行)
- `canonical_node/config_v9.py` (43行)
- `canonical_node/data_loader_v9.py` (19行)
- `canonical_7d/data_loader.py` (V8, 141行)
- `run_v9_experiments.py`, `run_v9_fixed_rerun.py`

---

## 1. NMAE计算正确性分析

### 1.1 计算公式

```python
# evaluation_v9.py 第6-15行
nmae_per_state = np.mean(errors / state_std, axis=0)  # 按时间步取均值
nmae_overall = np.mean(nmae_per_state)                 # 按状态维度取均值
```

NMAE = mean(|predicted - actual| / state_std)，对所有时间步和所有状态维度取平均。

### 1.2 发现的问题

**问题1: 失败段的NMAE被纳入平均，但计算窗口不同（严重）**

当模型在第k步崩溃时（k < h），该段的NMAE仅在k步上计算：

```python
# 第96-100行
predicted = np.array(predicted)          # 长度 k+1
real = real_states[:len(predicted)]      # 长度 k+1
n_valid = min(len(predicted), len(real)) - 1  # = k

if n_valid > 0:
    nmae = compute_nmae(predicted[1:n_valid+1], real[1:n_valid+1], state_std)
    nmae_list.append(nmae['overall'])  # 仅基于k步的NMAE
```

然后第114行对所有段取平均：
```python
nmae_mean = float(np.nanmean(valid_nmae))
```

**定量影响**：假设5段中有2段在H=500时分别在step 10和step 50崩溃。它们的NMAE分别基于10步和50步计算。而存活500步的段NMAE基于500步计算。三者被等权平均。这意味着早期崩溃的模型可能显示出**人工偏低的NMAE**（误差累积时间更短）。

**问题2: NMAE是累积误差指标，不同horizon直接比较无意义（中等）**

H=1的NMAE=0.006意味着单步预测平均误差为0.006个标准差。H=500的NMAE=0.998意味着500步累积误差平均为0.998个标准差。这两个数字不可直接比较——H=500的NMAE天然远大于H=1，因为误差随步数累积。这是ODE rollout评估的固有特性，但报告中未明确说明。

**问题3: 归一化因子可能不具代表性（低）**

state_std从训练数据计算（`data_loader.py`第69行：`state_std = np.std(train_states, axis=0)`）。如果测试集分布与训练集有偏差（例如测试集包含更多极端操控场景），归一化因子可能不具代表性。但由于按episode划分（见第3节），此风险较低。

### 1.3 NMAE计算的正确性

NMAE的数学计算本身**是正确的**。mean(|error| / std)是标准的归一化MAE。axis=0先对时间步取均值，然后对状态维度取均值，逻辑一致。代码实现无bug。

---

## 2. 生存率指标分析

### 2.1 生存率定义

```python
# 第107行
survival_list.append(survived and n_valid >= n - 1)
```

一段被认为"存活"需要同时满足：
1. 所有步骤未产生NaN/inf
2. 所有步骤未超出物理限制
3. 有效预测步数 >= horizon - 1

### 2.2 发现的问题

**问题4: 物理限制过于宽松（严重）**

```python
# config_v9.py 第19-23行
PHYSICAL_LIMITS = {
    'e_y': 5.0,       # 横向偏差5米 = 半个车道宽
    'e_psi': np.pi,   # 航向角偏差180度 = 完全反向
    'v': 5.0,         # 速度5 m/s = 18 km/h
    'theta': np.pi,   # 倾斜角180度 = 完全倒立
    'theta_dot': 10.0,# 倾斜角速度10 rad/s = 约573 deg/s
    'delta': np.pi/2, # 转向角90度 = 物理不可能
    'delta_dot': 10.0,# 转向角速度10 rad/s
}
```

**关键评估**：
- `e_y = 5.0m`：自行车横向偏离5米，已经完全驶出车道。正常骑行偏差应 < 0.5m
- `e_psi = pi`：航向偏差180度意味着自行车正在倒退。正常骑行偏差应 < 10度（约0.17 rad）
- `theta = pi`：倾斜角180度意味着自行车倒立。正常倾斜角应 < 15度（约0.26 rad）
- `delta = pi/2`：90度转向角在物理上不可能实现

这些限制仅捕获数值发散，而非物理合理的行为。**"100%生存"不等于"100%物理合理"**。

**问题5: 生存率是二值指标，信息量有限（中等）**

一个段要么100%存活，要么0%存活（对于单段）。虽然跨5段计算平均值，但这个指标无法区分：
- 模型恰好在限制边缘运行（接近不稳定）
- 模型远在限制内运行（真正稳定）

建议增加"安全裕度"指标：存活段中，状态距物理限制的最小距离。

**问题6: 生存率在早期崩溃时的定义有边界效应（低）**

如果模型在step 0就崩溃（n_valid=0），第104行会将NMAE记为NaN。但第107行仍然将其计为"未存活"。这本身是正确的，但如果所有5段都在step 0崩溃，`valid_nmae`为空，`nmae_mean`为NaN。代码处理了这种情况（第115行的空列表检查），但可能导致下游分析出现NaN传播。

### 2.3 生存率计算的正确性

生存率的数学计算**是正确的**。`np.mean(survived_flags)`确实是存活比例。定义清晰，实现无bug。

---

## 3. 数据泄漏分析

### 3.1 训练/测试划分

```python
# V8 data_loader.py 第47-59行
# 按episode划分，不按时间步划分
rng = np.random.RandomState(cfg.seed)  # seed=42
n_train = int(actual_n_episodes * cfg.train_ratio)  # 0.75
perm = rng.permutation(actual_n_episodes)
train_eps = set(perm[:n_train])  # 75% episodes -> ~44 episodes
test_eps = set(perm[n_train:])   # 25% episodes -> ~15 episodes
```

**无时间步级别泄漏**：训练集和测试集按episode划分，同一episode的所有时间步要么全在训练集，要么全在测试集。这确保了没有时间步级别的数据泄漏。

### 3.2 发现的问题

**问题7: 相邻episode可能存在信息泄漏（中等）**

数据集有59个episode（`generate_report.py`第54行）。如果episode在时间上是连续的（例如episode 1的结束状态是episode 2的初始状态），那么训练集中episode N的最后一步与测试集中episode N+1的第一步可能存在强相关性。

但由于episode划分是随机的（permutation），且有59个episode（~15个测试episode），相邻episode出现在训练/测试边界的概率约为 2/59 * 15 ≈ 51%。这意味着大约一半的测试episode可能与某个训练episode在时间上相邻。

**影响评估**：对于Neural ODE模型，初始条件是 rollout 的起点（`s0 = seg['states'][0]`），模型从测试episode的真实初始条件开始 rollout。即使初始条件来自测试集，模型在 rollout 过程中的误差累积仍然反映了泛化能力。因此，此问题对评估有效性的影响**有限**。

**问题8: test segments从test_mask正确采样（无问题）**

```python
# data_loader.py 第110-138行
test_indices = np.where(test_mask)[0]  # 只取test episodes的索引
# 从test episodes的连续片段中采样segments
```

测试segments确实来自测试episodes，**无数据泄漏**。

### 3.3 数据泄漏结论

**未发现严重的数据泄漏**。训练/测试按episode划分是正确的方法。state_std从训练数据计算也是正确的。

---

## 4. 测试样本量的统计显著性

### 4.1 当前配置

- N_SEGMENTS = 5（每个配置评估5段）
- 5段的NMAE均值作为最终指标
- 无置信区间报告

### 4.2 统计分析

**NMAE均值的置信区间**：

假设NMAE在5段上近似正态分布，95%置信区间为：
- CI = mean ± t(0.025, 4) * std / sqrt(5)
- t(0.025, 4) = 2.776
- CI半宽 = 2.776 * std / 2.236 = 1.241 * std

**定量示例**：如果v9_baseline在H=500的5段NMAE标准差为0.1，则95% CI半宽为0.124，即NMAE = 0.998 ± 0.124。这意味着NMAE的真实值可能在0.874到1.122之间。

**生存率的置信区间**：

对于二值指标（存活/未存活），5段的95% CI：
- 如果5/5存活（100%）：Wilson CI下界 = 1 - (1-0.05)^(1/5) ≈ 45%
- 即使5段全部存活，真实存活率的95% CI下界也仅为45%

这意味着**100%生存率在5段样本下不能可靠地证明模型真正稳定**。

### 4.3 问题9: 样本量不足（严重）

5段评估**不足以提供统计显著性**：
- NMAE的95% CI半宽约为1.24倍标准差
- 生存率的95% CI可能跨越45%-100%
- 无法区分配置之间的细微差异（如v9_baseline的0.998 vs v9_fixed_seq_contract的0.958）

**建议**：至少需要20段评估，或进行多次随机种子评估（如V8的StatisticalConfig建议10个eval seeds）。

---

## 5. 评估Horizon分析

### 5.1 当前配置

```python
# run_v9_fixed_rerun.py
HORIZONS = [1, 10, 50, 100, 200, 500]
```

### 5.2 评估

**问题10: Horizon选择基本合理，但缺少关键区间（低）**

- H=1: 单步精度（基本OK）
- H=10: 短期（0.33秒），OK
- H=50: 中期（1.67秒），OK
- H=100-500: 长期（3.3-16.7秒），覆盖了误差累积的关键区间

**缺失**：H=5和H=20在config_v9.py中定义但未在fixed_rerun中使用。H=1000在config中定义但也未使用。对于自行车路径跟踪，16.7秒（H=500）的horizon覆盖了一个典型的弯道通过时间，是合理的。

### 5.3 Horizon评估结论

Horizon选择**基本合理**，覆盖了从单步到长期的关键区间。

---

## 6. V9 Baseline "100%生存率"评估

### 6.1 数据

| Config | H=500 NMAE | H=500 Surv |
|--------|-----------|------------|
| v9_baseline | 0.998 | 100% |
| v9_fixed_seq | 0.476 | 0% |
| v9_fixed_seq_contract | 0.958 | 100% |

### 6.2 分析

**问题11: v9_baseline的"100%生存"可能具有误导性（严重）**

1. **物理限制过于宽松**（见问题4）：模型可能产生物理上不合理的状态（如e_y=4.9m, theta=2.5rad），但仍被计为"存活"
2. **5段样本量不足**（见问题9）：即使5段全部存活，95% CI下界仅约45%
3. **NMAE=0.998说明误差累积严重**：在H=500时NMAE接近1个标准差，意味着预测状态与真实状态有显著偏差，即使未超出宽松的物理限制

**v9_fixed_seq的"0%生存"同样需要谨慎解读**：可能是少数段在少数步骤超出限制，但由于生存率是二值的（一段内任何步骤超限即判定为未存活），0%可能实际上只差1-2个步骤。

### 6.3 对比分析

v9_fixed_seq的NMAE显著优于v9_baseline（H=500: 0.476 vs 0.998），但生存率为0%。这说明v9_fixed_seq的预测更准确，但偶尔会超出宽松的物理限制。v9_fixed_seq_contract通过contractive loss在NMAE（0.958）和生存率（100%）之间取得了平衡。

**结论：v9_baseline的"100%生存率"在宽松限制和小样本下意义有限，不应作为主要评估指标。NMAE更能反映模型的真实预测能力。**

---

## 7. 验收标准合理性

### 7.1 隐含的验收标准

从baseline结果和实验设计推断，验收标准为：
- H=500生存率 >= 100%
- NMAE在各horizon上尽可能低

### 7.2 问题12: 验收标准需要多维度定义（中等）

仅以生存率和NMAE作为验收标准不够全面。建议增加：
- **per-state NMAE**：识别哪些状态维度预测最差
- **安全裕度**：存活段中状态距物理限制的最小距离
- **误差增长速率**：NMAE随horizon的增长率（理想情况下应亚线性增长）
- **统计置信度**：报告NMAE和生存率的置信区间

### 7.3 问题13: 混合实验的评估标准不一致（低）

`run_hybrid_experiments.py`使用了略有不同的评估方法（第162-219行），虽然逻辑相同，但存在代码重复。建议统一使用`evaluation_v9.py`中的函数。

---

## 8. 总结

### 发现汇总

| 编号 | 问题 | 严重度 | 类别 |
|------|------|--------|------|
| 1 | 失败段NMAE计算窗口不同导致偏差 | 严重 | NMAE |
| 4 | 物理限制过于宽松，100%生存不等于物理合理 | 严重 | 生存率 |
| 9 | 5段样本量不足以提供统计显著性 | 严重 | 统计 |
| 11 | v9_baseline的100%生存率在宽松限制和小样本下意义有限 | 严重 | 综合 |
| 2 | NMAE是累积误差指标，不同horizon不可直接比较 | 中等 | NMAE |
| 5 | 生存率是二值指标，信息量有限 | 中等 | 生存率 |
| 7 | 相邻episode可能信息泄漏（影响有限） | 中等 | 数据 |
| 12 | 验收标准需要多维度定义 | 中等 | 方法论 |
| 3 | 归一化因子可能不具代表性 | 低 | NMAE |
| 6 | 生存率边界效应 | 低 | 生存率 |
| 10 | Horizon选择基本合理 | 低 | 评估设计 |
| 13 | 混合实验评估标准不一致 | 低 | 代码质量 |

### 整体评估

V9的评估方法论在**计算正确性上无bug**，NMAE和生存率的数学实现是正确的。但方法论存在三个严重问题：

1. **NMAE的跨horizon比较无意义**，且失败段的短窗口NMAE会拉低整体均值
2. **物理限制过于宽松**，生存率仅反映数值稳定性，不反映物理合理性
3. **5段样本量不足**，置信区间过宽，无法可靠区分配置差异

建议在后续实验中：
1. 报告per-state NMAE以识别薄弱维度
2. 收紧物理限制至工程合理范围（e_y < 1m, e_psi < 0.3 rad, theta < 0.3 rad）
3. 增加评估段数至至少20段
4. 报告置信区间
5. 增加"误差增长速率"指标（NMAE vs horizon的斜率）
