# C: V8 数据加载器深度审计

## 数据流图

```
stage2_dataset_150k.npz
    │
    ▼
[np.load] ──► obs(N,8), next_obs(N,8), action(N,), done(N,), n_episodes
    │
    ▼
[8D → 7D 映射] ──► 丢弃索引 5 (kappa) via IDX_7D_FROM_8D=[0,1,2,3,4,6,7]
    │                states_7d: (N,7), next_states_7d: (N,7)
    ▼
[delta 计算] ──► deltas_7d = next_states_7d - states_7d
    │
    ▼
[Episode 边界检测] ──► 扫描 done 标志，构建 ep_starts/ep_ends
    │
    ▼
[训练/测试划分] ──► 75/25 按 episode（seed=42），布尔掩码
    │
    ▼
[Std 计算] ──► state_std, action_std, delta_std（仅从训练数据）
    │
    ▼
[get_test_segments] ──► 查找连续测试 run，采样段
    │
    ▼
V9 通过 data_loader_v9.py 消费（薄委托）
```

## 1. 数据加载 (data_loader.py:8-98)

### 8D→7D 映射 (lines 26-29)

`IDX_7D_FROM_8D = [0, 1, 2, 3, 4, 6, 7]` 丢弃索引 5（kappa）。配置注释称 kappa "恒为零"。状态转换为 float64（line 27），对下游 std 计算和归一化的数值精度正确。

### Episode 检测 (lines 32-44)

算法线性扫描 `done` 数组。当 `done[i]` 为 True 时，记录 episode 边界 `[start, i+1)` 并将 start 推进到 `i+1`。

**Edge case（lines 41-43）**: 如果最终样本的 `done` 不为 True，追加尾部 episode `[start, len(done))`。数据集确认有 59 个 episodes。

**Episode 计数不一致（line 24 vs 45）**: 代码从文件加载 `n_episodes`（line 24）但随后计算 `actual_n_episodes = len(ep_starts)`（line 45）。返回的是文件中的 `n_episodes` 而非实际计数。

### Delta 计算 (line 29)

`deltas_7d = next_states_7d - states_7d`。在 episode 边界处，`next_obs[i]` 和 `obs[i]` 来自同一 episode 内的连续时间步（由数据集构造保证），因此 delta 是正确的。

## 2. 训练/测试划分 (lines 47-66)

按 episode 划分（非按样本），这对时间序列数据正确——防止同一 episode 的尾部在训练中出现而头部在测试中出现的信息泄露。

`train_ratio = 0.75` → 约 44 个训练 / 15 个测试 episode（共 59 个）。使用 `RandomState(seed=42)` 的置换保证可复现。

## 3. Std 计算 (lines 68-77)

```python
state_std = np.std(train_states, axis=0)     # shape (7,)
action_std = float(np.std(train_actions))     # 标量
delta_std = np.std(train_deltas, axis=0)      # shape (7,)
```

**[CONFIRMED]** 仅从训练数据计算。零 std 保护（<1e-10 替换为 1.0）防止除零但静默掩盖常量特征。

## 4. get_test_segments (lines 101-140)

### 关键发现: 段长度 vs episode 长度

V9 请求 `segment_length=1100`（config_v9.py:15）。每个测试段必须来自单个测试 episode 内至少 1101 个连续样本。短于 1101 样本的 episode 被静默排除。

平均 episode 长度约 2542 样本（150k/59），但范围可能从几百到 5000+。

### 段提取 (lines 132-137)

段从 `all_states` 和 `all_actions` 提取（非 `test_states`/`test_actions`），`start_idx` 是全局数据集索引。

## 5. V9 委托 (data_loader_v9.py)

```python
def load_7d_data(data_path='...stage2_dataset_150k.npz', seed=42):
    cfg = DataConfig(data_path=data_path, seed=seed)
    return _v8_load(cfg)
```

**[CONFIRMED]** 委托过程中无数据变换丢失。默认 segment_length 从 V8 的 200 改为 V9 的 1100。

## 6. 潜在问题汇总

| # | 严重度 | 位置 | 问题 |
|---|--------|------|------|
| 1 | HIGH | data_loader.py:89 | 返回文件中的 `n_episodes` 而非 `actual_n_episodes` |
| 2 | HIGH | data_loader_v9.py:15 | SEGMENT_LENGTH=1100 要求测试 episode 有 1101+ 连续样本 |
| 3 | MEDIUM | data_loader.py:8-98 | 无 NaN/Inf 验证 |
| 4 | MEDIUM | data_loader.py:8-98 | 无物理范围验证 |
| 5 | MEDIUM | data_loader_v9.py:3 | 硬编码绝对路径脆弱 |
| 6 | LOW | data_loader.py:74-77 | 零 std 保护静默替换常量特征 |
| 7 | LOW | neural_ode_v9.py:82-83 | 重复的 std 保护（data_loader 已应用） |

## 建议

1. 加载时添加 NaN/Inf 检查
2. 返回 `actual_n_episodes` 而非文件中的值
3. 评估前验证段可用性
4. 用相对导入替换硬编码 sys.path
5. 记录 kappa 假设的运行时断言
