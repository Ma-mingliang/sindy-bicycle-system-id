# 数据来源与哈希审计报告

> **生成时间**: 2026-06-24
> **目的**: 验证所有数据文件的完整性和一致性

---

## 1. 文件哈希清单

| 文件 | SHA-256 前缀 | 大小 (bytes) | 数组数 |
|------|-------------|-------------|--------|
| meijaard_openloop_data.npz | `cb2a0b04c260782b` | 8,967,432 | 4 |
| meijaard_openloop_data_v35.npz | `e15415f311a399ad` | 2,123,662 | 4 |
| meijaard_openloop_data_v5.npz | `185af07102dca8ba` | 8,162,040 | 4 |
| meijaard_sindy.npz | `bd3b6c1f73698f86` | 5,192 | 3 |
| meijaard_sindy_v35.npz | `1033e722d3f1767d` | 5,192 | 3 |
| meijaard_sindy_v5.npz | `867e7184eae35602` | 5,192 | 3 |
| sindy_full_model.npz | `899caf26b9402beb` | 1,750 | 2 |
| bicycle_data.npz | `e324d5f63f2f2abc` | 68,648 | 5 |
| bicycle_data_improved.npz | `d4990de3efed05f1` | 284,298 | 5 |
| sindy_model.npz | `35ce7967f48eb945` | 9,612 | 3 |

---

## 2. 配置文件校验

配置文件 `configs/reproducible_world_model_evaluation.yaml` 中声明的 SHA-256 前缀：

| 文件 | 声明前缀 | 实际前缀 | 状态 |
|------|---------|---------|------|
| meijaard_openloop_data_v35.npz | `e15415f311a399ad` | `e15415f311a399ad` | PASS |
| meijaard_sindy_v35.npz | `1033e722d3f1767d` | `1033e722d3f1767d` | PASS |
| sindy_full_model.npz | `899caf26b9402beb` | `899caf26b9402beb` | PASS |

---

## 3. 关键发现

1. **所有声明的 SHA-256 前缀匹配** — 数据文件未被篡改
2. **meijaard_openloop_data_v35.npz**: 29,478 样本, 4D 状态, 无 NaN/Inf
3. **meijaard_sindy_v35.npz**: 21×4 SINDy 系数矩阵
4. **sindy_full_model.npz**: 21×4 SINDy 系数矩阵
5. **10个数据文件全部存在**

---

## 4. 完整 SHA-256 值

```
meijaard_openloop_data.npz:     cb2a0b04c260782b...
meijaard_openloop_data_v35.npz: e15415f311a399ad...
meijaard_openloop_data_v5.npz:  185af07102dca8ba...
meijaard_sindy.npz:             bd3b6c1f73698f86...
meijaard_sindy_v35.npz:         1033e722d3f1767d...
meijaard_sindy_v5.npz:          867e7184eae35602...
sindy_full_model.npz:           899caf26b9402beb...
bicycle_data.npz:               e324d5f63f2f2abc...
bicycle_data_improved.npz:      d4990de3efed05f1...
sindy_model.npz:                35ce7967f48eb945...
```

完整哈希值见 `reports/data_provenance_hash_audit.json`。

---

*文档生成时间: 2026-06-24*
*审计工具: Claude Code*
