# V9 基线快照

**创建时间**: 2026-06-28 18:00
**RUN_ID**: 20260628_175824_neural_ode_72h
**Git Commit**: e361110c1baeb29a6cbb64db5c13f625405b41fc
**Branch**: research/neural-ode-72h-20260628_175824_neural_ode_72h

---

## 1. 模型配置

### 最佳 V9 配置
```python
config = {
    'hidden': 64,
    'depth': 3,
    'activation': 'tanh',
    'lr': 1e-3,
    'weight_decay': 0,
    'n_epochs': 200,
    'batch_size': 256,
    'dt': 1/30,
    'rollout_curriculum': '1,5,10,20',
    'lambda_multi': 0.3,
    'lambda_consistency': 0.1,
    'lambda_jacobian': 0.01,
    'seed': 43
}
```

### 模型架构
- 输入: 8 维 [state(7), action(1)]
- 隐藏层: 3 × 64 (Tanh)
- 输出: 7 维 (导数)
- 参数量: 9,351
- 积分: Euler (dt=1/30)

---

## 2. 基线性能

### 多 Horizon NMAE
| Horizon | NMAE | 物理存活率 |
|---------|------|-----------|
| H=1 | 0.0051 | 100% |
| H=10 | 0.0628 | 100% |
| H=50 | 0.4557 | 100% |
| H=100 | 0.5064 | 100% |
| H=200 | 0.4737 | 100% |
| H=500 | 0.5529 | 100% |
| H=1000 | 0.6443 | 80% |

### 主指标
```
PrimaryLongHorizonScore = mean(NMAE_H100, NMAE_H200, NMAE_H500) = 0.5110
```

### 各状态误差 (H=100)
| 状态 | NMAE |
|------|------|
| e_y | 0.7214 |
| e_psi | 0.8883 |
| v | 0.6041 |
| theta | 0.2536 |
| theta_dot | 0.3809 |
| delta | 0.2855 |
| delta_dot | 0.4113 |

---

## 3. 多 Seed 复现

| Seed | H=10 | H=50 | H=100 | H=200 | H=500 |
|------|------|------|-------|-------|-------|
| 42 | 0.0606 | 0.5901 | 0.6713 | 0.6669 | 0.8559 |
| 43 | 0.0534 | 0.5354 | 0.5777 | 0.5467 | 0.7145 |
| 44 | 0.0700 | 0.6011 | 0.6685 | 0.6804 | 0.7802 |

**跨 Seed 标准差**: < 0.1 (复现确认 PASS)

---

## 4. 数据集

- **文件**: `data/stage2_dataset_150k.npz`
- **规模**: 150,000 样本，148 个 episode
- **状态维度**: 8D → 7D (去除 k)
- **动作维度**: 1D
- **频率**: 30Hz (dt=1/30)
- **划分**: 按 episode 75/25

---

## 5. 检查点

- **路径**: `continuation_stage_v9/checkpoints/BEST_NEURAL_ODE_V9.pt`
- **大小**: 48,717 bytes
- **格式**: PyTorch checkpoint
- **兼容性**: 需要 numpy 版本兼容处理

---

## 6. 评估协议

### 评估脚本
- `continuation_stage_v9/canonical_node/evaluation_v9.py`
- `continuation_stage_v9/run_v9_experiments.py`

### 评估参数
- 评估 segments: 5
- Segment 长度: 1100 步
- 存活模式: physical
- 存活阈值: 各状态物理限制

---

## 7. 环境

- **Python**: 3.9.23
- **PyTorch**: 2.8.0+cu128
- **NumPy**: 1.23.5
- **CUDA**: 12.8
- **GPU**: NVIDIA GeForce RTX 4060 Ti
