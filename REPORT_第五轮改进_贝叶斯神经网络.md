# 第五轮改进：贝叶斯神经网络（MC Dropout）测试报告

> 测试时间：2026-06-22
> 改进方向：用MC Dropout近似贝叶斯神经网络

---

## 一、改进思路

### 1.1 原始方法

标准NN，无不确定性估计。

### 1.2 改进方法

用MC Dropout近似贝叶斯神经网络：
- 训练时使用Dropout
- 推理时保持Dropout开启，进行多次前向传播
- 用多次预测的均值和方差作为最终预测和不确定性
- 不确定性高时减少残差权重

**核心代码**：
```python
class MCDropoutResidualNet(nn.Module):
    def predict_with_uncertainty(self, s, a=None, n_samples=10):
        self.train()  # 保持Dropout开启
        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(s, a)
                predictions.append(pred.cpu().numpy())
        self.eval()

        predictions = np.array(predictions)
        mean = np.mean(predictions, axis=0)
        std = np.std(predictions, axis=0)
        return mean, std
```

---

## 二、测试结果

### 2.1 NeuralODE + MC Dropout

| 步数 | 纯基线 (rad) | MC Dropout (rad) | 改进倍率 |
|------|-------------|-----------------|----------|
| 1 | 0.00029 | 0.00152 | 5.16x (恶化) |
| 5 | 0.01832 | 0.00622 | 0.34x (改善) |
| 10 | 0.08923 | 0.07171 | 0.80x (改善) |
| 20 | 0.32880 | 0.28985 | 0.88x (改善) |
| 50 | 0.81747 | 0.69496 | 0.85x (改善) |
| 100 | 0.86248 | 0.54788 | 0.64x (改善) |
| 200 | 0.74895 | 0.56045 | 0.75x (改善) |
| 500 | 2.23629 | **1.92000** | 0.86x (改善) |

**不确定性统计**：
- 平均：3.7222
- 中位：0.0000
- 最小：0.0000
- 最大：30.0268

---

## 三、与其他方法对比

| 方法 | 500步MAE (rad) | 对比 |
|------|----------------|------|
| NeuralODE + OOD + DAgger + 0.3 | 0.48 | 基准 |
| NeuralODE + 多步损失 | **0.097** | 第三轮改进（最佳） |
| NeuralODE + 物理约束残差 | 0.284 | 第四轮改进（恶化） |
| NeuralODE + MC Dropout | 1.92 | 第五轮改进（改善14%） |

---

## 四、问题分析

### 4.1 为什么MC Dropout效果一般？

**原因**：
1. **不确定性估计太嘈杂**：平均3.72，最大30.03，方差很大
2. **不确定性高时残差权重被过度减少**：adaptive_scale可能降到0
3. **MC Dropout的不确定性估计不够准确**：10次采样可能不够

### 4.2 具体问题

- 不确定性中位为0，说明大部分时间不确定性很低
- 但最大值30.03，说明偶尔不确定性非常高
- 这导致残差权重不稳定

---

## 五、结论与建议

### 5.1 结论

1. **MC Dropout效果一般**（500步改善14%）
2. **不确定性估计太嘈杂**，导致残差权重不稳定
3. **多步损失训练**（第三轮改进）效果更好

### 5.2 后续改进方向

1. **增加采样次数**：从10次增加到50次或100次
2. **调整不确定性阈值**：当前阈值0.5可能太低
3. **尝试其他不确定性估计方法**：如Deep Ensemble

---

## 六、相关代码

| 文件 | 说明 |
|------|------|
| `test_bayesian_nn.py` | MC Dropout测试代码 |

---

## 七、运行命令

```bash
# 运行MC Dropout测试
"E:/Anaconda/envs/DL/python.exe" "D:/系统辨识作业/sindy_bicycle/test_bayesian_nn.py"
```

---

*报告生成时间：2026-06-22*
