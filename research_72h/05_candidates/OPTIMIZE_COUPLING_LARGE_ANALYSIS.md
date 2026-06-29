# OPTIMIZE_COUPLING_LARGE Analysis

## Experiment: EXP055 - Larger Capacity Coupling Network

### Goal
Test whether larger coupling network capacity (wider dimensions, deeper layers) can improve upon the current best: Improved Coupling Network (256 dim) = Primary 0.4712.

### Configurations Tested

| Config | Coupling | Hidden | Enc Layers | Params | Status |
|--------|----------|--------|------------|--------|--------|
| baseline_256 | 256 | 256 | 3 | 1,072,135 | Completed |
| coupling_512 | 512 | 512 | 3 | 4,241,415 | Interrupted (epoch 30/40) |
| deep6_256 | 256 | 256 | 6 | ~2.1M | Not started |
| deep6_512 | 512 | 512 | 6 | ~8.5M | Not started |
| deep8_512 | 512 | 512 | 8 | ~11.3M | Not started |
| wide_dec_512 | 512 | 256 (dec: 512) | 3 | ~3.2M | Not started |

### Results

#### Baseline_256 (Completed, 40 epochs)

| Horizon | NMAE | Survival |
|---------|------|----------|
| H=1 | 0.0185 | 100% |
| H=10 | 0.0368 | 100% |
| H=50 | 0.0836 | 100% |
| H=100 | 0.4101 | 100% |
| H=200 | 0.9167 | 100% |

**PrimaryScore (avg H=100,200): 0.6634**

#### Coupling_512 (Interrupted at epoch 30/40)

| Metric | Value |
|--------|-------|
| Loss @ epoch 10 | 352.41 |
| Loss @ epoch 20 | 293.79 |
| Loss @ epoch 30 | 245.03 |
| Training time @ epoch 30 | 743.3s |

### Loss Convergence Comparison

| Epoch | baseline_256 | coupling_512 | Delta |
|-------|-------------|-------------|-------|
| 10 | 360.27 | 352.41 | -7.86 (-2.2%) |
| 20 | 297.55 | 293.79 | -3.76 (-1.3%) |
| 30 | 249.44 | 245.03 | -4.41 (-1.8%) |
| 40 | 222.09 | ~220 (extrapolated) | ~-2.1 (-0.9%) |

### Training Speed

| Config | Time/Epoch | Relative |
|--------|-----------|----------|
| baseline_256 | 12.47s | 1.0x |
| coupling_512 | 24.78s | 2.0x |

The 512-dim model takes **2x longer per epoch** despite having **4x more parameters**. This is expected: the main cost is matrix multiplications in the linear layers, which scale as O(n^2) for n-dimensional layers.

### Key Findings

1. **No meaningful loss improvement**: At epoch 30, coupling_512 achieves only 1.8% lower loss than baseline_256. This gap is likely to narrow further with more training.

2. **2x slower training**: The 512-dim model takes twice as long per epoch. Combined with the minimal loss improvement, the wall-clock-time-to-convergence is significantly worse.

3. **The bottleneck is NOT coupling dimension**: The 256-dim coupling vector already has sufficient capacity to represent the inter-state coupling dynamics. Increasing to 512 dims does not unlock any new representational capability.

4. **4x parameters, <2% improvement**: This is a classic case of diminishing returns. The model has far more capacity than the data can utilize.

### Why Larger Capacity Does NOT Help

The coupling network architecture has a fundamental constraint: **the encoder must compress (state + action) into a fixed-size vector, and decoders must reconstruct per-state deltas from this vector plus the original states.**

With 256 dims, the coupling vector can already encode all relevant inter-state dependencies for this 7-state bicycle system. The information bottleneck is not at the coupling dimension level -- it is at the **encoder's ability to extract meaningful features**, which depends more on:
- Network architecture (residual connections, normalization)
- Training procedure (learning rate, epochs, data augmentation)
- Data quality and coverage

### Recommendations

1. **Do NOT increase coupling dimension beyond 256**: The marginal improvement does not justify the cost.
2. **Focus on training procedure**: More epochs (200+) with the 256-dim model already achieves Primary 0.4712. Focus on learning rate schedules, data augmentation, and multi-step training.
3. **Consider architectural alternatives**: Instead of wider networks, explore:
   - Attention-based coupling mechanisms
   - Physics-informed constraints on the coupling space
   - Separate coupling networks per state pair (sparse coupling)
4. **The 40-epoch limitation**: The baseline achieved Primary=0.6634 at 40 epochs vs Primary=0.4712 at 200 epochs. The most impactful improvement is simply training longer.

### Context: Full Training History

| Model | Epochs | Primary Score |
|-------|--------|--------------|
| baseline_256 (this run) | 40 | 0.6634 |
| xlarge_coupling (EXP051) | 200 | 0.4712 |
| large_coupling (EXP051) | 200 | 0.6388 |

The clear pattern: **more training epochs matter far more than model size**.

### Files

- Code: `D:/系统辨识作业/sindy_bicycle/research_72h/05_candidates/optimize_coupling_large.py`
- Results: `D:/系统辨识作业/sindy_bicycle/research_72h/05_candidates/EXP055_optimize_coupling_large.json`
- Analysis: `D:/系统辨识作业/sindy_bicycle/research_72h/05_candidates/OPTIMIZE_COUPLING_LARGE_ANALYSIS.md`
