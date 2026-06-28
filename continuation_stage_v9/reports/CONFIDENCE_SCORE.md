# Confidence Score

## Scoring Dimensions

### 1. Code Evidence (20/20)
- V9 multi-step bug confirmed by code inspection (lines 131-138 of neural_ode_v9.py)
- Shuffle=True confirmed (line 96)
- V12 fix confirmed (sequential segments, correct integration)
- All code paths traced and verified
- **Score: 20/20**

### 2. Log Evidence (18/20)
- Training logs show consistent convergence
- Survival loss = 0.000000 throughout (confirmed ineffective)
- Contractivity loss decreases during training (confirmed learning)
- Minor: validation errors not logged for all experiments
- **Score: 18/20**

### 3. Cross-Validation (16/20)
- Single-variable controls executed (shuffle, contractivity, lambda, clipping)
- Results consistent across experiments
- Multi-seed not performed (single seed only)
- 5 test segments is small sample size
- **Score: 16/20**

### 4. Experiment Verification (17/20)
- 8 unique configurations tested
- 3 experiment sets with incremental saves
- Results reproducible (deterministic with seed=43)
- Extended curriculum (400 epochs) not tested
- **Score: 17/20**

### 5. Regression Control (19/20)
- V9 baseline unchanged and verified
- No existing code modified (only additions)
- Checkpoints preserved
- All new files documented
- **Score: 19/20**

## Overall Confidence

| Dimension | Score | Weight | Weighted |
|-----------|-------|--------|----------|
| Code Evidence | 20/20 | 25% | 5.00 |
| Log Evidence | 18/20 | 20% | 3.60 |
| Cross-Validation | 16/20 | 20% | 3.20 |
| Experiment Verification | 17/20 | 20% | 3.40 |
| Regression Control | 19/20 | 15% | 2.85 |
| **Total** | | | **18.05/20** |

**Confidence: 90.25/100**

## Gap to 95% Confidence

To reach 95%, need:
1. Multi-seed validation (3+ seeds) — adds ~2 points to Cross-Validation
2. Extended curriculum experiments (400 epochs) — adds ~2 points to Experiment Verification
3. More test segments (10+) — adds ~1 point to Cross-Validation

## Confidence Assessment

**90.25% confidence** that:
- The shuffle bug is the primary cause of V9's training issues
- Contractivity is the only viable training-time mechanism for survival
- Safety-blended inference achieves the best NMAE-survival tradeoff
- The accuracy-stability tradeoff is fundamental but partially breakable

**NOT confident** that:
- All acceptance criteria can be met simultaneously
- The safety-blended approach is optimal (may be improvable with threshold tuning)
- V9 baseline's H=500 NMAE is a meaningful benchmark
