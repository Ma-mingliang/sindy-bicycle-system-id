# Counterfactual Analysis

## What If We Had Not Fixed the Shuffle Bug?

**V9 baseline** uses `shuffle=True` on individual timesteps. This means:
- Multi-step loss samples random timesteps, not contiguous trajectories
- The model cannot learn temporal dynamics from multi-step loss
- It essentially only learns single-step prediction

**Counterfactual**: If we kept shuffle=True but added contractivity:
- The model would still have broken multi-step loss
- Contractivity would only constrain single-step predictions
- Result: Likely similar to V9 baseline — conservative predictions with 100% survival

**Conclusion**: The shuffle fix is the necessary first step. Without it, no other fix matters because the multi-step training signal is fundamentally wrong.

## What If We Used Longer Training Curriculum?

Current training: max H=20 via curriculum '1,5,10,20'
Physical limits are exceeded at H=50+

**Counterfactual**: If we trained with curriculum '1,5,10,20,50,100':
- The model would see longer rollouts during training
- Survival-aware loss could actually penalize violations
- But: longer rollouts amplify any model error, making training unstable

**Evidence**: V12Config default curriculum is '1,5,10,20,50,100' with 400 epochs. Our experiments used '1,5,10,20' with 200 epochs to match V9 hyperparameters. The extended curriculum may help but requires more training time.

## What If V9's High NMAE at H=500 Is Actually Good?

V9 baseline: H=500 NMAE = 0.998

**Counterfactual**: What if 0.998 is the best achievable NMAE at H=500 for this system?
- Physical system may be inherently chaotic at 500 steps (16.7 seconds)
- Any model will accumulate ~1 std of error per state dimension
- The "improvement" from V12 (0.476) may be measuring something different

**Evidence**: V12 sequential's H=500 NMAE of 0.476 is 52% better than V9. But V12 has 0% survival. If we enforce survival (via contractivity or clipping), NMAE rises to ~0.95-1.5. This suggests the accuracy improvement is fragile — it only holds when the model is allowed to diverge.

**Conclusion**: The achievable NMAE with 100% survival at H=500 may be around 0.95, not 0.48. V9's 0.998 is close to this floor.

## What If We Changed the Physical Limits?

Current limits: e_y=5, e_psi=pi, v=5, delta=pi/2, delta_dot=10

**Counterfactual**: If we relaxed limits by 2x:
- More trajectories would survive
- V12 sequential might achieve higher survival without contractivity
- But: relaxed limits may not reflect actual physical constraints

**Verdict**: Not recommended — limits should reflect physical reality, not model capabilities.
