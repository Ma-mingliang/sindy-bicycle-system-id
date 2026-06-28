# Expert Challenges

## Challenge 1: "Contractivity Is the Wrong Approach"

**Expert argument**: Contractivity forces the Jacobian eigenvalues to be negative, which means ALL perturbations decay. This is overly restrictive — some state dimensions may naturally grow (e.g., theta accumulates over time). Forcing contractivity on all dimensions causes the model to under-predict natural dynamics.

**Supporting evidence**: H=50 NMAE worsens from 0.385 (no contractivity) to 0.647 (with contractivity). The model is learning "predict less change" rather than "predict accurate change that stays within limits."

## Challenge 2: "Inference-Time Clipping Is Fundamentally Flawed"

**Expert argument**: Hard clipping creates a discontinuity in the dynamics. The model was trained on smooth trajectories, but at inference time, clipping introduces jumps that the model has never seen. This causes the model to make increasingly wrong predictions after each clip, leading to error cascades.

**Supporting evidence**: clip_seq has H=500 NMAE of 1.505 — 3x worse than v9_fixed_seq without clipping.

## Challenge 3: "Safety Blending Is a Heuristic, Not a Solution"

**Expert argument**: The safety-blended model switches between two models based on a hand-tuned threshold (50% of physical limits). This is fragile:
- The threshold is arbitrary — why 50% and not 60% or 40%?
- The switch creates a discontinuity in the prediction function
- It doesn't address the root cause (model accuracy at long horizons)

**Counter-evidence**: safety_blend_0.5 actually achieves the best composite score (0.640) among all 100% survival configs. The heuristic works even if it's not principled.

## Challenge 4: "V9 Baseline's Results Are Being Treated as Ground Truth"

**Expert argument**: The acceptance criteria compare against V9 baseline, but V9 baseline may itself be a poor model. Its H=500 NMAE of 0.998 means the average prediction error is ~1 standard deviation — essentially random. Comparing against a bad baseline sets a low bar.

**Implication**: If we accept that H=500 NMAE ≈ 1.0 is the floor for this system with 100% survival, then our best solution (0.946) actually exceeds expectations.

## Challenge 5: "The Experiment Uses Only 5 Test Segments"

**Expert argument**: With only 5 test segments, the results have high variance. A single lucky or unlucky segment can swing the average by ±20%. The confidence in any conclusion is limited by the small sample size.

**Mitigation**: All configs use the same 5 segments, so relative comparisons are valid. Absolute performance claims need more segments.
