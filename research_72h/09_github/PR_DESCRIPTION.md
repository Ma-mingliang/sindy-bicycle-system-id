# Pull Request: Neural ODE 72h Research

## Title
Neural ODE 72h research: long-horizon dynamics prediction analysis

## Branch
`research/neural-ode-72h-20260628_175824_neural_ode_72h`

## Summary

This PR contains the results of a 72-hour research cycle on bicycle dynamics prediction using Neural ODE.

### Key Findings

1. **Root Cause Identified**: Training-test distribution mismatch
   - Model never learns to recover from its own errors
   - Anchored Shooting proves model CAN predict accurately (H=500 NMAE = 0.1916)

2. **GP 'Fake Stability' Discovered**
   - GP learns 'near-identity mapping' (delta≈0)
   - Long-term stability is fake - doesn't predict changes

3. **v9 'Bug' is Regularization**
   - Original v9's 'buggy' multi-step loss acts as regularization
   - Fixed version overfits training data

### Best Results

| Method | H=50 | H=100 | H=200 | H=500 | Primary |
|--------|------|-------|-------|-------|---------|
| v9 Baseline | 0.4557 | 0.5064 | 0.4737 | 0.5529 | 0.5110 |
| Noise-Augmented (no_noise) | 0.0565 | 0.1624 | 0.4825 | 0.8653 | 0.5034 |
| Constrained (no_constraint) | 0.1236 | 0.3395 | 0.6665 | 0.8045 | 0.6035 |

### Limitations

1. H=200/500 still worse than v9 baseline
2. OOD testing shows 0% survival rate
3. High variance across seeds

### Next Steps

1. Address training-test distribution mismatch with DAgger-like training
2. Improve OOD robustness
3. Continue research for full 144-hour cycle

## Files Changed

- `research_72h/` - All research files
  - `00_context/` - Project understanding
  - `01_baseline/` - V9 baseline reproduction
  - `02_diagnosis/` - Root cause analysis
  - `03_literature/` - Literature review
  - `04_agents/` - Subagent reports
  - `05_candidates/` - All experiment implementations and results
  - `06_experiments/` - Experiment registry
  - `07_models/` - Trained models
  - `08_review/` - Independent audit
  - `09_github/` - GitHub delivery
  - `final/` - Final reports
  - `loops/` - Loop summaries
  - `runtime/` - Runtime state

## Testing

- All experiments run with multiple seeds
- OOD testing completed
- Independent audit completed

## Checklist

- [x] Code follows project style
- [x] Tests pass
- [x] Documentation updated
- [x] No hardcoded secrets
- [x] No console.log statements

## Related Issues

- None

## Screenshots

- N/A

## Additional Notes

This is a research PR containing experimental code and results. The main findings are:

1. The root cause of long-horizon prediction failure is training-test distribution mismatch
2. GP models learn 'fake stability' by predicting near-zero deltas
3. The original v9 'bug' actually acts as regularization

Future work should focus on:
1. DAgger-like training to address distribution mismatch
2. OOD robustness improvements
3. Longer research cycles for more thorough exploration
