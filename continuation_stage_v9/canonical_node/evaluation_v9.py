"""V9 Evaluation: multi-step rollout, survival, NMAE."""
import numpy as np
from .config_v9 import STATE_DIM, STATE_NAMES_7D, STATE_LIMIT, PHYSICAL_LIMITS


def compute_nmae(predicted, actual, state_std):
    """Compute Normalized MAE per state and overall."""
    errors = np.abs(predicted - actual)
    # Normalize by state std
    nmae_per_state = np.mean(errors / state_std, axis=0)
    nmae_overall = np.mean(nmae_per_state)
    return {
        'overall': float(nmae_overall),
        **{name: float(nmae_per_state[i]) for i, name in enumerate(STATE_NAMES_7D)}
    }


def check_survival(state, mode='physical'):
    """Check if state survives (numerical or physical)."""
    if mode == 'numerical':
        return not (np.any(np.isnan(state)) or np.any(np.isinf(state))
                    or np.any(np.abs(state) > STATE_LIMIT))
    elif mode == 'physical':
        for i, name in enumerate(STATE_NAMES_7D):
            if name in PHYSICAL_LIMITS:
                if abs(state[i]) > PHYSICAL_LIMITS[name]:
                    return False
        return not (np.any(np.isnan(state)) or np.any(np.isinf(state)))
    return True


def compute_survival(survived_flags):
    """Compute survival rate from boolean array."""
    if len(survived_flags) == 0:
        return 0.0
    return float(np.mean(survived_flags))


def multi_step_evaluate(model, segments, state_std, horizons,
                        survival_mode='physical'):
    """Evaluate model on multiple segments at multiple horizons.

    Args:
        model: object with predict(s, tau) method
        segments: list of dicts with 'states' and 'actions'
        state_std: state standard deviations
        horizons: list of horizons to evaluate
        survival_mode: 'numerical', 'physical', or 'task'

    Returns:
        dict: {horizon: {nmae, survival_rate, per_state_nmae, per_step_errors}}
    """
    results = {}

    for h in horizons:
        nmae_list = []
        survival_list = []
        per_state_nmae = {name: [] for name in STATE_NAMES_7D}
        per_step_errors = []  # List of arrays, one per segment

        for seg in segments:
            s0 = seg['states'][0].copy()
            actions_seg = seg['actions']
            real_states = seg['states']

            n = min(h, len(actions_seg))
            predicted = [s0.copy()]
            s_cur = s0.copy()
            survived = True
            step_errors = []

            for step in range(n):
                try:
                    s_next = model.predict(s_cur, actions_seg[step])

                    if np.any(np.isnan(s_next)) or np.any(np.isinf(s_next)):
                        survived = False
                        break

                    if not check_survival(s_next, mode=survival_mode):
                        survived = False
                        break

                    # Compute per-step error
                    if step + 1 < len(real_states):
                        step_err = np.abs(s_next - real_states[step + 1]) / state_std
                        step_errors.append(step_err)

                    predicted.append(s_next.copy())
                    s_cur = s_next
                except Exception:
                    survived = False
                    break

            predicted = np.array(predicted)
            real = real_states[:len(predicted)]
            n_valid = min(len(predicted), len(real)) - 1

            if n_valid > 0:
                nmae = compute_nmae(predicted[1:n_valid+1], real[1:n_valid+1], state_std)
                nmae_list.append(nmae['overall'])
                for name in STATE_NAMES_7D:
                    per_state_nmae[name].append(nmae[name])
            else:
                nmae_list.append(float('nan'))

            survival_list.append(survived and n_valid >= n - 1)

            if step_errors:
                per_step_errors.append(np.array(step_errors))

        # Aggregate
        valid_nmae = [x for x in nmae_list if not np.isnan(x)]
        results[h] = {
            'nmae_mean': float(np.nanmean(valid_nmae)) if valid_nmae else float('nan'),
            'nmae_std': float(np.nanstd(valid_nmae)) if valid_nmae else float('nan'),
            'survival_rate': compute_survival(survival_list),
            'n_valid': len(valid_nmae),
            'per_state_nmae': {
                name: {
                    'mean': float(np.nanmean(per_state_nmae[name])) if per_state_nmae[name] else float('nan'),
                    'std': float(np.nanstd(per_state_nmae[name])) if per_state_nmae[name] else float('nan'),
                }
                for name in STATE_NAMES_7D
            },
            'per_step_errors': per_step_errors,
        }

    return results
