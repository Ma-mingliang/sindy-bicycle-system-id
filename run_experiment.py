"""
Complete experiment: improved data collection + SINDy identification + evaluation.

This script runs the full pipeline with improved parameters:
- More data with better coverage
- Lower threshold for more complete equations
- Better evaluation metrics
"""

import sys
import os
import numpy as np
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_collector import BicycleDataCollector, normalize_state, denormalize_state
from sindy_identification import (
    build_feature_matrix, build_polynomial_library, stlsq,
    evaluate_model, print_identified_equations, diagnose_errors,
    STATE_NAMES, FEATURE_NAMES
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def run_improved_experiment():
    print("=" * 70)
    print("  Improved SINDy Bicycle Experiment")
    print("=" * 70)

    # ============================================================
    # Step 1: Collect more data with better coverage
    # ============================================================
    print("\n[1/6] Collecting data with improved coverage...")
    dt = 1/30

    # Use multiple seeds for more diverse data
    all_states, all_actions, all_next_states = [], [], []
    for seed in [42, 123, 456, 789, 2024]:
        collector = BicycleDataCollector(dt=dt, seed=seed)
        dataset = collector.collect_dataset(n_episodes_per_scenario=50, t_episode=12.0)
        all_states.append(dataset['states'])
        all_actions.append(dataset['actions'])
        all_next_states.append(dataset['next_states'])

    states = np.vstack(all_states)
    actions = np.vstack(all_actions)
    next_states = np.vstack(all_next_states)

    print(f"\nTotal samples: {len(states)}")

    # Filter extreme values
    valid = np.all(np.abs(states) < 20, axis=1) & np.all(np.abs(next_states) < 20, axis=1)
    valid &= np.abs(states[:, 3]) < 0.8  # |theta| < 0.8 rad
    states = states[valid]
    actions = actions[valid]
    next_states = next_states[valid]
    print(f"After filtering: {len(states)} samples")

    np.savez('bicycle_data_improved.npz',
             states=states, actions=actions,
             next_states=next_states, dt=dt)

    # ============================================================
    # Step 2: Normalize
    # ============================================================
    print("\n[2/6] Normalizing data...")
    states_norm = normalize_state(states)
    next_states_norm = normalize_state(next_states)
    action_scale = max(np.std(actions), 1.0)
    actions_norm = actions / action_scale
    delta_states = next_states_norm - states_norm

    # ============================================================
    # Step 3: Build library
    # ============================================================
    print("\n[3/6] Building polynomial library...")
    X = build_feature_matrix(states_norm, actions_norm)
    Theta, lib_labels = build_polynomial_library(X, degree=2)
    print(f"Library size: {Theta.shape[1]} terms")

    # ============================================================
    # Step 4: Train/test split and identification
    # ============================================================
    print("\n[4/6] Running SINDy identification...")

    Theta_train, Theta_test, dY_train, dY_test = train_test_split(
        Theta, delta_states, test_size=0.2, random_state=42
    )
    _, _, s_train, s_test = train_test_split(
        Theta, states_norm, test_size=0.2, random_state=42
    )
    _, _, _, next_s_test = train_test_split(
        Theta, next_states, test_size=0.2, random_state=42
    )

    # Try multiple thresholds and alpha values
    best_overall = None
    best_score = -np.inf

    print(f"\n{'Thresh':>8} {'Alpha':>8} {'Avg R²':>10} {'RMSE':>12} {'Terms':>8} {'Score':>10}")
    print("-" * 65)

    for alpha in [0.01, 0.05, 0.1]:
        for thresh in [0.01, 0.02, 0.05, 0.08, 0.1, 0.15]:
            try:
                Xi = stlsq(Theta_train, dY_train, threshold=thresh,
                           max_iter=100, alpha=alpha)

                r2_scores, rmse_scores = evaluate_model(Theta_test, dY_test, Xi)
                avg_r2 = np.mean(r2_scores)
                avg_rmse = np.mean(rmse_scores)
                n_terms = int(np.sum(np.abs(Xi) > 1e-6))

                # Composite score
                sparsity = 1.0 / (1.0 + n_terms / 50.0)
                # Penalize if too few terms (underfitting)
                coverage = min(n_terms / 20.0, 1.0)
                score = avg_r2 * sparsity * coverage

                marker = ""
                if score > best_score:
                    best_score = score
                    best_overall = {
                        'Xi': Xi.copy(), 'threshold': thresh, 'alpha': alpha,
                        'r2_scores': r2_scores, 'rmse_scores': rmse_scores,
                        'avg_r2': avg_r2, 'avg_rmse': avg_rmse, 'n_terms': n_terms
                    }
                    marker = "<--"

                print(f"{thresh:>8.3f} {alpha:>8.3f} {avg_r2:>10.4f} {avg_rmse:>12.6f} "
                      f"{n_terms:>8d} {score:>10.4f} {marker}")

            except Exception as e:
                print(f"{thresh:>8.3f} {alpha:>8.3f} FAILED: {str(e)[:30]}")

    # ============================================================
    # Step 5: Best model analysis
    # ============================================================
    print("\n" + "=" * 70)
    print("[5/6] Best Model Analysis")
    print("=" * 70)

    Xi = best_overall['Xi']
    print(f"\nThreshold: {best_overall['threshold']}, Alpha: {best_overall['alpha']}")
    print(f"Avg R²: {best_overall['avg_r2']:.4f}, Avg RMSE: {best_overall['avg_rmse']:.6f}")
    print(f"Active terms: {best_overall['n_terms']}")

    print_identified_equations(Xi, lib_labels)

    # Per-state analysis
    print(f"\n{'State':>15} {'R²':>10} {'RMSE':>12} {'# Terms':>10} {'Top 3 Features'}")
    print("-" * 80)
    for i in range(Xi.shape[1]):
        r2 = best_overall['r2_scores'][i]
        rmse = best_overall['rmse_scores'][i]
        n = int(np.sum(np.abs(Xi[:, i]) > 1e-6))

        # Top 3 features
        top_idx = np.argsort(np.abs(Xi[:, i]))[::-1]
        top_terms = []
        for j in top_idx[:3]:
            if abs(Xi[j, i]) > 1e-6:
                label = lib_labels[j] if j < len(lib_labels) else f"f{j}"
                top_terms.append(f"{Xi[j, i]:+.3f}*{label}")
        top_str = " ".join(top_terms) if top_terms else "(none)"

        print(f"{STATE_NAMES[i]:>15} {r2:>10.4f} {rmse:>12.6f} {n:>10d} {top_str}")

    # ============================================================
    # Step 6: Save and visualize
    # ============================================================
    print("\n[6/6] Saving results...")

    np.savez('sindy_model_improved.npz',
             coefficients=Xi,
             lib_labels=lib_labels,
             threshold=best_overall['threshold'],
             alpha=best_overall['alpha'],
             feature_names=FEATURE_NAMES,
             state_names=STATE_NAMES,
             action_scale=action_scale)
    print("  Saved to sindy_model_improved.npz")

    # Generate visualizations
    try:
        from visualization import (
            plot_data_distribution,
            plot_coefficient_heatmap,
            plot_prediction_comparison,
            plot_error_diagnostics
        )

        fig_dir = os.path.join(os.path.dirname(__file__), 'figures', 'improved')
        os.makedirs(fig_dir, exist_ok=True)

        results = {
            'best_model': best_overall,
            'best_threshold': best_overall['threshold'],
            'lib_labels': lib_labels,
            'Theta_test': Theta_test,
            'dY_test': dY_test,
            's_test': s_test,
            'X_test': build_feature_matrix(s_test, np.zeros((len(s_test), 1))),
            'next_s_test': next_s_test
        }

        plot_data_distribution(states, actions, STATE_NAMES,
                               save_path=os.path.join(fig_dir, 'fig_data_distribution.png'))
        plot_coefficient_heatmap(Xi, STATE_NAMES, FEATURE_NAMES, lib_labels,
                                  save_path=os.path.join(fig_dir, 'fig_coefficients.png'))
        plot_prediction_comparison(results,
                                   save_path=os.path.join(fig_dir, 'fig_prediction.png'))
        plot_error_diagnostics(results, STATE_NAMES,
                               save_path=os.path.join(fig_dir, 'fig_error_diagnostics.png'))
        print("  Plots saved to figures/improved/")

    except Exception as e:
        print(f"  Visualization error: {e}")

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)

    return best_overall


if __name__ == '__main__':
    run_improved_experiment()
