"""
Main pipeline for SINDy-based bicycle system identification.

Reproduces the SINDy portion of the paper:
"基于 SINDy 与 TD-MPC2 的无人自行车系统辨识与模型强化学习研究"

Pipeline:
1. Generate bicycle dynamics data (stratified sampling)
2. Preprocess and normalize
3. Construct SINDy dictionary (polynomial library)
4. Run STLSQ sparse regression with threshold sweep
5. Evaluate and visualize results
"""

import sys
import os
import numpy as np
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_collector import BicycleDataCollector, normalize_state, denormalize_state
from sindy_identification import (
    run_sindy_identification, diagnose_errors,
    print_identified_equations, STATE_NAMES, FEATURE_NAMES
)


def main():
    print("=" * 70)
    print("  SINDy Bicycle System Identification")
    print("  Reproducing the SINDy pipeline from:")
    print("  '基于 SINDy 与 TD-MPC2 的无人自行车系统辨识与模型强化学习研究'")
    print("=" * 70)

    # ============================================================
    # Step 1: Data Collection
    # ============================================================
    print("\n" + "=" * 70)
    print("STEP 1: Data Collection (Stratified Sampling)")
    print("=" * 70)

    dt = 1/30  # 30 Hz as in the paper
    collector = BicycleDataCollector(dt=dt, seed=42)

    print("\nCollecting data across 3 scenarios:")
    print("  - Straight segments (k ≈ 0)")
    print("  - Curved segments (moderate k)")
    print("  - Sharp turns (large k)")
    print(f"  Sampling period: {dt:.4f} s (30 Hz)")

    t_start = time.time()
    dataset = collector.collect_dataset(
        n_episodes_per_scenario=40,
        t_episode=10.0
    )
    t_collect = time.time() - t_start

    states = dataset['states']
    actions = dataset['actions']
    next_states = dataset['next_states']
    rewards = dataset['rewards']

    print(f"\nData collection completed in {t_collect:.1f}s")
    print(f"Total samples: {len(states)}")

    np.savez('bicycle_data.npz',
             states=states, actions=actions,
             next_states=next_states, rewards=rewards, dt=dt)
    print("Raw data saved to bicycle_data.npz")

    # ============================================================
    # Step 2: Data Statistics
    # ============================================================
    print("\n" + "=" * 70)
    print("STEP 2: Data Statistics")
    print("=" * 70)

    print(f"\n{'State':>15} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 60)
    for i, name in enumerate(STATE_NAMES):
        s = states[:, i]
        print(f"{name:>15} {np.mean(s):>10.4f} {np.std(s):>10.4f} "
              f"{np.min(s):>10.4f} {np.max(s):>10.4f}")

    print(f"\n{'Action':>15} {np.mean(actions):>10.4f} {np.std(actions):>10.4f} "
          f"{np.min(actions):>10.4f} {np.max(actions):>10.4f}")

    # ============================================================
    # Step 3: Normalization
    # ============================================================
    print("\n" + "=" * 70)
    print("STEP 3: Data Normalization (as in paper)")
    print("=" * 70)
    print("\n  ey/10, eψ/1.57, v/5, θ/1.57, θ̇/10, k×8, δ/0.785, δ̇/3")

    states_norm = normalize_state(states)
    print(f"\n{'State':>15} {'Norm Mean':>10} {'Norm Std':>10}")
    print("-" * 40)
    for i, name in enumerate(STATE_NAMES):
        print(f"{name:>15} {np.mean(states_norm[:, i]):>10.4f} "
              f"{np.std(states_norm[:, i]):>10.4f}")

    # ============================================================
    # Step 4: SINDy Identification
    # ============================================================
    print("\n" + "=" * 70)
    print("STEP 4: SINDy Sparse Identification (STLSQ)")
    print("=" * 70)

    t_start = time.time()
    results = run_sindy_identification(
        states, actions, next_states,
        normalize=True,
        threshold_values=[0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3],
        alpha=0.1,
        max_iter=100,
        degree=2
    )
    t_id = time.time() - t_start
    print(f"\nIdentification completed in {t_id:.1f}s")

    # ============================================================
    # Step 5: Error Diagnostics
    # ============================================================
    if results['best_model'] is not None:
        print("\n" + "=" * 70)
        print("STEP 5: Error Diagnostics")
        print("=" * 70)
        diagnose_errors(results)

        # ============================================================
        # Step 6: Model Summary
        # ============================================================
        print("\n" + "=" * 70)
        print("STEP 6: Identified World Model Summary")
        print("=" * 70)

        Xi = results['best_model']['Xi']
        lib_labels = results['lib_labels']

        # Count active terms per state
        print(f"\n{'State':>15} {'# Active Terms':>15} {'Top Feature':>25}")
        print("-" * 60)
        for i in range(Xi.shape[1]):
            active = np.abs(Xi[:, i]) > 1e-6
            n_active = np.sum(active)
            if n_active > 0:
                top_idx = np.argmax(np.abs(Xi[:, i]))
                top_label = lib_labels[top_idx] if top_idx < len(lib_labels) else f"f{top_idx}"
                top_val = Xi[top_idx, i]
                print(f"{STATE_NAMES[i]:>15} {n_active:>15d} {f'{top_val:+.4f}*{top_label}':>25}")
            else:
                print(f"{STATE_NAMES[i]:>15} {0:>15d} {'(all zero)':>25}")

        # Save model
        np.savez('sindy_model.npz',
                 coefficients=Xi,
                 lib_labels=lib_labels,
                 threshold=results['best_threshold'],
                 feature_names=FEATURE_NAMES,
                 state_names=STATE_NAMES)
        print("\nModel saved to sindy_model.npz")

    # ============================================================
    # Step 7: Visualization
    # ============================================================
    print("\n" + "=" * 70)
    print("STEP 7: Generating Visualizations")
    print("=" * 70)

    try:
        from visualization import (
            plot_data_distribution,
            plot_coefficient_heatmap,
            plot_prediction_comparison,
            plot_error_diagnostics
        )

        fig_dir = os.path.join(os.path.dirname(__file__), 'figures', 'basic')
        os.makedirs(fig_dir, exist_ok=True)

        plot_data_distribution(states, actions, STATE_NAMES,
                               save_path=os.path.join(fig_dir, 'fig_data_distribution.png'))
        print("  [OK] figures/basic/fig_data_distribution.png")

        if results['best_model'] is not None:
            plot_coefficient_heatmap(results['best_model']['Xi'], STATE_NAMES, FEATURE_NAMES,
                                      results['lib_labels'],
                                      save_path=os.path.join(fig_dir, 'fig_coefficients.png'))
            print("  [OK] figures/basic/fig_coefficients.png")

            plot_prediction_comparison(results,
                                       save_path=os.path.join(fig_dir, 'fig_prediction.png'))
            print("  [OK] figures/basic/fig_prediction.png")

            plot_error_diagnostics(results, STATE_NAMES,
                                   save_path=os.path.join(fig_dir, 'fig_error_diagnostics.png'))
            print("  [OK] figures/basic/fig_error_diagnostics.png")

    except Exception as e:
        print(f"  Visualization error: {e}")

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\nOutputs:")
    print(f"  - bicycle_data.npz  : Raw data ({len(states)} samples)")
    print(f"  - sindy_model.npz   : SINDy model coefficients")
    print(f"  - fig_*.png         : Visualization plots")

    if results['best_model'] is not None:
        n_terms = np.sum(np.abs(results['best_model']['Xi']) > 1e-6)
        print(f"\nModel Summary:")
        print(f"  - Active terms: {n_terms}")
        print(f"  - Threshold: {results['best_threshold']}")
        print(f"  - Avg R²: {results['best_model']['avg_r2']:.4f}")

    return results


if __name__ == '__main__':
    results = main()
