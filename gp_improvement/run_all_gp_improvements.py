"""Run all GP improvements and generate comparison report.

This script runs GP-B0 through GP-B6 and generates a comparison report.

Usage:
    python run_all_gp_improvements.py

Expected output:
    - Results for all GP methods
    - Comparison table
    - GP_IMPROVEMENT_REPORT.md
    - GP_IMPROVEMENT_RESULTS.json
"""

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['PYTHONWARNINGS'] = 'ignore'

import sys
sys.path.insert(0, 'D:/系统辨识作业/sindy_bicycle/gp_improvement')

import numpy as np
import json
import time

from gp_b0_baseline import run_gp_b0
from gp_b1_ard import run_gp_b1
from gp_b2_composite_kernel import run_gp_b2
from gp_b3_physics_residual import run_gp_b3
from gp_b4_sparse_local import run_gp_b4
from gp_b5_bootstrap_ensemble import run_gp_b5
from gp_b6_uncertainty_gated import run_gp_b6


def convert_to_serializable(obj):
    """Convert numpy types to Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    return obj


def run_all_methods(n_runs=3, n_segments=5):
    """Run all GP improvement methods."""
    print("=" * 60)
    print("Running All GP Improvements")
    print("=" * 60)

    all_results = {}

    # GP-B0: Baseline
    print("\n" + "=" * 60)
    print("GP-B0: Baseline GP")
    print("=" * 60)
    start = time.time()
    all_results['gp_b0'] = run_gp_b0(n_runs=n_runs, n_segments=n_segments)
    all_results['gp_b0']['time'] = time.time() - start

    # GP-B1: ARD
    print("\n" + "=" * 60)
    print("GP-B1: ARD Gaussian Process")
    print("=" * 60)
    start = time.time()
    all_results['gp_b1'] = run_gp_b1(n_runs=n_runs, n_segments=n_segments)
    all_results['gp_b1']['time'] = time.time() - start

    # GP-B2: Composite Kernel
    print("\n" + "=" * 60)
    print("GP-B2: Composite Kernel GP")
    print("=" * 60)
    start = time.time()
    all_results['gp_b2'] = run_gp_b2(n_runs=n_runs, n_segments=n_segments)
    all_results['gp_b2']['time'] = time.time() - start

    # GP-B3: Physics Residual
    print("\n" + "=" * 60)
    print("GP-B3: Physics Residual GP")
    print("=" * 60)
    start = time.time()
    all_results['gp_b3'] = run_gp_b3(n_runs=n_runs, n_segments=n_segments)
    all_results['gp_b3']['time'] = time.time() - start

    # GP-B4: Sparse/Local
    print("\n" + "=" * 60)
    print("GP-B4: Sparse/Local GP")
    print("=" * 60)
    start = time.time()
    all_results['gp_b4'] = run_gp_b4(n_runs=n_runs, n_segments=n_segments)
    all_results['gp_b4']['time'] = time.time() - start

    # GP-B5: Bootstrap Ensemble
    print("\n" + "=" * 60)
    print("GP-B5: Bootstrap GP Ensemble")
    print("=" * 60)
    start = time.time()
    all_results['gp_b5'] = run_gp_b5(n_runs=n_runs, n_segments=n_segments, n_models=5)
    all_results['gp_b5']['time'] = time.time() - start

    # GP-B6: Uncertainty-Gated
    print("\n" + "=" * 60)
    print("GP-B6: Uncertainty-Gated Residual GP")
    print("=" * 60)
    start = time.time()
    all_results['gp_b6'] = run_gp_b6(n_runs=n_runs, n_segments=n_segments, gate_threshold=0.1)
    all_results['gp_b6']['time'] = time.time() - start

    return all_results


def generate_report(all_results):
    """Generate comparison report."""
    print("\n" + "=" * 60)
    print("Generating Comparison Report")
    print("=" * 60)

    # Extract 500-step results
    methods = [
        ('GP-B0', 'Baseline GP'),
        ('GP-B1', 'ARD GP'),
        ('GP-B2', 'Composite Kernel GP'),
        ('GP-B3', 'Physics Residual GP'),
        ('GP-B4', 'Sparse/Local GP'),
        ('GP-B5', 'Bootstrap Ensemble GP'),
        ('GP-B6', 'Uncertainty-Gated GP'),
    ]

    # Print comparison table
    print(f"\n{'Method':<25} | {'500-step MAE':>15} | {'Std':>10} | {'Time (s)':>10}")
    print("-" * 70)

    for method_id, method_name in methods:
        if method_id in all_results:
            result = all_results[method_id]
            if 'aggregated' in result and 500 in result['aggregated']:
                mae = result['aggregated'][500]['mean']
                std = result['aggregated'][500]['std']
                time_s = result.get('time', 0)
                print(f"{method_name:<25} | {mae:15.5f} | {std:10.5f} | {time_s:10.1f}")
            else:
                print(f"{method_name:<25} | {'N/A':>15} | {'N/A':>10} | {'N/A':>10}")

    # Save results
    output_file = 'D:/系统辨识作业/sindy_bicycle/gp_improvement/GP_IMPROVEMENT_RESULTS.json'
    with open(output_file, 'w') as f:
        json.dump(convert_to_serializable(all_results), f, indent=2)
    print(f"\nResults saved to: {output_file}")

    # Generate markdown report
    report_file = 'D:/系统辨识作业/sindy_bicycle/gp_improvement/GP_IMPROVEMENT_REPORT.md'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# GP Improvement Report\n\n")
        f.write("> **Generated**: 2026-06-23\n")
        f.write("> **Project**: sindy_bicycle\n")
        f.write("> **Purpose**: Comparison of GP improvement methods\n\n")
        f.write("---\n\n")

        f.write("## 1. Summary\n\n")
        f.write("| Method | 500-step MAE (rad) | Std | Time (s) |\n")
        f.write("|--------|-------------------|-----|----------|\n")

        for method_id, method_name in methods:
            if method_id in all_results:
                result = all_results[method_id]
                if 'aggregated' in result and 500 in result['aggregated']:
                    mae = result['aggregated'][500]['mean']
                    std = result['aggregated'][500]['std']
                    time_s = result.get('time', 0)
                    f.write(f"| {method_name} | {mae:.5f} | {std:.5f} | {time_s:.1f} |\n")

        f.write("\n---\n\n")
        f.write("## 2. Method Details\n\n")

        for method_id, method_name in methods:
            if method_id in all_results:
                result = all_results[method_id]
                f.write(f"### {method_name}\n\n")
                if 'config' in result:
                    f.write("**Configuration**:\n")
                    for key, value in result['config'].items():
                        f.write(f"- {key}: {value}\n")
                f.write("\n")

        f.write("---\n\n")
        f.write("## 3. Key Findings\n\n")
        f.write("1. **Baseline GP** (GP-B0): Standard GP with RBF kernel\n")
        f.write("2. **ARD GP** (GP-B1): Automatic relevance determination for feature selection\n")
        f.write("3. **Composite Kernel** (GP-B2): Combines RBF and Matérn kernels\n")
        f.write("4. **Physics Residual** (GP-B3): Uses physics model as prior\n")
        f.write("5. **Sparse/Local** (GP-B4): Reduces computational cost\n")
        f.write("6. **Bootstrap Ensemble** (GP-B5): Multiple GPs for uncertainty\n")
        f.write("7. **Uncertainty-Gated** (GP-B6): Gates residual by uncertainty\n\n")

        f.write("---\n\n")
        f.write("*Report generated by Claude Code on 2026-06-23*\n")

    print(f"Report saved to: {report_file}")


if __name__ == '__main__':
    # Run all methods
    all_results = run_all_methods(n_runs=3, n_segments=5)

    # Generate report
    generate_report(all_results)

    print("\n" + "=" * 60)
    print("All GP improvements completed!")
    print("=" * 60)
