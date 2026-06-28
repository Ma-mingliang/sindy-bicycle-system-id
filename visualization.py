"""
Visualization functions for SINDy bicycle system identification.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['SimSun', 'Times New Roman', 'DejaVu Serif']
rcParams['font.serif'] = ['Times New Roman', 'SimSun', 'DejaVu Serif']
rcParams['mathtext.fontset'] = 'stix'
rcParams['axes.unicode_minus'] = False


def plot_data_distribution(states, actions, state_names, save_path='fig_data_distribution.png'):
    """Plot data distribution across states and actions."""
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('Bicycle State-Action Data Distribution\n(自行车状态-动作数据分布)', fontsize=14)

    for i in range(8):
        ax = axes[i // 3, i % 3]
        ax.hist(states[:, i], bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='white')
        ax.set_title(f'{state_names[i]}', fontsize=11)
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.grid(True, alpha=0.3)

    ax = axes[2, 2]
    ax.hist(actions.flatten(), bins=50, density=True, alpha=0.7, color='coral', edgecolor='white')
    ax.set_title('Action (a)', fontsize=11)
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_coefficient_heatmap(Xi, state_names, feature_names, lib_labels,
                              save_path='fig_coefficients.png'):
    """Plot the SINDy coefficient matrix as a heatmap.

    Parameters
    ----------
    Xi : ndarray, shape (n_library, n_states)
        Coefficient matrix from SINDy
    """
    # Select only the rows with at least one non-zero coefficient
    active_mask = np.any(np.abs(Xi) > 1e-6, axis=1)
    Xi_active = Xi[active_mask]
    labels_active = [lib_labels[i] for i in range(len(lib_labels)) if active_mask[i]]

    fig, ax = plt.subplots(figsize=(max(12, len(labels_active) * 0.5), 6))

    vmax = np.max(np.abs(Xi_active)) if Xi_active.size > 0 else 1
    im = ax.imshow(Xi_active.T, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(labels_active)))
    ax.set_xticklabels(labels_active, rotation=60, ha='right', fontsize=7)
    ax.set_yticks(range(len(state_names)))
    ax.set_yticklabels([f'Δ{name}' for name in state_names], fontsize=10)

    # Annotate
    for i in range(Xi_active.shape[0]):
        for j in range(Xi_active.shape[1]):
            if abs(Xi_active[i, j]) > 1e-3:
                ax.text(i, j, f'{Xi_active[i, j]:.2f}',
                       ha='center', va='center', fontsize=6,
                       color='white' if abs(Xi_active[i, j]) > vmax * 0.5 else 'black')

    ax.set_title('SINDy Coefficient Matrix (Active Terms)\n(活跃项的SINDy系数矩阵)', fontsize=12)
    plt.colorbar(im, ax=ax, label='Coefficient')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_prediction_comparison(results, n_steps=200, save_path='fig_prediction.png'):
    """Compare SINDy predictions with ground truth."""
    best = results['best_model']
    Xi = best['Xi']
    Theta_test = results['Theta_test']
    dY_test = results['dY_test']

    dY_pred = Theta_test @ Xi

    fig, axes = plt.subplots(4, 2, figsize=(16, 14))
    fig.suptitle('SINDy Prediction vs Ground Truth\n'
                 f'(SINDy预测 vs 真实值, threshold={results["best_threshold"]}, '
                 f'avg R²={best["avg_r2"]:.4f})',
                 fontsize=13)

    state_names = ['ey', 'eψ', 'v', 'θ', 'θ̇', 'k', 'δ', 'δ̇']

    for i in range(8):
        ax = axes[i // 2, i % 2]
        idx = np.arange(min(n_steps, len(dY_test)))

        ax.plot(idx, dY_test[:n_steps, i], 'b-', alpha=0.7, label='Ground Truth', linewidth=1)
        ax.plot(idx, dY_pred[:n_steps, i], 'r--', alpha=0.7, label='SINDy Prediction', linewidth=1)

        rmse = np.sqrt(np.mean((dY_test[:, i] - dY_pred[:, i])**2))
        ss_res = np.sum((dY_test[:, i] - dY_pred[:, i])**2)
        ss_tot = np.sum((dY_test[:, i] - np.mean(dY_test[:, i]))**2)
        r2 = 1 - ss_res / (ss_tot + 1e-10)

        ax.set_title(f'Δ{state_names[i]}  (RMSE={rmse:.4f}, R²={r2:.3f})', fontsize=10)
        ax.set_xlabel('Sample')
        ax.set_ylabel('Δ value')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_error_diagnostics(results, state_names, save_path='fig_error_diagnostics.png'):
    """Plot error diagnostics."""
    best = results['best_model']
    Xi = best['Xi']
    Theta_test = results['Theta_test']
    dY_test = results['dY_test']
    s_test = results['s_test']

    dY_pred = Theta_test @ Xi
    errors = dY_test - dY_pred

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Error Diagnostics (误差诊断)', fontsize=14)

    # 1. RMSE per state
    ax = axes[0, 0]
    rmse_per_state = [np.sqrt(np.mean(errors[:, i]**2)) for i in range(8)]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, 8))
    ax.bar(state_names, rmse_per_state, color=colors, edgecolor='white')
    ax.set_title('RMSE per State Dimension')
    ax.set_ylabel('RMSE')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')

    # 2. Error vs state magnitude
    ax = axes[0, 1]
    for i in [0, 3, 6]:  # ey, theta, delta
        abs_state = np.abs(s_test[:, i])
        abs_error = np.abs(errors[:, i])
        ax.scatter(abs_state, abs_error, alpha=0.2, s=5, label=state_names[i])
    ax.set_xlabel('|State| (normalized)')
    ax.set_ylabel('|Error|')
    ax.set_title('Error vs State Magnitude')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Error correlation
    ax = axes[1, 0]
    if errors.shape[0] > 1:
        corr = np.corrcoef(errors.T)
        im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_xticks(range(8))
        ax.set_xticklabels(state_names, rotation=45, fontsize=8)
        ax.set_yticks(range(8))
        ax.set_yticklabels(state_names, fontsize=8)
        ax.set_title('Error Correlation Matrix')
        plt.colorbar(im, ax=ax)

    # 4. Residual histogram
    ax = axes[1, 1]
    ax.hist(errors.flatten(), bins=100, density=True, alpha=0.7,
            color='steelblue', edgecolor='white')
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    ax.set_title('Residual Distribution (All States)')
    ax.set_xlabel('Residual')
    ax.set_ylabel('Density')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
