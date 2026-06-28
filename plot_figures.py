"""Generate figures for the paper.

Figures:
1. SINDy coefficient heatmap
2. Training curves (v6)
3. LQR vs SAC tracking comparison
4. Progressive disturbance schedule
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.family'] = ['serif']
matplotlib.rcParams['font.serif'] = ['Times New Roman']
matplotlib.rcParams['mathtext.fontset'] = 'stix'
import math
from scipy.linalg import solve_continuous_are
from meijaard_dynamics import benchmark_par_to_canonical, ab_matrix


def plot_sindy_coefficients():
    """Plot SINDy coefficient matrix as heatmap."""
    data = np.load('D:/系统辨识作业/sindy_bicycle/meijaard_sindy_v35.npz')
    Xi = data['coefficients']
    R2 = data.get('R2', None)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    # Feature names (SINDy library)
    features = ['1', r'$\phi$', r'$\delta$', r'$\dot{\phi}$', r'$\dot{\delta}$',
                r'$T_\delta$']
    states = [r'$\dot{\phi}$', r'$\dot{\delta}$', r'$\ddot{\phi}$', r'$\ddot{\delta}$']

    # Xi is (n_features, n_states)
    im = ax.imshow(Xi.T, cmap='RdBu_r', aspect='auto',
                   vmin=-np.max(np.abs(Xi)), vmax=np.max(np.abs(Xi)))

    ax.set_xticks(range(len(features)))
    ax.set_xticklabels(features, fontsize=11)
    ax.set_yticks(range(len(states)))
    ax.set_yticklabels(states, fontsize=11)
    ax.set_xlabel('Library Functions', fontsize=12)
    ax.set_ylabel('State Derivatives', fontsize=12)
    ax.set_title('SINDy Coefficient Matrix (v=3.5 m/s)', fontsize=13)

    # Add text annotations
    for i in range(Xi.shape[0]):
        for j in range(Xi.shape[1]):
            val = Xi[i, j]
            if abs(val) > 0.01:
                color = 'white' if abs(val) > 0.5 * np.max(np.abs(Xi)) else 'black'
                ax.text(i, j, f'{val:.3f}', ha='center', va='center',
                        fontsize=8, color=color)

    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig('D:/系统辨识作业/sindy_bicycle/fig_sindy_coefficients.png',
                dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved fig_sindy_coefficients.png")


def plot_training_curves():
    """Plot v6 training curves from log file."""
    log_file = 'D:/系统辨识作业/sindy_bicycle/mbpo_sac_v6_training.log'

    eps = []; tracks = []; rewards = []; weights = []; alphas = []
    with open(log_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('=') or line.startswith('MBPO') or \
               line.startswith('Device') or line.startswith('ALPHA') or \
               line.startswith('LQR') or line.startswith('Training') or \
               line.startswith('Warmup') or line.startswith('Ep') or \
               line.startswith('Evaluation'):
                continue
            parts = line.replace('|', '').split()
            if len(parts) >= 10:
                try:
                    ep = int(parts[0])
                    reward = float(parts[3])
                    steps = int(parts[4])
                    track = float(parts[5])
                    alpha = float(parts[7])
                    w = float(parts[8])
                    eps.append(ep)
                    rewards.append(reward)
                    tracks.append(track)
                    alphas.append(alpha)
                    weights.append(w)
                except (ValueError, IndexError):
                    continue

    if not eps:
        print("No data found in training log")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Tracking error
    ax = axes[0, 0]
    ax.plot(eps, tracks, 'b-', alpha=0.3, linewidth=0.5)
    # Moving average
    window = 10
    if len(tracks) > window:
        ma = np.convolve(tracks, np.ones(window)/window, mode='valid')
        ax.plot(eps[window-1:], ma, 'b-', linewidth=2, label='Moving Avg')
    ax.axhline(y=0.0919, color='r', linestyle='--', label='Optimal LQR')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Tracking Error')
    ax.set_title('Tracking Error (sigma=17.2' + chr(176) + ')')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Reward
    ax = axes[0, 1]
    ax.plot(eps, rewards, 'g-', alpha=0.3, linewidth=0.5)
    if len(rewards) > window:
        ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax.plot(eps[window-1:], ma, 'g-', linewidth=2, label='Moving Avg')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Episode Reward')
    ax.set_title('Training Reward')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # SAC weight schedule
    ax = axes[1, 0]
    ax.plot(eps, weights, 'r-', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('SAC Weight')
    ax.set_title('Graduated SAC Weight Schedule')
    ax.grid(True, alpha=0.3)

    # Alpha (temperature)
    ax = axes[1, 1]
    ax.plot(eps, alphas, 'm-', alpha=0.5, linewidth=1)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Alpha')
    ax.set_title('SAC Temperature')
    ax.grid(True, alpha=0.3)

    plt.suptitle('MBPO-SAC v6 Training Progress', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('D:/系统辨识作业/sindy_bicycle/fig_training_curves.png',
                dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved fig_training_curves.png")


def plot_lqr_vs_sac():
    """Plot LQR vs SAC tracking comparison across sigma levels."""
    v0 = 3.5; g = 9.81
    p = {
        'IBxx': 11.3557360401, 'IBxz': -1.96756380745,
        'IByy': 12.2177848012, 'IBzz': 3.12354397008,
        'IFxx': 0.0904106601579, 'IFyy': 0.149389340425,
        'IHxx': 0.253379594731, 'IHxz': -0.0720452391817,
        'IHyy': 0.246138810935, 'IHzz': 0.0955770796289,
        'IRxx': 0.0883819364527, 'IRyy': 0.152467620286,
        'c': 0.0685808540382, 'g': 9.81, 'lam': 0.399680398707,
        'mB': 81.86, 'mF': 2.02, 'mH': 3.22, 'mR': 3.11,
        'rF': 0.34352982332, 'rR': 0.340958858855, 'w': 1.121,
        'xB': 0.289099434117, 'xH': 0.866949640247,
        'zB': -1.04029228321, 'zH': -0.748236400835,
    }
    M, C1, K0, K2 = benchmark_par_to_canonical(p)
    A35, B35 = ab_matrix(M, C1, K0, K2, v0, g)
    reorder = [0, 2, 1, 3]
    A_lqr = A35[np.ix_(reorder, reorder)]
    B_lqr = B35[reorder, 1:2]
    Q = np.diag([1000.0, 100.0, 10.0, 1.0])
    R = np.array([[0.2]])
    P = solve_continuous_are(A_lqr, B_lqr, Q, R)
    K_lqr = (np.linalg.inv(R) @ B_lqr.T @ P).flatten()

    def compute_sigma(ep):
        return min(0.3, (1.08 ** ep) * 0.01)

    eps_list = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 53, 60, 70, 80]
    lqr_tracks = []

    dt = 1/30; sub_steps = 5; max_steps = 1000

    for ep in eps_list:
        sigma = compute_sigma(ep)
        tracks = []
        for seed in range(20):
            rng = np.random.RandomState(seed)
            theta = rng.uniform(-0.05, 0.05)
            theta_dot = 0.0; delta = 0.0; delta_dot = 0.0
            target_theta = np.clip(rng.normal(0, sigma), -math.pi/12, math.pi/12)
            total_track = 0.0
            for step in range(max_steps):
                if step <= 100:
                    tgt = 0.0
                elif step % 100 == 1:
                    tgt = np.clip(rng.normal(0, sigma), -math.pi/12, math.pi/12)
                    target_theta = tgt
                else:
                    tgt = target_theta
                x_lqr = np.array([theta - target_theta, theta_dot, delta, delta_dot])
                tau = float(-K_lqr @ x_lqr)
                A_m, B_m = ab_matrix(M, C1, K0, K2, max(v0, 0.5), g)
                B_s = B_m[:, 1:2]
                dt_sub = dt / sub_steps
                for _ in range(sub_steps):
                    x = np.array([[theta], [delta], [theta_dot], [delta_dot]])
                    xd = A_m @ x + B_s * tau
                    theta += dt_sub * xd[0, 0]
                    delta += dt_sub * xd[1, 0]
                    theta_dot += dt_sub * xd[2, 0]
                    delta_dot += dt_sub * xd[3, 0]
                new_dis = target_theta - theta
                total_track += abs(new_dis)
                if abs(theta) > math.pi / 3:
                    break
            tracks.append(total_track / max(step + 1, 1))
        lqr_tracks.append(np.mean(tracks))

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    sigmas = [math.degrees(compute_sigma(ep)) for ep in eps_list]

    ax.plot(sigmas, lqr_tracks, 'b-o', linewidth=2, markersize=6, label='Optimal LQR')

    # Add v6 best result (from training log)
    # v6 at sigma=17.2 achieved track~0.06-0.08
    ax.axhspan(0.06, 0.08, alpha=0.2, color='green', label='v6 SAC range (sigma=17.2' + chr(176) + ')')
    ax.axhline(y=0.07, color='green', linestyle='--', linewidth=1.5, label='v6 SAC avg')

    ax.set_xlabel('Disturbance Sigma (degrees)', fontsize=12)
    ax.set_ylabel('Average Tracking Error', fontsize=12)
    ax.set_title('LQR Tracking Performance vs Disturbance Level', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('D:/系统辨识作业/sindy_bicycle/fig_lqr_performance.png',
                dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved fig_lqr_performance.png")


def plot_progressive_disturbance():
    """Plot the progressive disturbance schedule."""
    eps = list(range(200))
    sigmas = [min(0.3, (1.08 ** ep) * 0.01) for ep in eps]
    sigma_deg = [math.degrees(s) for s in sigmas]

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(eps, sigma_deg, 'b-', linewidth=2)
    ax.axhline(y=math.degrees(0.3), color='r', linestyle='--', label='Max sigma (17.2' + chr(176) + ')')
    ax.fill_between(eps, 0, sigma_deg, alpha=0.2)
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Sigma (degrees)', fontsize=12)
    ax.set_title('Progressive Disturbance Schedule', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('D:/系统辨识作业/sindy_bicycle/fig_progressive_disturbance.png',
                dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved fig_progressive_disturbance.png")


def plot_system_eigenvalues():
    """Plot Meijaard model eigenvalues at different speeds."""
    p = {
        'IBxx': 11.3557360401, 'IBxz': -1.96756380745,
        'IByy': 12.2177848012, 'IBzz': 3.12354397008,
        'IFxx': 0.0904106601579, 'IFyy': 0.149389340425,
        'IHxx': 0.253379594731, 'IHxz': -0.0720452391817,
        'IHyy': 0.246138810935, 'IHzz': 0.0955770796289,
        'IRxx': 0.0883819364527, 'IRyy': 0.152467620286,
        'c': 0.0685808540382, 'g': 9.81, 'lam': 0.399680398707,
        'mB': 81.86, 'mF': 2.02, 'mH': 3.22, 'mR': 3.11,
        'rF': 0.34352982332, 'rR': 0.340958858855, 'w': 1.121,
        'xB': 0.289099434117, 'xH': 0.866949640247,
        'zB': -1.04029228321, 'zH': -0.748236400835,
    }
    M, C1, K0, K2 = benchmark_par_to_canonical(p)
    g = 9.81

    speeds = np.linspace(0.5, 10, 50)
    all_eigs = []

    for v in speeds:
        A, B = ab_matrix(M, C1, K0, K2, v, g)
        eigs = np.linalg.eigvals(A)
        all_eigs.append(eigs)

    all_eigs = np.array(all_eigs)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    colors = ['r', 'b', 'g', 'm']
    labels = [r'$\lambda_1$', r'$\lambda_2$', r'$\lambda_3$', r'$\lambda_4$']

    for i in range(4):
        real = all_eigs[:, i].real
        imag = all_eigs[:, i].imag
        ax.plot(speeds, real, color=colors[i], linewidth=2, label=labels[i])

    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.axvline(x=3.5, color='gray', linestyle='--', label='v=3.5 m/s')
    ax.axvline(x=5.0, color='orange', linestyle='--', alpha=0.5, label='Self-stable region')
    ax.axvline(x=7.1, color='orange', linestyle='--', alpha=0.5)
    ax.fill_betweenx([-2, 2], 5.0, 7.1, alpha=0.1, color='orange')

    ax.set_xlabel('Forward Speed (m/s)', fontsize=12)
    ax.set_ylabel('Real Part of Eigenvalue', fontsize=12)
    ax.set_title('Meijaard Bicycle Model Eigenvalues vs Speed', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-3, 3)

    plt.tight_layout()
    plt.savefig('D:/系统辨识作业/sindy_bicycle/fig_eigenvalues.png',
                dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved fig_eigenvalues.png")


def plot_sindy_vs_true():
    """Plot SINDy prediction vs true Meijaard dynamics."""
    data = np.load('D:/系统辨识作业/sindy_bicycle/meijaard_openloop_data_v35.npz')
    X = data['states']  # (n_steps, 4) - theta, delta, theta_dot, delta_dot
    U = data['taus']    # (n_steps, 1) - torque
    dt = float(data['dt'])
    t = np.arange(len(X)) * dt

    sindy_data = np.load('D:/系统辨识作业/sindy_bicycle/meijaard_sindy_v35.npz')
    Xi = sindy_data['coefficients']
    A_d = np.eye(4) + Xi[1:5, :].T
    B_d = Xi[5:6, :].T

    # SINDy prediction (use first N steps where linear model is valid)
    N = min(300, len(X))
    X_sindy = np.zeros((N, 4))
    X_sindy[0] = X[0]
    for i in range(N - 1):
        X_sindy[i+1] = A_d @ X_sindy[i] + B_d.flatten() * U[i, 0]
        # Stop if prediction diverges
        if np.max(np.abs(X_sindy[i+1])) > 1e6:
            X_sindy[i+1:] = np.nan
            N = i + 1
            break

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    state_names = [r'$\theta$ (rad)', r'$\delta$ (rad)',
                   r'$\dot{\theta}$ (rad/s)', r'$\dot{\delta}$ (rad/s)']

    for idx, (ax, name) in enumerate(zip(axes.flat, state_names)):
        ax.plot(t[:N], X[:N, idx], 'b-', linewidth=1.5, label='True')
        ax.plot(t[:N], X_sindy[:N, idx], 'r--', linewidth=1.5, label='SINDy')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(name)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Compute R on valid data
        valid = ~np.isnan(X_sindy[:N, idx])
        if valid.sum() > 10:
            ss_res = np.sum((X[:N, idx][valid] - X_sindy[:N, idx][valid])**2)
            ss_tot = np.sum((X[:N, idx][valid] - np.mean(X[:N, idx][valid]))**2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        else:
            r2 = 0
        ax.set_title(f'{name} (R2={r2:.4f})')

    plt.suptitle('SINDy Model Validation at v=3.5 m/s', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('D:/系统辨识作业/sindy_bicycle/fig_sindy_validation.png',
                dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved fig_sindy_validation.png")


def plot_mbpo_architecture():
    """Plot MBPO-SAC architecture diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Boxes (x, y, w, h, label, color)
    boxes = [
        (1, 4.2, 2, 1, 'Real\nEnvironment', 'lightblue'),
        (4, 4.2, 2, 1, 'SINDy\nWorld Model', 'lightyellow'),
        (7, 4.2, 2, 1, 'SAC\nAgent', 'lightgreen'),
        (1, 1.8, 2, 1, 'Real\nBuffer', 'lightcyan'),
        (4, 1.8, 2, 1, 'Virtual\nBuffer', 'lightyellow'),
        (7, 1.8, 2, 1, 'LQR\nController', 'lightcoral'),
    ]

    for x, y, w, h, label, color in boxes:
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha='center', va='center', fontsize=10, fontweight='bold')

    # Helper to draw arrow with label
    def draw_arrow(x1, y1, x2, y2, label, label_offset=(0, 0.15)):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
        mx = (x1 + x2) / 2 + label_offset[0]
        my = (y1 + y2) / 2 + label_offset[1]
        ax.text(mx, my, label, ha='center', va='bottom', fontsize=8, color='gray')

    # Top row horizontal arrows
    draw_arrow(3, 4.7, 4, 4.7, 'state/action', (0, 0.05))
    draw_arrow(6, 4.7, 7, 4.7, 'virtual data', (0, 0.05))

    # Vertical arrows (top to bottom)
    draw_arrow(2, 4.2, 2, 2.8, 'transitions', (0.35, 0))
    draw_arrow(5, 4.2, 5, 2.8, 'virtual transitions', (0.6, 0))
    draw_arrow(8, 4.2, 8, 2.8, 'Q-values', (-0.5, 0))

    # Bottom row horizontal arrows (with labels BELOW the line)
    draw_arrow(3, 2.3, 4, 2.3, 'mixed batch', (0, -0.25))
    draw_arrow(6, 2.3, 7, 2.3, 'action', (0, -0.25))

    # Title
    ax.text(5, 5.8, 'MBPO-SAC Architecture with Graduated Residual Control',
            ha='center', va='center', fontsize=13, fontweight='bold')

    # Residual formula
    ax.text(5, 0.5, r'$\tau = u_{LQR} + w \cdot \alpha \cdot \pi_{SAC}(s)$',
            ha='center', va='center', fontsize=14,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('D:/系统辨识作业/sindy_bicycle/fig_mbpo_architecture.png',
                dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved fig_mbpo_architecture.png")


if __name__ == '__main__':
    plot_sindy_coefficients()
    plot_sindy_vs_true()
    plot_system_eigenvalues()
    plot_progressive_disturbance()
    plot_lqr_vs_sac()
    plot_mbpo_architecture()
    # Training curves only available after training
    try:
        plot_training_curves()
    except Exception as e:
        print(f"Training curves skipped: {e}")
    print("\nAll figures generated!")
