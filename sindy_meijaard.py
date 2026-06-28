"""SINDy identification for Meijaard 2007 benchmark bicycle.

Identifies discrete state transition model:
  s_{t+1} = s_t + f_SINDy(s_t, tau)

where s = [phi, delta, phi_dot, delta_dot], tau = steering torque.

Uses STLSQ (Sequentially Thresholded Least Squares) with polynomial library.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# State names for Meijaard model
STATE_NAMES = ['phi', 'delta', 'phi_dot', 'delta_dot']
FEATURE_NAMES = STATE_NAMES + ['tau']


def build_polynomial_library(X, degree=2):
    """Build polynomial library up to given degree.

    Includes: constant, linear, quadratic (pairwise interactions + squares).

    Args:
        X: ndarray (n_samples, n_features)
        degree: max polynomial degree

    Returns:
        Theta: ndarray (n_samples, n_library_terms)
        labels: list of str
    """
    n_samples, n_features = X.shape
    features = []
    labels = []

    # Degree 0: constant
    features.append(np.ones((n_samples, 1)))
    labels.append('1')

    # Degree 1: linear
    for i in range(n_features):
        features.append(X[:, i:i+1])
        name = FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f'x{i}'
        labels.append(name)

    # Degree 2: quadratic
    if degree >= 2:
        for i in range(n_features):
            for j in range(i, n_features):
                features.append(X[:, i:i+1] * X[:, j:j+1])
                name_i = FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f'x{i}'
                name_j = FEATURE_NAMES[j] if j < len(FEATURE_NAMES) else f'x{j}'
                if i == j:
                    labels.append(f'{name_i}^2')
                else:
                    labels.append(f'{name_i}*{name_j}')

    Theta = np.hstack(features)
    return Theta, labels


def stlsq(Theta, Y, threshold, max_iter=50, alpha=0.0):
    """Sequentially Thresholded Least Squares.

    Args:
        Theta: library matrix (n_samples, n_library)
        Y: target (n_samples, n_states)
        threshold: sparsity threshold
        max_iter: max iterations
        alpha: L2 regularization

    Returns:
        Xi: coefficient matrix (n_library, n_states)
    """
    n_library = Theta.shape[1]
    n_states = Y.shape[1]
    Xi = np.zeros((n_library, n_states))

    for k in range(n_states):
        y = Y[:, k]

        # First iteration: fit ALL terms
        if alpha > 0:
            lhs = Theta.T @ Theta + alpha * np.eye(n_library)
            rhs = Theta.T @ y
            xi = np.linalg.solve(lhs, rhs)
        else:
            xi = np.linalg.lstsq(Theta, y, rcond=None)[0]

        # Threshold small coefficients
        xi[np.abs(xi) <= threshold] = 0.0

        for _ in range(max_iter):
            # Active set
            active = np.abs(xi) > threshold
            if not np.any(active):
                xi = np.zeros(n_library)
                break

            # Solve least squares on active set
            Theta_active = Theta[:, active]
            if alpha > 0:
                lhs = Theta_active.T @ Theta_active + alpha * np.eye(Theta_active.shape[1])
                rhs = Theta_active.T @ y
                xi_active = np.linalg.solve(lhs, rhs)
            else:
                xi_active = np.linalg.lstsq(Theta_active, y, rcond=None)[0]

            # Update full coefficient vector
            xi_new = np.zeros(n_library)
            xi_new[active] = xi_active

            # Threshold
            xi_new[np.abs(xi_new) <= threshold] = 0.0

            # Check convergence
            if np.allclose(xi, xi_new, atol=1e-10):
                break
            xi = xi_new

        Xi[:, k] = xi

    return Xi


def identify_meijaard_sindy(data_path, threshold=0.01, alpha=0.0, degree=2):
    """Run SINDy identification on Meijaard data.

    Args:
        data_path: path to .npz file with states, taus, next_states
        threshold: STLSQ sparsity threshold
        alpha: L2 regularization
        degree: polynomial degree

    Returns:
        Xi: coefficient matrix
        labels: feature labels
        metrics: dict with train/test errors
    """
    data = np.load(data_path)
    states = data['states']       # (N, 4)
    taus = data['taus']           # (N, 1)
    next_states = data['next_states']  # (N, 4)

    print(f"Loaded {len(states)} transitions from {data_path}")
    print(f"Speed: {data['v']} m/s, dt: {data['dt']} s")

    # Compute deltas: s_{t+1} - s_t
    deltas = next_states - states

    # Build feature matrix: [phi, delta, phi_dot, delta_dot, tau]
    X = np.hstack([states, taus])

    # Build library
    Theta, labels = build_polynomial_library(X, degree=degree)
    print(f"Library size: {Theta.shape[1]} terms")
    print(f"Terms: {labels}")

    # Train/test split
    Theta_train, Theta_test, Y_train, Y_test = train_test_split(
        Theta, deltas, test_size=0.2, random_state=42)
    print(f"Train: {len(Theta_train)}, Test: {len(Theta_test)}")

    # Run STLSQ
    print(f"\nRunning STLSQ (threshold={threshold}, alpha={alpha})...")
    Xi = stlsq(Theta_train, Y_train, threshold=threshold, alpha=alpha)

    # Evaluate
    Y_pred_train = Theta_train @ Xi
    Y_pred_test = Theta_test @ Xi

    metrics = {}
    for i, name in enumerate(STATE_NAMES):
        rmse_train = np.sqrt(mean_squared_error(Y_train[:, i], Y_pred_train[:, i]))
        rmse_test = np.sqrt(mean_squared_error(Y_test[:, i], Y_pred_test[:, i]))
        r2 = r2_score(Y_test[:, i], Y_pred_test[:, i])
        metrics[name] = {'rmse_train': rmse_train, 'rmse_test': rmse_test, 'r2': r2}
        print(f"  {name:12s}: RMSE_train={rmse_train:.6f}, RMSE_test={rmse_test:.6f}, R²={r2:.6f}")

    # Count non-zero coefficients
    n_active = np.sum(np.abs(Xi) > threshold)
    print(f"\nActive coefficients: {n_active}/{Xi.size}")

    # Print significant coefficients
    print(f"\nSignificant coefficients (|c| > {threshold}):")
    for j, name in enumerate(STATE_NAMES):
        active_idx = np.where(np.abs(Xi[:, j]) > threshold)[0]
        if len(active_idx) > 0:
            print(f"  d({name}):")
            for idx in active_idx:
                print(f"    {labels[idx]:20s} = {Xi[idx, j]:+.8f}")

    return Xi, labels, metrics


def main():
    print("=" * 60)
    print("SINDy Identification: Meijaard 2007 Bicycle")
    print("=" * 60)

    # Try different thresholds
    for threshold in [0.001, 0.005, 0.01, 0.02, 0.05]:
        print(f"\n{'='*60}")
        print(f"Threshold = {threshold}")
        print(f"{'='*60}")
        Xi, labels, metrics = identify_meijaard_sindy(
            'D:/系统辨识作业/sindy_bicycle/meijaard_openloop_data.npz',
            threshold=threshold, alpha=0.0)

        # Save best model
        if threshold == 0.01:
            out_path = 'D:/系统辨识作业/sindy_bicycle/meijaard_sindy.npz'
            np.savez(out_path, coefficients=Xi, labels=labels,
                     state_names=STATE_NAMES, feature_names=labels,
                     v=5.5, dt=1/30)
            print(f"\nSaved to {out_path}")

    # Also identify at v=5.0
    print(f"\n{'='*60}")
    print(f"Identifying at v=5.0 m/s")
    print(f"{'='*60}")
    Xi5, labels5, metrics5 = identify_meijaard_sindy(
        'D:/系统辨识作业/sindy_bicycle/meijaard_openloop_data_v5.npz',
        threshold=0.01, alpha=0.0)

    out_path5 = 'D:/系统辨识作业/sindy_bicycle/meijaard_sindy_v5.npz'
    np.savez(out_path5, coefficients=Xi5, labels=labels5,
             state_names=STATE_NAMES, feature_names=labels5,
             v=5.0, dt=1/30)
    print(f"\nSaved to {out_path5}")


if __name__ == '__main__':
    main()
