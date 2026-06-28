"""
SINDy-based discrete state transition identification for bicycle dynamics.

Implements the SINDy pipeline described in the paper:
1. Dictionary construction (constant, linear, quadratic, mechanism-related cross terms)
2. Sparse regression with STLSQ (Sequentially Thresholded Least Squares)
3. Model selection based on error-sparsity-stability tradeoff
4. Error diagnostics

The identified model maps:
    s_{t+1} = s_t + f_SINDy(s_t, a_t) + ξ_t

where s is the 8D path-tracking state and a is the steering control input.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# State and feature names
STATE_NAMES = ['ey', 'epsi', 'v', 'theta', 'theta_dot', 'k', 'delta', 'delta_dot']
FEATURE_NAMES = STATE_NAMES + ['a']  # state + action


def build_feature_matrix(states, actions):
    """
    Build the augmented feature matrix [states, actions].
    """
    return np.hstack([states, actions])


def build_polynomial_library(X, degree=2):
    """
    Build polynomial library (dictionary) up to given degree.

    Includes:
    - Bias term (constant)
    - Linear terms (all features)
    - Quadratic terms (all pairwise interactions + squares)

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
    degree : int
        Maximum polynomial degree

    Returns
    -------
    Theta : ndarray, shape (n_samples, n_library_terms)
    feature_labels : list of str
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
        labels.append(FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f'x{i}')

    # Degree 2: quadratic (interactions + squares)
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
    """
    Sequentially Thresholded Least Squares (STLSQ).

    The core SINDy algorithm:
    1. Solve least squares
    2. Threshold small coefficients to zero
    3. Re-solve on active set
    4. Repeat until convergence

    Parameters
    ----------
    Theta : ndarray, shape (n_samples, n_library)
        Library matrix
    Y : ndarray, shape (n_samples, n_states)
        Target (delta states)
    threshold : float
        Sparsity threshold
    max_iter : int
        Maximum iterations
    alpha : float
        L2 regularization (Tikhonov)

    Returns
    -------
    Xi : ndarray, shape (n_library, n_states)
        Coefficient matrix
    """
    n_library = Theta.shape[1]
    n_states = Y.shape[1]
    Xi = np.zeros((n_library, n_states))

    for state_idx in range(n_states):
        y = Y[:, state_idx]
        # Initial least squares
        coef = np.linalg.lstsq(Theta, y, rcond=None)[0]

        for iteration in range(max_iter):
            # Threshold small coefficients
            coef_old = coef.copy()
            small = np.abs(coef) < threshold
            coef[small] = 0

            # Re-solve on active set (non-zero coefficients)
            active = ~small
            if not np.any(active):
                break

            Theta_active = Theta[:, active]
            if alpha > 0:
                # Ridge regression on active set
                ATA = Theta_active.T @ Theta_active + alpha * np.eye(Theta_active.shape[1])
                ATy = Theta_active.T @ y
                coef[active] = np.linalg.solve(ATA, ATy)
            else:
                coef[active] = np.linalg.lstsq(Theta_active, y, rcond=None)[0]

            # Check convergence
            if np.allclose(coef, coef_old, atol=1e-10):
                break

        Xi[:, state_idx] = coef

    return Xi


def evaluate_model(Theta, Y, Xi):
    """
    Evaluate model: R² and RMSE per state.
    """
    Y_pred = Theta @ Xi
    r2_scores = []
    rmse_scores = []
    for i in range(Y.shape[1]):
        r2_scores.append(r2_score(Y[:, i], Y_pred[:, i]))
        rmse_scores.append(np.sqrt(mean_squared_error(Y[:, i], Y_pred[:, i])))
    return np.array(r2_scores), np.array(rmse_scores)


def run_sindy_identification(states, actions, next_states,
                              normalize=True,
                              threshold_values=None,
                              alpha=0.1,
                              max_iter=50,
                              degree=2):
    """
    Run the full SINDy identification pipeline.

    Parameters
    ----------
    states : ndarray, shape (n, 8)
    actions : ndarray, shape (n, 1)
    next_states : ndarray, shape (n, 8)
    normalize : bool
    threshold_values : list of float
    alpha : float
    max_iter : int
    degree : int
        Polynomial degree for dictionary

    Returns
    -------
    results : dict
    """
    if threshold_values is None:
        threshold_values = [0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]

    # Step 1: Normalize
    if normalize:
        from data_collector import normalize_state
        states_norm = normalize_state(states)
        next_states_norm = normalize_state(next_states)
        action_scale = max(np.std(actions), 1.0)
        actions_norm = actions / action_scale
    else:
        states_norm = states
        next_states_norm = next_states
        actions_norm = actions
        action_scale = 1.0

    # Step 2: Compute delta states
    delta_states = next_states_norm - states_norm

    # Step 3: Build feature matrix and library
    X = build_feature_matrix(states_norm, actions_norm)
    Theta, lib_labels = build_polynomial_library(X, degree=degree)

    # Step 4: Train/test split
    (Theta_train, Theta_test,
     dY_train, dY_test,
     s_train, s_test,
     X_train, X_test) = train_test_split(
        Theta, delta_states, states_norm, X, test_size=0.2, random_state=42
    )

    # Also get raw next states for test set
    _, _, _, next_s_test = train_test_split(
        Theta, next_states, test_size=0.2, random_state=42
    )

    print("=" * 60)
    print("SINDy Discrete State Transition Identification")
    print("=" * 60)
    print(f"\nTraining samples: {len(Theta_train)}")
    print(f"Test samples: {len(Theta_test)}")
    print(f"Library terms: {Theta.shape[1]} (up to degree {degree})")
    print(f"Target dimensions: {dY_train.shape[1]} (delta state)")

    # Step 5: Threshold sweep
    best_model = None
    best_score = -np.inf
    best_threshold = None
    all_results = []

    print(f"\n{'Threshold':>10} {'Avg R²':>10} {'Avg RMSE':>12} {'N terms':>10} {'Status':>10}")
    print("-" * 60)

    for thresh in threshold_values:
        try:
            # Run STLSQ
            Xi = stlsq(Theta_train, dY_train, threshold=thresh,
                       max_iter=max_iter, alpha=alpha)

            # Evaluate on test set
            r2_scores, rmse_scores = evaluate_model(Theta_test, dY_test, Xi)
            avg_r2 = np.mean(r2_scores)
            avg_rmse = np.mean(rmse_scores)

            # Count non-zero terms
            n_terms = np.sum(np.abs(Xi) > 1e-10)

            # Composite score
            sparsity_bonus = 1.0 / (1.0 + n_terms / 30.0)
            score = avg_r2 * sparsity_bonus

            result = {
                'threshold': thresh,
                'Xi': Xi.copy(),
                'r2_scores': r2_scores,
                'rmse_scores': rmse_scores,
                'avg_r2': avg_r2,
                'avg_rmse': avg_rmse,
                'n_terms': int(n_terms),
                'score': score
            }
            all_results.append(result)

            status = ""
            if score > best_score:
                best_score = score
                best_model = result
                best_threshold = thresh
                status = "<-- BEST"

            print(f"{thresh:>10.3f} {avg_r2:>10.4f} {avg_rmse:>12.6f} {n_terms:>10d} {status:>10}")

        except Exception as e:
            print(f"{thresh:>10.3f} {'FAILED':>10} - {str(e)[:40]}")
            all_results.append({'threshold': thresh, 'error': str(e)})

    print("=" * 60)

    if best_model is not None:
        print(f"\nBest threshold: {best_threshold}")
        print(f"Best avg R²: {best_model['avg_r2']:.4f}")
        print(f"Best avg RMSE: {best_model['avg_rmse']:.6f}")
        print_identified_equations(best_model['Xi'], lib_labels)

    return {
        'best_model': best_model,
        'best_threshold': best_threshold,
        'best_score': best_score,
        'all_results': all_results,
        'lib_labels': lib_labels,
        'action_scale': action_scale,
        'normalize': normalize,
        'Theta_test': Theta_test,
        'dY_test': dY_test,
        's_test': s_test,
        'X_test': X_test,
        'next_s_test': next_s_test,
        'Theta_all': Theta,
        'dY_all': delta_states,
        's_all': states_norm
    }


def print_identified_equations(Xi, lib_labels):
    """
    Print the identified equations in a readable format.
    """
    print("\n" + "=" * 60)
    print("Identified Discrete State Transition Equations")
    print("=" * 60)

    n_states = Xi.shape[1]
    for i in range(n_states):
        coefs = Xi[:, i]
        terms = []
        for j, c in enumerate(coefs):
            if abs(c) > 1e-6:
                label = lib_labels[j] if j < len(lib_labels) else f"f{j}"
                terms.append(f"{c:+.4f}*{label}")

        if terms:
            eq_str = f"  Δ{STATE_NAMES[i]} = " + " ".join(terms[:10])  # Show top 10
            if len(terms) > 10:
                eq_str += f"  ... (+{len(terms)-10} more)"
        else:
            eq_str = f"  Δ{STATE_NAMES[i]} = 0"
        print(eq_str)

    print("\n" + "=" * 60)
    print("Note: Δs = s_{t+1} - s_t (state change per time step)")
    print("Equations in normalized space" if True else "")


def diagnose_errors(results):
    """
    Error diagnostics as described in the paper.
    """
    best = results['best_model']
    Xi = best['Xi']
    Theta_test = results['Theta_test']
    dY_test = results['dY_test']

    dY_pred = Theta_test @ Xi

    print("\n" + "=" * 60)
    print("Error Diagnostics")
    print("=" * 60)

    print(f"\n{'State':>15} {'RMSE':>12} {'Max Error':>12} {'R²':>10}")
    print("-" * 55)

    for i in range(dY_test.shape[1]):
        rmse = np.sqrt(mean_squared_error(dY_test[:, i], dY_pred[:, i]))
        max_err = np.max(np.abs(dY_test[:, i] - dY_pred[:, i]))
        r2 = r2_score(dY_test[:, i], dY_pred[:, i])
        print(f"{STATE_NAMES[i]:>15} {rmse:>12.6f} {max_err:>12.6f} {r2:>10.4f}")

    # Error by state magnitude
    print("\nError by State Magnitude (normalized):")
    s_test = results['s_test']
    for i, name in enumerate(STATE_NAMES):
        low_mask = np.abs(s_test[:, i]) < 0.3
        high_mask = np.abs(s_test[:, i]) >= 0.3
        if np.sum(low_mask) > 0 and np.sum(high_mask) > 0:
            err_low = np.sqrt(mean_squared_error(dY_test[low_mask, i], dY_pred[low_mask, i]))
            err_high = np.sqrt(mean_squared_error(dY_test[high_mask, i], dY_pred[high_mask, i]))
            print(f"  {name:>15}: low-mag={err_low:.6f}, high-mag={err_high:.6f}")


if __name__ == '__main__':
    print("Loading data...")
    try:
        data = np.load('bicycle_data.npz')
        states = data['states']
        actions = data['actions']
        next_states = data['next_states']
        print(f"Loaded {len(states)} samples")
    except FileNotFoundError:
        print("Generating data...")
        from data_collector import BicycleDataCollector
        collector = BicycleDataCollector(dt=1/30, seed=42)
        dataset = collector.collect_dataset(n_episodes_per_scenario=30, t_episode=8.0)
        states = dataset['states']
        actions = dataset['actions']
        next_states = dataset['next_states']

    results = run_sindy_identification(
        states, actions, next_states,
        normalize=True,
        threshold_values=[0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3],
        alpha=0.1,
        degree=2
    )

    if results['best_model'] is not None:
        diagnose_errors(results)
