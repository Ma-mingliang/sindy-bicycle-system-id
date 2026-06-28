"""Test physics-informed SINDy model in Gymnasium environment."""

import os
import sys
import math
import json
import numpy as np
import scipy.linalg as la

sys.path.insert(0, os.path.dirname(__file__))

from path_tracking_env import PathTrackingEnv


def normalize_state(raw_state):
    scales = np.array([10.0, 1.57, 5.0, 1.57, 10.0, 1/8.0, 0.785, 3.0])
    normed = raw_state.copy()
    for i in range(8):
        normed[i] = np.clip(normed[i], -scales[i]*2, scales[i]*2) / scales[i]
    return normed.astype(np.float32)


def build_physics_informed_library(state_norm, action_norm):
    x = np.concatenate([state_norm, action_norm])
    n = len(x)
    terms = [1.0]
    terms.extend(x)
    for i in range(n):
        for j in range(i, n):
            terms.append(x[i] * x[j])

    theta_norm = state_norm[3]
    v_norm = state_norm[2]
    delta_norm = state_norm[6]

    terms.append(theta_norm * v_norm**2)
    terms.append(delta_norm * v_norm**2)
    terms.append(delta_norm * theta_norm * v_norm)
    terms.append(theta_norm**3)

    return np.array(terms, dtype=np.float64)


class PhysicsInformedSINDyModel:
    def __init__(self, Xi, action_scale=1.41):
        self.Xi = Xi
        self.action_scale = action_scale

    def predict_next_state(self, state_raw, action_raw):
        state_norm = normalize_state(state_raw)
        action_norm = np.atleast_1d(float(np.asarray(action_raw).ravel()[0] / self.action_scale))
        lib = build_physics_informed_library(state_norm, action_norm)
        delta_norm = lib @ self.Xi
        scales = np.array([10.0, 1.57, 5.0, 1.57, 10.0, 1/8.0, 0.785, 3.0])
        return state_raw + delta_norm * scales


def compute_lqr_gains(A, B, Q, R):
    P = la.solve_continuous_are(A, B, Q, R)
    return (np.linalg.inv(R) @ B.T @ P).flatten()


def run_gymnasium_test(model, num_seeds=50, max_steps=2000, K_lqr=None):
    """Test physics-informed SINDy in Gymnasium environment."""
    returns, lengths, ets = [], [], []

    for seed in range(num_seeds):
        env = PathTrackingEnv(
            dynamics_model='linearized',
            max_episode_steps=max_steps,
            speed=3.0,
            dt=1/30,
        )
        obs, info = env.reset(seed=seed)

        K = K_lqr if K_lqr is not None else env.K_lqr

        ep_return = 0.0
        terminated = False
        truncated = False

        for step in range(max_steps):
            fx = env._x + env.wheelbase * math.cos(env._heading)
            fy = env._y + env.wheelbase * math.sin(env._heading)
            path_info = env.path.get_closest_point(fx, fy, env._heading)
            ey = path_info['lateral_error']
            epsi = path_info['course_error_angle']
            k = path_info['curvature']

            from stanley_controller import steady_state_calculation
            delta_stanley = -(math.atan(0.6 * ey / max(env._v, 0.5)) + 0.4 * epsi)
            target_roll = steady_state_calculation(delta_stanley, env._v)
            x_lqr = np.array([env._theta - target_roll, env._theta_dot, env._delta, env._delta_dot])
            u_total = float(-K @ x_lqr)

            full_state = np.array([ey, epsi, env._v, env._theta, env._theta_dot, k, env._delta, env._delta_dot])
            next_state = model.predict_next_state(full_state, u_total)

            env._theta = float(np.clip(next_state[3], -1.57, 1.57))
            env._theta_dot = next_state[4]
            env._delta = float(np.clip(next_state[6], -0.785, 0.785))
            env._delta_dot = next_state[7]
            env._v = next_state[2]

            env._heading += env._v * (-env._delta / env.wheelbase) * env.dt
            env._x += env._v * math.cos(env._heading) * env.dt
            env._y += env._v * math.sin(env._heading) * env.dt

            terminated = abs(env._theta) > math.pi / 4
            reward = -abs(ey) - 0.1 * abs(env._theta)
            if terminated:
                reward = -10.0

            ep_return += reward
            env._step_count += 1
            truncated = env._step_count >= max_steps

            if terminated or truncated:
                break

        env.close()
        returns.append(ep_return)
        lengths.append(step + 1)
        ets.append(terminated)

    return {
        'return_mean': float(np.mean(returns)),
        'return_std': float(np.std(returns)),
        'length_mean': float(np.mean(lengths)),
        'length_std': float(np.std(lengths)),
        'et_rate': float(np.mean(ets)),
    }


def run_linearized_baseline(num_seeds=50, max_steps=2000):
    """Run linearized Whipple baseline for comparison."""
    returns, lengths, ets = [], [], []

    for seed in range(num_seeds):
        env = PathTrackingEnv(
            dynamics_model='linearized',
            max_episode_steps=max_steps,
            speed=3.0,
            dt=1/30,
        )
        obs, info = env.reset(seed=seed)

        ep_return = 0.0
        terminated = False
        truncated = False

        for step in range(max_steps):
            action = env.action_space.sample() * 0
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward

            if terminated or truncated:
                break

        env.close()
        returns.append(ep_return)
        lengths.append(step + 1)
        ets.append(terminated)

    return {
        'return_mean': float(np.mean(returns)),
        'return_std': float(np.std(returns)),
        'length_mean': float(np.mean(lengths)),
        'length_std': float(np.std(lengths)),
        'et_rate': float(np.mean(ets)),
    }


def run_hybrid_gymnasium_test(sindy_model, num_seeds=50, max_steps=2000, K_lqr=None):
    """Test hybrid model (SINDy gravity + true damping) in Gymnasium environment."""
    returns, lengths, ets = [], [], []

    # True damping coefficient in normalized space
    dt = 1/30
    g, h = 9.81, 0.8
    scale_theta_dot = 10.0
    scale_theta = 1.57
    true_damping = 2.0
    true_gravity_coeff = (g / h) * dt * scale_theta / scale_theta_dot
    true_damping_coeff = -true_damping * dt * scale_theta / scale_theta_dot

    for seed in range(num_seeds):
        env = PathTrackingEnv(
            dynamics_model='linearized',
            max_episode_steps=max_steps,
            speed=3.0,
            dt=dt,
        )
        obs, info = env.reset(seed=seed)

        K = K_lqr if K_lqr is not None else env.K_lqr

        ep_return = 0.0
        terminated = False
        truncated = False

        for step in range(max_steps):
            fx = env._x + env.wheelbase * math.cos(env._heading)
            fy = env._y + env.wheelbase * math.sin(env._heading)
            path_info = env.path.get_closest_point(fx, fy, env._heading)
            ey = path_info['lateral_error']
            epsi = path_info['course_error_angle']
            k = path_info['curvature']

            from stanley_controller import steady_state_calculation
            delta_stanley = -(math.atan(0.6 * ey / max(env._v, 0.5)) + 0.4 * epsi)
            target_roll = steady_state_calculation(delta_stanley, env._v)
            x_lqr = np.array([env._theta - target_roll, env._theta_dot, env._delta, env._delta_dot])
            u_total = float(-K @ x_lqr)

            full_state = np.array([ey, epsi, env._v, env._theta, env._theta_dot, k, env._delta, env._delta_dot])

            # Use SINDy for most dimensions, but override theta_dot with true damping
            state_norm = normalize_state(full_state)
            action_norm = np.atleast_1d(float(u_total / 1.41))
            lib = build_physics_informed_library(state_norm, action_norm)
            delta_norm = lib @ sindy_model.Xi

            # Override theta_dot dynamics
            theta_norm = state_norm[3]
            theta_dot_norm = state_norm[4]
            corrected_td = true_gravity_coeff * theta_norm + true_damping_coeff * theta_dot_norm
            # Add SINDy's nonlinear terms (last 4 physics-informed terms)
            corrected_td += lib[-4:] @ sindy_model.Xi[-4:, 4]
            delta_norm[4] = corrected_td

            scales = np.array([10.0, 1.57, 5.0, 1.57, 10.0, 1/8.0, 0.785, 3.0])
            next_state = full_state + delta_norm * scales

            env._theta = float(np.clip(next_state[3], -1.57, 1.57))
            env._theta_dot = next_state[4]
            env._delta = float(np.clip(next_state[6], -0.785, 0.785))
            env._delta_dot = next_state[7]
            env._v = next_state[2]

            env._heading += env._v * (-env._delta / env.wheelbase) * env.dt
            env._x += env._v * math.cos(env._heading) * env.dt
            env._y += env._v * math.sin(env._heading) * env.dt

            terminated = abs(env._theta) > math.pi / 4
            reward = -abs(ey) - 0.1 * abs(env._theta)
            if terminated:
                reward = -10.0

            ep_return += reward
            env._step_count += 1
            truncated = env._step_count >= max_steps

            if terminated or truncated:
                break

        env.close()
        returns.append(ep_return)
        lengths.append(step + 1)
        ets.append(terminated)

    return {
        'return_mean': float(np.mean(returns)),
        'return_std': float(np.std(returns)),
        'length_mean': float(np.mean(lengths)),
        'length_std': float(np.std(lengths)),
        'et_rate': float(np.mean(ets)),
    }


def main():
    out_dir = r'D:/系统辨识作业/sindy_bicycle/runs/physics_informed_dynamics'

    # Load physics-informed SINDy model
    model_path = os.path.join(out_dir, 'physics_informed_sindy.npz')
    data = np.load(model_path, allow_pickle=True)
    Xi = data['coefficients']
    action_scale = float(data['action_scale'])

    sindy_model = PhysicsInformedSINDyModel(Xi, action_scale=action_scale)
    print(f"Loaded physics-informed SINDy model")

    # Physical parameters
    speed = 3.0
    g, h, wheelbase = 9.81, 0.8, 1.0
    omega_n, zeta, trail = 5.0, 0.5, 0.15
    B = np.array([[0], [0], [0], [1.0]])
    Q = np.diag([1000, 100, 10, 1])
    R = np.array([[0.2]])

    # True LQR gains
    A_true = np.array([
        [0, 1, 0, 0],
        [g/h, -2.0, -speed**2/(h*wheelbase), 0],
        [0, 0, 0, 1],
        [omega_n**2*trail*speed/wheelbase, 0, -omega_n**2, -2*zeta*omega_n]
    ])
    K_true = compute_lqr_gains(A_true, B, Q, R)
    print(f"True LQR gains:  K = [{', '.join(f'{k:.1f}' for k in K_true)}]")

    # SINDy-specific LQR gains
    scale_td, scale_t, dt = 10.0, 1.57, 1/30
    g_h_sindy = 0.030923 * scale_td / scale_t / dt
    damp_sindy = 0.059057 * scale_td / scale_t / dt

    A_sindy = np.array([
        [0, 1, 0, 0],
        [g_h_sindy, -damp_sindy, -speed**2/(h*wheelbase), 0],
        [0, 0, 0, 1],
        [omega_n**2*trail*speed/wheelbase, 0, -omega_n**2, -2*zeta*omega_n]
    ])
    K_sindy = compute_lqr_gains(A_sindy, B, Q, R)
    print(f"SINDy LQR gains: K = [{', '.join(f'{k:.1f}' for k in K_sindy)}]")

    # Test 1: Linearized baseline
    print("\n" + "="*60)
    print("Test 1: Linearized Whipple baseline")
    print("="*60)
    r1 = run_linearized_baseline(num_seeds=50, max_steps=2000)
    print(f"Length={r1['length_mean']:.1f} +/- {r1['length_std']:.1f}, "
          f"Return={r1['return_mean']:.2f}, ET={r1['et_rate']:.0%}")

    # Test 2: SINDy model + true LQR (mismatched)
    print("\n" + "="*60)
    print("Test 2: SINDy dynamics + true LQR (mismatched)")
    print("="*60)
    r2 = run_gymnasium_test(sindy_model, num_seeds=50, max_steps=2000, K_lqr=K_true)
    print(f"Length={r2['length_mean']:.1f} +/- {r2['length_std']:.1f}, "
          f"Return={r2['return_mean']:.2f}, ET={r2['et_rate']:.0%}")

    # Test 3: SINDy model + SINDy LQR (matched)
    print("\n" + "="*60)
    print("Test 3: SINDy dynamics + SINDy LQR (matched)")
    print("="*60)
    r3 = run_gymnasium_test(sindy_model, num_seeds=50, max_steps=2000, K_lqr=K_sindy)
    print(f"Length={r3['length_mean']:.1f} +/- {r3['length_std']:.1f}, "
          f"Return={r3['return_mean']:.2f}, ET={r3['et_rate']:.0%}")

    # Test 4: Hybrid model (SINDy gravity + true damping) + true LQR
    print("\n" + "="*60)
    print("Test 4: Hybrid (SINDy gravity + true damping) + true LQR")
    print("="*60)
    r4 = run_hybrid_gymnasium_test(sindy_model, num_seeds=50, max_steps=2000, K_lqr=K_true)
    print(f"Length={r4['length_mean']:.1f} +/- {r4['length_std']:.1f}, "
          f"Return={r4['return_mean']:.2f}, ET={r4['et_rate']:.0%}")

    # Save results
    results = {
        'linearized_baseline': r1,
        'sindy_mismatched_lqr': r2,
        'sindy_matched_lqr': r3,
        'hybrid_true_lqr': r4,
    }
    results_path = os.path.join(out_dir, 'gymnasium_test_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"{'Model':<40} {'Length':>10} {'Return':>10} {'ET':>6}")
    print("-" * 66)
    print(f"{'Linearized Whipple (baseline)':<40} {r1['length_mean']:>10.1f} {r1['return_mean']:>10.2f} {r1['et_rate']:>5.0%}")
    print(f"{'SINDy + true LQR (mismatched)':<40} {r2['length_mean']:>10.1f} {r2['return_mean']:>10.2f} {r2['et_rate']:>5.0%}")
    print(f"{'SINDy + SINDy LQR (matched)':<40} {r3['length_mean']:>10.1f} {r3['return_mean']:>10.2f} {r3['et_rate']:>5.0%}")
    print(f"{'Hybrid + true LQR':<40} {r4['length_mean']:>10.1f} {r4['return_mean']:>10.2f} {r4['et_rate']:>5.0%}")


if __name__ == '__main__':
    main()
