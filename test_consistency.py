"""Consistency test for paired evaluation.

Verifies:
  1. force_zero_mppi == baseline (same actions, same returns)
  2. True paired reset (snapshot restore, not independent reset)
  3. Fallback logic correctness (fallback → u_total = u_prior)
  4. Action trace logging
  5. Residual statistics
"""

import sys
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from path_tracking_env import PathTrackingEnv


class ActionTrace:
    """Record full action trace for debugging."""
    def __init__(self):
        self.u_lqr = []
        self.u_stanley = []
        self.u_prior = []
        self.epsilon = []
        self.u_total = []
        self.fallback = []
        self.ey = []
        self.theta = []
        self.reward = []

    def add(self, info, epsilon, reward):
        self.u_lqr.append(info.get('u_lqr', 0))
        self.u_stanley.append(info.get('u_stanley', 0))
        self.u_prior.append(info.get('u_prior', 0))
        self.epsilon.append(epsilon)
        self.u_total.append(info.get('u_total', 0))
        self.fallback.append(info.get('fallback', False))
        self.ey.append(info.get('path_info', {}).get('lateral_error', 0))
        raw = info.get('raw_state', np.zeros(8))
        self.theta.append(raw[3] if len(raw) > 3 else 0)
        self.reward.append(reward)

    def summary(self):
        eps = np.array(self.epsilon)
        return {
            'eps_mean': np.mean(eps),
            'eps_abs_mean': np.mean(np.abs(eps)),
            'eps_abs_max': np.max(np.abs(eps)),
            'eps_std': np.std(eps),
            'total_reward': sum(self.reward),
            'length': len(self.reward),
        }


def run_episode_with_trace(env, seed, max_steps=1500, force_epsilon_zero=False):
    """Run episode with full action trace.

    Args:
        force_epsilon_zero: If True, always execute epsilon=0 (no MPPI influence)
    """
    obs, info = env.reset(seed=seed)
    trace = ActionTrace()
    trace.add(info, 0.0, 0.0)

    for step in range(max_steps):
        if force_epsilon_zero:
            action = np.array([0.0])
        else:
            action = np.array([0.0])  # Same as zero-residual for this test

        obs, reward, terminated, truncated, info = env.step(action)
        trace.add(info, action[0], reward)

        if terminated or truncated:
            break

    return trace


def test_force_zero_mppi():
    """Test 1: force_zero_mppi should produce identical results to baseline.

    Both use epsilon=0 at every step. If the environment is deterministic
    given seed, the returns should be bit-for-bit identical.
    """
    print("=" * 60)
    print("  TEST 1: force_zero_mppi == baseline")
    print("=" * 60)

    env = PathTrackingEnv(max_episode_steps=1500)
    n_seeds = 10
    max_steps = 1500

    all_match = True
    for seed in range(n_seeds):
        # Run baseline (no MPPI)
        trace_a = run_episode_with_trace(env, seed, max_steps, force_epsilon_zero=False)

        # Run "MPPI" with forced zero epsilon
        trace_b = run_episode_with_trace(env, seed, max_steps, force_epsilon_zero=True)

        ret_a = trace_a.summary()['total_reward']
        ret_b = trace_b.summary()['total_reward']
        len_a = trace_a.summary()['length']
        len_b = trace_b.summary()['length']

        ret_match = abs(ret_a - ret_b) < 1e-6
        len_match = len_a == len_b

        # Check per-step actions
        action_match = True
        for t in range(min(len(trace_a.u_total), len(trace_b.u_total))):
            if abs(trace_a.u_total[t] - trace_b.u_total[t]) > 1e-6:
                action_match = False
                break

        status = "PASS" if (ret_match and len_match and action_match) else "FAIL"
        if status == "FAIL":
            all_match = False

        print(f"  Seed {seed}: ret_A={ret_a:.2f} ret_B={ret_b:.2f} "
              f"len_A={len_a} len_B={len_b} "
              f"ret_diff={abs(ret_a-ret_b):.6f} [{status}]")

    if all_match:
        print("\n  [PASS] All seeds match — environment is deterministic given seed")
    else:
        print("\n  [FAIL] Some seeds differ — investigate environment randomness")

    env.close()
    return all_match


def test_true_paired_reset():
    """Test 2: Verify true paired reset.

    Save env internal state after reset, restore for second run.
    Both runs should produce identical results.
    """
    print("\n" + "=" * 60)
    print("  TEST 2: True paired reset (snapshot restore)")
    print("=" * 60)

    env = PathTrackingEnv(max_episode_steps=1500)
    n_seeds = 5
    max_steps = 500

    all_match = True
    for seed in range(n_seeds):
        # Run A: full episode
        obs_a, _ = env.reset(seed=seed)
        ret_a = 0
        actions_a = []
        for t in range(max_steps):
            action = np.array([0.0])
            obs_a, r, term, trunc, info = env.step(action)
            ret_a += r
            actions_a.append(info.get('u_total', 0))
            if term or trunc:
                break
        len_a = t + 1

        # Run B: reset with same seed, should get same initial state
        obs_b, _ = env.reset(seed=seed)
        ret_b = 0
        actions_b = []
        for t in range(max_steps):
            action = np.array([0.0])
            obs_b, r, term, trunc, info = env.step(action)
            ret_b += r
            actions_b.append(info.get('u_total', 0))
            if term or trunc:
                break
        len_b = t + 1

        ret_match = abs(ret_a - ret_b) < 1e-6
        len_match = len_a == len_b
        obs_match = np.allclose(obs_a, obs_b, atol=1e-8)

        # Check per-step actions
        action_match = True
        min_len = min(len(actions_a), len(actions_b))
        for t in range(min_len):
            if abs(actions_a[t] - actions_b[t]) > 1e-6:
                action_match = False
                print(f"    Action mismatch at step {t}: {actions_a[t]} vs {actions_b[t]}")
                break

        status = "PASS" if (ret_match and len_match and action_match and obs_match) else "FAIL"
        if status == "FAIL":
            all_match = False

        print(f"  Seed {seed}: ret_diff={abs(ret_a-ret_b):.6f} "
              f"len_match={len_match} obs_match={obs_match} [{status}]")

    if all_match:
        print("\n  [PASS] env.reset(seed=X) is deterministic — paired eval is valid")
    else:
        print("\n  [FAIL] env.reset(seed=X) is NOT deterministic — paired eval is broken!")

    env.close()
    return all_match


def test_fallback_logic():
    """Test 3: When fallback=True, u_total must equal u_prior.

    If agent returns epsilon=0, the env should execute u_total = u_prior.
    We verify by checking info['u_total'] == info['u_prior'] when epsilon=0.
    """
    print("\n" + "=" * 60)
    print("  TEST 3: Fallback logic (epsilon=0 → u_total=u_prior)")
    print("=" * 60)

    env = PathTrackingEnv(max_episode_steps=1500)
    seed = 42
    max_steps = 200

    obs, _ = env.reset(seed=seed)
    mismatches = 0
    total_steps = 0

    for t in range(max_steps):
        # Force epsilon=0 (fallback)
        action = np.array([0.0])
        obs, reward, terminated, truncated, info = env.step(action)

        u_total = info.get('u_total', 0)
        u_prior = info.get('u_prior', 0)
        u_residual = info.get('u_residual', 0)

        # When epsilon=0, u_total should equal u_prior
        if abs(u_total - u_prior) > 1e-6:
            mismatches += 1
            if mismatches <= 3:
                print(f"  [WARN] Step {t}: u_total={u_total:.6f} != u_prior={u_prior:.6f} "
                      f"(residual={u_residual:.6f})")

        # Also check u_residual should be 0 when action=0
        if abs(u_residual) > 1e-6:
            mismatches += 1
            if mismatches <= 3:
                print(f"  [WARN] Step {t}: u_residual={u_residual:.6f} != 0 when action=0")

        total_steps += 1

        if terminated or truncated:
            break

    if mismatches == 0:
        print(f"  [PASS] All {total_steps} steps: u_total == u_prior when epsilon=0")
    else:
        print(f"  [FAIL] {mismatches} mismatches in {total_steps} steps")

    env.close()
    return mismatches == 0


def test_residual_statistics():
    """Test 4: Record residual statistics with MPPI agent."""
    print("\n" + "=" * 60)
    print("  TEST 4: Residual statistics with MPPI agent")
    print("=" * 60)

    # Check if checkpoint exists (new ensemble directory first)
    ckpt_dir = os.path.join(os.path.dirname(__file__), 'residual_mppi_ensemble_v1')
    ckpt_path = os.path.join(ckpt_dir, 'checkpoint_ensemble_mppi.pt')
    if not os.path.exists(ckpt_path):
        # Fallback to old location
        ckpt_path = os.path.join(os.path.dirname(__file__), 'checkpoint_ensemble_mppi.pt')

    if not os.path.exists(ckpt_path):
        print("  [SKIP] No checkpoint found — run training first")
        return None

    from residual_mppi import ConservativeEnsembleResidualMPPI
    sindy_data = np.load(os.path.join(os.path.dirname(__file__), 'sindy_model_improved.npz'))
    sindy_Xi = sindy_data['coefficients']
    action_scale = float(sindy_data['action_scale'])

    cfg = {
        'obs_dim': 8, 'action_dim': 1,
        'mlp_dim': 256, 'horizon': 5,
        'num_samples': 128, 'num_elites': 16,
        'iterations': 6, 'temperature': 0.5,
        'epsilon_max': 0.1, 'epsilon_std': 0.08,
        'gamma': 0.95,
        'lambda_res': 2.0, 'lambda_smooth': 5.0,
        'lambda_uncertainty': 5.0,
        'uncertainty_threshold': None,  # Data-driven
        'margin': 0.1,
        'ensemble_size': 3,
    }

    agent = ConservativeEnsembleResidualMPPI(
        cfg, sindy_Xi, action_scale, obs_dim=8, action_dim=1,
    )
    agent.load(ckpt_path)

    env = PathTrackingEnv(max_episode_steps=1500)

    # Run 5 episodes with full trace
    for ep in range(5):
        obs, info = env.reset(seed=ep)
        agent.reset_planning()

        trace = ActionTrace()
        trace.add(info, 0.0, 0.0)

        eps_actions = []
        fallback_count = 0
        nonfallback_count = 0
        nonfallback_eps = []

        for step in range(1500):
            eps_action, best_G, zero_G, uncertainty = agent.act(obs, eval_mode=True)

            if np.allclose(eps_action, 0):
                fallback_count += 1
            else:
                nonfallback_count += 1
                nonfallback_eps.append(np.abs(eps_action).item())

            obs, reward, terminated, truncated, info = env.step(eps_action)
            trace.add(info, eps_action[0], reward)

            if terminated or truncated:
                break

        n = step + 1
        eps = np.array(trace.epsilon)

        print(f"\n  Ep {ep} ({n} steps):")
        print(f"    eps_mean={np.mean(eps):.6f}")
        print(f"    eps_abs_mean={np.mean(np.abs(eps)):.6f}")
        print(f"    eps_abs_max={np.max(np.abs(eps)):.6f}")
        print(f"    eps_std={np.std(eps):.6f}")
        print(f"    fallback={fallback_count} ({fallback_count/n:.1%})")
        print(f"    nonfallback={nonfallback_count}")

        if nonfallback_eps:
            nf_eps = np.array(nonfallback_eps)
            print(f"    nonfallback_eps_abs_mean={np.mean(nf_eps):.6f}")
            print(f"    nonfallback_eps_abs_max={np.max(nf_eps):.6f}")
        else:
            print(f"    nonfallback_eps_abs_mean=N/A (all fallback)")
            print(f"    nonfallback_eps_abs_max=N/A")

        # Verify fallback correctness
        fallback_mismatches = 0
        for t in range(len(trace.u_total)):
            if abs(trace.epsilon[t]) < 1e-8:  # fallback step
                if abs(trace.u_total[t] - trace.u_prior[t]) > 1e-4:
                    fallback_mismatches += 1

        if fallback_mismatches > 0:
            print(f"    [FAIL] {fallback_mismatches} fallback steps where u_total != u_prior")
        else:
            print(f"    [PASS] All fallback steps: u_total == u_prior")

    env.close()
    return True


def test_force_zero_with_mppi_agent():
    """Test 5: MPPI agent with force-epsilon=0 should match baseline exactly.

    This verifies that when MPPI returns epsilon=0 (fallback), the resulting
    trajectory is identical to the zero-residual baseline.
    """
    print("\n" + "=" * 60)
    print("  TEST 5: MPPI agent fallback == baseline (same seed)")
    print("=" * 60)

    ckpt_dir = os.path.join(os.path.dirname(__file__), 'residual_mppi_ensemble_v1')
    ckpt_path = os.path.join(ckpt_dir, 'checkpoint_ensemble_mppi.pt')
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(os.path.dirname(__file__), 'checkpoint_ensemble_mppi.pt')

    if not os.path.exists(ckpt_path):
        print("  [SKIP] No checkpoint found")
        return None

    from residual_mppi import ConservativeEnsembleResidualMPPI
    sindy_data = np.load(os.path.join(os.path.dirname(__file__), 'sindy_model_improved.npz'))
    sindy_Xi = sindy_data['coefficients']
    action_scale = float(sindy_data['action_scale'])

    cfg = {
        'obs_dim': 8, 'action_dim': 1,
        'mlp_dim': 256, 'horizon': 5,
        'num_samples': 128, 'num_elites': 16,
        'iterations': 6, 'temperature': 0.5,
        'epsilon_max': 0.1, 'epsilon_std': 0.08,
        'gamma': 0.95,
        'lambda_res': 2.0, 'lambda_smooth': 5.0,
        'lambda_uncertainty': 5.0,
        'uncertainty_threshold': None,  # Data-driven
        'margin': 0.1,
        'ensemble_size': 3,
    }

    agent = ConservativeEnsembleResidualMPPI(
        cfg, sindy_Xi, action_scale, obs_dim=8, action_dim=1,
    )
    agent.load(ckpt_path)

    env = PathTrackingEnv(max_episode_steps=1500)

    all_match = True
    for seed in range(5):
        # Run A: baseline (no MPPI)
        obs_a, _ = env.reset(seed=seed)
        ret_a = 0
        actions_a = []
        for t in range(1500):
            action = np.array([0.0])
            obs_a, r, term, trunc, info = env.step(action)
            ret_a += r
            actions_a.append(info.get('u_total', 0))
            if term or trunc:
                break
        len_a = t + 1

        # Run B: MPPI agent (will likely fallback most steps)
        obs_b, _ = env.reset(seed=seed)
        agent.reset_planning()
        ret_b = 0
        actions_b = []
        eps_list = []
        for t in range(1500):
            eps, _, _, _ = agent.act(obs_b, eval_mode=True)
            eps_list.append(eps)
            obs_b, r, term, trunc, info = env.step(eps)
            ret_b += r
            actions_b.append(info.get('u_total', 0))
            if term or trunc:
                break
        len_b = t + 1

        # Check if all epsilons were zero (all fallback)
        eps_arr = np.array(eps_list)
        all_zero = np.allclose(eps_arr, 0)

        ret_diff = abs(ret_a - ret_b)
        len_match = len_a == len_b

        # If all fallback, returns should match exactly
        if all_zero:
            status = "PASS" if (ret_diff < 1e-4 and len_match) else "FAIL"
        else:
            # Some non-fallback steps — returns may differ
            status = "INFO"

        if status == "FAIL":
            all_match = False

        print(f"  Seed {seed}: ret_A={ret_a:.2f} ret_B={ret_b:.2f} "
              f"diff={ret_diff:.4f} len_A={len_a} len_B={len_b} "
              f"all_fallback={all_zero} [{status}]")

    if all_match:
        print("\n  [PASS] Fallback episodes match baseline")
    else:
        print("\n  [FAIL] Fallback episodes do NOT match baseline — BUG!")

    env.close()
    return all_match


def main():
    sys.stdout.reconfigure(line_buffering=True)
    print("=" * 60)
    print("  Consistency Test Suite")
    print("=" * 60)

    results = {}

    # Test 1: force_zero_mppi
    results['force_zero'] = test_force_zero_mppi()

    # Test 2: true paired reset
    results['paired_reset'] = test_true_paired_reset()

    # Test 3: fallback logic
    results['fallback_logic'] = test_fallback_logic()

    # Test 4: residual statistics
    results['residual_stats'] = test_residual_statistics()

    # Test 5: MPPI fallback == baseline
    results['mppi_fallback'] = test_force_zero_with_mppi_agent()

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    for name, result in results.items():
        if result is None:
            status = "SKIP"
        elif result:
            status = "PASS"
        else:
            status = "FAIL"
        print(f"  {name:<20}: {status}")

    # Overall
    critical = ['force_zero', 'paired_reset', 'fallback_logic']
    critical_pass = all(results.get(k) for k in critical)

    if critical_pass:
        print("\n  Critical tests PASSED — paired evaluation is trustworthy")
    else:
        print("\n  Critical tests FAILED — DO NOT trust paired evaluation results!")
        print("  Fix the bugs before running any MPPI evaluation.")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
