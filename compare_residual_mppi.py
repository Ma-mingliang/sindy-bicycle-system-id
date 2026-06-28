"""Paired comparison: Zero-residual baseline vs Conservative Ensemble Residual-MPPI.

50 episodes with same seeds, detailed metrics including uncertainty, fallback reason analysis.
"""

import sys
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
import numpy as np
import torch
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from path_tracking_env import PathTrackingEnv
from residual_mppi import ConservativeEnsembleResidualMPPI

CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'residual_mppi_ensemble_v1')


def run_episode_simple(env, agent, seed, max_steps=1500, use_mppi=True):
    """Run one episode, collect metrics including fallback reasons."""
    obs, _ = env.reset(seed=seed)

    if use_mppi and agent is not None:
        agent.reset_planning()

    total_r = 0
    ey_list, theta_list = [], []
    u_list, eps_list = [], []
    fallback_count = 0
    nonfallback_count = 0
    uncertainty_list = []
    predicted_adv_list = []

    for step in range(max_steps):
        if use_mppi and agent is not None:
            eps_action, best_G, zero_G, uncertainty = agent.act(obs, eval_mode=True)
            predicted_adv = best_G - zero_G
            predicted_adv_list.append(predicted_adv)
            uncertainty_list.append(uncertainty)

            if np.allclose(eps_action, 0):
                fallback_count += 1
            else:
                nonfallback_count += 1
            action = eps_action
        else:
            action = np.array([0.0])

        obs, reward, terminated, truncated, info = env.step(action)
        total_r += reward

        pi = info.get('path_info', {})
        ey_list.append(pi.get('lateral_error', 0))

        raw = info.get('raw_state', np.zeros(8))
        theta_list.append(raw[3] if len(raw) > 3 else 0)

        u_list.append(info.get('u_total', 0))
        eps_list.append(info.get('u_residual', 0) if use_mppi else 0)

        if terminated or truncated:
            break

    n = step + 1
    fb_reasons = agent.get_fb_reasons() if (use_mppi and agent is not None) else {}

    return {
        'return': total_r,
        'length': n,
        'ey_rms': np.sqrt(np.mean(np.array(ey_list)**2)),
        'theta_rms': np.sqrt(np.mean(np.array(theta_list)**2)),
        'u_abs_mean': np.mean(np.abs(u_list)),
        'eps_abs_mean': np.mean(np.abs(eps_list)),
        'fallback_count': fallback_count,
        'fallback_rate': fallback_count / n,
        'nonfallback_count': nonfallback_count,
        'uncertainty_mean': np.mean(uncertainty_list) if uncertainty_list else 0.0,
        'predicted_advantage_mean': np.mean(predicted_adv_list) if predicted_adv_list else 0.0,
        'fb_reasons': fb_reasons,
    }


def main():
    sys.stdout.reconfigure(line_buffering=True)
    print("=" * 70)
    print("  Conservative Ensemble Residual-MPPI vs Zero-residual")
    print("=" * 70)

    # Load SINDy model
    sindy_data = np.load(os.path.join(os.path.dirname(__file__), 'sindy_model_improved.npz'))
    sindy_Xi = sindy_data['coefficients']
    action_scale = float(sindy_data['action_scale'])

    # Config: uncertainty_threshold will be overridden by checkpoint metadata
    cfg = {
        'obs_dim': 8, 'action_dim': 1,
        'mlp_dim': 256,
        'horizon': 5,
        'num_samples': 128, 'num_elites': 16,
        'iterations': 6, 'temperature': 0.5,
        'epsilon_max': 0.1, 'epsilon_std': 0.08,
        'gamma': 0.95,
        'lambda_res': 2.0, 'lambda_smooth': 5.0,
        'lambda_uncertainty': 5.0,
        'uncertainty_threshold': None,  # Data-driven from checkpoint
        'margin': 0.1,
        'ensemble_size': 3,
    }

    # Load trained agent
    ckpt_path = os.path.join(CKPT_DIR, 'checkpoint_ensemble_mppi.pt')
    if not os.path.exists(ckpt_path):
        print(f"  ERROR: No checkpoint found at {ckpt_path}")
        print("  Run train_residual_mppi.py first.")
        return

    agent = ConservativeEnsembleResidualMPPI(
        cfg, sindy_Xi, action_scale, obs_dim=8, action_dim=1,
    )
    agent.load(ckpt_path)

    # Load checkpoint metadata to get uncertainty threshold
    state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if 'uncertainty_threshold_p95' in state_dict:
        agent.uncertainty_threshold = state_dict['uncertainty_threshold_p95']
        cfg['uncertainty_threshold'] = state_dict['uncertainty_threshold_p95']
        print(f"  Uncertainty threshold (from ckpt p95): {cfg['uncertainty_threshold']:.6f}")

    env = PathTrackingEnv(max_episode_steps=1500)

    n_episodes = 50
    print(f"\n  Running {n_episodes} paired episodes...")
    print(f"  Config: horizon={cfg['horizon']}, epsilon_max={cfg['epsilon_max']}, "
          f"ensemble_size={cfg['ensemble_size']}")

    zero_results, mppi_results = [], []

    for ep in range(n_episodes):
        # Zero-residual
        zero_r = run_episode_simple(env, None, seed=ep, use_mppi=False)
        zero_results.append(zero_r)

        # MPPI
        mppi_r = run_episode_simple(env, agent, seed=ep, use_mppi=True)
        mppi_results.append(mppi_r)

        adv = mppi_r['return'] - zero_r['return']
        fb = mppi_r['fb_reasons']
        fb_str = (f"r={fb.get('one_step_reward',0):.2f} "
                  f"rk={fb.get('risk',0):.2f} "
                  f"u={fb.get('uncertainty',0):.2f} "
                  f"m={fb.get('margin',0):.2f}") if fb else ""
        print(f"  Ep {ep:>2d}: zero={zero_r['return']:>7.1f}  mppi={mppi_r['return']:>7.1f}  "
              f"adv={adv:>+7.1f}  fb={mppi_r['fallback_rate']:.0%}  "
              f"nf={mppi_r['nonfallback_count']:>3d}  {fb_str}")

    # === Aggregate statistics ===
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)

    zero_ret = np.array([r['return'] for r in zero_results])
    mppi_ret = np.array([r['return'] for r in mppi_results])
    adv = mppi_ret - zero_ret

    print(f"\n  {'Metric':<30} | {'Zero-residual':>15} | {'Ensemble MPPI':>15} | {'Advantage':>12}")
    print(f"  {'-'*30} | {'-'*15} | {'-'*15} | {'-'*12}")

    def row(name, zero_val, mppi_val, higher_is_better=True):
        diff = mppi_val - zero_val
        ok = "+" if (diff > 0) == higher_is_better else ""
        print(f"  {name:<30} | {zero_val:>15.2f} | {mppi_val:>15.2f} | {ok}{diff:>+11.2f}")

    row("Return (mean)", np.mean(zero_ret), np.mean(mppi_ret), True)
    row("Return (std)", np.std(zero_ret), np.std(mppi_ret), False)
    row("Return (median)", np.median(zero_ret), np.median(mppi_ret), True)

    mppi_eps = np.array([r['eps_abs_mean'] for r in mppi_results])
    mppi_fallback = np.array([r['fallback_rate'] for r in mppi_results])
    mppi_nf_count = np.array([r['nonfallback_count'] for r in mppi_results])
    mppi_unc = np.array([r['uncertainty_mean'] for r in mppi_results])
    mppi_pred_adv = np.array([r['predicted_advantage_mean'] for r in mppi_results])

    print(f"\n  {'Metric':<30} | {'Value':>15}")
    print(f"  {'-'*30} | {'-'*15}")
    print(f"  {'Fallback rate (mean)':<30} | {np.mean(mppi_fallback):>15.1%}")
    print(f"  {'Non-fallback steps (mean)':<30} | {np.mean(mppi_nf_count):>15.1f}")
    print(f"  {'Ensemble uncertainty (mean)':<30} | {np.mean(mppi_unc):>15.6f}")
    print(f"  {'Predicted advantage (mean)':<30} | {np.mean(mppi_pred_adv):>15.6f}")

    # Fallback reason aggregation
    all_fb = [r['fb_reasons'] for r in mppi_results if r['fb_reasons']]
    if all_fb:
        print(f"\n  Fallback reason breakdown (mean across episodes):")
        for key in ['one_step_reward', 'risk', 'uncertainty', 'margin']:
            vals = [r.get(key, 0) for r in all_fb]
            print(f"    {key:<20}: {np.mean(vals):.3f}")

    # Win/loss/draw
    wins = int(np.sum(adv > 0))
    losses = int(np.sum(adv < 0))
    draws = int(np.sum(adv == 0))
    print(f"\n  Win/Loss/Draw: {wins}/{losses}/{draws}")
    print(f"  Win rate: {wins/n_episodes*100:.1f}%")

    # Paired t-test
    from scipy import stats
    t_stat, p_value = stats.ttest_rel(mppi_ret, zero_ret)
    print(f"\n  Paired t-test: t={t_stat:.3f}, p={p_value:.4f}")
    if p_value < 0.001:
        print("  Significance: *** (p < 0.001)")
    elif p_value < 0.01:
        print("  Significance: ** (p < 0.01)")
    elif p_value < 0.05:
        print("  Significance: * (p < 0.05)")
    else:
        print("  Significance: n.s. (p >= 0.05)")

    ci = stats.t.interval(0.95, len(adv)-1, loc=np.mean(adv), scale=stats.sem(adv))
    print(f"  95% CI for advantage: [{ci[0]:.2f}, {ci[1]:.2f}]")

    # Non-fallback episodes
    nf_episodes = [i for i in range(n_episodes) if mppi_results[i]['nonfallback_count'] > 0]
    if nf_episodes:
        nf_adv = adv[nf_episodes]
        print(f"\n  Episodes with non-fallback actions: {len(nf_episodes)}/{n_episodes}")
        print(f"  Mean advantage (non-fallback eps): {np.mean(nf_adv):.2f}")

    # Config
    print(f"\n  --- Configuration ---")
    for k, v in cfg.items():
        print(f"  {k}: {v}")

    env.close()
    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
