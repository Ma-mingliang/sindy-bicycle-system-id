"""Comprehensive ablation: margin sweep + penalty sweep with full diagnostics.

Metrics per config:
  fallback_rate, fallback_by_reward (diag), fallback_by_risk,
  fallback_by_uncertainty, fallback_by_margin, nonfallback_steps,
  eps_abs_mean, eps_abs_max, clip_rate, predicted_advantage_mean,
  real_advantage_mean, early_termination_count
"""
import sys, os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from path_tracking_env import PathTrackingEnv
from residual_mppi import ConservativeEnsembleResidualMPPI

CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'residual_mppi_ensemble_v1')


def run_episode(env, agent, seed, max_steps=1500):
    """Run one episode, return all diagnostics."""
    obs, _ = env.reset(seed=seed)
    if agent is not None:
        agent.reset_planning()
    total_r = 0
    early_term = False

    for step in range(max_steps):
        if agent is not None:
            eps, best_G, zero_G, unc = agent.act(obs, eval_mode=True)
            action = eps
        else:
            action = np.array([0.0])
        obs, r, t, tr, info = env.step(action)
        total_r += r
        if t:
            early_term = True
        if t or tr:
            break

    n = step + 1
    fb = agent.get_fb_reasons() if agent else {}
    return {
        'return': total_r, 'length': n,
        'early_term': early_term,
        'fb': fb,
    }


def eval_config(sindy_Xi, action_scale, cfg, env, n_episodes=10, unc_threshold=None):
    """Run n_episodes with given config, return aggregate diagnostics."""
    agent = ConservativeEnsembleResidualMPPI(
        cfg, sindy_Xi, action_scale, obs_dim=8, action_dim=1)
    ckpt_path = os.path.join(CKPT_DIR, 'checkpoint_ensemble_mppi.pt')
    agent.load(ckpt_path)
    if unc_threshold is not None:
        agent.uncertainty_threshold = unc_threshold

    zero_rets, mppi_rets = [], []
    all_fb, all_early = [], []
    all_nf, all_eps_mean, all_eps_max, all_clip = [], [], [], []
    all_pred_adv = []

    for ep in range(n_episodes):
        zero_r = run_episode(env, None, seed=ep)
        mppi_r = run_episode(env, agent, seed=ep)

        zero_rets.append(zero_r['return'])
        mppi_rets.append(mppi_r['return'])
        all_early.append(mppi_r['early_term'])
        all_fb.append(mppi_r['fb'])

        fb = mppi_r['fb']
        all_nf.append(fb.get('nonfallback', 0))
        all_eps_mean.append(fb.get('eps_abs_mean', 0))
        all_eps_max.append(fb.get('eps_abs_max', 0))
        all_clip.append(fb.get('clip_rate', 0))
        all_pred_adv.append(fb.get('predicted_adv_mean', 0))

    zero_ret = np.array(zero_rets)
    mppi_ret = np.array(mppi_rets)
    adv = mppi_ret - zero_ret

    # Aggregate fallback reasons
    avg_fb = {}
    for key in ['one_step_reward', 'risk', 'uncertainty', 'margin']:
        avg_fb[key] = np.mean([r.get(key, 0) for r in all_fb])

    return {
        'zero_mean': float(np.mean(zero_ret)),
        'mppi_mean': float(np.mean(mppi_ret)),
        'real_advantage': float(np.mean(adv)),
        'early_term_count': int(np.sum(all_early)),
        'fb_rate': float(np.mean([r.get('total_steps', 0) - r.get('nonfallback', 0)
                                   for r in all_fb])) / max(float(np.mean([r.get('total_steps', 1) for r in all_fb])), 1),
        'fb_by_reward': avg_fb['one_step_reward'],
        'fb_by_risk': avg_fb['risk'],
        'fb_by_uncertainty': avg_fb['uncertainty'],
        'fb_by_margin': avg_fb['margin'],
        'nf_steps': float(np.mean(all_nf)),
        'eps_abs_mean': float(np.mean(all_eps_mean)),
        'eps_abs_max': float(np.max(all_eps_max)),
        'clip_rate': float(np.mean(all_clip)),
        'predicted_adv_mean': float(np.mean(all_pred_adv)),
    }


def print_header():
    print(f"  {'config':<28} | {'fb%':>5} | {'rwd':>5} | {'risk':>5} | {'unc':>5} | "
          f"{'marg':>5} | {'nf':>5} | {'|e|':>6} | {'clip':>5} | "
          f"{'pred':>7} | {'real':>7} | {'ET':>3}")
    print(f"  {'-'*28} | {'-'*5} | {'-'*5} | {'-'*5} | {'-'*5} | "
          f"{'-'*5} | {'-'*5} | {'-'*6} | {'-'*5} | "
          f"{'-'*7} | {'-'*7} | {'-'*3}")


def print_row(name, r):
    print(f"  {name:<28} | {r['fb_rate']:>5.0%} | {r['fb_by_reward']:>.3f} | "
          f"{r['fb_by_risk']:>.3f} | {r['fb_by_uncertainty']:>.3f} | {r['fb_by_margin']:>.3f} | "
          f"{r['nf_steps']:>5.1f} | {r['eps_abs_mean']:>.4f} | {r['clip_rate']:>.3f} | "
          f"{r['predicted_adv_mean']:>+7.3f} | {r['real_advantage']:>+7.2f} | {r['early_term_count']:>3}")


def main():
    sys.stdout.reconfigure(line_buffering=True)

    sindy_data = np.load(os.path.join(os.path.dirname(__file__), 'sindy_model_improved.npz'))
    sindy_Xi = sindy_data['coefficients']
    action_scale = float(sindy_data['action_scale'])

    # Load uncertainty threshold from checkpoint
    ckpt_path = os.path.join(CKPT_DIR, 'checkpoint_ensemble_mppi.pt')
    state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    unc_threshold = state_dict.get('uncertainty_threshold_p95', 0.01)

    env = PathTrackingEnv(max_episode_steps=1500)
    n_eps = 10

    # ============================================================
    # Part 1: Margin sweep (fixed lambda_res=2.0, lambda_smooth=5.0)
    # ============================================================
    print("=" * 90)
    print("  PART 1: Margin Sweep")
    print(f"  Fixed: horizon=5, epsilon_max=0.1, gamma=0.95, lambda_res=2.0, lambda_smooth=5.0")
    print(f"  Uncertainty threshold (val p95): {unc_threshold:.6f}")
    print(f"  One-step reward gate: DIAGNOSTIC ONLY")
    print(f"  Episodes per config: {n_eps}")
    print("=" * 90)

    base_cfg = {
        'obs_dim': 8, 'action_dim': 1, 'mlp_dim': 256,
        'horizon': 5, 'num_samples': 128, 'num_elites': 16,
        'iterations': 6, 'temperature': 0.5,
        'epsilon_max': 0.1, 'epsilon_std': 0.08,
        'gamma': 0.95,
        'lambda_res': 2.0, 'lambda_smooth': 5.0,
        'lambda_uncertainty': 5.0,
        'uncertainty_threshold': None,
        'ensemble_size': 3,
    }

    margins = [0.1, 0.05, 0.02, 0.01, 0.0]
    margin_results = {}

    print_header()
    for margin in margins:
        cfg = {**base_cfg, 'margin': margin}
        r = eval_config(sindy_Xi, action_scale, cfg, env, n_eps, unc_threshold)
        margin_results[margin] = r
        print_row(f"margin={margin}", r)

    # ============================================================
    # Part 2: Penalty sweep (if margin sweep shows promise)
    # ============================================================
    best_margin = min(margins, key=lambda m: margin_results[m]['fb_rate'])

    print(f"\n  Best margin from Part 1: {best_margin} (fb_rate={margin_results[best_margin]['fb_rate']:.0%})")

    # Always do penalty sweep for diagnostics
    print("\n" + "=" * 90)
    print(f"  PART 2: Penalty Sweep (margin={best_margin})")
    print(f"  Varying lambda_res and lambda_smooth")
    print("=" * 90)

    penalty_configs = [
        (0.5, 1.0),
        (0.5, 2.0),
        (1.0, 1.0),
        (1.0, 2.0),
        (1.0, 5.0),
        (2.0, 1.0),
        (2.0, 2.0),
    ]

    penalty_results = {}
    print_header()
    for lr, ls in penalty_configs:
        cfg = {**base_cfg, 'margin': best_margin, 'lambda_res': lr, 'lambda_smooth': ls}
        r = eval_config(sindy_Xi, action_scale, cfg, env, n_eps, unc_threshold)
        penalty_results[(lr, ls)] = r
        print_row(f"res={lr},smooth={ls}", r)

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 90)
    print("  SUMMARY: Top configs by real_advantage")
    print("=" * 90)

    all_configs = []
    for m, r in margin_results.items():
        all_configs.append((f"margin={m}", r))
    for (lr, ls), r in penalty_results.items():
        all_configs.append((f"margin={best_margin},res={lr},smooth={ls}", r))

    # Sort by real advantage
    all_configs.sort(key=lambda x: x[1]['real_advantage'], reverse=True)

    print(f"\n  {'rank':>4} {'config':<35} | {'fb%':>5} | {'nf':>5} | "
          f"{'real_adv':>8} | {'pred_adv':>8} | {'ET':>3}")
    print(f"  {'-'*4} {'-'*35} | {'-'*5} | {'-'*5} | {'-'*8} | {'-'*8} | {'-'*3}")

    for i, (name, r) in enumerate(all_configs[:10]):
        print(f"  {i+1:>4} {name:<35} | {r['fb_rate']:>5.0%} | {r['nf_steps']:>5.1f} | "
              f"{r['real_advantage']:>+8.2f} | {r['predicted_adv_mean']:>+8.3f} | {r['early_term_count']:>3}")

    # Recommendation
    print("\n  Recommendation criteria:")
    print("    real_advantage >= 0, early_term not increasing, fb_rate < 90%, eps not clipping")

    candidates = [(n, r) for n, r in all_configs
                  if r['real_advantage'] >= 0 and r['fb_rate'] < 0.9]
    if candidates:
        print(f"\n  Candidates meeting criteria: {len(candidates)}")
        for n, r in candidates[:3]:
            print(f"    {n}: real_adv={r['real_advantage']:+.2f}, fb={r['fb_rate']:.0%}, "
                  f"nf={r['nf_steps']:.1f}, clip={r['clip_rate']:.0%}")
    else:
        print("\n  No config meets all criteria — need further investigation.")

    env.close()
    print("\n" + "=" * 90)


if __name__ == '__main__':
    main()
