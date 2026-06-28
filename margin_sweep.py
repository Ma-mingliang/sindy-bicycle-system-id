"""Margin sweep: test margin = 0.1, 0.05, 0.02, 0.01 with paired evaluation."""
import sys, os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from path_tracking_env import PathTrackingEnv
from residual_mppi import ConservativeEnsembleResidualMPPI

CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'residual_mppi_ensemble_v1')


def run_episode(env, agent, seed, max_steps=1500):
    obs, _ = env.reset(seed=seed)
    if agent is not None:
        agent.reset_planning()
    total_r = 0
    fb_count, nf_count = 0, 0
    eps_list, unc_list = [], []

    for step in range(max_steps):
        if agent is not None:
            eps, best_G, zero_G, unc = agent.act(obs, eval_mode=True)
            if np.allclose(eps, 0):
                fb_count += 1
            else:
                nf_count += 1
                eps_list.append(np.abs(eps).item())
            unc_list.append(unc)
            action = eps
        else:
            action = np.array([0.0])

        obs, r, t, tr, info = env.step(action)
        total_r += r
        if t or tr:
            break

    n = step + 1
    fb_reasons = agent.get_fb_reasons() if agent else {}
    return {
        'return': total_r, 'length': n,
        'fb_rate': fb_count / n, 'nf_count': nf_count,
        'eps_mean': np.mean(eps_list) if eps_list else 0.0,
        'unc_mean': np.mean(unc_list) if unc_list else 0.0,
        'fb_reasons': fb_reasons,
    }


def main():
    sys.stdout.reconfigure(line_buffering=True)

    sindy_data = np.load(os.path.join(os.path.dirname(__file__), 'sindy_model_improved.npz'))
    sindy_Xi = sindy_data['coefficients']
    action_scale = float(sindy_data['action_scale'])

    cfg = {
        'obs_dim': 8, 'action_dim': 1, 'mlp_dim': 256,
        'horizon': 5, 'num_samples': 128, 'num_elites': 16,
        'iterations': 6, 'temperature': 0.5,
        'epsilon_max': 0.1, 'epsilon_std': 0.08,
        'gamma': 0.95, 'lambda_res': 2.0, 'lambda_smooth': 5.0,
        'lambda_uncertainty': 5.0, 'uncertainty_threshold': None,
        'margin': 0.1, 'ensemble_size': 3,
    }

    ckpt_path = os.path.join(CKPT_DIR, 'checkpoint_ensemble_mppi.pt')
    env = PathTrackingEnv(max_episode_steps=1500)

    # Load checkpoint once for threshold
    agent_base = ConservativeEnsembleResidualMPPI(cfg, sindy_Xi, action_scale, obs_dim=8, action_dim=1)
    agent_base.load(ckpt_path)
    state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    unc_threshold = state_dict.get('uncertainty_threshold_p95', 0.01)

    margins = [0.1, 0.05, 0.02, 0.01]
    n_episodes = 20

    print("=" * 80)
    print(f"  Margin Sweep: {n_episodes} episodes per margin")
    print(f"  Uncertainty threshold: {unc_threshold:.6f}")
    print("=" * 80)

    all_results = {}

    for margin in margins:
        cfg['margin'] = margin
        agent = ConservativeEnsembleResidualMPPI(cfg, sindy_Xi, action_scale, obs_dim=8, action_dim=1)
        agent.load(ckpt_path)
        agent.uncertainty_threshold = unc_threshold

        print(f"\n  --- margin = {margin} ---")

        zero_returns, mppi_returns = [], []
        all_fb_reasons = []
        all_fb_rates, all_nf_counts, all_eps_means = [], [], []

        for ep in range(n_episodes):
            # Zero-residual
            zero_r = run_episode(env, None, seed=ep)
            zero_returns.append(zero_r['return'])

            # MPPI
            mppi_r = run_episode(env, agent, seed=ep)
            mppi_returns.append(mppi_r['return'])
            all_fb_rates.append(mppi_r['fb_rate'])
            all_nf_counts.append(mppi_r['nf_count'])
            all_eps_means.append(mppi_r['eps_mean'])
            all_fb_reasons.append(mppi_r['fb_reasons'])

            adv = mppi_r['return'] - zero_r['return']
            fb = mppi_r['fb_reasons']
            print(f"  Ep {ep:>2d}: zero={zero_r['return']:>7.1f}  mppi={mppi_r['return']:>7.1f}  "
                  f"adv={adv:>+7.1f}  fb={mppi_r['fb_rate']:.0%}  nf={mppi_r['nf_count']:>3d}  "
                  f"|eps|={mppi_r['eps_mean']:.3f}  "
                  f"r={fb.get('one_step_reward',0):.2f} rk={fb.get('risk',0):.2f} "
                  f"m={fb.get('margin',0):.2f}")

        zero_ret = np.array(zero_returns)
        mppi_ret = np.array(mppi_returns)
        adv = mppi_ret - zero_ret

        # Aggregate
        avg_fb = {}
        for key in ['one_step_reward', 'risk', 'uncertainty', 'margin']:
            vals = [r.get(key, 0) for r in all_fb_reasons]
            avg_fb[key] = np.mean(vals)

        wins = int(np.sum(adv > 0))
        losses = int(np.sum(adv < 0))

        from scipy import stats
        t_stat, p_value = stats.ttest_rel(mppi_ret, zero_ret)
        ci = stats.t.interval(0.95, len(adv)-1, loc=np.mean(adv), scale=stats.sem(adv))

        print(f"\n  margin={margin} SUMMARY:")
        print(f"    Zero-residual mean: {np.mean(zero_ret):.2f}")
        print(f"    MPPI mean:          {np.mean(mppi_ret):.2f}")
        print(f"    Real advantage:     {np.mean(adv):+.2f}")
        print(f"    Win/Loss:           {wins}/{losses}")
        print(f"    t={t_stat:.3f}, p={p_value:.4f}")
        print(f"    95% CI: [{ci[0]:.2f}, {ci[1]:.2f}]")
        print(f"    Fallback rate:      {np.mean(all_fb_rates):.1%}")
        print(f"    Non-fallback steps: {np.mean(all_nf_counts):.1f}")
        print(f"    |eps| mean:         {np.mean(all_eps_means):.4f}")
        print(f"    Fallback reasons:   reward={avg_fb['one_step_reward']:.3f}  "
              f"risk={avg_fb['risk']:.3f}  unc={avg_fb['uncertainty']:.3f}  "
              f"margin={avg_fb['margin']:.3f}")

        all_results[margin] = {
            'zero_mean': np.mean(zero_ret), 'mppi_mean': np.mean(mppi_ret),
            'advantage': np.mean(adv), 'wins': wins, 'losses': losses,
            'p_value': p_value, 'fb_rate': np.mean(all_fb_rates),
            'nf_count': np.mean(all_nf_counts), 'eps_mean': np.mean(all_eps_means),
            'fb_reasons': avg_fb,
        }

    # Final comparison
    print("\n" + "=" * 80)
    print("  MARGIN SWEEP COMPARISON")
    print("=" * 80)
    print(f"  {'margin':<8} | {'zero_mean':>10} | {'mppi_mean':>10} | {'advantage':>10} | "
          f"{'W/L':>6} | {'p_value':>8} | {'fb_rate':>8} | {'nf_steps':>8} | {'|eps|':>8}")
    print(f"  {'-'*8} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*6} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*8}")

    for margin in margins:
        r = all_results[margin]
        print(f"  {margin:<8} | {r['zero_mean']:>10.2f} | {r['mppi_mean']:>10.2f} | {r['advantage']:>+10.2f} | "
              f"{r['wins']}/{r['losses']:>4} | {r['p_value']:>8.4f} | {r['fb_rate']:>7.1%} | "
              f"{r['nf_count']:>8.1f} | {r['eps_mean']:>8.4f}")

    print(f"\n  Fallback reasons by margin:")
    print(f"  {'margin':<8} | {'reward':>8} | {'risk':>8} | {'unc':>8} | {'margin':>8}")
    print(f"  {'-'*8} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*8}")
    for margin in margins:
        fb = all_results[margin]['fb_reasons']
        print(f"  {margin:<8} | {fb['one_step_reward']:>8.3f} | {fb['risk']:>8.3f} | "
              f"{fb['uncertainty']:>8.3f} | {fb['margin']:>8.3f}")

    env.close()
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
