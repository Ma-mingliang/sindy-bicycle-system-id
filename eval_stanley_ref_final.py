"""Stage 6: Final 50-seed paired evaluation for stanley_ref residual-MPPI.

Uses best config from Stage 5 tuning.
"""
import sys, os, json
from datetime import datetime
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from path_tracking_env import PathTrackingEnv
from residual_mppi import ConservativeEnsembleResidualMPPI

CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'residual_mppi_stanley_ref')


def run_episode(env, agent, seed, max_steps=1500):
    obs, _ = env.reset(seed=seed)
    if agent is not None:
        agent.reset_planning()
    total_r = 0
    early_term = False
    ey_list, epsi_list, theta_list, theta_dot_list = [], [], [], []
    u_list = []

    for step in range(max_steps):
        if agent is not None:
            eps, best_G, zero_G, unc = agent.act(obs, eval_mode=True)
            action = eps
        else:
            action = np.array([0.0])
        obs, r, t, tr, info = env.step(action)
        total_r += r

        raw = info.get('raw_state', np.zeros(8))
        ey_list.append(abs(raw[0]))
        epsi_list.append(abs(raw[1]))
        theta_list.append(abs(raw[3]))
        theta_dot_list.append(abs(raw[4]))
        u_list.append(abs(info.get('u_total', 0)))

        if t:
            early_term = True
        if t or tr:
            break

    n = step + 1
    fb = agent.get_fb_reasons() if agent else {}
    return {
        'return': total_r, 'length': n,
        'early_term': early_term,
        'ey_rms': float(np.sqrt(np.mean(np.array(ey_list)**2))),
        'epsi_rms': float(np.sqrt(np.mean(np.array(epsi_list)**2))),
        'theta_rms': float(np.sqrt(np.mean(np.array(theta_list)**2))),
        'theta_max': float(np.max(np.array(theta_list))) if theta_list else 0.0,
        'theta_dot_rms': float(np.sqrt(np.mean(np.array(theta_dot_list)**2))),
        'control_energy': float(np.mean(np.array(u_list)**2)),
        'fb': fb,
    }


def main():
    sys.stdout.reconfigure(line_buffering=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    sindy_data = np.load(os.path.join(os.path.dirname(__file__), 'sindy_model_improved.npz'))
    sindy_Xi = sindy_data['coefficients']
    action_scale = float(sindy_data['action_scale'])

    ckpt_path = os.path.join(CKPT_DIR, 'checkpoint_ensemble_mppi.pt')
    state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    unc_threshold = state_dict.get('val_uncertainty_p95', 0.001)

    # Best config from Stage 5
    cfg = {
        'obs_dim': 8, 'action_dim': 1, 'mlp_dim': 256,
        'horizon': 8, 'num_samples': 128, 'num_elites': 16,
        'iterations': 6, 'temperature': 0.5,
        'epsilon_max': 0.1, 'epsilon_std': 0.08,
        'gamma': 0.95,
        'lambda_res': 0.5, 'lambda_smooth': 5.0,
        'lambda_uncertainty': 5.0,
        'uncertainty_threshold': None,
        'ensemble_size': 3,
        'risk_mode': 'tolerance',
        'ey_tol': 0.02, 'theta_tol': 0.02,
        'margin': 0.01,
    }

    print("=" * 100)
    print("  Stage 6: Final Paired Evaluation (50 seeds)")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Config: margin=0.01, lambda_res=0.5, horizon=8, eps_max=0.1")
    print(f"  Timestamp: {timestamp}")
    print("=" * 100)

    # Load best LQR config if available
    best_lqr_path = os.path.join(os.path.dirname(__file__), 'configs', 'best_lqr.json')
    lqr_kwargs = {}
    if os.path.exists(best_lqr_path):
        import json as _json
        with open(best_lqr_path) as f:
            best = _json.load(f)
        lqr_kwargs = {'lqr_Q': best['Q_diag'], 'lqr_R': best['R_val']}
        print(f"  LQR: {best['label']} (Q={best['Q_diag']}, R={best['R_val']})")

    env = PathTrackingEnv(max_episode_steps=1500, residual_injection='stanley_ref',
                          **lqr_kwargs)
    agent = ConservativeEnsembleResidualMPPI(
        cfg, sindy_Xi, action_scale, obs_dim=8, action_dim=1)
    agent.load(ckpt_path)
    agent.uncertainty_threshold = unc_threshold

    n_episodes = 50
    zero_results, mppi_results = [], []

    for ep in range(n_episodes):
        zero_r = run_episode(env, None, seed=ep)
        mppi_r = run_episode(env, agent, seed=ep)

        zero_results.append(zero_r)
        mppi_results.append(mppi_r)

        if (ep + 1) % 10 == 0:
            z_ret = np.mean([r['return'] for r in zero_results])
            m_ret = np.mean([r['return'] for r in mppi_results])
            z_et = sum(r['early_term'] for r in zero_results)
            m_et = sum(r['early_term'] for r in mppi_results)
            print(f"  [{ep+1}/{n_episodes}] zero_ret={z_ret:.2f}, mppi_ret={m_ret:.2f}, "
                  f"zero_ET={z_et}, mppi_ET={m_et}")

    env.close()

    # Compute statistics
    z_ret = np.array([r['return'] for r in zero_results])
    m_ret = np.array([r['return'] for r in mppi_results])
    adv = m_ret - z_ret

    z_ey = np.array([r['ey_rms'] for r in zero_results])
    m_ey = np.array([r['ey_rms'] for r in mppi_results])
    z_ep = np.array([r['epsi_rms'] for r in zero_results])
    m_ep = np.array([r['epsi_rms'] for r in mppi_results])
    z_th = np.array([r['theta_rms'] for r in zero_results])
    m_th = np.array([r['theta_rms'] for r in mppi_results])
    z_th_max = np.array([r['theta_max'] for r in zero_results])
    m_th_max = np.array([r['theta_max'] for r in mppi_results])

    z_et = sum(r['early_term'] for r in zero_results)
    m_et = sum(r['early_term'] for r in mppi_results)

    # Paired t-test
    from scipy import stats
    t_stat, p_val = stats.ttest_rel(m_ret, z_ret)

    # Wilcoxon signed-rank
    try:
        w_stat, w_pval = stats.wilcoxon(m_ret, z_ret)
    except:
        w_stat, w_pval = 0, 1.0

    print("\n" + "=" * 100)
    print("  RESULTS SUMMARY")
    print("=" * 100)

    print(f"\n  {'Metric':<25} {'Zero (baseline)':>18} {'MPPI (ours)':>18} {'Delta':>12} {'p-value':>10}")
    print(f"  {'-'*25} {'-'*18} {'-'*18} {'-'*12} {'-'*10}")

    print(f"  {'Return (mean)':<25} {np.mean(z_ret):>18.2f} {np.mean(m_ret):>18.2f} "
          f"{np.mean(adv):>+12.2f} {p_val:>10.4f}")
    print(f"  {'Return (std)':<25} {np.std(z_ret):>18.2f} {np.std(m_ret):>18.2f}")
    print(f"  {'ET count':<25} {z_et:>18d} {m_et:>18d} {m_et-z_et:>+12d}")
    print(f"  {'ey_rms (mean)':<25} {np.mean(z_ey):>18.4f} {np.mean(m_ey):>18.4f} "
          f"{np.mean(m_ey-z_ey):>+12.4f}")
    print(f"  {'epsi_rms (mean)':<25} {np.mean(z_ep):>18.4f} {np.mean(m_ep):>18.4f} "
          f"{np.mean(m_ep-z_ep):>+12.4f}")
    print(f"  {'theta_rms (mean)':<25} {np.mean(z_th):>18.4f} {np.mean(m_th):>18.4f} "
          f"{np.mean(m_th-z_th):>+12.4f}")
    print(f"  {'theta_max (max)':<25} {np.max(z_th_max):>18.4f} {np.max(m_th_max):>18.4f} "
          f"{np.max(m_th_max)-np.max(z_th_max):>+12.4f}")

    print(f"\n  Paired t-test: t={t_stat:.3f}, p={p_val:.6f}")
    print(f"  Wilcoxon: W={w_stat:.1f}, p={w_pval:.6f}")
    print(f"  Effect size (Cohen's d): {np.mean(adv)/np.std(adv):.3f}")

    # Success criteria check
    print(f"\n  Success Criteria:")
    print(f"    [PASS] real_advantage > 0: {np.mean(adv):+.2f}" if np.mean(adv) > 0
          else f"    [FAIL] real_advantage > 0: {np.mean(adv):+.2f}")
    print(f"    [PASS] ET not increased: {m_et} <= {z_et}" if m_et <= z_et
          else f"    [FAIL] ET increased: {m_et} > {z_et}")
    print(f"    [PASS] theta_rms not worsened: {np.mean(m_th):.4f} <= {np.mean(z_th):.4f}"
          if np.mean(m_th) <= np.mean(z_th)
          else f"    [FAIL] theta_rms worsened: {np.mean(m_th):.4f} > {np.mean(z_th):.4f}")

    # Fallback stats
    fb_data = [r['fb'] for r in mppi_results if r['fb']]
    if fb_data:
        avg_nf = np.mean([r.get('nonfallback', 0) for r in fb_data])
        avg_total = np.mean([r.get('total_steps', 1) for r in fb_data])
        fb_rate = 1 - avg_nf / max(avg_total, 1)
        print(f"    Fallback rate: {fb_rate:.1%}")
        print(f"    Avg nonfallback steps: {avg_nf:.1f}/{avg_total:.0f}")

    print("\n" + "=" * 100)

    # Save results
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(log_dir, exist_ok=True)

    results = {
        'timestamp': timestamp,
        'config': cfg,
        'n_episodes': n_episodes,
        'zero_return_mean': float(np.mean(z_ret)),
        'zero_return_std': float(np.std(z_ret)),
        'mppi_return_mean': float(np.mean(m_ret)),
        'mppi_return_std': float(np.std(m_ret)),
        'advantage_mean': float(np.mean(adv)),
        'advantage_std': float(np.std(adv)),
        'zero_et': int(z_et),
        'mppi_et': int(m_et),
        'zero_ey_rms': float(np.mean(z_ey)),
        'mppi_ey_rms': float(np.mean(m_ey)),
        'zero_epsi_rms': float(np.mean(z_ep)),
        'mppi_epsi_rms': float(np.mean(m_ep)),
        'zero_theta_rms': float(np.mean(z_th)),
        'mppi_theta_rms': float(np.mean(m_th)),
        't_stat': float(t_stat),
        'p_value': float(p_val),
        'cohens_d': float(np.mean(adv)/np.std(adv)) if np.std(adv) > 0 else 0,
    }

    results_path = os.path.join(log_dir, f'stage_6_final_eval_{timestamp}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {results_path}")

    # Update Stage 4 report with final numbers
    report_path = os.path.join(log_dir, 'stage_4_mppi_diagnostics.md')
    print("=" * 100)


if __name__ == '__main__':
    main()
