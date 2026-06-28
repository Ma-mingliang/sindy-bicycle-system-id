"""Stage 5: Fine-grained tuning for stanley_ref residual-MPPI.

Sweeps:
  1. Margin: [0.0, 0.001, 0.005, 0.01, 0.02]
  2. Lambda_res: [0.2, 0.5, 1.0, 2.0]
  3. Sampling: [64, 128, 256]
  4. Epsilon_max: [0.05, 0.08, 0.10, 0.15]
  5. Horizon: [3, 5, 8, 10]
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
    ey_list, epsi_list, theta_list = [], [], []

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
        'fb': fb,
    }


def evaluate_config(sindy_Xi, action_scale, unc_threshold, cfg, n_episodes=10):
    """Evaluate one config over n_episodes."""
    env = PathTrackingEnv(max_episode_steps=1500, residual_injection='stanley_ref')
    agent = ConservativeEnsembleResidualMPPI(
        cfg, sindy_Xi, action_scale, obs_dim=8, action_dim=1)
    ckpt_path = os.path.join(CKPT_DIR, 'checkpoint_ensemble_mppi.pt')
    agent.load(ckpt_path)
    agent.uncertainty_threshold = unc_threshold

    zero_rets, mppi_rets = [], []
    all_fb, all_early, all_len = [], [], []
    all_ey, all_ep, all_th, all_th_max = [], [], [], []

    for ep in range(n_episodes):
        zero_r = run_episode(env, None, seed=ep)
        mppi_r = run_episode(env, agent, seed=ep)

        zero_rets.append(zero_r['return'])
        mppi_rets.append(mppi_r['return'])
        all_early.append(mppi_r['early_term'])
        all_fb.append(mppi_r['fb'])
        all_len.append(mppi_r['length'])
        all_ey.append(mppi_r['ey_rms'])
        all_ep.append(mppi_r['epsi_rms'])
        all_th.append(mppi_r['theta_rms'])
        all_th_max.append(mppi_r['theta_max'])

    env.close()

    zero_ret = np.array(zero_rets)
    mppi_ret = np.array(mppi_rets)
    adv = mppi_ret - zero_ret

    avg_fb = {}
    for key in ['one_step_reward', 'risk', 'uncertainty', 'margin']:
        avg_fb[key] = float(np.mean([r.get(key, 0) for r in all_fb]))

    fb_rate = float(np.mean([r.get('total_steps',0)-r.get('nonfallback',0)
                              for r in all_fb])) / max(float(np.mean([r.get('total_steps',1) for r in all_fb])),1)
    nf = float(np.mean([r.get('nonfallback',0) for r in all_fb]))
    eps_mean = float(np.mean([r.get('eps_abs_mean',0) for r in all_fb]))

    return {
        'fb_rate': fb_rate,
        'nf': nf,
        'eps_mean': eps_mean,
        'real_adv': float(np.mean(adv)),
        'et_count': int(np.sum(all_early)),
        'ey_rms': float(np.mean(all_ey)),
        'epsi_rms': float(np.mean(all_ep)),
        'theta_rms': float(np.mean(all_th)),
        'theta_max': float(np.max(all_th_max)),
        'return_mean': float(np.mean(mppi_ret)),
    }


def print_header():
    print(f"  {'config':>30} | {'fb%':>5} | {'nf':>6} | {'|e|':>7} | {'real':>7} | "
          f"{'ET':>3} | {'ey_rms':>7} | {'ep_rms':>7} | {'th_rms':>7} | {'th_max':>7}")
    print(f"  {'-'*30} | {'-'*5} | {'-'*6} | {'-'*7} | {'-'*7} | "
          f"{'-'*3} | {'-'*7} | {'-'*7} | {'-'*7} | {'-'*7}")


def print_result(name, r):
    print(f"  {name:>30} | {r['fb_rate']:>5.0%} | {r['nf']:>6.1f} | {r['eps_mean']:>.5f} | "
          f"{r['real_adv']:>+7.2f} | {r['et_count']:>3} | {r['ey_rms']:>.4f} | "
          f"{r['epsi_rms']:>.4f} | {r['theta_rms']:>.4f} | {r['theta_max']:>.4f}")


def main():
    sys.stdout.reconfigure(line_buffering=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    sindy_data = np.load(os.path.join(os.path.dirname(__file__), 'sindy_model_improved.npz'))
    sindy_Xi = sindy_data['coefficients']
    action_scale = float(sindy_data['action_scale'])

    ckpt_path = os.path.join(CKPT_DIR, 'checkpoint_ensemble_mppi.pt')
    state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    unc_threshold = state_dict.get('val_uncertainty_p95', 0.001)

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
        'risk_mode': 'tolerance',
        'ey_tol': 0.02, 'theta_tol': 0.02,
        'margin': 0.001,
    }

    all_results = {}

    # ========== Sweep 1: Margin ==========
    print("=" * 100)
    print(f"  Stage 5: Tuning Sweep (stanley_ref)")
    print(f"  Timestamp: {timestamp}")
    print("=" * 100)

    print("\n  [1/5] Margin Sweep")
    print_header()
    margin_results = {}
    for margin in [0.0, 0.001, 0.005, 0.01, 0.02]:
        cfg = {**base_cfg, 'margin': margin}
        r = evaluate_config(sindy_Xi, action_scale, unc_threshold, cfg, n_episodes=10)
        margin_results[margin] = r
        print_result(f"margin={margin}", r)
    all_results['margin'] = {str(k): v for k, v in margin_results.items()}

    # Pick best margin
    best_margin = max(margin_results.items(),
                      key=lambda x: x[1]['real_adv'] if x[1]['et_count'] <= 5 else -999)[0]
    print(f"\n  Best margin: {best_margin} (real_adv={margin_results[best_margin]['real_adv']:+.2f})")

    # ========== Sweep 2: Lambda_res ==========
    print("\n  [2/5] Lambda_res Sweep")
    print_header()
    lambda_results = {}
    for lam in [0.2, 0.5, 1.0, 2.0, 5.0]:
        cfg = {**base_cfg, 'margin': best_margin, 'lambda_res': lam}
        r = evaluate_config(sindy_Xi, action_scale, unc_threshold, cfg, n_episodes=10)
        lambda_results[lam] = r
        print_result(f"lambda_res={lam}", r)
    all_results['lambda_res'] = {str(k): v for k, v in lambda_results.items()}

    best_lambda = max(lambda_results.items(),
                      key=lambda x: x[1]['real_adv'] if x[1]['et_count'] <= 5 else -999)[0]
    print(f"\n  Best lambda_res: {best_lambda} (real_adv={lambda_results[best_lambda]['real_adv']:+.2f})")

    # ========== Sweep 3: Num samples ==========
    print("\n  [3/5] Sampling Sweep")
    print_header()
    sample_results = {}
    for ns in [64, 128, 256]:
        cfg = {**base_cfg, 'margin': best_margin, 'lambda_res': best_lambda, 'num_samples': ns}
        r = evaluate_config(sindy_Xi, action_scale, unc_threshold, cfg, n_episodes=10)
        sample_results[ns] = r
        print_result(f"num_samples={ns}", r)
    all_results['num_samples'] = {str(k): v for k, v in sample_results.items()}

    best_ns = max(sample_results.items(),
                  key=lambda x: x[1]['real_adv'] if x[1]['et_count'] <= 5 else -999)[0]
    print(f"\n  Best num_samples: {best_ns} (real_adv={sample_results[best_ns]['real_adv']:+.2f})")

    # ========== Sweep 4: Epsilon_max ==========
    print("\n  [4/5] Epsilon_max Sweep")
    print_header()
    eps_results = {}
    for emax in [0.05, 0.08, 0.10, 0.15]:
        cfg = {**base_cfg, 'margin': best_margin, 'lambda_res': best_lambda,
               'num_samples': best_ns, 'epsilon_max': emax, 'epsilon_std': emax * 0.8}
        r = evaluate_config(sindy_Xi, action_scale, unc_threshold, cfg, n_episodes=10)
        eps_results[emax] = r
        print_result(f"epsilon_max={emax}", r)
    all_results['epsilon_max'] = {str(k): v for k, v in eps_results.items()}

    best_emax = max(eps_results.items(),
                    key=lambda x: x[1]['real_adv'] if x[1]['et_count'] <= 5 else -999)[0]
    print(f"\n  Best epsilon_max: {best_emax} (real_adv={eps_results[best_emax]['real_adv']:+.2f})")

    # ========== Sweep 5: Horizon ==========
    print("\n  [5/5] Horizon Sweep")
    print_header()
    horizon_results = {}
    for h in [3, 5, 8, 10]:
        cfg = {**base_cfg, 'margin': best_margin, 'lambda_res': best_lambda,
               'num_samples': best_ns, 'epsilon_max': best_emax,
               'epsilon_std': best_emax * 0.8, 'horizon': h}
        r = evaluate_config(sindy_Xi, action_scale, unc_threshold, cfg, n_episodes=10)
        horizon_results[h] = r
        print_result(f"horizon={h}", r)
    all_results['horizon'] = {str(k): v for k, v in horizon_results.items()}

    best_h = max(horizon_results.items(),
                 key=lambda x: x[1]['real_adv'] if x[1]['et_count'] <= 5 else -999)[0]
    print(f"\n  Best horizon: {best_h} (real_adv={horizon_results[best_h]['real_adv']:+.2f})")

    # ========== Summary ==========
    best_cfg = {
        'margin': best_margin,
        'lambda_res': best_lambda,
        'num_samples': best_ns,
        'epsilon_max': best_emax,
        'epsilon_std': best_emax * 0.8,
        'horizon': best_h,
    }
    best_r = horizon_results[best_h]

    print("\n" + "=" * 100)
    print("  Best Configuration:")
    for k, v in best_cfg.items():
        print(f"    {k}: {v}")
    print(f"\n  Performance:")
    print(f"    real_advantage: {best_r['real_adv']:+.2f}")
    print(f"    ET: {best_r['et_count']}/10")
    print(f"    ey_rms: {best_r['ey_rms']:.4f}")
    print(f"    epsi_rms: {best_r['epsi_rms']:.4f}")
    print(f"    theta_rms: {best_r['theta_rms']:.4f}")
    print(f"    theta_max: {best_r['theta_max']:.4f}")
    print(f"    fallback_rate: {best_r['fb_rate']:.0%}")
    print(f"    nonfallback_steps: {best_r['nf']:.1f}")

    # Save results
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    results_path = os.path.join(log_dir, f'stage_5_tuning_{timestamp}.json')
    with open(results_path, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'best_config': best_cfg,
            'best_performance': best_r,
            'all_results': all_results,
        }, f, indent=2)
    print(f"\n  Saved: {results_path}")
    print("=" * 100)


if __name__ == '__main__':
    main()
