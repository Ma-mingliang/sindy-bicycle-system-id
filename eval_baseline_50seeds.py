"""Stage 0: Confirm current LQR+Stanley baseline (50 seeds).

Runs zero-residual episodes with stanley_ref injection mode.
Collects all metrics: return, ET, ey_rms, epsi_rms, theta_rms, theta_max,
control_energy, u_lqr/u_stanley/u_total statistics.
"""
import sys, os, json, csv
from datetime import datetime
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from path_tracking_env import PathTrackingEnv


def run_episode(env, seed, max_steps=1500):
    obs, _ = env.reset(seed=seed)
    total_r = 0
    early_term = False
    ey_list, epsi_list, theta_list, theta_dot_list = [], [], [], []
    u_lqr_list, u_stanley_list, u_total_list = [], [], []
    step_count = 0

    for step in range(max_steps):
        action = np.array([0.0])  # zero residual
        obs, r, t, tr, info = env.step(action)
        total_r += r

        raw = info.get('raw_state', np.zeros(8))
        ey_list.append(abs(raw[0]))
        epsi_list.append(abs(raw[1]))
        theta_list.append(abs(raw[3]))
        theta_dot_list.append(abs(raw[4]))
        u_lqr_raw = info.get('u_lqr')
        u_lqr_list.append(abs(u_lqr_raw) if u_lqr_raw is not None else 0.0)
        u_stanley_list.append(abs(info.get('u_stanley', 0) or 0))
        u_total_list.append(abs(info.get('u_total', 0)))

        step_count += 1
        if t:
            early_term = True
        if t or tr:
            break

    n = step_count
    return {
        'return': total_r,
        'length': n,
        'early_term': early_term,
        'ey_rms': float(np.sqrt(np.mean(np.array(ey_list)**2))),
        'epsi_rms': float(np.sqrt(np.mean(np.array(epsi_list)**2))),
        'theta_rms': float(np.sqrt(np.mean(np.array(theta_list)**2))),
        'theta_max': float(np.max(np.array(theta_list))) if theta_list else 0.0,
        'theta_dot_rms': float(np.sqrt(np.mean(np.array(theta_dot_list)**2))),
        'control_energy': float(np.mean(np.array(u_total_list)**2)),
        'u_lqr_mean': float(np.mean(u_lqr_list)),
        'u_lqr_max': float(np.max(u_lqr_list)),
        'u_stanley_mean': float(np.mean(u_stanley_list)),
        'u_stanley_max': float(np.max(u_stanley_list)),
        'u_total_mean': float(np.mean(u_total_list)),
        'u_total_max': float(np.max(u_total_list)),
    }


def main():
    sys.stdout.reconfigure(line_buffering=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print("=" * 90)
    print("  Stage 0: Baseline Evaluation (50 seeds, zero-residual, stanley_ref)")
    print(f"  Timestamp: {timestamp}")
    print("=" * 90)

    env = PathTrackingEnv(max_episode_steps=1500, residual_injection='stanley_ref')

    n_episodes = 50
    results = []

    for ep in range(n_episodes):
        r = run_episode(env, seed=ep)
        results.append(r)

        if (ep + 1) % 10 == 0:
            avg_ret = np.mean([x['return'] for x in results])
            et_count = sum(x['early_term'] for x in results)
            print(f"  [{ep+1}/{n_episodes}] avg_return={avg_ret:.2f}, ET={et_count}/{ep+1}")

    env.close()

    # Aggregate statistics
    ret = np.array([r['return'] for r in results])
    lengths = np.array([r['length'] for r in results])
    et_count = sum(r['early_term'] for r in results)

    ey_rms = np.array([r['ey_rms'] for r in results])
    epsi_rms = np.array([r['epsi_rms'] for r in results])
    theta_rms = np.array([r['theta_rms'] for r in results])
    theta_max = np.array([r['theta_max'] for r in results])
    ctrl_energy = np.array([r['control_energy'] for r in results])

    u_lqr_mean = np.array([r['u_lqr_mean'] for r in results])
    u_lqr_max = np.array([r['u_lqr_max'] for r in results])
    u_stanley_mean = np.array([r['u_stanley_mean'] for r in results])
    u_stanley_max = np.array([r['u_stanley_max'] for r in results])
    u_total_mean = np.array([r['u_total_mean'] for r in results])
    u_total_max = np.array([r['u_total_max'] for r in results])

    print("\n" + "=" * 90)
    print("  BASELINE RESULTS SUMMARY")
    print("=" * 90)

    print(f"\n  {'Metric':<30} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
    print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")

    def row(name, arr):
        print(f"  {name:<30} {np.mean(arr):>12.4f} {np.std(arr):>12.4f} "
              f"{np.min(arr):>12.4f} {np.max(arr):>12.4f}")

    row("Return", ret)
    row("Episode length", lengths)
    row("ey_rms", ey_rms)
    row("epsi_rms", epsi_rms)
    row("theta_rms", theta_rms)
    row("theta_max", theta_max)
    row("theta_dot_rms", np.array([r['theta_dot_rms'] for r in results]))
    row("Control energy", ctrl_energy)
    row("u_lqr_mean", u_lqr_mean)
    row("u_lqr_max", u_lqr_max)
    row("u_stanley_mean", u_stanley_mean)
    row("u_stanley_max", u_stanley_max)
    row("u_total_mean", u_total_mean)
    row("u_total_max", u_total_max)

    print(f"\n  Early terminations: {et_count}/{n_episodes} ({100*et_count/n_episodes:.1f}%)")

    # Save results
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    # JSON
    summary = {
        'timestamp': timestamp,
        'n_episodes': n_episodes,
        'injection_mode': 'stanley_ref',
        'action': 'zero',
        'return_mean': float(np.mean(ret)),
        'return_std': float(np.std(ret)),
        'return_min': float(np.min(ret)),
        'return_max': float(np.max(ret)),
        'episode_length_mean': float(np.mean(lengths)),
        'episode_length_std': float(np.std(lengths)),
        'early_termination_count': int(et_count),
        'early_termination_rate': float(et_count / n_episodes),
        'ey_rms_mean': float(np.mean(ey_rms)),
        'ey_rms_std': float(np.std(ey_rms)),
        'epsi_rms_mean': float(np.mean(epsi_rms)),
        'epsi_rms_std': float(np.std(epsi_rms)),
        'theta_rms_mean': float(np.mean(theta_rms)),
        'theta_rms_std': float(np.std(theta_rms)),
        'theta_max_max': float(np.max(theta_max)),
        'theta_dot_rms_mean': float(np.mean(np.array([r['theta_dot_rms'] for r in results]))),
        'control_energy_mean': float(np.mean(ctrl_energy)),
        'u_lqr_mean': float(np.mean(u_lqr_mean)),
        'u_lqr_max': float(np.max(u_lqr_max)),
        'u_stanley_mean': float(np.mean(u_stanley_mean)),
        'u_stanley_max': float(np.max(u_stanley_max)),
        'u_total_mean': float(np.mean(u_total_mean)),
        'u_total_max': float(np.max(u_total_max)),
        'per_seed': [
            {
                'seed': i,
                'return': float(results[i]['return']),
                'length': int(results[i]['length']),
                'early_term': bool(results[i]['early_term']),
                'ey_rms': float(results[i]['ey_rms']),
                'epsi_rms': float(results[i]['epsi_rms']),
                'theta_rms': float(results[i]['theta_rms']),
                'theta_max': float(results[i]['theta_max']),
                'control_energy': float(results[i]['control_energy']),
            }
            for i in range(n_episodes)
        ],
    }

    json_path = os.path.join(results_dir, 'stage0_baseline_50seeds.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved JSON: {json_path}")

    # CSV
    csv_path = os.path.join(results_dir, 'stage0_baseline_50seeds.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['seed', 'return', 'length', 'early_term', 'ey_rms',
                         'epsi_rms', 'theta_rms', 'theta_max', 'theta_dot_rms',
                         'control_energy', 'u_lqr_mean', 'u_lqr_max',
                         'u_stanley_mean', 'u_stanley_max', 'u_total_mean', 'u_total_max'])
        for i, r in enumerate(results):
            writer.writerow([i, r['return'], r['length'], r['early_term'],
                             r['ey_rms'], r['epsi_rms'], r['theta_rms'], r['theta_max'],
                             r['theta_dot_rms'], r['control_energy'],
                             r['u_lqr_mean'], r['u_lqr_max'],
                             r['u_stanley_mean'], r['u_stanley_max'],
                             r['u_total_mean'], r['u_total_max']])
    print(f"  Saved CSV: {csv_path}")

    # Markdown report
    log_dir = os.path.join(base_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    report_path = os.path.join(log_dir, 'stage0_baseline.md')
    with open(report_path, 'w') as f:
        f.write("# Stage 0: Baseline Evaluation\n\n")
        f.write(f"**Timestamp**: {timestamp}\n")
        f.write(f"**Config**: 50 seeds, zero-residual, stanley_ref injection\n\n")
        f.write("## Results\n\n")
        f.write("| Metric | Mean | Std | Min | Max |\n")
        f.write("|--------|------|-----|-----|-----|\n")
        f.write(f"| Return | {np.mean(ret):.2f} | {np.std(ret):.2f} | {np.min(ret):.2f} | {np.max(ret):.2f} |\n")
        f.write(f"| Episode length | {np.mean(lengths):.1f} | {np.std(lengths):.1f} | {np.min(lengths):.0f} | {np.max(lengths):.0f} |\n")
        f.write(f"| ey_rms | {np.mean(ey_rms):.4f} | {np.std(ey_rms):.4f} | {np.min(ey_rms):.4f} | {np.max(ey_rms):.4f} |\n")
        f.write(f"| epsi_rms | {np.mean(epsi_rms):.4f} | {np.std(epsi_rms):.4f} | {np.min(epsi_rms):.4f} | {np.max(epsi_rms):.4f} |\n")
        f.write(f"| theta_rms | {np.mean(theta_rms):.4f} | {np.std(theta_rms):.4f} | {np.min(theta_rms):.4f} | {np.max(theta_rms):.4f} |\n")
        f.write(f"| theta_max | {np.max(theta_max):.4f} | — | — | — |\n")
        f.write(f"| Control energy | {np.mean(ctrl_energy):.4f} | {np.std(ctrl_energy):.4f} | — | — |\n")
        f.write(f"| u_lqr_mean | {np.mean(u_lqr_mean):.4f} | — | — | — |\n")
        f.write(f"| u_stanley_mean | {np.mean(u_stanley_mean):.4f} | — | — | — |\n")
        f.write(f"| u_total_mean | {np.mean(u_total_mean):.4f} | — | — | — |\n\n")
        f.write(f"**Early terminations**: {et_count}/{n_episodes} ({100*et_count/n_episodes:.1f}%)\n\n")
        f.write("## Next Steps\n\n")
        f.write("Stage 1: LQR Q/R re-tuning sweep to prioritize safety.\n")
    print(f"  Saved report: {report_path}")

    print("\n" + "=" * 90)


if __name__ == '__main__':
    main()
