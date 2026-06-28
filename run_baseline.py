"""Stage 0: Baseline verification — LQR+Stanley zero-residual, 50 seeds."""
import sys, os, json, csv
from datetime import datetime
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from path_tracking_env import PathTrackingEnv

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_baseline_episode(env, seed, max_steps=1500):
    obs, _ = env.reset(seed=seed)
    total_r = 0
    ey_list, epsi_list, theta_list, theta_dot_list = [], [], [], []
    u_lqr_list, u_stanley_list, u_total_list = [], [], []
    early_term = False

    for step in range(max_steps):
        action = np.array([0.0])
        obs, r, t, tr, info = env.step(action)
        total_r += r

        raw = info.get('raw_state', np.zeros(8))
        ey_list.append(abs(raw[0]))
        epsi_list.append(abs(raw[1]))
        theta_list.append(abs(raw[3]))
        theta_dot_list.append(abs(raw[4]))
        u_lqr_list.append(abs(info.get('u_lqr', 0) or 0))
        u_stanley_list.append(abs(info.get('u_stanley', 0)))
        u_total_list.append(abs(info.get('u_total', 0)))

        if t:
            early_term = True
        if t or tr:
            break

    n = step + 1
    return {
        'seed': seed, 'return': total_r, 'length': n,
        'early_term': early_term,
        'ey_rms': float(np.sqrt(np.mean(np.array(ey_list)**2))),
        'epsi_rms': float(np.sqrt(np.mean(np.array(epsi_list)**2))),
        'theta_rms': float(np.sqrt(np.mean(np.array(theta_list)**2))),
        'theta_max': float(np.max(np.array(theta_list))),
        'theta_dot_max': float(np.max(np.array(theta_dot_list))),
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
    env = PathTrackingEnv(max_episode_steps=1500)
    seeds = list(range(50))
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print("=" * 80)
    print("  Stage 0: Baseline Verification (LQR+Stanley, zero residual)")
    print(f"  Seeds: {len(seeds)}, Timestamp: {timestamp}")
    print("=" * 80)

    results = []
    for i, seed in enumerate(seeds):
        r = run_baseline_episode(env, seed)
        results.append(r)
        if (i + 1) % 10 == 0:
            print(f"  Completed {i+1}/{len(seeds)} seeds")

    # Aggregate
    rets = [r['return'] for r in results]
    lens = [r['length'] for r in results]
    ets = [r['early_term'] for r in results]
    ey_rs = [r['ey_rms'] for r in results]
    ep_rs = [r['epsi_rms'] for r in results]
    th_rs = [r['theta_rms'] for r in results]
    th_max = [r['theta_max'] for r in results]
    energy = [r['control_energy'] for r in results]

    summary = {
        'timestamp': timestamp,
        'n_seeds': len(seeds),
        'return_mean': float(np.mean(rets)),
        'return_std': float(np.std(rets)),
        'length_mean': float(np.mean(lens)),
        'length_std': float(np.std(lens)),
        'early_term_count': int(np.sum(ets)),
        'early_term_rate': float(np.mean(ets)),
        'ey_rms_mean': float(np.mean(ey_rs)),
        'ey_rms_std': float(np.std(ey_rs)),
        'epsi_rms_mean': float(np.mean(ep_rs)),
        'epsi_rms_std': float(np.std(ep_rs)),
        'theta_rms_mean': float(np.mean(th_rs)),
        'theta_rms_std': float(np.std(th_rs)),
        'theta_max_mean': float(np.mean(th_max)),
        'theta_max_max': float(np.max(th_max)),
        'control_energy_mean': float(np.mean(energy)),
        'u_lqr_mean': float(np.mean([r['u_lqr_mean'] for r in results])),
        'u_lqr_max': float(np.max([r['u_lqr_max'] for r in results])),
        'u_stanley_mean': float(np.mean([r['u_stanley_mean'] for r in results])),
        'u_stanley_max': float(np.max([r['u_stanley_max'] for r in results])),
        'u_total_mean': float(np.mean([r['u_total_mean'] for r in results])),
        'u_total_max': float(np.max([r['u_total_max'] for r in results])),
    }

    # Print
    print(f"\n  {'Metric':<30} | {'Value':>12}")
    print(f"  {'-'*30} | {'-'*12}")
    for k, v in summary.items():
        if k not in ('timestamp', 'n_seeds'):
            print(f"  {k:<30} | {v:>12.4f}")

    # Save JSON
    json_path = os.path.join(RESULTS_DIR, 'baseline_lqr_stanley_50seeds.json')
    with open(json_path, 'w') as f:
        json.dump({'summary': summary, 'per_seed': results}, f, indent=2)

    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, 'baseline_lqr_stanley_50seeds.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    # Save MD
    md_path = os.path.join(RESULTS_DIR, 'baseline_lqr_stanley_50seeds.md')
    with open(md_path, 'w') as f:
        f.write("# Baseline: LQR+Stanley Zero-Residual (50 Seeds)\n\n")
        f.write(f"**Timestamp**: {timestamp}\n\n")
        f.write("## Summary\n\n")
        f.write(f"| Metric | Value |\n|--------|-------|\n")
        for k, v in summary.items():
            if k not in ('timestamp', 'n_seeds'):
                f.write(f"| {k} | {v:.4f} |\n")
        f.write(f"\n## Risk Assessment\n\n")
        if summary['early_term_count'] > 10:
            f.write(f"- **WARNING**: ET={summary['early_term_count']}/50 — baseline itself is unstable\n")
        if summary['theta_max_max'] > 1.2:
            f.write(f"- **WARNING**: theta_max_max={summary['theta_max_max']:.3f} — close to termination (pi/2=1.57)\n")
        if summary['theta_rms_mean'] > 1.0:
            f.write(f"- **WARNING**: theta_rms_mean={summary['theta_rms_mean']:.3f} — high roll angle\n")
        f.write(f"\n## Conclusion\n\n")
        f.write(f"Baseline return: {summary['return_mean']:.2f} +/- {summary['return_std']:.2f}\n")
        f.write(f"Baseline ET: {summary['early_term_count']}/50\n")

    env.close()
    print(f"\n  Saved: {json_path}")
    print(f"  Saved: {csv_path}")
    print(f"  Saved: {md_path}")
    print("=" * 80)


if __name__ == '__main__':
    main()
