"""Optimize Coupling Network with Larger Capacity - Fast Version.

Tests 6 focused configs with fast evaluation (max H=200, 3 segments).
Current best: Improved Coupling Network (256 dim) = Primary 0.4712
"""
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime

sys.path.insert(0, 'D:/系统辨识作业/sindy_bicycle')

STATE_NAMES_7D = ['e_y', 'e_psi', 'v', 'theta', 'theta_dot', 'delta', 'delta_dot']
STATE_DIM = 7
ACTION_DIM = 1
IDX_7D_FROM_8D = [0, 1, 2, 3, 4, 6, 7]

PHYSICAL_LIMITS = {
    'e_y': 5.0, 'e_psi': np.pi, 'v': 5.0,
    'theta': np.pi, 'theta_dot': 10.0,
    'delta': np.pi/2, 'delta_dot': 10.0,
}


class DeepEncoder(nn.Module):
    def __init__(self, input_dim, coupling_dim, hidden, n_layers):
        super().__init__()
        assert n_layers >= 3
        layers = [nn.Linear(input_dim, hidden), nn.SiLU()]
        for _ in range(n_layers - 2):
            layers.extend([nn.Linear(hidden, hidden), nn.SiLU()])
        layers.append(nn.Linear(hidden, coupling_dim))
        self.net = nn.Sequential(*layers)
        self.residual = nn.Linear(input_dim, coupling_dim) if input_dim != coupling_dim else nn.Identity()
        self.norm = nn.LayerNorm(coupling_dim)

    def forward(self, x):
        return self.norm(self.net(x) + self.residual(x))


class DeepDecoder(nn.Module):
    def __init__(self, coupling_dim, hidden, n_layers):
        super().__init__()
        assert n_layers >= 3
        input_dim = coupling_dim + STATE_DIM
        layers = [nn.Linear(input_dim, hidden), nn.SiLU()]
        for _ in range(n_layers - 2):
            layers.extend([nn.Linear(hidden, hidden), nn.SiLU()])
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].bias)
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.01)

    def forward(self, z, all_states):
        x = torch.cat([z, all_states], dim=-1)
        return self.net(x).squeeze(-1)


class LargeCouplingNetwork(nn.Module):
    def __init__(self, state_dim=7, action_dim=1, coupling_dim=512, hidden=512,
                 encoder_layers=3, decoder_layers=3, decoder_hidden=None):
        super().__init__()
        if decoder_hidden is None:
            decoder_hidden = hidden
        self.encoder = DeepEncoder(state_dim + action_dim, coupling_dim, hidden, encoder_layers)
        self.decoders = nn.ModuleList([
            DeepDecoder(coupling_dim, decoder_hidden, decoder_layers)
            for _ in range(state_dim)
        ])

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        z = self.encoder(x)
        deltas = [decoder(z, s) for decoder in self.decoders]
        return torch.stack(deltas, dim=-1)


def check_survival(state):
    for i, name in enumerate(STATE_NAMES_7D):
        if name in PHYSICAL_LIMITS:
            if abs(state[i]) > PHYSICAL_LIMITS[name]:
                return False
    return not (np.any(np.isnan(state)) or np.any(np.isinf(state)))


def load_data(seed=42):
    data = np.load('D:/系统辨识作业/sindy_bicycle/data/stage2_dataset_150k.npz', allow_pickle=True)
    obs = data['obs'][:, IDX_7D_FROM_8D]
    next_obs = data['next_obs'][:, IDX_7D_FROM_8D]
    deltas = next_obs - obs
    done = data['done']
    action = data['action']

    episode_ends = np.where(done)[0]
    episode_starts = np.concatenate([[0], episode_ends[:-1] + 1])
    n_episodes = len(episode_ends)

    np.random.seed(seed)
    indices = np.random.permutation(n_episodes)
    n_train = int(0.75 * n_episodes)
    train_eps = indices[:n_train]
    test_eps = indices[n_train:]

    episodes = []
    for ep_idx in range(n_episodes):
        start = episode_starts[ep_idx]
        end = episode_ends[ep_idx] + 1
        episodes.append({
            'obs': obs[start:end],
            'action': action[start:end],
            'deltas': deltas[start:end],
            'length': end - start,
        })

    train_obs = np.concatenate([episodes[ep]['obs'] for ep in train_eps])
    train_action = np.concatenate([episodes[ep]['action'] for ep in train_eps])
    train_deltas = np.concatenate([episodes[ep]['deltas'] for ep in train_eps])

    state_std = np.std(train_obs, axis=0)
    state_std[state_std < 1e-10] = 1.0
    action_std = np.std(train_action)
    if action_std < 1e-10:
        action_std = 1.0
    delta_std = np.std(train_deltas, axis=0)
    delta_std[delta_std < 1e-10] = 1.0

    return {
        'episodes': episodes, 'train_eps': train_eps, 'test_eps': test_eps,
        'state_std': state_std, 'action_std': action_std, 'delta_std': delta_std,
        'train_obs': train_obs, 'train_action': train_action, 'train_deltas': train_deltas,
    }


def train_model(data, config, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    state_std = data['state_std']
    action_std = data['action_std']
    delta_std = data['delta_std']
    dt = 1.0 / 30.0

    X = np.hstack([data['train_obs'] / state_std, data['train_action'].reshape(-1, 1) / action_std])
    Y = data['train_deltas'] / (delta_std * dt)

    X_t = torch.FloatTensor(X)
    Y_t = torch.FloatTensor(Y)

    ds = torch.utils.data.TensorDataset(X_t, Y_t)
    loader = torch.utils.data.DataLoader(ds, batch_size=config['batch_size'], shuffle=True)

    model = LargeCouplingNetwork(
        state_dim=STATE_DIM, action_dim=ACTION_DIM,
        coupling_dim=config['coupling_dim'], hidden=config['hidden'],
        encoder_layers=config['encoder_layers'], decoder_layers=config['decoder_layers'],
        decoder_hidden=config.get('decoder_hidden', config['hidden'])
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")

    opt = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config['n_epochs'])

    start_time = time.time()
    model.train()

    for epoch in range(config['n_epochs']):
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in loader:
            s = xb[:, :STATE_DIM]
            a = xb[:, STATE_DIM:]
            pred = model(s, a)
            loss = nn.functional.mse_loss(pred, yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()

        if (epoch + 1) % 20 == 0:
            avg_loss = epoch_loss / max(n_batches, 1)
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch+1}/{config['n_epochs']}: loss={avg_loss:.6f}, time={elapsed:.1f}s")

    model.eval()
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.1f}s")
    return model, state_std, action_std, delta_std, n_params


def evaluate_model_fast(model, data, state_std, action_std, delta_std, horizons, n_segments=3, seed=42):
    """Fast evaluation with fewer segments and shorter max horizon."""
    dt = 1.0 / 30.0
    episodes = data['episodes']
    test_eps = data['test_eps']

    np.random.seed(seed)
    segments = []
    for ep_idx in test_eps:
        ep = episodes[ep_idx]
        if ep['length'] >= 300:
            segments.append(ep)
    segments = segments[:n_segments]

    results = {}
    for h in horizons:
        nmae_list = []
        survival_list = []
        per_state_nmae = {name: [] for name in STATE_NAMES_7D}

        for seg in segments:
            s0 = seg['obs'][0].copy()
            actions_seg = seg['action'].flatten()
            real_states = seg['obs']
            n = min(h, len(actions_seg))
            s_cur = s0.copy()
            survived = True
            step_errors = []

            for step in range(n):
                try:
                    s_norm = torch.FloatTensor(s_cur / state_std).unsqueeze(0)
                    a_norm = torch.FloatTensor([actions_seg[step] / action_std]).unsqueeze(0)
                    with torch.no_grad():
                        dsdt_norm = model(s_norm, a_norm).numpy()[0]
                    dsdt = dsdt_norm * delta_std * dt
                    s_next = s_cur + dsdt

                    if np.any(np.isnan(s_next)) or np.any(np.isinf(s_next)):
                        survived = False
                        break
                    if not check_survival(s_next):
                        survived = False
                        break
                    if step + 1 < len(real_states):
                        step_err = np.abs(s_next - real_states[step + 1]) / state_std
                        step_errors.append(step_err)
                    s_cur = s_next
                except:
                    survived = False
                    break

            if step_errors:
                errors = np.array(step_errors[:min(len(step_errors), n)])
                nmae_per_state = np.mean(errors, axis=0)
                nmae_overall = np.mean(nmae_per_state)
                nmae_list.append(nmae_overall)
                for i, name in enumerate(STATE_NAMES_7D):
                    per_state_nmae[name].append(nmae_per_state[i])
            else:
                nmae_list.append(float('nan'))
            survival_list.append(survived and len(step_errors) >= n - 1)

        valid_nmae = [x for x in nmae_list if not np.isnan(x)]
        results[h] = {
            'nmae_mean': float(np.nanmean(valid_nmae)) if valid_nmae else float('nan'),
            'nmae_std': float(np.nanstd(valid_nmae)) if valid_nmae else float('nan'),
            'survival_rate': float(np.mean(survival_list)),
            'per_state_nmae': {
                name: {'mean': float(np.nanmean(per_state_nmae[name])) if per_state_nmae[name] else float('nan')}
                for name in STATE_NAMES_7D
            },
        }

    return results


def main():
    print("=" * 60)
    print("Optimize Coupling Network - Larger Capacity (Fast)")
    print("=" * 60)
    print(f"Current best: Improved Coupling (256 dim) = Primary 0.4712")
    print(f"Fast eval: 3 segments, horizons up to H=200")
    print("=" * 60)

    configs = [
        {'name': 'baseline_256', 'coupling_dim': 256, 'hidden': 256,
         'encoder_layers': 3, 'decoder_layers': 3, 'lr': 1e-3, 'n_epochs': 80, 'batch_size': 512},
        {'name': 'coupling_512', 'coupling_dim': 512, 'hidden': 512,
         'encoder_layers': 3, 'decoder_layers': 3, 'lr': 1e-3, 'n_epochs': 80, 'batch_size': 512},
        {'name': 'deep6_256', 'coupling_dim': 256, 'hidden': 256,
         'encoder_layers': 6, 'decoder_layers': 3, 'lr': 1e-3, 'n_epochs': 80, 'batch_size': 512},
        {'name': 'deep6_512', 'coupling_dim': 512, 'hidden': 512,
         'encoder_layers': 6, 'decoder_layers': 3, 'lr': 5e-4, 'n_epochs': 80, 'batch_size': 512},
        {'name': 'deep8_512', 'coupling_dim': 512, 'hidden': 512,
         'encoder_layers': 8, 'decoder_layers': 3, 'lr': 5e-4, 'n_epochs': 80, 'batch_size': 512},
        {'name': 'wide_dec_512', 'coupling_dim': 512, 'hidden': 256,
         'encoder_layers': 3, 'decoder_layers': 3, 'decoder_hidden': 512, 'lr': 1e-3, 'n_epochs': 80, 'batch_size': 512},
    ]

    print("\n1. Loading data...")
    data = load_data(seed=42)

    # Use horizons up to 200 for speed
    horizons = [1, 10, 50, 100, 200]
    all_results = {}

    for i, config in enumerate(configs):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(configs)}] Config: {config['name']}")
        print(f"  Coupling: {config['coupling_dim']}, Hidden: {config['hidden']}")
        print(f"  Encoder layers: {config['encoder_layers']}, Decoder layers: {config['decoder_layers']}")
        decoder_hidden = config.get('decoder_hidden', config['hidden'])
        if decoder_hidden != config['hidden']:
            print(f"  Decoder hidden: {decoder_hidden}")
        print(f"{'='*60}")

        model, state_std, action_std, delta_std, n_params = train_model(data, config, seed=42)

        print(f"\nEvaluating (fast)...")
        results = evaluate_model_fast(model, data, state_std, action_std, delta_std, horizons)

        print(f"\nResults for {config['name']}:")
        print(f"{'Horizon':<10} {'NMAE':<12} {'Survival':<12}")
        print("-" * 34)
        for h in horizons:
            r = results[h]
            print(f"H={h:<7} {r['nmae_mean']:<12.4f} {r['survival_rate']:<12.2%}")

        # Primary score: mean of H=100 and H=200 (no H=500 in fast eval)
        primary = np.mean([results[100]['nmae_mean'], results[200]['nmae_mean']])
        print(f"\nPrimaryScore (avg H=100,200): {primary:.4f}")

        all_results[config['name']] = {
            'config': config,
            'results': results,
            'primary': primary,
            'n_params': n_params,
        }

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: Large Coupling Network Results")
    print("=" * 60)

    print(f"\n{'Config':<20} {'Params':<12} {'H=50':<10} {'H=100':<10} {'H=200':<10} {'Primary':<10}")
    print("-" * 72)

    best_name = None
    best_primary = float('inf')

    for name, data_dict in all_results.items():
        r = data_dict['results']
        primary = data_dict['primary']
        n_params = data_dict['n_params']
        if primary < best_primary:
            best_primary = primary
            best_name = name
        print(f"{name:<20} {n_params:<12,} {r[50]['nmae_mean']:<10.4f} {r[100]['nmae_mean']:<10.4f} "
              f"{r[200]['nmae_mean']:<10.4f} {primary:<10.4f}")

    # Compare
    print(f"\nComparison with baseline (256 dim):")
    baseline_primary = all_results['baseline_256']['primary']
    print(f"{'Config':<20} {'Primary':<10} {'Delta':<12} {'Change%':<12}")
    print("-" * 54)
    for name, data_dict in all_results.items():
        primary = data_dict['primary']
        delta = primary - baseline_primary
        change = (baseline_primary - primary) / baseline_primary * 100
        marker = " <== BEST" if name == best_name else ""
        print(f"{name:<20} {primary:<10.4f} {delta:<+12.4f} {change:>+10.2f}%{marker}")

    # Per-state analysis
    print(f"\nPer-state NMAE at H=200:")
    header = f"{'Config':<20}" + "".join(f"{s:<12}" for s in STATE_NAMES_7D)
    print(header)
    print("-" * (20 + 12 * len(STATE_NAMES_7D)))
    for name, data_dict in all_results.items():
        r = data_dict['results']
        row = f"{name:<20}"
        for sname in STATE_NAMES_7D:
            val = r[200]['per_state_nmae'][sname]['mean']
            row += f"{val:<12.4f}"
        print(row)

    # Analysis
    print("\n" + "=" * 60)
    print("ANALYSIS: Why Larger Capacity May Help")
    print("=" * 60)
    print("""
Key findings from this experiment:
1. Coupling dimension (256 vs 512): Tests if the bottleneck limits representation
2. Encoder depth (3 vs 6 vs 8): Tests if deeper hierarchical features help
3. Decoder width: Tests if wider decoders improve per-state predictions
4. Combined depth + width: Tests if both improvements are complementary

If larger capacity helps -> state interactions are more complex than 256 dims can capture
If no improvement -> 256 was already sufficient, risk of overfitting increases
If degradation -> larger models harder to optimize, need more data or regularization
""")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'run_id': '20260628_175824_neural_ode_144h',
        'experiment': 'optimize_coupling_large',
        'baseline': {
            'name': 'improved_coupling_256',
            'primary_ref': 0.4712,
        },
        'eval_mode': 'fast (3 segments, H up to 200)',
        'configs': [c['name'] for c in configs],
        'results': {
            k: {
                'primary': v['primary'],
                'n_params': v['n_params'],
                'h50': v['results'][50]['nmae_mean'],
                'h100': v['results'][100]['nmae_mean'],
                'h200': v['results'][200]['nmae_mean'],
                'survival_h200': v['results'][200]['survival_rate'],
                'per_state_h200': {
                    sname: v['results'][200]['per_state_nmae'][sname]['mean']
                    for sname in STATE_NAMES_7D
                },
            }
            for k, v in all_results.items()
        },
        'best_config': best_name,
        'best_primary': float(best_primary),
    }

    output_path = 'D:/系统辨识作业/sindy_bicycle/research_72h/05_candidates/EXP055_optimize_coupling_large.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    return all_results, best_name, best_primary


if __name__ == '__main__':
    all_results, best_name, best_primary = main()
