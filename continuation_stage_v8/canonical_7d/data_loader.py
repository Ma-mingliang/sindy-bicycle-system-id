"""Load and prepare 7D real bicycle data."""
import numpy as np
from pathlib import Path
from typing import Tuple
from .config import DataConfig, STATE_NAMES_7D, IDX_7D_FROM_8D


def load_7d_data(config: DataConfig = None) -> dict:
    """Load stage2 dataset and extract 7D state.

    Returns dict with:
        train_states, train_actions, train_deltas,
        test_states, test_actions, test_deltas,
        state_std, action_std, delta_std,
        n_episodes, episode_boundaries
    """
    cfg = config or DataConfig()
    data = np.load(cfg.data_path, allow_pickle=True)

    obs_8d = data['obs']  # (N, 8)
    next_obs_8d = data['next_obs']  # (N, 8)
    actions = data['action'].flatten()  # (N,)
    done = data['done']  # (N,)
    n_episodes = int(data['n_episodes'])

    # Extract 7D: drop kappa (index 5)
    states_7d = obs_8d[:, IDX_7D_FROM_8D].astype(np.float64)
    next_states_7d = next_obs_8d[:, IDX_7D_FROM_8D].astype(np.float64)
    deltas_7d = next_states_7d - states_7d
    actions = actions.astype(np.float64)

    # Find episode boundaries from done flags
    ep_starts = []
    ep_ends = []
    start = 0
    for i in range(len(done)):
        if done[i]:
            ep_starts.append(start)
            ep_ends.append(i + 1)
            start = i + 1
    if start < len(done):
        ep_starts.append(start)
        ep_ends.append(len(done))

    actual_n_episodes = len(ep_starts)

    # Train/test split by episodes
    rng = np.random.RandomState(cfg.seed)
    n_train = int(actual_n_episodes * cfg.train_ratio)
    perm = rng.permutation(actual_n_episodes)
    train_eps = set(perm[:n_train])
    test_eps = set(perm[n_train:])

    train_mask = np.zeros(len(states_7d), dtype=bool)
    test_mask = np.zeros(len(states_7d), dtype=bool)
    for ep_idx in train_eps:
        train_mask[ep_starts[ep_idx]:ep_ends[ep_idx]] = True
    for ep_idx in test_eps:
        test_mask[ep_starts[ep_idx]:ep_ends[ep_idx]] = True

    train_states = states_7d[train_mask]
    train_actions = actions[train_mask]
    train_deltas = deltas_7d[train_mask]
    test_states = states_7d[test_mask]
    test_actions = actions[test_mask]
    test_deltas = deltas_7d[test_mask]

    # Compute std from training data
    state_std = np.std(train_states, axis=0)
    action_std = float(np.std(train_actions))
    delta_std = np.std(train_deltas, axis=0)

    # Avoid division by zero
    state_std[state_std < 1e-10] = 1.0
    delta_std[delta_std < 1e-10] = 1.0
    if action_std < 1e-10:
        action_std = 1.0

    return {
        'train_states': train_states,
        'train_actions': train_actions,
        'train_deltas': train_deltas,
        'test_states': test_states,
        'test_actions': test_actions,
        'test_deltas': test_deltas,
        'state_std': state_std,
        'action_std': action_std,
        'delta_std': delta_std,
        'n_episodes': n_episodes,
        'episode_starts': ep_starts,
        'episode_ends': ep_ends,
        'train_mask': train_mask,
        'test_mask': test_mask,
        'all_states': states_7d,
        'all_actions': actions,
        'all_deltas': deltas_7d,
        'all_done': done,
    }


def get_test_segments(data: dict, n_segments: int = 10,
                      segment_length: int = 200, seed: int = 42) -> list:
    """Extract test segments for multi-step evaluation.

    Returns list of dicts, each with:
        states: (segment_length+1, 7)
        actions: (segment_length,)
        start_idx: int
    """
    rng = np.random.RandomState(seed)
    test_mask = data['test_mask']
    test_indices = np.where(test_mask)[0]

    # Find continuous runs in test data
    runs = []
    run_start = test_indices[0]
    for i in range(1, len(test_indices)):
        if test_indices[i] != test_indices[i - 1] + 1:
            runs.append((run_start, test_indices[i - 1] + 1))
            run_start = test_indices[i]
    runs.append((run_start, test_indices[-1] + 1))

    # Sample segments from runs
    segments = []
    attempts = 0
    while len(segments) < n_segments and attempts < n_segments * 10:
        attempts += 1
        run_idx = rng.randint(len(runs))
        r_start, r_end = runs[run_idx]
        if r_end - r_start < segment_length + 1:
            continue
        offset = rng.randint(r_start, r_end - segment_length)
        seg = {
            'states': data['all_states'][offset:offset + segment_length + 1],
            'actions': data['all_actions'][offset:offset + segment_length],
            'start_idx': offset,
        }
        segments.append(seg)

    return segments
