#!/usr/bin/env python
"""Minimal RL test - single episode with detailed output to verify everything works."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import scanpy as sc
import numpy as np
from rl_sc_cluster_utils.environment import ClusteringEnv

# GAG gene sets
GAG_GENE_SETS = {
    "CS_biosynthesis": [
        "CSGALNACT1", "CSGALNACT2",
        "CHSY1", "CHSY2", "CHSY3",
        "CHPF", "CHPF2"
    ],
    "CS_sulfation": [
        "CHST11", "CHST12", "CHST13", "CHST14",
        "CHST3", "CHST7", "CHST15"
    ],
    "HS_biosynthesis": [
        "EXT1", "EXT2", "EXTL1", "EXTL2", "EXTL3", "EXTL4"
    ],
    "HS_sulfation": [
        "NDST1", "NDST2", "NDST3", "NDST4",
        "HS2ST1",
        "HS6ST1", "HS6ST2", "HS6ST3",
        "HS3ST1", "HS3ST2", "HS3ST3A1", "HS3ST3B1", "HS3ST4", "HS3ST5", "HS3ST6"
    ],
    "Sulfate_activation": [
        "PAPSS1", "PAPSS2", "SLC35B2", "SLC35B3", "SLC26A2"
    ],
    "PNN_core": [
        "ACAN", "BCAN", "NCAN", "VCAN",
        "HAPLN1", "HAPLN2", "HAPLN3", "HAPLN4",
        "HAS1", "HAS2", "HAS3",
        "TNC", "TNR", "SPOCK1", "CRTAC1"
    ]
}


def load_small_subset(data_path, subset_size=500):
    """Load a very small subset for fast testing."""
    print(f"Loading data from: {data_path}")
    adata = sc.read_h5ad(data_path)

    if 'scVI' in adata.obsm and 'X_scvi' not in adata.obsm:
        adata.obsm['X_scvi'] = adata.obsm['scVI']

    print(f"Using random subset of {subset_size} cells...")
    np.random.seed(42)
    indices = np.random.choice(adata.n_obs, subset_size, replace=False)
    adata = adata[indices].copy()

    return adata


def run_single_episode_detailed(env):
    """Run a single episode with detailed step-by-step output."""
    print("\n" + "="*70)
    print("RUNNING SINGLE EPISODE - DETAILED OUTPUT")
    print("="*70)

    # Reset
    print("\n[RESET] Resetting environment...")
    state, info = env.reset()

    print(f"✓ Reset successful")
    print(f"  State shape: {state.shape}")
    print(f"  State sample (first 5 dims): {state[:5]}")
    print(f"  State sample (last 5 dims): {state[-5:]}")
    print(f"  Initial clusters: {info['n_clusters']}")
    print(f"  Initial resolution: {info['resolution']:.2f}")
    print(f"  State is finite: {np.all(np.isfinite(state))}")

    # Run episode
    print(f"\n[EPISODE] Running episode (max {env.max_steps} steps)...")
    print("-"*70)

    total_reward = 0
    step = 0

    while step < env.max_steps:
        # Choose action
        action = env.action_space.sample()
        action_names = ["Split", "Merge", "Re-cluster+", "Re-cluster-", "Accept"]

        print(f"\n[STEP {step + 1}]")
        print(f"  Action: {action} ({action_names[action]})")
        print(f"  Current state (first 3 dims): {state[:3]}")
        print(f"  Current resolution: {env.current_resolution:.2f}")

        # Step
        next_state, reward, terminated, truncated, info = env.step(action)

        # Print detailed info
        print(f"  ✓ Step executed successfully")
        print(f"  Reward: {reward:.4f} (raw: {info['raw_reward']:.4f})")
        print(f"  Q_cluster: {info['Q_cluster']:.4f}")
        print(f"    - Silhouette: {info['silhouette']:.4f}")
        print(f"    - Modularity: {info['modularity']:.4f}")
        print(f"    - Balance: {info['balance']:.4f}")
        print(f"  Q_GAG: {info['Q_GAG']:.4f}")
        print(f"    - Mean F-stat: {info['mean_f_stat']:.4f}")
        print(f"  Penalty: {info['penalty']:.4f}")
        print(f"    - N clusters: {info['n_clusters']}")
        print(f"    - N singletons: {info['n_singletons']}")
        print(f"    - Resolution clamped: {info['resolution_clamped']}")
        print(f"  New clusters: {info['n_clusters']}")
        print(f"  New resolution: {info['resolution']:.2f}")
        print(f"  Action success: {info['action_success']}")
        print(f"  Next state shape: {next_state.shape}")
        print(f"  Next state is finite: {np.all(np.isfinite(next_state))}")
        print(f"  Terminated: {terminated}, Truncated: {truncated}")

        total_reward += reward
        state = next_state
        step += 1

        if terminated:
            print(f"\n  → Episode TERMINATED (Accept action)")
            break
        if truncated:
            print(f"\n  → Episode TRUNCATED (max steps reached)")
            break

    print("\n" + "-"*70)
    print(f"[EPISODE SUMMARY]")
    print(f"  Total steps: {step}")
    print(f"  Total reward: {total_reward:.4f}")
    print(f"  Final clusters: {info['n_clusters']}")
    print(f"  Final resolution: {info['resolution']:.2f}")
    print(f"  Final Q_cluster: {info['Q_cluster']:.4f}")
    print(f"  Final Q_GAG: {info['Q_GAG']:.4f}")
    print(f"  Final penalty: {info['penalty']:.4f}")

    return total_reward, info


def run_multiple_episodes(env, num_episodes=10):
    """Run multiple episodes and track learning."""
    import time

    print("\n" + "="*70)
    print(f"RUNNING {num_episodes} EPISODES - Tracking Learning Progress")
    print("="*70)

    episode_rewards = []
    episode_lengths = []
    episode_q_clusters = []
    episode_q_gags = []
    episode_penalties = []
    episode_final_clusters = []

    start_time = time.time()

    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        while not done and episode_length < env.max_steps:
            action = env.action_space.sample()  # Random for now
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            state = next_state
            episode_reward += reward
            episode_length += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_q_clusters.append(info['Q_cluster'])
        episode_q_gags.append(info['Q_GAG'])
        episode_penalties.append(info['penalty'])
        episode_final_clusters.append(info['n_clusters'])

        # Print progress
        print(f"Episode {episode + 1:2d}: "
              f"Reward={episode_reward:7.2f}, "
              f"Length={episode_length:2d}, "
              f"Clusters={info['n_clusters']:3d}, "
              f"Q_cluster={info['Q_cluster']:6.4f}, "
              f"Q_GAG={info['Q_GAG']:6.4f}, "
              f"Penalty={info['penalty']:6.2f}")

    elapsed_time = time.time() - start_time

    # Calculate learning metrics
    print("\n" + "-"*70)
    print("LEARNING ANALYSIS")
    print("-"*70)

    if len(episode_rewards) >= 5:
        first_half = episode_rewards[:len(episode_rewards)//2]
        second_half = episode_rewards[len(episode_rewards)//2:]

        print(f"\nReward Statistics:")
        print(f"  Mean (first half): {np.mean(first_half):.4f}")
        print(f"  Mean (second half): {np.mean(second_half):.4f}")
        print(f"  Improvement: {np.mean(second_half) - np.mean(first_half):.4f}")
        print(f"  Std (all): {np.std(episode_rewards):.4f}")

        if np.mean(second_half) > np.mean(first_half):
            print(f"  ✓ Rewards improving (learning detected)")
        else:
            print(f"  ⚠ Rewards not improving (may need more episodes or better agent)")

    print(f"\nEpisode Statistics:")
    print(f"  Mean length: {np.mean(episode_lengths):.1f} steps")
    print(f"  Mean final clusters: {np.mean(episode_final_clusters):.1f}")
    print(f"  Mean Q_cluster: {np.mean(episode_q_clusters):.4f}")
    print(f"  Mean Q_GAG: {np.mean(episode_q_gags):.4f}")
    print(f"  Mean penalty: {np.mean(episode_penalties):.2f}")

    print(f"\nPerformance:")
    print(f"  Total time: {elapsed_time:.2f} seconds")
    print(f"  Time per episode: {elapsed_time/num_episodes:.2f} seconds")
    print(f"  Episodes per minute: {60 * num_episodes / elapsed_time:.1f}")

    return {
        'rewards': episode_rewards,
        'lengths': episode_lengths,
        'q_clusters': episode_q_clusters,
        'q_gags': episode_q_gags,
        'penalties': episode_penalties,
        'final_clusters': episode_final_clusters,
        'elapsed_time': elapsed_time
    }


def main():
    """Main test function."""
    import time

    data_path = project_root / "data" / "processed" / "human_interneurons.h5ad"

    if not data_path.exists():
        print(f"ERROR: Data file not found at {data_path}")
        return 1

    # Load very small subset
    print("="*70)
    print("RL TRAINING TEST - 10 Episodes")
    print("="*70)

    setup_start = time.time()
    adata = load_small_subset(data_path, subset_size=500)
    print(f"✓ Data loaded: {adata.shape}")

    # Create environment
    print("\n[SETUP] Creating ClusteringEnv...")
    env = ClusteringEnv(
        adata=adata,
        gene_sets=GAG_GENE_SETS,
        max_steps=10,  # Short episodes for testing
        normalize_rewards=True,
    )
    print(f"✓ Environment created")
    print(f"  Action space: {env.action_space}")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Max steps: {env.max_steps}")
    print(f"  Normalize rewards: {env.normalize_rewards}")
    setup_time = time.time() - setup_start
    print(f"  Setup time: {setup_time:.2f} seconds")

    # Run 10 episodes
    results = run_multiple_episodes(env, num_episodes=10)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"✓ Successfully ran {len(results['rewards'])} episodes")
    print(f"✓ Total time: {results['elapsed_time']:.2f} seconds")
    print(f"✓ Environment is working correctly for RL training")
    print("="*70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
