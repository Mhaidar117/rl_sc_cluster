#!/usr/bin/env python
"""Example script showing how to use ClusteringEnv with real data."""

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


def load_data(data_path, use_subset=True, subset_size=10000):
    """Load and prepare AnnData."""
    print(f"Loading data from: {data_path}")
    adata = sc.read_h5ad(data_path)

    # Map embedding key if needed
    if 'scVI' in adata.obsm and 'X_scvi' not in adata.obsm:
        adata.obsm['X_scvi'] = adata.obsm['scVI']

    if use_subset:
        print(f"Using subset of {subset_size} cells for faster testing...")
        adata = adata[:subset_size].copy()

    return adata


def run_episode(env, max_steps=15, strategy='random'):
    """Run a single episode."""
    state, info = env.reset()

    print(f"\nEpisode started:")
    print(f"  Initial clusters: {info['n_clusters']}")
    print(f"  Resolution: {info['resolution']:.2f}")

    episode_reward = 0
    history = []

    for step in range(max_steps):
        # Choose action based on strategy
        if strategy == 'random':
            action = env.action_space.sample()
        elif strategy == 'explore':
            # Explore: alternate between actions
            action = step % 4  # Actions 0-3 (not 4=accept)
        else:
            action = 0  # Default: split

        state, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        history.append({
            'step': step + 1,
            'action': action,
            'reward': reward,
            'raw_reward': info['raw_reward'],
            'n_clusters': info['n_clusters'],
            'Q_cluster': info['Q_cluster'],
            'Q_GAG': info['Q_GAG'],
            'penalty': info['penalty'],
        })

        if step % 3 == 0:  # Print every 3 steps
            print(f"  Step {step+1}: action={action}, "
                  f"reward={reward:.4f}, clusters={info['n_clusters']}, "
                  f"Q_cluster={info['Q_cluster']:.4f}, "
                  f"Q_GAG={info['Q_GAG']:.4f}")

        if terminated or truncated:
            print(f"\nEpisode ended at step {step+1}: "
                  f"terminated={terminated}, truncated={truncated}")
            break

    print(f"\nEpisode summary:")
    print(f"  Total reward: {episode_reward:.4f}")
    print(f"  Final clusters: {info['n_clusters']}")
    print(f"  Final resolution: {info['resolution']:.2f}")
    print(f"  Final Q_cluster: {info['Q_cluster']:.4f}")
    print(f"  Final Q_GAG: {info['Q_GAG']:.4f}")
    print(f"  Final penalty: {info['penalty']:.4f}")

    return history, info


def main():
    """Main example."""
    data_path = project_root / "data" / "processed" / "human_interneurons.h5ad"

    if not data_path.exists():
        print(f"ERROR: Data file not found at {data_path}")
        return 1

    # Load data
    adata = load_data(data_path, use_subset=True, subset_size=10000)
    print(f"Data shape: {adata.shape}")

    # Create environment
    print("\nCreating ClusteringEnv...")
    env = ClusteringEnv(
        adata=adata,
        gene_sets=GAG_GENE_SETS,
        max_steps=15,
        normalize_rewards=True,  # Default: enabled
    )

    # Run a few episodes
    print("\n" + "="*60)
    print("Running episodes...")
    print("="*60)

    for episode in range(3):
        print(f"\n{'='*60}")
        print(f"Episode {episode + 1}")
        print(f"{'='*60}")

        history, final_info = run_episode(env, max_steps=15, strategy='explore')

    print("\n" + "="*60)
    print("Example completed!")
    print("="*60)
    print("\nNext steps:")
    print("1. Train an RL agent (e.g., PPO) on this environment")
    print("2. Tune reward weights (alpha, beta, delta) if needed")
    print("3. Use larger subset or full dataset for training")

    return 0


if __name__ == "__main__":
    sys.exit(main())
