#!/usr/bin/env python
"""Test PPO with 1000 cells, 10 episodes, save all metrics."""

import sys
from pathlib import Path
import time
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("ERROR: stable-baselines3 not installed")
    print("Install with: pip install stable-baselines3")
    sys.exit(1)

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


def load_subset(data_path, n_cells=1000):
    """Load subset of dataset."""
    print(f"[LOAD] Loading data from: {data_path}")
    adata = sc.read_h5ad(data_path)

    if 'scVI' in adata.obsm and 'X_scvi' not in adata.obsm:
        adata.obsm['X_scvi'] = adata.obsm['scVI']

    print(f"[LOAD] Using first {n_cells} cells (out of {adata.n_obs})")
    adata = adata[:n_cells].copy()

    # Compute neighbors if missing (crucial for modularity score)
    if "neighbors" not in adata.uns:
        print("[LOAD] Computing neighbors graph for modularity...")
        sc.pp.neighbors(adata, use_rep='X_scvi' if 'X_scvi' in adata.obsm else None)

    print(f"[LOAD] ✓ Data loaded: {adata.shape}")
    return adata


def plot_trajectories(all_step_data, output_path):
    """Plot trajectories for all 10 episodes."""
    print(f"[PLOT] Creating trajectory visualization...")
    df = pd.DataFrame(all_step_data)

    # Setup grid: 10 rows (one per episode), 4 columns (Actions, Reward, Raw Metrics, Weighted Components)
    fig = plt.figure(figsize=(25, 30))
    gs = gridspec.GridSpec(10, 4, width_ratios=[1, 1, 1, 1])

    episodes = sorted(df['episode'].unique())

    action_labels = {0: 'Split', 1: 'Merge', 2: 'Res+', 3: 'Res-', 4: 'Accept'}
    action_colors = {0: 'red', 1: 'blue', 2: 'green', 3: 'orange', 4: 'purple'}

    for i, ep in enumerate(episodes):
        ep_data = df[df['episode'] == ep].sort_values('step')
        steps = ep_data['step']

        # 1. Action Timeline
        ax0 = plt.subplot(gs[i, 0])
        actions = ep_data['action']
        colors = [action_colors[a] for a in actions]
        ax0.scatter(steps, actions, c=colors, s=100)
        ax0.plot(steps, actions, c='gray', alpha=0.3, linestyle='--')
        ax0.set_yticks(list(action_labels.keys()))
        ax0.set_yticklabels(list(action_labels.values()))
        ax0.set_title(f"Episode {ep}: Actions")
        ax0.set_ylim(-0.5, 4.5)
        ax0.grid(True, alpha=0.3)

        # 2. Reward Trajectory
        ax1 = plt.subplot(gs[i, 1])
        rewards = ep_data['reward']
        ax1.plot(steps, rewards, marker='o', c='black')
        ax1.axhline(0, color='red', linestyle='--', alpha=0.5)
        ax1.set_title(f"Episode {ep}: Reward")
        ax1.grid(True, alpha=0.3)

        # 3. Raw Metrics (Silhouette, Modularity, GAG)
        ax2 = plt.subplot(gs[i, 2])
        # Plot silhouette (scaled -1 to 1)
        ax2.plot(steps, ep_data['silhouette'], label='Silhouette', marker='x', c='blue', linestyle='--')
        ax2.plot(steps, ep_data['modularity'], label='Modularity', marker='+', c='cyan', linestyle='--')
        ax2.plot(steps, ep_data['Q_GAG'], label='Q_GAG (Raw)', marker='o', c='orange', linewidth=2)
        ax2.legend()
        ax2.set_title(f"Episode {ep}: Raw Metrics")
        ax2.grid(True, alpha=0.3)

        # 4. Weighted Components (What the agent sees)
        ax3 = plt.subplot(gs[i, 3])
        # Reconstruct weighted values based on params (Alpha=0.2, Delta=0.01)
        # And the Non-Linear GAG used in wrapper: (Q_GAG * 6)^2
        w_cluster = ep_data['Q_cluster'] * 0.2
        w_gag_nonlinear = (ep_data['Q_GAG'] * 6.0) ** 2
        w_penalty = ep_data['penalty'] * 0.01

        ax3.plot(steps, w_cluster, label='0.2 * Q_cluster', c='blue')
        ax3.plot(steps, w_gag_nonlinear, label='(Q_GAG * 6)^2', c='orange', linewidth=2)
        ax3.plot(steps, w_penalty, label='0.01 * Penalty', c='red', linestyle=':')

        ax3.legend()
        ax3.set_title(f"Episode {ep}: Effective Potentials")
        ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[PLOT] ✓ Saved to: {output_path}")


import gymnasium as gym

class DeltaRewardWrapper(gym.Wrapper):
    """
    Reward wrapper that changes the reward to be the improvement in quality
    rather than the absolute quality.

    R_t = Potential(S_t) - Potential(S_{t-1})
    """
    def __init__(self, env):
        super().__init__(env)
        self.last_potential = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # Calculate initial custom potential
        raw_gag = info.get('Q_GAG', 0.0)
        # Scaling reduced from 10.0 to 6.0 to balance with Q_cluster
        transformed_gag = (raw_gag * 6.0) ** 2
        q_cluster = info.get('Q_cluster', 0.0)
        penalty = info.get('penalty', 0.0)
        weighted_penalty = penalty * 0.01

        self.last_potential = (0.2 * q_cluster) + (1.0 * transformed_gag) - weighted_penalty

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Calculate NON-LINEAR GAG potential
        raw_gag = info.get('Q_GAG', 0.0)
        # Scaling reduced from 10.0 to 6.0
        transformed_gag = (raw_gag * 6.0) ** 2

        q_cluster = info.get('Q_cluster', 0.0)
        penalty = info.get('penalty', 0.0)
        weighted_penalty = penalty * 0.01

        # Custom Potential
        current_potential = (0.2 * q_cluster) + (1.0 * transformed_gag) - weighted_penalty

        # Calculate shaping reward
        if action == 4:
            if self.env.current_step < 5:
                shaping_reward = -5.0
            else:
                shaping_reward = 0.0
        else:
            shaping_reward = current_potential - self.last_potential
            # Exploration bonus
            shaping_reward += 0.2

        self.last_potential = current_potential

        step_penalty = 0.0
        final_reward = shaping_reward + step_penalty

        return obs, final_reward, terminated, truncated, info

def make_env(adata, gene_sets):
    """Create environment factory."""
    def _init():
        env = ClusteringEnv(
            adata=adata,
            gene_sets=gene_sets,
            max_steps=15,
            normalize_rewards=False, # We handle normalization/shaping in the wrapper
            reward_delta=0.01,  # Reduced penalty weight
            reward_alpha=0.2,   # Lower weight on standard clustering quality
            reward_beta=2.0,    # Super-High weight on GAG enrichment (Biology)
        )
        return DeltaRewardWrapper(env)
    return _init


def run_episodes_with_ppo(env, num_episodes=10, total_timesteps=4000):
    """Run episodes using PPO, collect all metrics."""
    print(f"\n[PPO] Creating PPO model...")

    # Create vectorized env with our wrapper
    vec_env = DummyVecEnv([make_env(env.adata, env.gene_sets)])

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=3e-4,
        n_steps=200,    # Increased n_steps
        batch_size=50,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.15,  # Increased entropy to force exploration (was 0.05)
        vf_coef=0.5,
        device="cpu",
        verbose=1,
        tensorboard_log=None,
    )
    print(f"[PPO] ✓ Model created")

    print(f"\n[PPO] Training for {total_timesteps} timesteps...")
    print(f"[PPO] Using Delta-Reward (Improvement-based) and Entropy=0.05")
    print(f"[PPO] Training started...", flush=True)

    training_start = time.time()
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    training_time = time.time() - training_start

    print(f"[PPO] ✓ Training completed in {training_time:.2f} seconds ({training_time/60:.1f} minutes)")

    # Now run episodes with trained model to collect metrics
    print(f"\n[COLLECT] Collecting metrics from {num_episodes} episodes...")
    print(f"[COLLECT] Running episodes...", flush=True)
    print(f"[COLLECT] NOTE: Using RAW rewards (no normalization) for evaluation", flush=True)

    # Create environment WITHOUT reward normalization for evaluation
    # This gives us true reward values, not normalized ones
    def make_eval_env():
        eval_env = ClusteringEnv(
            adata=env.adata,
            gene_sets=env.gene_sets,
            max_steps=15,
            normalize_rewards=False,  # Disable normalization for evaluation
            reward_delta=0.01,  # Match training penalty weight
            reward_alpha=0.2,   # Match training
            reward_beta=2.0,    # Match training
        )
        return eval_env

    vec_env = DummyVecEnv([make_eval_env])
    all_step_data = []  # Collect ALL steps from ALL episodes

    for episode in range(num_episodes):
        print(f"\n[EPISODE {episode+1}/{num_episodes}] Starting...", flush=True)
        obs = vec_env.reset()
        episode_reward = 0
        episode_length = 0

        done = np.array([False])  # Initialize as array for vectorized env
        action_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}  # Track action distribution
        last_step_data = {}

        while not done[0] and episode_length < 15:
            action, _ = model.predict(obs, deterministic=False)
            action_counts[int(action[0])] += 1
            obs, reward, done, info = vec_env.step(action)

            # Extract metrics from info
            step_info = info[0]  # Unwrap from vectorized env

            # Collect all metrics including per-gene-set F-stats
            step_data = {
                'episode': episode + 1,
                'step': episode_length + 1,
                'action': int(action[0]),
                'reward': float(reward[0]),
                'raw_reward': float(step_info.get('raw_reward', 0)),
                'Q_cluster': float(step_info.get('Q_cluster', 0)),
                'Q_GAG': float(step_info.get('Q_GAG', 0)),
                'penalty': float(step_info.get('penalty', 0)),
                'silhouette': float(step_info.get('silhouette', 0)),
                'modularity': float(step_info.get('modularity', 0)),
                'balance': float(step_info.get('balance', 0)),
                'n_clusters': int(step_info.get('n_clusters', 0)),
                'n_singletons': int(step_info.get('n_singletons', 0)),
                'mean_f_stat': float(step_info.get('mean_f_stat', 0)),
                'resolution': float(step_info.get('resolution', 0)),
                'resolution_clamped': bool(step_info.get('resolution_clamped', False)),
            }

            # Add per-gene-set F-statistics
            f_stats = step_info.get('f_stats', {})
            for gene_set_name in GAG_GENE_SETS.keys():
                val = f_stats.get(gene_set_name, 0.0)
                step_data[f'f_stat_{gene_set_name}'] = float(val) if hasattr(val, '__float__') else 0.0

            all_step_data.append(step_data)
            last_step_data = step_data

            episode_reward += reward[0]
            episode_length += 1

            if done[0]:
                break

        print(f"[EPISODE {episode+1}/{num_episodes}] Complete: "
              f"reward={episode_reward:.4f}, length={episode_length}, "
              f"clusters={last_step_data.get('n_clusters', 0)}, "
              f"Q_cluster={last_step_data.get('Q_cluster', 0):.4f}, "
              f"Q_GAG={last_step_data.get('Q_GAG', 0):.4f}, "
              f"actions={action_counts}", flush=True)

    return all_step_data, model


def main():
    """Main test."""
    print("="*70)
    print("PPO TEST - 1000 cells, 10 episodes")
    print("="*70)

    data_path = project_root / "data" / "processed" / "human_interneurons.h5ad"
    if not data_path.exists():
        print(f"ERROR: {data_path} not found")
        return 1

    # Load 1000 cells
    adata = load_subset(data_path, n_cells=1000)

    # Create environment
    print(f"\n[ENV] Creating environment...")
    env = ClusteringEnv(
        adata=adata,
        gene_sets=GAG_GENE_SETS,
        max_steps=15,
        normalize_rewards=True,
    )
    print(f"[ENV] ✓ Environment created")

    # Test it works
    state, info = env.reset()
    print(f"[ENV] ✓ Reset test: {info['n_clusters']} clusters")
    print(f"[ENV] ✓ F-stats available: {list(info.get('f_stats', {}).keys())}")

    # Run episodes with PPO
    episode_metrics, model = run_episodes_with_ppo(env, num_episodes=10, total_timesteps=1000)

    # Save metrics
    print(f"\n[SAVE] Saving metrics...")
    output_dir = project_root / "results" / "ppo_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as CSV
    df = pd.DataFrame(episode_metrics)
    csv_path = output_dir / "episode_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"[SAVE] ✓ Metrics saved to: {csv_path}")
    print(f"[SAVE]   Columns: {list(df.columns)}")

    # JSON saving disabled - data already converted to native Python types in step_data

    # Save model
    model_path = output_dir / "ppo_model.zip"
    model.save(str(model_path))
    print(f"[SAVE] ✓ Model saved to: {model_path}")

    # Print summary
    print(f"\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    # Calculate episode-level stats from step data
    episode_rewards = df.groupby('episode')['reward'].sum()
    episode_lengths = df.groupby('episode')['step'].max()

    print(f"Episodes completed: {df['episode'].nunique()}")
    print(f"Mean reward: {episode_rewards.mean():.4f}")
    print(f"Mean length: {episode_lengths.mean():.1f}")
    print(f"Mean clusters: {df['n_clusters'].mean():.1f}")
    print(f"Mean Q_cluster: {df['Q_cluster'].mean():.4f}")
    print(f"Mean Q_GAG: {df['Q_GAG'].mean():.4f}")
    print(f"Mean penalty: {df['penalty'].mean():.2f}")

    # Print F-stats per gene set
    print(f"\nMean F-statistics per gene set:")
    for gene_set_name in GAG_GENE_SETS.keys():
        col_name = f'f_stat_{gene_set_name}'
        if col_name in df.columns:
            mean_f = df[col_name].mean()
            print(f"  {gene_set_name}: {mean_f:.4f}")

    # Plot trajectories
    plot_path = output_dir / "trajectories.png"
    try:
        plot_trajectories(episode_metrics, str(plot_path))
    except Exception as e:
        print(f"[PLOT] Error creating plot: {e}")

    print("="*70)
    print("✓ TEST COMPLETED SUCCESSFULLY!")
    print("="*70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
