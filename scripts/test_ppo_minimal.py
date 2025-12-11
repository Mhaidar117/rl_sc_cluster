#!/usr/bin/env python
"""Test PPO with 10000 cells, 20 episodes, save all metrics."""

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
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("ERROR: stable-baselines3 not installed")
    print("Install with: pip install stable-baselines3")
    sys.exit(1)

import scanpy as sc
import numpy as np
from tqdm import tqdm
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


def plot_resolution_clusters(all_step_data, output_path):
    """Plot Resolution vs Number of Clusters Heatmap."""
    print(f"[PLOT] Creating resolution vs clusters heatmap...")
    df = pd.DataFrame(all_step_data)

    # 1. Prepare Data for Heatmap
    # We want mean reward for each (resolution, n_clusters) pair
    # Resolution is rounded to 0.1 to bin it effectively
    df['res_bin'] = df['resolution'].round(2) # Keep 2 decimal places to handle small float errors

    # Pivot table: Index=n_clusters, Columns=res_bin, Values=raw_reward
    pivot_table = df.pivot_table(
        index='n_clusters',
        columns='res_bin',
        values='raw_reward',
        aggfunc='mean'
    )

    # Sort index descending (high clusters at top)
    pivot_table = pivot_table.sort_index(ascending=False)

    # 2. Plotting
    fig, ax = plt.subplots(figsize=(12, 10))

    # Use imshow for heatmap
    # We need to handle NaNs (unvisited states) - let's mask them
    # And we need to explicitly set extent or labels because imshow works on array indices

    # Extract data as array
    data_array = pivot_table.to_numpy()

    # Create colormap
    cmap = plt.cm.viridis
    cmap.set_bad('white') # Set NaNs to white or gray

    # Plot
    im = ax.imshow(data_array, aspect='auto', cmap=cmap, interpolation='nearest')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Mean Raw Reward', rotation=270, labelpad=20)

    # Set ticks and labels
    # X-axis: Resolution bins
    x_labels = pivot_table.columns
    # We don't want every single label if there are too many, but here resolution is likely 0.1, 0.2...0.8
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels)

    # Y-axis: n_clusters
    y_labels = pivot_table.index
    # If too many Y labels, show every Nth
    if len(y_labels) > 20:
        step = len(y_labels) // 20 + 1
        y_ticks = np.arange(0, len(y_labels), step)
        y_tick_labels = y_labels[::step]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_tick_labels)
    else:
        ax.set_yticks(np.arange(len(y_labels)))
        ax.set_yticklabels(y_labels)

    ax.set_xlabel('Resolution')
    ax.set_ylabel('Number of Clusters')
    ax.set_title('State Quality Heatmap: Reward vs (Resolution, Clusters)')

    # Add annotations for visited count if not too crowded?
    # Maybe too cluttered. Let's skip for now.

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[PLOT] ✓ Saved to: {output_path}")


def plot_trajectories(all_step_data, output_path):
    """Plot trajectories for all episodes."""
    print(f"[PLOT] Creating trajectory visualization...")
    df = pd.DataFrame(all_step_data)

    episodes = sorted(df['episode'].unique())
    num_episodes = len(episodes)

    # Setup grid: dynamic rows (one per episode), 4 columns (Actions, Reward, Raw Metrics, Weighted Components)
    # Scale figure height based on number of episodes (3 units per episode)
    fig = plt.figure(figsize=(25, 3 * num_episodes))
    gs = gridspec.GridSpec(num_episodes, 4, width_ratios=[1, 1, 1, 1])

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
        # Non-Linear GAG transformation: (Q_GAG * 6)^2 (now in RewardCalculator)
        w_cluster = ep_data['Q_cluster'] * 0.2
        # Use Q_GAG_transformed if available, otherwise reconstruct
        if 'Q_GAG_transformed' in ep_data.columns:
            w_gag_nonlinear = ep_data['Q_GAG_transformed']
        else:
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


def plot_best_episode_umaps(adata, best_episode_num, initial_clusters, final_clusters, output_path):
    """
    Plot UMAP visualizations for the best episode: initial and final clustering states.

    Parameters
    ----------
    adata : AnnData
        Annotated data object
    best_episode_num : int
        Episode number (1-indexed) of the best episode
    initial_clusters : np.ndarray
        Initial clustering labels for the best episode
    final_clusters : np.ndarray
        Final clustering labels for the best episode
    output_path : str
        Path to save the plot
    """
    print(f"[PLOT] Creating UMAP plots for best episode {best_episode_num}...")

    # Ensure UMAP coordinates exist
    if 'X_umap' not in adata.obsm:
        print("[PLOT] Computing UMAP coordinates...")
        if 'X_scvi' in adata.obsm:
            sc.tl.umap(adata, use_rep='X_scvi')
        else:
            sc.tl.umap(adata)

    # Create figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Temporarily store original clusters
    original_clusters = adata.obs['clusters'].copy()

    # Plot 1: Initial clustering
    adata.obs['clusters'] = initial_clusters.astype(str)
    sc.pl.umap(adata, color='clusters', ax=axes[0], show=False,
               title=f'Episode {best_episode_num}: Initial Clustering',
               legend_loc='right margin')

    # Plot 2: Final clustering
    adata.obs['clusters'] = final_clusters.astype(str)
    sc.pl.umap(adata, color='clusters', ax=axes[1], show=False,
               title=f'Episode {best_episode_num}: Final Clustering',
               legend_loc='right margin')

    # Restore original clusters
    adata.obs['clusters'] = original_clusters

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] ✓ Saved to: {output_path}")


class TrainingMetricsCallback(BaseCallback):
    """Callback to track training metrics during PPO learning."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.timesteps = []

    def _on_step(self) -> bool:
        # Get episode statistics from info
        if 'episode' in self.locals.get('infos', [{}])[0]:
            ep_info = self.locals['infos'][0]['episode']
            self.episode_rewards.append(ep_info['r'])
            self.episode_lengths.append(ep_info['l'])
            self.timesteps.append(self.num_timesteps)
        return True


def plot_training_convergence(callback, output_path):
    """Plot training convergence curves."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Episode Rewards Over Time
    if callback.episode_rewards:
        axes[0, 0].plot(callback.timesteps, callback.episode_rewards, alpha=0.6, label='Episode Reward')
        # Moving average
        window = min(10, len(callback.episode_rewards) // 4)
        if window > 1:
            moving_avg = np.convolve(callback.episode_rewards,
                                     np.ones(window)/window, mode='valid')
            axes[0, 0].plot(callback.timesteps[window-1:], moving_avg,
                           'r-', linewidth=2, label=f'Moving Avg ({window})')
        axes[0, 0].set_xlabel('Training Timesteps')
        axes[0, 0].set_ylabel('Episode Reward')
        axes[0, 0].set_title('Training: Episode Rewards')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

    # 2. Episode Lengths
    if callback.episode_lengths:
        axes[0, 1].plot(callback.timesteps, callback.episode_lengths, alpha=0.6)
        axes[0, 1].set_xlabel('Training Timesteps')
        axes[0, 1].set_ylabel('Episode Length')
        axes[0, 1].set_title('Training: Episode Lengths')
        axes[0, 1].grid(True, alpha=0.3)

    # 3. Reward Distribution (Histogram)
    if callback.episode_rewards:
        axes[1, 0].hist(callback.episode_rewards, bins=20, edgecolor='black', alpha=0.7)
        axes[1, 0].axvline(np.mean(callback.episode_rewards), color='r',
                          linestyle='--', label=f'Mean: {np.mean(callback.episode_rewards):.2f}')
        axes[1, 0].set_xlabel('Episode Reward')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Training: Reward Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

    # 4. Learning Curve (First vs Last Half)
    if len(callback.episode_rewards) >= 4:
        mid = len(callback.episode_rewards) // 2
        first_half = callback.episode_rewards[:mid]
        second_half = callback.episode_rewards[mid:]

        axes[1, 1].boxplot([first_half, second_half],
                          labels=['First Half', 'Second Half'])
        axes[1, 1].set_ylabel('Episode Reward')
        axes[1, 1].set_title('Training: Learning Progress')
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[PLOT] ✓ Training convergence plot saved to: {output_path}")


def plot_evaluation_convergence(episode_metrics, output_path):
    """Plot evaluation episode convergence."""
    df = pd.DataFrame(episode_metrics)

    # Group by episode
    episode_rewards = df.groupby('episode')['reward'].sum()
    episode_q_gag = df.groupby('episode')['Q_GAG'].last()
    episode_q_cluster = df.groupby('episode')['Q_cluster'].last()
    episode_clusters = df.groupby('episode')['n_clusters'].last()

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Episode Rewards
    episodes = episode_rewards.index
    axes[0, 0].plot(episodes, episode_rewards.values, 'o-', linewidth=2, markersize=6)
    axes[0, 0].axhline(episode_rewards.mean(), color='r', linestyle='--',
                       label=f'Mean: {episode_rewards.mean():.2f}')
    axes[0, 0].fill_between(episodes,
                           episode_rewards.mean() - episode_rewards.std(),
                           episode_rewards.mean() + episode_rewards.std(),
                           alpha=0.2, label=f'±1 std')
    axes[0, 0].set_xlabel('Evaluation Episode')
    axes[0, 0].set_ylabel('Total Episode Reward')
    axes[0, 0].set_title('Evaluation: Episode Rewards')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Q_GAG Over Episodes
    axes[0, 1].plot(episodes, episode_q_gag.values, 'o-', linewidth=2,
                   markersize=6, color='orange', label='Q_GAG')
    axes[0, 1].set_xlabel('Evaluation Episode')
    axes[0, 1].set_ylabel('Final Q_GAG')
    axes[0, 1].set_title('Evaluation: GAG Enrichment')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Q_cluster Over Episodes
    axes[1, 0].plot(episodes, episode_q_cluster.values, 'o-', linewidth=2,
                   markersize=6, color='blue', label='Q_cluster')
    axes[1, 0].set_xlabel('Evaluation Episode')
    axes[1, 0].set_ylabel('Final Q_cluster')
    axes[1, 0].set_title('Evaluation: Clustering Quality')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Final Clusters
    axes[1, 1].plot(episodes, episode_clusters.values, 'o-', linewidth=2,
                   markersize=6, color='green', label='n_clusters')
    axes[1, 1].set_xlabel('Evaluation Episode')
    axes[1, 1].set_ylabel('Final Number of Clusters')
    axes[1, 1].set_title('Evaluation: Cluster Count')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[PLOT] ✓ Evaluation convergence plot saved to: {output_path}")


def make_env(adata, gene_sets):
    """Create environment factory."""
    def _init():
        env = ClusteringEnv(
            adata=adata,
            gene_sets=gene_sets,
            max_steps=50,  # Increased episode length
            normalize_rewards=False,  # Reward shaping handled in RewardCalculator
            reward_mode="shaped",  # Use shaped mode to avoid negative rewards
            reward_alpha=0.2,   # Lower weight on standard clustering quality
            reward_beta=2.0,    # High weight on GAG enrichment (Biology)
            reward_delta=0.01,  # Reduced penalty weight
            gag_nonlinear=True,  # Apply non-linear GAG transformation
            gag_scale=6.0,       # Scaling factor for GAG transformation
            exploration_bonus=0.2,  # Exploration bonus (for improvement mode)
            early_termination_penalty=-5.0,  # Penalty for early Accept
            min_steps_before_accept=20,  # Minimum steps before Accept allowed
        )
        return env
    return _init


def run_episodes_with_ppo(env, num_episodes=10, total_timesteps=4000):
    """Run episodes using PPO, collect all metrics."""
    print(f"\n[PPO] Creating PPO model...")

    # Create vectorized env
    vec_env = DummyVecEnv([make_env(env.adata, env.gene_sets)])

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=3e-4,
        n_steps=100,    # Adjusted for smaller dataset
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
    print(f"[PPO] Using shaped reward mode (avoids negative rewards)")
    print(f"[PPO] Training started...", flush=True)

    # Create callback for training metrics
    training_callback = TrainingMetricsCallback()

    training_start = time.time()
    model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=training_callback)
    training_time = time.time() - training_start

    print(f"[PPO] ✓ Training completed in {training_time:.2f} seconds ({training_time/60:.1f} minutes)")

    # Now run episodes with trained model to collect metrics
    print(f"\n[COLLECT] Collecting metrics from {num_episodes} episodes...")
    print(f"[COLLECT] NOTE: Using RAW rewards (no normalization) for evaluation", flush=True)

    # Create environment WITHOUT reward normalization for evaluation
    # This gives us true reward values, not normalized ones
    # Use same reward configuration as training for consistency
    def make_eval_env():
        eval_env = ClusteringEnv(
            adata=env.adata,
            gene_sets=env.gene_sets,
            max_steps=50,  # Match training
            normalize_rewards=False,  # Disable normalization for evaluation
            reward_mode="shaped",  # Match training
            reward_alpha=0.2,   # Match training
            reward_beta=2.0,    # Match training
            reward_delta=0.01,  # Match training penalty weight
            gag_nonlinear=True,  # Match training
            gag_scale=6.0,       # Match training
            early_termination_penalty=-5.0,  # Match training
            min_steps_before_accept=20,  # Match training
        )
        return eval_env

    vec_env = DummyVecEnv([make_eval_env])
    all_step_data = []  # Collect ALL steps from ALL episodes

    # Store initial and final clustering states for each episode
    episode_clustering_states = {}  # {episode_num: {'initial': labels, 'final': labels}}
    episode_rewards = {}  # {episode_num: total_reward}

    # Use tqdm for progress bar
    episode_pbar = tqdm(range(num_episodes), desc="[COLLECT] Running episodes", unit="episode")
    for episode in episode_pbar:
        obs = vec_env.reset()

        # Get initial clustering state (after reset, before any steps)
        eval_env = vec_env.envs[0]
        initial_clusters = eval_env.adata.obs['clusters'].copy().values
        episode_clustering_states[episode + 1] = {'initial': initial_clusters}

        episode_reward = 0
        episode_length = 0

        done = np.array([False])  # Initialize as array for vectorized env
        action_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}  # Track action distribution
        last_step_data = {}

        while not done[0] and episode_length < 50:  # Match max_steps
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
                'raw_reward': float(step_info.get('raw_reward', step_info.get('reward', 0))),
                'Q_cluster': float(step_info.get('Q_cluster', 0)),
                'Q_GAG': float(step_info.get('Q_GAG', 0)),  # Raw GAG
                'Q_GAG_transformed': float(step_info.get('Q_GAG_transformed', step_info.get('Q_GAG', 0))),  # Transformed GAG
                'penalty': float(step_info.get('penalty', 0)),
                'silhouette': float(step_info.get('silhouette', 0)),  # Raw silhouette
                'silhouette_for_reward': float(step_info.get('silhouette_for_reward', step_info.get('silhouette', 0))),
                'modularity': float(step_info.get('modularity', 0)),
                'balance': float(step_info.get('balance', 0)),
                'n_clusters': int(step_info.get('n_clusters', 0)),
                'n_singletons': int(step_info.get('n_singletons', 0)),
                'mean_f_stat': float(step_info.get('mean_f_stat', 0)),
                'resolution': float(step_info.get('resolution', 0)),
                'resolution_clamped': bool(step_info.get('resolution_clamped', False)),
                'reward_mode': str(step_info.get('reward_mode', 'unknown')),
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

        # Store final clustering state and episode reward
        final_clusters = eval_env.adata.obs['clusters'].copy().values
        episode_clustering_states[episode + 1]['final'] = final_clusters
        episode_rewards[episode + 1] = episode_reward

        # Update progress bar with episode info
        episode_pbar.set_postfix({
            'reward': f'{episode_reward:.2f}',
            'clusters': last_step_data.get('n_clusters', 0),
            'length': episode_length
        })

    episode_pbar.close()
    print(f"[COLLECT] ✓ Completed {num_episodes} episodes")

    return all_step_data, model, episode_clustering_states, episode_rewards, training_callback


def main():
    """Main test."""
    print("="*70)
    print("PPO TEST - 100 cells, 5 episodes")
    print("="*70)

    data_path = project_root / "data" / "processed" / "human_interneurons.h5ad"
    if not data_path.exists():
        print(f"ERROR: {data_path} not found")
        return 1

    # Load 100 cells
    adata = load_subset(data_path, n_cells=100)

    # Create environment
    print(f"\n[ENV] Creating environment...")
    env = ClusteringEnv(
        adata=adata,
        gene_sets=GAG_GENE_SETS,
        max_steps=50,
        normalize_rewards=True,
    )
    print(f"[ENV] ✓ Environment created")

    # Test it works
    state, info = env.reset()
    print(f"[ENV] ✓ Reset test: {info['n_clusters']} clusters")
    print(f"[ENV] ✓ F-stats available: {list(info.get('f_stats', {}).keys())}")

    # Run episodes with PPO
    episode_metrics, model, episode_clustering_states, episode_rewards_dict, training_callback = run_episodes_with_ppo(
        env, num_episodes=5, total_timesteps=500
    )

    # Save metrics
    print(f"\n[SAVE] Saving metrics...")
    output_dir = project_root / "results" / "ppo_test_100c_5ep"
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
    res_plot_path = output_dir / "resolution_clusters.png"
    training_plot_path = output_dir / "training_convergence.png"
    eval_plot_path = output_dir / "evaluation_convergence.png"
    umap_plot_path = output_dir / "best_episode_umaps.png"

    try:
        plot_trajectories(episode_metrics, str(plot_path))
        plot_resolution_clusters(episode_metrics, str(res_plot_path))

        # Plot training convergence
        plot_training_convergence(training_callback, str(training_plot_path))

        # Plot evaluation convergence
        plot_evaluation_convergence(episode_metrics, str(eval_plot_path))

        # Plot UMAPs for best episode
        if episode_rewards_dict:
            best_episode_num = max(episode_rewards_dict, key=episode_rewards_dict.get)
            best_episode_reward = episode_rewards_dict[best_episode_num]
            print(f"\n[EVAL] Best episode: {best_episode_num} (reward: {best_episode_reward:.4f})")

            best_data = episode_clustering_states[best_episode_num]
            plot_best_episode_umaps(
                env.adata,
                best_episode_num,
                best_data['initial'],
                best_data['final'],
                str(umap_plot_path)
            )
    except Exception as e:
        print(f"[PLOT] Error creating plot: {e}")
        import traceback
        traceback.print_exc()

    print("="*70)
    print("✓ TEST COMPLETED SUCCESSFULLY!")
    print("="*70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
