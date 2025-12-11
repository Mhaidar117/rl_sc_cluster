#!/usr/bin/env python
"""Test RL training with a simple agent using Gymnasium wrappers."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import scanpy as sc
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics
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


class SimpleEpsilonGreedyAgent:
    """Very simple epsilon-greedy agent for testing."""

    def __init__(self, action_space, epsilon=0.3, learning_rate=0.1, discount=0.95):
        self.action_space = action_space
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.discount = discount

        # Simple Q-table: state -> action -> value
        self.q_table = {}
        self.state_hash_size = 10

    def _hash_state(self, state):
        """Convert continuous state to discrete hash."""
        discretized = tuple(
            int(np.clip(s * self.state_hash_size, 0, self.state_hash_size - 1))
            for s in state[:10]  # Use first 10 dims for hash
        )
        return discretized

    def get_action(self, state, training=True):
        """Choose action using epsilon-greedy policy."""
        state_hash = self._hash_state(state)

        if state_hash not in self.q_table:
            self.q_table[state_hash] = np.zeros(self.action_space.n)

        if training and np.random.random() < self.epsilon:
            return self.action_space.sample()
        else:
            return np.argmax(self.q_table[state_hash])

    def update(self, state, action, reward, next_state, done):
        """Update Q-values using Q-learning."""
        state_hash = self._hash_state(state)
        next_state_hash = self._hash_state(next_state)

        if state_hash not in self.q_table:
            self.q_table[state_hash] = np.zeros(self.action_space.n)
        if next_state_hash not in self.q_table:
            self.q_table[next_state_hash] = np.zeros(self.action_space.n)

        current_q = self.q_table[state_hash][action]
        if done:
            target_q = reward
        else:
            target_q = reward + self.discount * np.max(self.q_table[next_state_hash])

        self.q_table[state_hash][action] = (
            current_q + self.learning_rate * (target_q - current_q)
        )


def load_data(data_path, subset_size=2000):
    """Load and prepare a small subset for fast testing."""
    print(f"Loading data from: {data_path}")
    adata = sc.read_h5ad(data_path)

    if 'scVI' in adata.obsm and 'X_scvi' not in adata.obsm:
        adata.obsm['X_scvi'] = adata.obsm['scVI']

    print(f"Using random subset of {subset_size} cells...")
    np.random.seed(42)
    indices = np.random.choice(adata.n_obs, subset_size, replace=False)
    adata = adata[indices].copy()

    return adata


def train_agent(env, agent, num_episodes=20):
    """Train agent and track progress."""
    print(f"\n{'='*60}")
    print(f"Training for {num_episodes} episodes")
    print(f"{'='*60}\n")

    episode_rewards = []
    episode_lengths = []

    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        while not done:
            action = agent.get_action(state, training=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.update(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward
            episode_length += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        if (episode + 1) % 5 == 0:
            recent_rewards = episode_rewards[-5:]
            recent_lengths = episode_lengths[-5:]
            print(f"Episode {episode + 1:3d}: "
                  f"Reward: {episode_reward:7.2f} (mean: {np.mean(recent_rewards):7.2f}), "
                  f"Length: {episode_length:2d} (mean: {np.mean(recent_lengths):.1f}), "
                  f"Final clusters: {info['n_clusters']:2d}")

    return episode_rewards, episode_lengths


def main():
    """Main training test."""
    data_path = project_root / "data" / "processed" / "human_interneurons.h5ad"

    if not data_path.exists():
        print(f"ERROR: Data file not found at {data_path}")
        return 1

    adata = load_data(data_path, subset_size=2000)
    print(f"Data shape: {adata.shape}")

    print("\nCreating ClusteringEnv...")
    env = ClusteringEnv(
        adata=adata,
        gene_sets=GAG_GENE_SETS,
        max_steps=15,
        normalize_rewards=True,
    )

    # Wrap with RecordEpisodeStatistics
    env = RecordEpisodeStatistics(env)

    agent = SimpleEpsilonGreedyAgent(
        env.action_space,
        epsilon=0.3,
        learning_rate=0.1,
        discount=0.95
    )

    episode_rewards, episode_lengths = train_agent(env, agent, num_episodes=20)

    print(f"\n{'='*60}")
    print("Training Summary")
    print(f"{'='*60}")
    print(f"Total episodes: {len(episode_rewards)}")
    print(f"Mean reward (first 5): {np.mean(episode_rewards[:5]):.4f}")
    print(f"Mean reward (last 5):  {np.mean(episode_rewards[-5:]):.4f}")
    print(f"Mean episode length: {np.mean(episode_lengths):.1f}")
    print(f"Q-table size: {len(agent.q_table)} states")

    if len(episode_rewards) >= 10:
        first_half = np.mean(episode_rewards[:len(episode_rewards)//2])
        second_half = np.mean(episode_rewards[len(episode_rewards)//2:])
        improvement = second_half - first_half
        print(f"\nReward improvement: {improvement:.4f}")
        if improvement > 0:
            print("✓ Agent appears to be learning (rewards improving)")
        else:
            print("⚠ Agent rewards not improving (may need more training)")

    # Test trained agent (no exploration)
    print(f"\n{'='*60}")
    print("Testing trained agent (no exploration)")
    print(f"{'='*60}")

    state, info = env.reset()
    test_reward = 0
    test_length = 0

    while True:
        action = agent.get_action(state, training=False)  # No exploration
        state, reward, terminated, truncated, info = env.step(action)
        test_reward += reward
        test_length += 1

        if terminated or truncated:
            break

    print(f"Test episode reward: {test_reward:.4f}")
    print(f"Test episode length: {test_length}")
    print(f"Final clusters: {info['n_clusters']}")
    print(f"Final Q_cluster: {info['Q_cluster']:.4f}")
    print(f"Final Q_GAG: {info['Q_GAG']:.4f}")

    print(f"\n{'='*60}")
    print("✓ Training test completed successfully!")
    print(f"{'='*60}")
    print("\nThe environment is working correctly for RL training.")
    print("You can now use a more sophisticated RL algorithm if needed.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
