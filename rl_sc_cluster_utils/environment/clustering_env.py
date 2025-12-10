"""Gymnasium-compatible RL environment for scRNA-seq cluster refinement."""

from typing import Any, Dict, List, Optional, Tuple

from anndata import AnnData
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import scanpy as sc

from .actions import ActionExecutor
from .rewards import RewardCalculator, RewardNormalizer
from .state import StateExtractor


class ClusteringEnv(gym.Env):
    """
    Gymnasium-compatible RL environment for scRNA-seq cluster refinement.

    State: 35-dimensional vector encoding clustering state
    Actions: 5 discrete actions (split, merge, re-cluster, accept)
    Reward: Composite of clustering quality and GAG enrichment

    Reward Formula:
        R = α·Q_cluster + β·Q_GAG - δ·Penalty

    Where:
    - Q_cluster = 0.5·silhouette + 0.3·modularity + 0.2·balance
    - Q_GAG = mean(log1p(f_stat) / 10.0) across gene sets
    - Penalty = degenerate states + singletons + bounds violations

    Parameters
    ----------
    adata : AnnData
        Annotated data object containing single-cell data
    gene_sets : dict, optional
        Dictionary mapping gene set names to lists of gene names
        (default: None, which uses empty gene sets)
    max_steps : int, optional
        Maximum number of steps per episode (default: 15)
    normalize_state : bool, optional
        Whether to normalize state vector (default: False)
    normalize_rewards : bool, optional
        Whether to normalize rewards using running statistics (default: True)
    render_mode : str, optional
        Render mode for visualization (default: None)
    reward_alpha : float, optional
        Weight for clustering quality in reward (default: 0.6)
    reward_beta : float, optional
        Weight for GAG enrichment in reward (default: 0.4)
    reward_delta : float, optional
        Weight for penalties in reward (default: 1.0)
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        adata: AnnData,
        gene_sets: Optional[Dict[str, List[str]]] = None,
        max_steps: int = 15,
        normalize_state: bool = False,
        normalize_rewards: bool = True,
        render_mode: Optional[str] = None,
        reward_alpha: float = 0.6,
        reward_beta: float = 0.4,
        reward_delta: float = 1.0,
    ) -> None:
        super().__init__()

        # Store parameters
        self.adata = adata.copy()  # Make a copy to avoid modifying original
        self.gene_sets = gene_sets or {}
        self.max_steps = max_steps
        self.normalize_state = normalize_state
        self.normalize_rewards = normalize_rewards
        self.render_mode = render_mode

        # Initialize state extractor
        self.state_extractor = StateExtractor(
            self.adata, self.gene_sets, normalize=self.normalize_state
        )

        # Initialize action executor
        self.action_executor = ActionExecutor(
            self.adata,
            min_resolution=0.1,
            max_resolution=2.0,
        )

        # Initialize reward calculator
        self.reward_calculator = RewardCalculator(
            self.adata,
            self.gene_sets,
            alpha=reward_alpha,
            beta=reward_beta,
            delta=reward_delta,
        )

        # Initialize reward normalizer (if enabled)
        self.reward_normalizer: Optional[RewardNormalizer] = None
        if self.normalize_rewards:
            self.reward_normalizer = RewardNormalizer()

        # Action space: 5 discrete actions
        # 0: Split worst cluster
        # 1: Merge closest pair
        # 2: Re-cluster resolution +0.1
        # 3: Re-cluster resolution -0.1
        # 4: Accept (terminate episode)
        self.action_space = spaces.Discrete(5)

        # Observation space: 35-dimensional vector
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(35,), dtype=np.float64
        )

        # Episode tracking
        self.current_step = 0
        self.state: Optional[np.ndarray] = None
        self.current_resolution = 0.5  # Initial Leiden resolution
        self._initial_clustering_done = False
        self._previous_reward: Optional[float] = None

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment to initial state.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility
        options : dict, optional
            Additional options for reset

        Returns
        -------
        state : np.ndarray
            Initial state vector (35 dimensions)
        info : dict
            Additional information
        """
        super().reset(seed=seed)

        # Reset episode tracking
        self.current_step = 0
        self.current_resolution = 0.5
        self._previous_reward = None

        # Reset reward normalizer (if enabled)
        if self.reward_normalizer is not None:
            self.reward_normalizer.reset()

        # Perform initial clustering if not already done or if we need to reset
        # Check if neighbors graph exists, if not compute it
        if "neighbors" not in self.adata.uns:
            # Compute neighbors graph if not present
            if "X_scvi" in self.adata.obsm:
                sc.pp.neighbors(self.adata, use_rep="X_scvi", n_neighbors=15)
            else:
                # Use PCA if no embeddings
                if "X_pca" not in self.adata.obsm:
                    sc.pp.pca(self.adata)
                sc.pp.neighbors(self.adata, n_neighbors=15)

        # Perform initial Leiden clustering
        # Use igraph flavor for future compatibility
        sc.tl.leiden(
            self.adata,
            resolution=self.current_resolution,
            key_added="clusters",
            flavor="igraph",
            n_iterations=2,
            directed=False,
        )
        self._initial_clustering_done = True

        # Ensure cluster IDs are numeric
        from .actions import convert_cluster_ids_to_numeric

        convert_cluster_ids_to_numeric(self.adata)

        # Extract state using StateExtractor
        self.state = self.state_extractor.extract_state(
            self.adata, self.current_step, self.max_steps
        )

        # Get number of clusters for info
        n_clusters = len(self.adata.obs["clusters"].unique())

        info = {
            "step": self.current_step,
            "resolution": self.current_resolution,
            "n_clusters": n_clusters,
        }

        return self.state, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Parameters
        ----------
        action : int
            Action to take (0-4)

        Returns
        -------
        state : np.ndarray
            Next state vector (35 dimensions)
        reward : float
            Reward for the action (may be normalized if normalize_rewards=True)
        terminated : bool
            Whether episode is terminated (Accept action)
        truncated : bool
            Whether episode is truncated (max steps reached)
        info : dict
            Additional information including reward components
        """
        # Validate action (Gymnasium compliance: raise ValueError for out-of-bounds)
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}. Must be in range [0, 4].")

        # Execute action
        action_result = self.action_executor.execute(action, self.current_resolution)

        # Update resolution if it changed
        if action_result["new_resolution"] != self.current_resolution:
            self.current_resolution = action_result["new_resolution"]

        # Increment step counter
        self.current_step += 1

        # Extract new state
        next_state = self.state_extractor.extract_state(
            self.adata, self.current_step, self.max_steps
        )
        self.state = next_state

        # Compute reward using RewardCalculator
        raw_reward, reward_info = self.reward_calculator.compute_reward(
            self.adata,
            previous_reward=self._previous_reward,
            resolution_clamped=action_result["resolution_clamped"],
        )

        # Apply normalization if enabled
        if self.reward_normalizer is not None:
            reward = self.reward_normalizer.update_and_normalize(raw_reward)
        else:
            reward = raw_reward

        # Update previous reward for next step
        self._previous_reward = raw_reward

        # Check termination conditions
        terminated = action == 4  # Accept action
        truncated = self.current_step >= self.max_steps

        # Get number of clusters for info
        n_clusters = (
            len(self.adata.obs["clusters"].unique()) if "clusters" in self.adata.obs else 0
        )

        # Build info dict with action results and reward components
        info = {
            "action": action,
            "step": self.current_step,
            "terminated": terminated,
            "truncated": truncated,
            "resolution": self.current_resolution,
            "n_clusters": n_clusters,
            "action_success": action_result["success"],
            "action_error": action_result["error"],
            "resolution_clamped": action_result["resolution_clamped"],
            "no_change": action_result["no_change"],
            # Reward components
            "raw_reward": raw_reward,
            "Q_cluster": reward_info["Q_cluster"],
            "Q_GAG": reward_info["Q_GAG"],
            "penalty": reward_info["penalty"],
            "silhouette": reward_info["silhouette"],
            "modularity": reward_info["modularity"],
            "balance": reward_info["balance"],
            "n_singletons": reward_info["n_singletons"],
            "mean_f_stat": reward_info["mean_f_stat"],
        }

        return next_state, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        """
        Render the environment.

        Returns
        -------
        rgb_array : np.ndarray or None
            RGB array for visualization (if render_mode='rgb_array')
        """
        if self.render_mode == "human":
            # Future: UMAP visualization with cluster borders
            pass
        elif self.render_mode == "rgb_array":
            # Future: Return RGB array for video recording
            return None
        return None

    def close(self) -> None:
        """Clean up resources."""
        # Nothing to clean up yet
        pass
