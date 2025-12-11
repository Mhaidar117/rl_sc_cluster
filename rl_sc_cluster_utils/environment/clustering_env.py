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

    Reward Formula (configurable modes):
        - Absolute: R = α·Q_cluster + β·Q_GAG_transformed - δ·Penalty
        - Improvement: R = (current_potential - previous_potential) + exploration_bonus
        - Shaped: R = raw_reward - baseline + offset (default, avoids negative rewards)

    Where:
    - Q_cluster = 0.5·silhouette_for_reward + 0.3·modularity + 0.2·balance
    - Q_GAG = mean(log1p(f_stat) / 10.0) across gene sets (raw)
    - Q_GAG_transformed = (Q_GAG * gag_scale)² if gag_nonlinear=True
    - Penalty = degenerate states + singletons + bounds violations
    - Early termination: Penalty applied if Accept action taken before min_steps_before_accept

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
        Weight for clustering quality in reward (default: 0.2)
    reward_beta : float, optional
        Weight for GAG enrichment in reward (default: 2.0)
    reward_delta : float, optional
        Weight for penalties in reward (default: 0.01)
    reward_mode : str, optional
        Reward mode: "absolute", "improvement", or "shaped" (default: "shaped")
    gag_nonlinear : bool, optional
        Apply non-linear transformation (Q_GAG * scale)² (default: True)
    gag_scale : float, optional
        Scaling factor for GAG transformation (default: 6.0)
    exploration_bonus : float, optional
        Bonus per step for improvement mode (default: 0.2)
    silhouette_shift : float, optional
        Shift amount to keep silhouette non-negative (default: 0.5)
    early_termination_penalty : float, optional
        Penalty for Accept action before minimum steps (default: -5.0)
    min_steps_before_accept : int, optional
        Minimum steps before Accept action is allowed without penalty (default: 10)
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
        reward_alpha: float = 0.2,
        reward_beta: float = 2.0,
        reward_delta: float = 0.01,
        reward_mode: str = "shaped",
        gag_nonlinear: bool = True,
        gag_scale: float = 6.0,
        exploration_bonus: float = 0.2,
        silhouette_shift: float = 0.5,
        early_termination_penalty: float = -5.0,
        min_steps_before_accept: int = 10,
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

        # Initialize action executor (with gene_sets for GAG-aware actions)
        self.action_executor = ActionExecutor(
            self.adata,
            gene_sets=self.gene_sets,
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
            reward_mode=reward_mode,
            gag_nonlinear=gag_nonlinear,
            gag_scale=gag_scale,
            exploration_bonus=exploration_bonus,
            silhouette_shift=silhouette_shift,
            early_termination_penalty=early_termination_penalty,
            min_steps_before_accept=min_steps_before_accept,
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

        # Reset reward calculator state
        self.reward_calculator.reset()

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
            action=action,
            current_step=self.current_step,
        )

        # Apply normalization if enabled (but NOT for early termination penalty)
        if self.reward_normalizer is not None and not reward_info.get(
            "early_termination_penalty_applied", False
        ):
            reward = self.reward_normalizer.update_and_normalize(raw_reward)
        else:
            reward = raw_reward

        # Apply early termination penalty AFTER normalization to bypass normalizer
        # This ensures the penalty value remains stable and predictable
        if reward_info.get("early_termination_penalty_applied", False):
            reward = self.reward_calculator.early_termination_penalty

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
            "Q_GAG": reward_info["Q_GAG"],  # Raw GAG
            "Q_GAG_transformed": reward_info.get("Q_GAG_transformed", reward_info["Q_GAG"]),
            "penalty": reward_info["penalty"],
            "early_termination_penalty_applied": reward_info.get(
                "early_termination_penalty_applied", False
            ),
            "silhouette": reward_info["silhouette"],  # Raw silhouette (preserved)
            "silhouette_for_reward": reward_info.get(
                "silhouette_for_reward", reward_info["silhouette"]
            ),
            "modularity": reward_info["modularity"],
            "balance": reward_info["balance"],
            "n_singletons": reward_info["n_singletons"],
            "mean_f_stat": reward_info["mean_f_stat"],
            "f_stats": reward_info.get("f_stats", {}),  # Per-gene-set F-statistics
            "reward_mode": reward_info.get("reward_mode", "unknown"),
            "baseline": reward_info.get("baseline"),  # For shaped mode
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
