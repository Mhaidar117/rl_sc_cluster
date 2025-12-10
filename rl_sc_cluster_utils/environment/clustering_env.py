"""Gymnasium-compatible RL environment for scRNA-seq cluster refinement."""

from typing import Any, Dict, List, Optional, Tuple

from anndata import AnnData
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import scanpy as sc

from .actions import ActionExecutor
from .state import StateExtractor


class ClusteringEnv(gym.Env):
    """
    Gymnasium-compatible RL environment for scRNA-seq cluster refinement.

    State: 35-dimensional vector encoding clustering state
    Actions: 5 discrete actions (split, merge, re-cluster, accept)
    Reward: Composite of clustering quality and GAG enrichment

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
        Whether to normalize rewards (default: False)
    render_mode : str, optional
        Render mode for visualization (default: None)
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        adata: AnnData,
        gene_sets: Optional[Dict[str, List[str]]] = None,
        max_steps: int = 15,
        normalize_state: bool = False,
        normalize_rewards: bool = False,
        render_mode: Optional[str] = None,
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
            Reward for the action
        terminated : bool
            Whether episode is terminated (Accept action)
        truncated : bool
            Whether episode is truncated (max steps reached)
        info : dict
            Additional information
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

        # Placeholder: constant reward (will be implemented in Stage 4)
        # TODO: Use action_result["resolution_clamped"] for penalty in Stage 4
        reward = 0.0

        # Check termination conditions
        terminated = action == 4  # Accept action
        truncated = self.current_step >= self.max_steps

        # Get number of clusters for info
        n_clusters = (
            len(self.adata.obs["clusters"].unique()) if "clusters" in self.adata.obs else 0
        )

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
