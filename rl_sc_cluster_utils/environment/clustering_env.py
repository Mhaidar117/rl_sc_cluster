"""Gymnasium-compatible RL environment for scRNA-seq cluster refinement."""

from typing import Any, Dict, Optional, Tuple

from anndata import AnnData
import gymnasium as gym
from gymnasium import spaces
import numpy as np


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
        max_steps: int = 15,
        normalize_state: bool = False,
        normalize_rewards: bool = False,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()

        # Store parameters
        self.adata = adata
        self.max_steps = max_steps
        self.normalize_state = normalize_state
        self.normalize_rewards = normalize_rewards
        self.render_mode = render_mode

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

        # Placeholder: return zero vector
        # Will be replaced with actual state extraction in Stage 2
        self.state = np.zeros(35, dtype=np.float64)

        # Reset episode tracking
        self.current_step = 0
        self.current_resolution = 0.5

        info = {
            "step": self.current_step,
            "resolution": self.current_resolution,
            "n_clusters": 1,  # Placeholder
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
        # Validate action
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}. Must be in range [0, 4].")

        # Placeholder: no-op actions
        # State remains unchanged
        # Will be replaced with actual action execution in Stage 3
        next_state = self.state.copy()

        # Placeholder: constant reward
        # Will be replaced with actual reward computation in Stage 4
        reward = 0.0

        # Increment step counter
        self.current_step += 1

        # Check termination conditions
        terminated = action == 4  # Accept action
        truncated = self.current_step >= self.max_steps

        info = {
            "action": action,
            "step": self.current_step,
            "terminated": terminated,
            "truncated": truncated,
            "resolution": self.current_resolution,
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
