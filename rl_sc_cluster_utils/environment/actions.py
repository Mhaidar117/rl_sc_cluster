"""Action execution for the clustering environment."""

from typing import Dict, Optional, Tuple

from anndata import AnnData
import numpy as np
import scanpy as sc
from sklearn.metrics import silhouette_score


def convert_cluster_ids_to_numeric(adata: AnnData) -> None:
    """
    Convert cluster IDs from string to numeric format.

    Leiden clustering returns string IDs. This function converts them to
    numeric IDs for easier manipulation and consistency.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with cluster labels in adata.obs['clusters']
    """
    if "clusters" not in adata.obs:
        return

    # Check if already numeric
    if adata.obs["clusters"].dtype in [np.int8, np.int16, np.int32, np.int64]:
        return

    # Convert string IDs to numeric
    unique_clusters = sorted(adata.obs["clusters"].unique())
    mapping = {cluster: i for i, cluster in enumerate(unique_clusters)}
    adata.obs["clusters"] = adata.obs["clusters"].map(mapping).astype(int)


class ActionExecutor:
    """
    Execute actions that modify clustering state.

    Actions:
    - 0: Split worst cluster
    - 1: Merge closest pair
    - 2: Re-cluster resolution +0.1
    - 3: Re-cluster resolution -0.1
    - 4: Accept (no-op, handled in step())

    Parameters
    ----------
    adata : AnnData
        Annotated data object containing single-cell data
    min_resolution : float, optional
        Minimum resolution for Leiden clustering (default: 0.1)
    max_resolution : float, optional
        Maximum resolution for Leiden clustering (default: 2.0)
    """

    def __init__(
        self,
        adata: AnnData,
        min_resolution: float = 0.1,
        max_resolution: float = 2.0,
    ) -> None:
        self.adata = adata
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution

        # Cache embeddings
        if "X_scvi" in adata.obsm:
            self._embeddings = adata.obsm["X_scvi"]
        elif adata.X is not None:
            self._embeddings = adata.X if isinstance(adata.X, np.ndarray) else adata.X.toarray()
        else:
            raise ValueError("AnnData must contain either obsm['X_scvi'] or X matrix")

    def execute(self, action: int, current_resolution: float) -> Dict[str, any]:  # noqa: ANN401
        """
        Execute an action and return execution results.

        Parameters
        ----------
        action : int
            Action to execute (0-4)
        current_resolution : float
            Current Leiden clustering resolution

        Returns
        -------
        result : dict
            Dictionary containing:
            - success (bool): Whether action executed successfully
            - error (str or None): Error message if action failed
            - resolution_clamped (bool): Whether resolution was clamped
            - no_change (bool): Whether action had no effect
            - new_resolution (float): New resolution after action
        """
        # Validate action semantics
        is_valid, error_msg = self._validate_action(action, current_resolution)
        if not is_valid:
            return {
                "success": False,
                "error": error_msg,
                "resolution_clamped": False,
                "no_change": True,
                "new_resolution": current_resolution,
            }

        # Execute action
        if action == 0:
            result = self._split_worst_cluster(current_resolution)
        elif action == 1:
            result = self._merge_closest_pair()
        elif action == 2:
            result = self._increment_resolution(current_resolution)
        elif action == 3:
            result = self._decrement_resolution(current_resolution)
        elif action == 4:
            result = {
                "success": True,
                "error": None,
                "resolution_clamped": False,
                "no_change": False,
                "new_resolution": current_resolution,
            }
        else:
            # Should not happen (validated in step())
            return {
                "success": False,
                "error": f"Unknown action: {action}",
                "resolution_clamped": False,
                "no_change": True,
                "new_resolution": current_resolution,
            }

        # Ensure cluster IDs are numeric after action
        convert_cluster_ids_to_numeric(self.adata)

        return result

    def _validate_action(
        self, action: int, current_resolution: float
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate that an action is semantically valid.

        Parameters
        ----------
        action : int
            Action to validate
        current_resolution : float
            Current resolution

        Returns
        -------
        is_valid : bool
            Whether action is valid
        error_msg : str or None
            Error message if invalid
        """
        if "clusters" not in self.adata.obs:
            return False, "No clusters found in adata.obs['clusters']"

        n_clusters = len(self.adata.obs["clusters"].unique())

        if action == 0:  # Split
            if n_clusters == 1:
                return False, "Cannot split: only 1 cluster"
            # Check for non-singleton clusters
            cluster_sizes = self.adata.obs["clusters"].value_counts()
            if (cluster_sizes >= 2).sum() == 0:
                return False, "Cannot split: all clusters are singletons"
        elif action == 1:  # Merge
            if n_clusters == 1:
                return False, "Cannot merge: only 1 cluster"
        # Actions 2, 3, 4 are always valid (if action_space.contains passed)

        return True, None

    def _split_worst_cluster(self, current_resolution: float) -> Dict[str, any]:
        """
        Split the cluster with lowest silhouette score.

        Parameters
        ----------
        current_resolution : float
            Current Leiden resolution

        Returns
        -------
        result : dict
            Execution result dictionary
        """
        # Ensure clusters are numeric (not categorical) for easier manipulation
        convert_cluster_ids_to_numeric(self.adata)

        # Check if we have more than one cluster
        n_clusters = len(self.adata.obs["clusters"].unique())
        if n_clusters == 1:
            return {
                "success": False,
                "error": "Cannot split: only 1 cluster",
                "resolution_clamped": False,
                "no_change": True,
                "new_resolution": current_resolution,
            }

        # Find worst cluster
        worst_cluster = self._find_worst_cluster()
        if worst_cluster is None:
            return {
                "success": False,
                "error": "Could not find cluster to split",
                "resolution_clamped": False,
                "no_change": True,
                "new_resolution": current_resolution,
            }

        # Extract subgraph for cluster
        cluster_mask = (self.adata.obs["clusters"] == worst_cluster).values
        if cluster_mask.sum() < 2:
            return {
                "success": False,
                "error": "Cannot split singleton cluster",
                "resolution_clamped": False,
                "no_change": True,
                "new_resolution": current_resolution,
            }

        # Extract subgraph
        # Scanpy stores connectivities in obsp, not uns
        if "connectivities" in self.adata.obsp:
            neighbors = self.adata.obsp["connectivities"]
        elif "connectivities" in self.adata.uns.get("neighbors", {}):
            # Fallback: old format (for compatibility)
            neighbors = self.adata.uns["neighbors"]["connectivities"]
        else:
            raise ValueError(
                "Neighbors graph not found in obsp['connectivities'] or uns['neighbors']['connectivities']"
            )

        # Subgraph: connections within this cluster
        subgraph = neighbors[cluster_mask, :][:, cluster_mask]

        # Create temporary AnnData for sub-clustering
        cluster_adata = self.adata[cluster_mask].copy()
        # Preserve neighbors structure for scanpy
        cluster_adata.obsp["connectivities"] = subgraph
        cluster_adata.uns["neighbors"] = {
            "connectivities_key": "connectivities",
            "params": self.adata.uns.get("neighbors", {}).get("params", {}),
        }

        # Sub-cluster at higher resolution
        subcluster_resolution = min(self.max_resolution, current_resolution + 0.2)
        sc.tl.leiden(
            cluster_adata,
            resolution=subcluster_resolution,
            key_added="subclusters",
            flavor="igraph",
            n_iterations=2,
            directed=False,
        )

        # Check if sub-clustering produced multiple clusters
        n_subclusters = len(cluster_adata.obs["subclusters"].unique())
        if n_subclusters == 1:
            # No-op: sub-clustering didn't split
            return {
                "success": True,
                "error": None,
                "resolution_clamped": False,
                "no_change": True,
                "new_resolution": current_resolution,
            }

        # Map sub-clusters back to original adata
        # Get current max cluster ID (handle categorical)
        if self.adata.obs["clusters"].dtype.name == "category":
            max_cluster_id = int(self.adata.obs["clusters"].cat.categories.max())
        else:
            max_cluster_id = int(self.adata.obs["clusters"].max())

        # Convert subcluster labels to numeric
        convert_cluster_ids_to_numeric(cluster_adata)

        # Assign new cluster IDs (offset by max_cluster_id + 1)
        new_labels = cluster_adata.obs["subclusters"].values.astype(int) + max_cluster_id + 1

        # Update labels
        self.adata.obs.loc[cluster_mask, "clusters"] = new_labels

        return {
            "success": True,
            "error": None,
            "resolution_clamped": False,
            "no_change": False,
            "new_resolution": current_resolution,
        }

    def _find_worst_cluster(self) -> Optional[int]:
        """
        Find cluster with lowest mean silhouette score.

        Returns
        -------
        worst_cluster : int or None
            Cluster ID with lowest silhouette, or None if not found
        """
        cluster_silhouettes = {}

        for cluster_id in self.adata.obs["clusters"].unique():
            cluster_mask = self.adata.obs["clusters"] == cluster_id
            cluster_size = cluster_mask.sum()

            # Skip singletons
            if cluster_size < 2:
                continue

            # Compute silhouette for this cluster
            try:
                cluster_embeddings = self._embeddings[cluster_mask]
                cluster_labels = self.adata.obs["clusters"][cluster_mask]

                # Need at least 2 unique labels for silhouette
                if len(np.unique(cluster_labels)) < 2:
                    # All same label, use 0 as silhouette (worst)
                    cluster_silhouettes[cluster_id] = 0.0
                else:
                    # Silhouette score for this cluster
                    sil = silhouette_score(cluster_embeddings, cluster_labels)
                    cluster_silhouettes[cluster_id] = sil
            except Exception:
                # If silhouette fails, use 0 as silhouette (worst)
                cluster_silhouettes[cluster_id] = 0.0

        if not cluster_silhouettes:
            return None

        # Return cluster with lowest silhouette
        worst_cluster = min(cluster_silhouettes, key=cluster_silhouettes.get)
        return worst_cluster

    def _merge_closest_pair(self) -> Dict[str, any]:
        """
        Merge the two clusters with minimum centroid distance.

        Returns
        -------
        result : dict
            Execution result dictionary
        """
        # Compute centroids
        centroids = self._compute_centroids()
        if len(centroids) < 2:
            return {
                "success": False,
                "error": "Cannot merge: less than 2 clusters",
                "resolution_clamped": False,
                "no_change": True,
                "new_resolution": 0.5,  # Default, not used when no_change=True
            }

        # Find closest pair
        closest_pair = self._find_closest_clusters(centroids)
        if closest_pair is None:
            return {
                "success": False,
                "error": "Could not find clusters to merge",
                "resolution_clamped": False,
                "no_change": True,
                "new_resolution": 0.5,  # Default, not used when no_change=True
            }

        cluster1, cluster2 = closest_pair

        # Merge clusters (use smaller ID as new label)
        new_label = min(cluster1, cluster2)

        mask1 = self.adata.obs["clusters"] == cluster1
        mask2 = self.adata.obs["clusters"] == cluster2

        self.adata.obs.loc[mask1, "clusters"] = new_label
        self.adata.obs.loc[mask2, "clusters"] = new_label

        return {
            "success": True,
            "error": None,
            "resolution_clamped": False,
            "no_change": False,
            "new_resolution": self.adata.uns.get("_current_resolution", 0.5),
        }

    def _compute_centroids(self) -> Dict[int, np.ndarray]:
        """
        Compute centroid of each cluster in embedding space.

        Returns
        -------
        centroids : dict
            Dictionary mapping cluster ID to centroid vector
        """
        centroids = {}

        for cluster_id in self.adata.obs["clusters"].unique():
            cluster_mask = self.adata.obs["clusters"] == cluster_id
            cluster_embeddings = self._embeddings[cluster_mask]
            centroids[cluster_id] = np.mean(cluster_embeddings, axis=0)

        return centroids

    def _find_closest_clusters(
        self, centroids: Dict[int, np.ndarray]
    ) -> Optional[Tuple[int, int]]:
        """
        Find pair of clusters with minimum Euclidean distance.

        Parameters
        ----------
        centroids : dict
            Dictionary mapping cluster ID to centroid vector

        Returns
        -------
        closest_pair : tuple or None
            Tuple of (cluster1, cluster2) with minimum distance
        """
        if len(centroids) < 2:
            return None

        min_dist = np.inf
        closest_pair = None

        cluster_ids = list(centroids.keys())
        for i, c1 in enumerate(cluster_ids):
            for c2 in cluster_ids[i + 1 :]:
                dist = np.linalg.norm(centroids[c1] - centroids[c2])
                if dist < min_dist:
                    min_dist = dist
                    closest_pair = (c1, c2)

        return closest_pair

    def _increment_resolution(self, current_resolution: float) -> Dict[str, any]:
        """
        Increment resolution by 0.1 and re-cluster.

        Parameters
        ----------
        current_resolution : float
            Current resolution

        Returns
        -------
        result : dict
            Execution result dictionary
        """
        # Increment resolution with clamping
        new_resolution = min(self.max_resolution, current_resolution + 0.1)
        clamped = (
            new_resolution == self.max_resolution and current_resolution < self.max_resolution
        )

        # Re-cluster with new resolution
        sc.tl.leiden(
            self.adata,
            resolution=new_resolution,
            key_added="clusters",
            flavor="igraph",
            n_iterations=2,
            directed=False,
        )

        return {
            "success": True,
            "error": None,
            "resolution_clamped": clamped,
            "no_change": False,
            "new_resolution": new_resolution,
        }

    def _decrement_resolution(self, current_resolution: float) -> Dict[str, any]:
        """
        Decrement resolution by 0.1 and re-cluster.

        Parameters
        ----------
        current_resolution : float
            Current resolution

        Returns
        -------
        result : dict
            Execution result dictionary
        """
        # Decrement resolution with clamping
        new_resolution = max(self.min_resolution, current_resolution - 0.1)
        clamped = (
            new_resolution == self.min_resolution and current_resolution > self.min_resolution
        )

        # Re-cluster with new resolution
        sc.tl.leiden(
            self.adata,
            resolution=new_resolution,
            key_added="clusters",
            flavor="igraph",
            n_iterations=2,
            directed=False,
        )

        return {
            "success": True,
            "error": None,
            "resolution_clamped": clamped,
            "no_change": False,
            "new_resolution": new_resolution,
        }
