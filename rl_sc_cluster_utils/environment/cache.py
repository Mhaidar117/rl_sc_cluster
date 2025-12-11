"""Caching utilities for clustering-dependent metrics."""

from typing import Dict, List, Optional

import pandas as pd


class ClusteringCache:
    """
    LRU cache for clustering-dependent metrics.

    Uses hash-based keys derived from clustering state to cache expensive
    computations like silhouette scores, modularity, GAG metrics, etc.

    Parameters
    ----------
    max_size : int, optional
        Maximum number of entries in cache (default: 100)
    """

    def __init__(self, max_size: int = 100) -> None:
        self.max_size = max_size
        self._cache: Dict[str, Dict] = {}
        self._access_order: List[str] = []
        self.hits = 0
        self.misses = 0

    def _hash_clustering(self, cluster_labels) -> str:
        """
        Create stable hash key from clustering labels.

        Uses sorted cluster label counts to create a deterministic hash
        that is invariant to label ordering.

        Parameters
        ----------
        cluster_labels : array-like
            Cluster labels for each cell

        Returns
        -------
        hash_key : str
            String representation of hash
        """
        # Convert to Series and get value counts, sorted by cluster ID
        counts = pd.Series(cluster_labels).value_counts().sort_index()
        # Create tuple of (cluster_id, count) pairs for stable hashing
        key_tuple = tuple(zip(counts.index, counts.values))
        return str(hash(key_tuple))

    def get(self, cluster_labels, metric_type: str) -> Optional[Dict]:
        """
        Retrieve cached metric if available.

        Parameters
        ----------
        cluster_labels : array-like
            Cluster labels for current state
        metric_type : str
            Type of metric (e.g., 'q_cluster', 'q_gag')

        Returns
        -------
        cached_value : dict or None
            Cached metric value if found, None otherwise
        """
        hash_key = self._hash_clustering(cluster_labels)
        cache_key = f"{hash_key}_{metric_type}"

        if cache_key in self._cache:
            self.hits += 1
            # Update access order (move to end = most recently used)
            if cache_key in self._access_order:
                self._access_order.remove(cache_key)
            self._access_order.append(cache_key)
            return self._cache[cache_key]

        self.misses += 1
        return None

    def set(self, cluster_labels, metric_type: str, value: Dict) -> None:
        """
        Cache metric value with LRU eviction.

        Parameters
        ----------
        cluster_labels : array-like
            Cluster labels for current state
        metric_type : str
            Type of metric (e.g., 'q_cluster', 'q_gag')
        value : dict
            Metric value to cache
        """
        hash_key = self._hash_clustering(cluster_labels)
        cache_key = f"{hash_key}_{metric_type}"

        # LRU eviction: remove oldest entry if cache is full
        if len(self._cache) >= self.max_size and cache_key not in self._cache:
            oldest = self._access_order.pop(0)
            del self._cache[oldest]

        # Add/update entry
        self._cache[cache_key] = value
        # Update access order
        if cache_key in self._access_order:
            self._access_order.remove(cache_key)
        self._access_order.append(cache_key)

    def clear(self) -> None:
        """Clear all cached values and reset statistics."""
        self._cache.clear()
        self._access_order.clear()
        self.hits = 0
        self.misses = 0

    def get_stats(self) -> Dict[str, float]:
        """
        Get cache performance statistics.

        Returns
        -------
        stats : dict
            Dictionary with keys: hits, misses, hit_rate, size
        """
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "size": len(self._cache),
        }
