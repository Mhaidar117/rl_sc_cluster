#!/usr/bin/env python
"""Test script to verify ClusteringEnv works with real data."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import scanpy as sc
import numpy as np
from rl_sc_cluster_utils.environment import ClusteringEnv

# GAG gene sets (from notebook)
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


def load_and_prepare_data(data_path, use_subset=False, subset_size=5000):
    """Load and prepare AnnData for environment."""
    print(f"Loading data from: {data_path}")
    adata = sc.read_h5ad(data_path)

    print(f"Data shape: {adata.shape}")
    print(f"Available obsm keys: {list(adata.obsm.keys())}")

    # Map embedding key if needed
    if 'scVI' in adata.obsm and 'X_scvi' not in adata.obsm:
        print("Mapping 'scVI' to 'X_scvi'...")
        adata.obsm['X_scvi'] = adata.obsm['scVI']

    # Use subset for faster testing
    if use_subset:
        print(f"Using subset of {subset_size} cells for testing...")
        adata = adata[:subset_size].copy()

    # Check gene names format
    print(f"Gene names sample: {list(adata.var_names[:5])}")

    return adata


def check_gene_sets(adata, gene_sets):
    """Check which genes from gene sets are present in data."""
    print("\nChecking gene set coverage:")
    all_genes = set(adata.var_names)

    for set_name, genes in gene_sets.items():
        found = [g for g in genes if g in all_genes]
        coverage = len(found) / len(genes) * 100
        print(f"  {set_name}: {len(found)}/{len(genes)} genes ({coverage:.1f}%)")
        if len(found) == 0:
            print(f"    WARNING: No genes found for {set_name}!")

    return all_genes


def test_environment(adata, gene_sets, max_steps=5):
    """Test the environment with real data."""
    print("\n" + "="*60)
    print("Testing ClusteringEnv")
    print("="*60)

    try:
        # Create environment
        print("\nCreating environment...")
        env = ClusteringEnv(
            adata=adata,
            gene_sets=gene_sets,
            max_steps=max_steps,
            normalize_rewards=True
        )
        print("✓ Environment created successfully")

        # Reset
        print("\nResetting environment...")
        state, info = env.reset()
        print(f"✓ Reset successful")
        print(f"  State shape: {state.shape}")
        print(f"  Initial clusters: {info['n_clusters']}")
        print(f"  Resolution: {info['resolution']}")

        # Run a few steps
        print(f"\nRunning {max_steps} steps...")
        total_reward = 0

        for step in range(max_steps):
            # Use a simple action sequence for testing
            action = 2 if step % 2 == 0 else 3  # Alternate increment/decrement

            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            print(f"  Step {step+1}: action={action}, reward={reward:.4f}, "
                  f"clusters={info['n_clusters']}, "
                  f"Q_cluster={info['Q_cluster']:.4f}, "
                  f"Q_GAG={info['Q_GAG']:.4f}, "
                  f"penalty={info['penalty']:.4f}")

            if terminated or truncated:
                print(f"  Episode ended: terminated={terminated}, truncated={truncated}")
                break

        print(f"\n✓ Test completed successfully!")
        print(f"  Total reward: {total_reward:.4f}")
        print(f"  Final clusters: {info['n_clusters']}")

        return True

    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    data_path = project_root / "data" / "processed" / "human_interneurons.h5ad"

    if not data_path.exists():
        print(f"ERROR: Data file not found at {data_path}")
        print("Please ensure the data file exists.")
        return 1

    # Load data (use subset for faster testing)
    adata = load_and_prepare_data(data_path, use_subset=True, subset_size=5000)

    # Check gene sets
    all_genes = check_gene_sets(adata, GAG_GENE_SETS)

    # Test environment
    success = test_environment(adata, GAG_GENE_SETS, max_steps=5)

    if success:
        print("\n" + "="*60)
        print("✓ All tests passed! Environment is ready for real data.")
        print("="*60)
        return 0
    else:
        print("\n" + "="*60)
        print("✗ Tests failed. Please check the errors above.")
        print("="*60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
