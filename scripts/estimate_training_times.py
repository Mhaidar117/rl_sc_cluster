#!/usr/bin/env python
"""Estimate training times for different dataset sizes and episode counts."""

import pandas as pd
import numpy as np

# Known benchmarks
# 1000 cells, 10 episodes = 5-6 minutes (using 5.5 minutes = 330 seconds)
# 10000 cells, 20 episodes = ~2 hours (7200 seconds)
BASE_CELLS = 1000
BASE_EPISODES = 10
BASE_TIME_SECONDS = 330  # 5.5 minutes

# Observed scaling from actual runs
# From 1000→10000 cells: 33 sec/ep → 360 sec/ep = 10.91x for 10x cells
# This gives exponent ≈ 1.038 (n^1.038, nearly linear!)
OBSERVED_EXPONENT = 1.038

# Dataset sizes
FULL_DATASET_CELLS = 121427
HALF_DATASET_CELLS = 60713  # 50% of full dataset

# Episode counts to estimate
EPISODE_COUNTS = [10, 50, 100, 200, 300, 500, 1000]

# Scaling factors for computation time with cell count
# Observed: n^1.038 (from actual runs, nearly linear - best case)
# Conservative: n^1.8 (closer to quadratic, worst case if scaling degrades)
SCALING_EXPONENT_OBSERVED = 1.038
SCALING_EXPONENT_CONSERVATIVE = 1.8


def estimate_time_per_episode(n_cells: int, exponent: float) -> float:
    """
    Estimate time per episode for n_cells.

    Uses power law scaling: time = base_time * (n_cells / base_cells)^exponent
    """
    time_per_episode_base = BASE_TIME_SECONDS / BASE_EPISODES
    scaling_factor = (n_cells / BASE_CELLS) ** exponent
    return time_per_episode_base * scaling_factor


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    elif seconds < 86400:
        hours = seconds / 3600
        return f"{hours:.1f}h"
    else:
        days = seconds / 86400
        return f"{days:.1f}d"


def create_time_table():
    """Create a table of estimated training times."""
    results = []

    for n_episodes in EPISODE_COUNTS:
        # Full dataset - observed (based on actual runs)
        time_per_episode_full_obs = estimate_time_per_episode(FULL_DATASET_CELLS, SCALING_EXPONENT_OBSERVED)
        total_time_full_obs = time_per_episode_full_obs * n_episodes

        # Full dataset - conservative
        time_per_episode_full_cons = estimate_time_per_episode(FULL_DATASET_CELLS, SCALING_EXPONENT_CONSERVATIVE)
        total_time_full_cons = time_per_episode_full_cons * n_episodes

        # Half dataset - observed
        time_per_episode_half_obs = estimate_time_per_episode(HALF_DATASET_CELLS, SCALING_EXPONENT_OBSERVED)
        total_time_half_obs = time_per_episode_half_obs * n_episodes

        # Half dataset - conservative
        time_per_episode_half_cons = estimate_time_per_episode(HALF_DATASET_CELLS, SCALING_EXPONENT_CONSERVATIVE)
        total_time_half_cons = time_per_episode_half_cons * n_episodes

        results.append({
            'Episodes': n_episodes,
            'Full Dataset (Observed)': format_time(total_time_full_obs),
            'Full Dataset (Conservative)': format_time(total_time_full_cons),
            '50% Dataset (Observed)': format_time(total_time_half_obs),
            '50% Dataset (Conservative)': format_time(total_time_half_cons),
            'Full Dataset (Obs, sec)': total_time_full_obs,
            'Full Dataset (Cons, sec)': total_time_full_cons,
            '50% Dataset (Obs, sec)': total_time_half_obs,
            '50% Dataset (Cons, sec)': total_time_half_cons,
        })

    df = pd.DataFrame(results)
    return df


def print_table(df: pd.DataFrame):
    """Print formatted table."""
    print("=" * 90)
    print("ESTIMATED TRAINING TIMES")
    print("=" * 90)
    print(f"\nBaseline: {BASE_CELLS} cells, {BASE_EPISODES} episodes = {BASE_TIME_SECONDS/60:.1f} minutes")
    print(f"Observed: 10000 cells, 20 episodes = ~2 hours")
    print(f"Scaling assumptions:")
    print(f"  - Observed: Time per episode ∝ (n_cells)^{SCALING_EXPONENT_OBSERVED} (from actual runs)")
    print(f"  - Conservative: Time per episode ∝ (n_cells)^{SCALING_EXPONENT_CONSERVATIVE} (worst case)")
    print(f"\nFull dataset: {FULL_DATASET_CELLS:,} cells")
    print(f"50% dataset: {HALF_DATASET_CELLS:,} cells")
    print("\n" + "=" * 90)

    # Print main table
    display_df = df[['Episodes', 'Full Dataset (Observed)', 'Full Dataset (Conservative)',
                     '50% Dataset (Observed)', '50% Dataset (Conservative)']].copy()
    print(display_df.to_string(index=False))

    print("\n" + "=" * 90)
    print("\nTime per episode breakdown:")
    print(f"Full dataset (observed): {format_time(estimate_time_per_episode(FULL_DATASET_CELLS, SCALING_EXPONENT_OBSERVED))} per episode")
    print(f"Full dataset (conservative): {format_time(estimate_time_per_episode(FULL_DATASET_CELLS, SCALING_EXPONENT_CONSERVATIVE))} per episode")
    print(f"50% dataset (observed): {format_time(estimate_time_per_episode(HALF_DATASET_CELLS, SCALING_EXPONENT_OBSERVED))} per episode")
    print(f"50% dataset (conservative): {format_time(estimate_time_per_episode(HALF_DATASET_CELLS, SCALING_EXPONENT_CONSERVATIVE))} per episode")
    print("=" * 90)


def main():
    """Main function."""
    df = create_time_table()
    print_table(df)

    # Save to CSV
    output_path = "training_time_estimates.csv"
    df.to_csv(output_path, index=False)
    print(f"\n✓ Saved detailed estimates to: {output_path}")


if __name__ == "__main__":
    main()
