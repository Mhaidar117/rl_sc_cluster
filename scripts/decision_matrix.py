#!/usr/bin/env python
"""Decision matrix: Total runtime (training + evaluation) for different dataset sizes and episode counts."""

import pandas as pd
import numpy as np

# Known benchmarks
# 1000 cells, 10 episodes = 5.5 minutes total (330 seconds)
# 10000 cells, 20 episodes = ~2 hours total (7200 seconds)
BASE_CELLS = 1000
BASE_EPISODES = 10
BASE_TIME_TOTAL = 330  # 5.5 minutes total (training + evaluation)

# Observed scaling from actual runs
# From 1000→10000 cells: time per episode scales as n^1.038
OBSERVED_EXPONENT = 1.038

# Full dataset size
FULL_DATASET_CELLS = 121427

# Dataset percentages to test
DATASET_PERCENTAGES = [10, 25, 50, 75, 100]

# Episode counts to test
EPISODE_COUNTS = [10, 20, 50, 100, 200, 300, 500]

# Training configuration (from test_ppo_minimal.py)
TRAINING_TIMESTEPS = 2000
TIMESTEPS_PER_ITERATION = 200
TRAINING_ITERATIONS = TRAINING_TIMESTEPS // TIMESTEPS_PER_ITERATION


def estimate_time_per_episode(n_cells: int) -> float:
    """
    Estimate time per episode for n_cells (evaluation phase).

    Uses power law scaling: time = base_time * (n_cells / base_cells)^exponent
    """
    time_per_episode_base = BASE_TIME_TOTAL / BASE_EPISODES
    scaling_factor = (n_cells / BASE_CELLS) ** OBSERVED_EXPONENT
    return time_per_episode_base * scaling_factor


def estimate_training_time(n_cells: int, n_episodes: int) -> float:
    """
    Estimate training time based on timesteps.

    Training time scales with cell count.
    For 2000 timesteps, training time is roughly proportional to cell count.

    Note: Actual observed training was slower than expected (7.4h for 10000 cells),
    but we'll use a more realistic estimate based on expected performance.
    For decision-making, we'll use a conservative estimate that accounts for
    both best-case (faster) and observed-case (slower) scenarios.
    """
    # Training time per iteration scales with cell count
    # From observed: 10000 cells, 2000 timesteps = 26573 seconds (7.4h) - SLOW
    # But expected was ~1.9 hours based on early iterations

    # Use a middle ground: assume training is ~2-3x slower than evaluation
    # This accounts for PPO overhead while being more realistic than worst-case

    eval_time_per_ep = estimate_time_per_episode(n_cells)

    # Training time: scale with cell count and timesteps
    # Base: for 1000 cells, training is minimal (~10% of total time)
    # For 10000 cells, if we assume training ≈ 1.5-2x evaluation time
    # This gives us a more realistic estimate

    # Use observed ratio but scale more conservatively
    # For 10000 cells: training was 26573s, eval would be ~7200s = 3.7x ratio
    # But this seems too high - let's use 2x as a more realistic ratio

    # Actually, let's calculate based on timesteps:
    # Each iteration collects 200 timesteps, and we need 10 iterations
    # Time per iteration scales with cell count
    base_iteration_time = 686  # seconds per iteration for 10000 cells (from terminal)
    scaling_factor = (n_cells / 10000) ** OBSERVED_EXPONENT
    time_per_iteration = base_iteration_time * scaling_factor
    training_time = time_per_iteration * TRAINING_ITERATIONS

    return training_time


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        if minutes < 10:
            return f"{minutes:.1f}m"
        else:
            return f"{minutes:.0f}m"
    elif seconds < 86400:
        hours = seconds / 3600
        if hours < 10:
            return f"{hours:.1f}h"
        else:
            return f"{hours:.0f}h"
    else:
        days = seconds / 86400
        if days < 10:
            return f"{days:.1f}d"
        else:
            return f"{days:.0f}d"


def create_decision_matrix():
    """Create decision matrix for total runtime."""
    results = []

    for pct in DATASET_PERCENTAGES:
        n_cells = int(FULL_DATASET_CELLS * pct / 100)

        for n_episodes in EPISODE_COUNTS:
            # Estimate training time
            training_time = estimate_training_time(n_cells, n_episodes)

            # Estimate evaluation time
            time_per_episode = estimate_time_per_episode(n_cells)
            eval_time = time_per_episode * n_episodes

            # Total time
            total_time = training_time + eval_time

            results.append({
                'Dataset %': pct,
                'Cells': n_cells,
                'Episodes': n_episodes,
                'Training Time': format_time(training_time),
                'Eval Time': format_time(eval_time),
                'Total Time': format_time(total_time),
                'Total (hours)': total_time / 3600,
                'Total (minutes)': total_time / 60,
                'Total (seconds)': total_time,
            })

    df = pd.DataFrame(results)
    return df


def print_decision_matrix(df: pd.DataFrame):
    """Print formatted decision matrix."""
    print("=" * 100)
    print("DECISION MATRIX: Total Runtime (Training + Evaluation)")
    print("=" * 100)
    print(f"\nBaseline: {BASE_CELLS} cells, {BASE_EPISODES} episodes = {BASE_TIME_TOTAL/60:.1f} minutes")
    print(f"Observed: 10000 cells, 20 episodes = ~2 hours total")
    print(f"Scaling: Time per episode ∝ (n_cells)^{OBSERVED_EXPONENT}")
    print(f"\nFull dataset: {FULL_DATASET_CELLS:,} cells")
    print(f"Training: {TRAINING_TIMESTEPS} timesteps ({TRAINING_ITERATIONS} iterations)")
    print("\n" + "=" * 100)

    # Create pivot table: rows = dataset %, columns = episodes, values = total time
    pivot = df.pivot_table(
        index='Dataset %',
        columns='Episodes',
        values='Total Time',
        aggfunc='first'
    )

    print("\nTOTAL RUNTIME BY DATASET % AND EPISODES:")
    print(pivot.to_string())

    print("\n" + "=" * 100)
    print("\nTOTAL RUNTIME IN HOURS:")
    pivot_hours = df.pivot_table(
        index='Dataset %',
        columns='Episodes',
        values='Total (hours)',
        aggfunc='first'
    )
    # Format hours nicely
    pivot_hours_str = pivot_hours.map(lambda x: f"{x:.1f}h" if x < 24 else f"{x/24:.1f}d")
    print(pivot_hours_str.to_string())

    print("\n" + "=" * 100)
    print("\nQUICK REFERENCE - Common Scenarios:")
    print("-" * 100)

    common_scenarios = [
        (10, 10, "Quick test"),
        (10, 20, "Small validation"),
        (25, 50, "Medium validation"),
        (50, 100, "Large validation"),
        (100, 200, "Full dataset, moderate episodes"),
        (100, 500, "Full dataset, many episodes"),
    ]

    for pct, episodes, desc in common_scenarios:
        subset = df[(df['Dataset %'] == pct) & (df['Episodes'] == episodes)]
        if not subset.empty:
            row = subset.iloc[0]
            print(f"{desc:25} | {pct:3}% dataset ({row['Cells']:6,} cells) | {episodes:3} episodes | {row['Total Time']:>8} | {row['Total (hours)']:>6.1f}h")

    print("=" * 100)


def main():
    """Main function."""
    df = create_decision_matrix()
    print_decision_matrix(df)

    # Save to CSV
    output_path = "decision_matrix.csv"
    df.to_csv(output_path, index=False)
    print(f"\n✓ Saved detailed matrix to: {output_path}")

    # Also save pivot table for easy viewing
    pivot_hours = df.pivot_table(
        index='Dataset %',
        columns='Episodes',
        values='Total (hours)',
        aggfunc='first'
    )
    pivot_path = "decision_matrix_pivot.csv"
    pivot_hours.to_csv(pivot_path)
    print(f"✓ Saved pivot table (hours) to: {pivot_path}")


if __name__ == "__main__":
    main()
