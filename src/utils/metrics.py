"""
Metrics Calculator for Bin Packing Evaluation

Comprehensive metrics for evaluating packing quality and performance.
Paper Reference: Section 5 - Experimental Results
"""

import numpy as np
from typing import Dict, List, Any
from ..environment.container import Container


class MetricsCalculator:
    """
    Calculate various metrics for bin packing evaluation.

    Metrics include:
    - Space utilization (primary metric from paper)
    - Packing ratio (items packed / total items)
    - Average height (compactness measure)
    - Stability score (center of gravity)
    - Invalid action rate
    - Average episode length

    Paper Reference: Section 5.1 - Evaluation Metrics
    """

    @staticmethod
    def calculate_space_utilization(container: Container) -> float:
        """
        Calculate space utilization ratio.

        Space utilization = Packed volume / Container volume

        Paper Reference: Equation (8) - Primary evaluation metric
        "Space utilization is the most important metric for bin packing,
        representing the efficiency of packing."

        Args:
            container: Container with packed items

        Returns:
            Utilization ratio [0, 1]
        """
        return container.utilization

    @staticmethod
    def calculate_packing_ratio(num_packed: int, total_items: int) -> float:
        """
        Calculate packing ratio.

        Packing ratio = Number of packed items / Total items

        Args:
            num_packed: Number of successfully packed items
            total_items: Total number of items

        Returns:
            Packing ratio [0, 1]
        """
        return num_packed / total_items if total_items > 0 else 0.0

    @staticmethod
    def calculate_average_height(container: Container) -> float:
        """
        Calculate average height of packed items.

        Lower average height indicates more compact packing.

        Args:
            container: Container with packed items

        Returns:
            Average height (normalized by container height)
        """
        if not container.packed_items:
            return 0.0

        total_height = sum(
            item.position[2] + item.dimensions[2]
            for item in container.packed_items
        )
        avg_height = total_height / len(container.packed_items)

        # Normalize by container height
        return avg_height / container.height

    @staticmethod
    def calculate_stability(container: Container) -> float:
        """
        Calculate packing stability score.

        Args:
            container: Container with packed items

        Returns:
            Stability score [0, 1], higher is better
        """
        return container.get_stability_score()

    @staticmethod
    def calculate_compactness(container: Container) -> float:
        """
        Calculate packing compactness.

        Compactness measures how tightly items are packed vertically.

        Args:
            container: Container with packed items

        Returns:
            Compactness score (lower is more compact)
        """
        return container.get_compactness() / container.height

    @staticmethod
    def calculate_surface_flatness(container: Container) -> float:
        """
        Calculate surface flatness (variance of height map).

        Lower variance indicates flatter, more uniform packing.

        Args:
            container: Container with packed items

        Returns:
            Normalized standard deviation of height map
        """
        if container.height_map.size == 0:
            return 0.0

        std_dev = np.std(container.height_map)
        return std_dev / container.height

    @staticmethod
    def calculate_episode_metrics(
        episode_rewards: List[float],
        episode_utilizations: List[float],
        episode_lengths: List[int],
    ) -> Dict[str, float]:
        """
        Calculate aggregate metrics across multiple episodes.

        Args:
            episode_rewards: List of episode rewards
            episode_utilizations: List of utilization ratios
            episode_lengths: List of episode lengths

        Returns:
            Dictionary of aggregate metrics
        """
        return {
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "mean_utilization": np.mean(episode_utilizations),
            "std_utilization": np.std(episode_utilizations),
            "max_utilization": np.max(episode_utilizations),
            "min_utilization": np.min(episode_utilizations),
            "mean_length": np.mean(episode_lengths),
            "std_length": np.std(episode_lengths),
        }

    @staticmethod
    def calculate_all_metrics(container: Container, info: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate all available metrics for a container.

        Args:
            container: Container with packed items
            info: Additional information dictionary

        Returns:
            Dictionary of all metrics
        """
        metrics = {
            "utilization": MetricsCalculator.calculate_space_utilization(container),
            "packing_ratio": MetricsCalculator.calculate_packing_ratio(
                container.num_packed,
                info.get("total_items", container.num_packed),
            ),
            "average_height": MetricsCalculator.calculate_average_height(container),
            "stability": MetricsCalculator.calculate_stability(container),
            "compactness": MetricsCalculator.calculate_compactness(container),
            "surface_flatness": MetricsCalculator.calculate_surface_flatness(container),
            "max_height": container.get_max_height() / container.height,
            "num_packed": container.num_packed,
        }

        return metrics

    @staticmethod
    def print_metrics(metrics: Dict[str, float], title: str = "Metrics"):
        """
        Print metrics in a formatted table.

        Args:
            metrics: Dictionary of metrics
            title: Table title
        """
        print(f"\n{'='*50}")
        print(f"{title:^50}")
        print(f"{'='*50}")

        for key, value in metrics.items():
            if isinstance(value, float):
                if "ratio" in key or "utilization" in key or "stability" in key:
                    print(f"{key:.<40} {value:>8.2%}")
                else:
                    print(f"{key:.<40} {value:>8.4f}")
            else:
                print(f"{key:.<40} {value:>8}")

        print(f"{'='*50}\n")
