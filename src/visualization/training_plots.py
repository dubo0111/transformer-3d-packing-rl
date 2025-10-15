"""
Training Visualization and Plotting

Utilities for visualizing training progress, metrics, and performance.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator


class TrainingPlotter:
    """
    Visualize training metrics and progress.

    Features:
    - Plot training curves (loss, reward, utilization)
    - Compare multiple training runs
    - Export plots as HTML or images
    - Load metrics from TensorBoard logs

    Example:
        >>> plotter = TrainingPlotter()
        >>> plotter.plot_training_curves(metrics_dict)
        >>> plotter.save_html("training_curves.html")
    """

    def __init__(self):
        self.fig = None

    def plot_training_curves(
        self,
        metrics: Dict[str, List[float]],
        steps: Optional[List[int]] = None,
        title: str = "Training Curves",
        smooth: bool = True,
        window: int = 10,
    ) -> go.Figure:
        """
        Plot training curves for multiple metrics.

        Args:
            metrics: Dictionary of {metric_name: values}
            steps: List of step numbers (x-axis)
            title: Plot title
            smooth: Whether to apply smoothing
            window: Smoothing window size

        Returns:
            Plotly figure
        """
        if steps is None:
            # Assume metrics are indexed by step
            max_len = max(len(v) for v in metrics.values())
            steps = list(range(max_len))

        # Create subplots
        n_metrics = len(metrics)
        rows = (n_metrics + 1) // 2
        cols = min(2, n_metrics)

        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=list(metrics.keys()),
        )

        for idx, (metric_name, values) in enumerate(metrics.items()):
            row = idx // cols + 1
            col = idx % cols + 1

            # Optionally smooth the curve
            if smooth and len(values) > window:
                smoothed_values = self._smooth(values, window)
            else:
                smoothed_values = values

            # Add trace
            fig.add_trace(
                go.Scatter(
                    x=steps[:len(values)],
                    y=values,
                    mode="lines",
                    name=metric_name,
                    opacity=0.3 if smooth else 1.0,
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

            if smooth:
                fig.add_trace(
                    go.Scatter(
                        x=steps[:len(smoothed_values)],
                        y=smoothed_values,
                        mode="lines",
                        name=f"{metric_name} (smoothed)",
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                )

        fig.update_layout(
            title=title,
            height=300 * rows,
            showlegend=False,
        )

        self.fig = fig
        return fig

    def plot_metric_comparison(
        self,
        metrics_dict: Dict[str, Dict[str, List[float]]],
        metric_name: str,
        title: Optional[str] = None,
    ) -> go.Figure:
        """
        Compare a single metric across multiple runs.

        Args:
            metrics_dict: {run_name: {metric_name: values}}
            metric_name: Name of metric to compare
            title: Plot title

        Returns:
            Plotly figure
        """
        fig = go.Figure()

        for run_name, metrics in metrics_dict.items():
            if metric_name in metrics:
                values = metrics[metric_name]
                steps = list(range(len(values)))

                fig.add_trace(
                    go.Scatter(
                        x=steps,
                        y=values,
                        mode="lines",
                        name=run_name,
                    )
                )

        if title is None:
            title = f"{metric_name.capitalize()} Comparison"

        fig.update_layout(
            title=title,
            xaxis_title="Step",
            yaxis_title=metric_name.capitalize(),
            showlegend=True,
        )

        self.fig = fig
        return fig

    def plot_episode_statistics(
        self,
        episode_rewards: List[float],
        episode_utilizations: List[float],
        episode_lengths: List[float],
        title: str = "Episode Statistics",
    ) -> go.Figure:
        """
        Plot episode-level statistics.

        Args:
            episode_rewards: List of episode rewards
            episode_utilizations: List of utilization rates
            episode_lengths: List of episode lengths
            title: Plot title

        Returns:
            Plotly figure
        """
        episodes = list(range(len(episode_rewards)))

        fig = make_subplots(
            rows=3,
            cols=1,
            subplot_titles=["Episode Reward", "Space Utilization", "Episode Length"],
        )

        # Reward
        fig.add_trace(
            go.Scatter(x=episodes, y=episode_rewards, mode="lines+markers", name="Reward"),
            row=1,
            col=1,
        )

        # Utilization
        fig.add_trace(
            go.Scatter(x=episodes, y=episode_utilizations, mode="lines+markers", name="Utilization"),
            row=2,
            col=1,
        )

        # Length
        fig.add_trace(
            go.Scatter(x=episodes, y=episode_lengths, mode="lines+markers", name="Length"),
            row=3,
            col=1,
        )

        fig.update_layout(
            title=title,
            height=800,
            showlegend=False,
        )

        self.fig = fig
        return fig

    def plot_utilization_distribution(
        self,
        utilizations: List[float],
        title: str = "Utilization Distribution",
    ) -> go.Figure:
        """
        Plot distribution of space utilization.

        Args:
            utilizations: List of utilization values
            title: Plot title

        Returns:
            Plotly figure
        """
        fig = go.Figure()

        fig.add_trace(
            go.Histogram(
                x=utilizations,
                nbinsx=30,
                name="Utilization",
            )
        )

        mean_util = np.mean(utilizations)
        fig.add_vline(
            x=mean_util,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {mean_util:.2%}",
        )

        fig.update_layout(
            title=title,
            xaxis_title="Space Utilization",
            yaxis_title="Frequency",
            showlegend=False,
        )

        self.fig = fig
        return fig

    @staticmethod
    def _smooth(values: List[float], window: int) -> List[float]:
        """
        Apply moving average smoothing.

        Args:
            values: Values to smooth
            window: Window size

        Returns:
            Smoothed values
        """
        if len(values) < window:
            return values

        smoothed = []
        for i in range(len(values)):
            start = max(0, i - window // 2)
            end = min(len(values), i + window // 2 + 1)
            smoothed.append(np.mean(values[start:end]))

        return smoothed

    def load_tensorboard_logs(
        self,
        log_dir: str,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, List]]:
        """
        Load metrics from TensorBoard logs.

        Args:
            log_dir: Path to TensorBoard log directory
            tags: List of metric tags to load (None = all)

        Returns:
            Dictionary of {tag: {"steps": [...], "values": [...]}}
        """
        ea = event_accumulator.EventAccumulator(log_dir)
        ea.Reload()

        available_tags = ea.Tags()["scalars"]
        if tags is None:
            tags = available_tags

        metrics = {}
        for tag in tags:
            if tag in available_tags:
                events = ea.Scalars(tag)
                steps = [e.step for e in events]
                values = [e.value for e in events]
                metrics[tag] = {"steps": steps, "values": values}

        return metrics

    def save_html(self, filepath: str):
        """
        Save current figure as HTML.

        Args:
            filepath: Path to save HTML file
        """
        if self.fig is None:
            raise ValueError("No figure to save.")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        self.fig.write_html(str(filepath))
        print(f"Plot saved to: {filepath}")

    def save_image(self, filepath: str, width: int = 1200, height: int = 800):
        """
        Save current figure as static image.

        Args:
            filepath: Path to save image (png, jpg, svg, pdf)
            width: Image width in pixels
            height: Image height in pixels
        """
        if self.fig is None:
            raise ValueError("No figure to save.")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        self.fig.write_image(str(filepath), width=width, height=height)
        print(f"Image saved to: {filepath}")

    def show(self):
        """Display current figure."""
        if self.fig is None:
            raise ValueError("No figure to show.")

        self.fig.show()
