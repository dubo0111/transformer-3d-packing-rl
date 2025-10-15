"""
Evaluation Script for TAP-Net

Evaluate trained model and generate visualizations.

Usage:
    python evaluate.py --checkpoint checkpoints/best.pt
    python evaluate.py --checkpoint checkpoints/best.pt --visualize --save-html
    python evaluate.py --checkpoint checkpoints/best.pt --n-episodes 50
"""

import argparse
import torch
from pathlib import Path

from src.environment.packing_env import PackingEnv
from src.models.tap_net import TAPNet
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.metrics import MetricsCalculator
from src.visualization.plotly_3d import PackingVisualizer
from src.visualization.training_plots import TrainingPlotter


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate TAP-Net")

    # Model
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (auto-detected from checkpoint dir if not provided)",
    )

    # Evaluation
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=10,
        help="Number of episodes to evaluate",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic policy (greedy)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device for evaluation",
    )

    # Visualization
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate 3D visualizations",
    )
    parser.add_argument(
        "--save-html",
        action="store_true",
        help="Save visualizations as HTML",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save outputs",
    )

    return parser.parse_args()


def main():
    """Main evaluation loop."""
    args = parse_args()

    # Setup logger
    logger = setup_logger("evaluation", log_file="logs/evaluation.log")
    logger.info("=" * 80)
    logger.info("TAP-Net Evaluation")
    logger.info("=" * 80)

    # Load configuration
    if args.config is None:
        # Auto-detect config from checkpoint directory
        checkpoint_path = Path(args.checkpoint)
        config_path = checkpoint_path.parent / "config.yaml"
        if not config_path.exists():
            config_path = "config/default.yaml"
    else:
        config_path = args.config

    logger.info(f"Loading configuration from: {config_path}")
    config = load_config(str(config_path))

    # Create environment
    logger.info("Creating environment...")
    env = PackingEnv(
        container_size=tuple(config["environment"]["container_size"]),
        grid_size=config["environment"]["grid_size"],
        max_items=config["environment"]["max_items"],
        item_size_range=tuple(config["environment"]["item_size_range"]),
        enable_action_mask=config["environment"]["enable_action_mask"],
        reward_type=config["environment"]["reward_type"],
        normalize_state=config["environment"]["normalize_state"],
        seed=42,
    )

    # Create model
    logger.info("Creating TAP-Net model...")
    model = TAPNet(
        grid_size=config["environment"]["grid_size"],
        item_feature_dim=5,
        d_model=config["model"]["d_model"],
        nhead=config["model"]["nhead"],
        num_layers=config["model"]["num_layers"],
        dim_feedforward=config["model"]["dim_feedforward"],
        dropout=config["model"]["dropout"],
        share_encoder=config["model"]["share_encoder"],
    )

    # Load checkpoint
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    model.load(args.checkpoint, device=args.device)
    model.to(args.device)
    model.eval()

    logger.info("Model loaded successfully")

    # Run evaluation
    logger.info(f"Evaluating over {args.n_episodes} episodes...")

    episode_metrics_list = []
    containers_for_viz = []

    for episode_idx in range(args.n_episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        done = False
        steps = 0

        with torch.no_grad():
            while not done:
                # Parse observation
                grid_size = env.grid_size
                height_map_size = grid_size * grid_size
                height_map = obs[:height_map_size].reshape(grid_size, grid_size)
                item_features = obs[height_map_size:height_map_size + 5]

                # Get action mask
                action_mask = env._get_action_mask()

                # Convert to tensors
                height_map_tensor = torch.from_numpy(height_map).unsqueeze(0).to(args.device)
                item_features_tensor = torch.from_numpy(item_features).unsqueeze(0).to(args.device)
                action_mask_tensor = torch.from_numpy(action_mask).unsqueeze(0).to(args.device)

                # Get action
                action, _, _ = model.get_action(
                    height_map_tensor,
                    item_features_tensor,
                    action_mask_tensor,
                    deterministic=args.deterministic,
                )

                # Step environment
                obs, reward, terminated, truncated, info = env.step(action.item())
                episode_reward += reward
                steps += 1
                done = terminated or truncated

        # Calculate metrics
        container = env.get_container_state()
        metrics = MetricsCalculator.calculate_all_metrics(container, info)
        metrics["episode_reward"] = episode_reward
        metrics["episode_length"] = steps

        episode_metrics_list.append(metrics)

        # Save container for visualization (first 5 episodes)
        if args.visualize and episode_idx < 5:
            import copy
            containers_for_viz.append(copy.deepcopy(container))

        logger.info(f"Episode {episode_idx + 1}/{args.n_episodes}: "
                   f"Utilization={metrics['utilization']:.2%}, "
                   f"Packed={int(metrics['num_packed'])}, "
                   f"Reward={episode_reward:.3f}")

    # Aggregate metrics
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 80)

    # Calculate mean and std for all metrics
    import numpy as np

    all_keys = episode_metrics_list[0].keys()
    aggregate_metrics = {}

    for key in all_keys:
        values = [m[key] for m in episode_metrics_list]
        aggregate_metrics[f"mean_{key}"] = np.mean(values)
        aggregate_metrics[f"std_{key}"] = np.std(values)
        if key in ["utilization", "packing_ratio"]:
            aggregate_metrics[f"max_{key}"] = np.max(values)
            aggregate_metrics[f"min_{key}"] = np.min(values)

    # Print results
    MetricsCalculator.print_metrics(aggregate_metrics, "Evaluation Metrics")

    # Generate visualizations
    if args.visualize:
        logger.info("Generating visualizations...")
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        visualizer = PackingVisualizer()

        for idx, container in enumerate(containers_for_viz):
            logger.info(f"Visualizing episode {idx + 1}...")

            fig = visualizer.visualize_container(
                container,
                show_height_map=False,
                show_container_bounds=True,
                show_labels=True,
                title=f"Episode {idx + 1} (Utilization: {container.utilization:.1%})",
            )

            if args.save_html:
                html_path = output_dir / f"episode_{idx + 1}.html"
                visualizer.save_html(str(html_path))
                logger.info(f"  Saved to: {html_path}")
            else:
                visualizer.show()

        logger.info(f"Visualizations saved to: {output_dir}")

    # Save metrics to file
    metrics_file = Path(args.output_dir) / "evaluation_metrics.txt"
    metrics_file.parent.mkdir(parents=True, exist_ok=True)

    with open(metrics_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("TAP-Net Evaluation Results\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Episodes: {args.n_episodes}\n")
        f.write(f"Deterministic: {args.deterministic}\n\n")

        f.write("Aggregate Metrics:\n")
        f.write("-" * 80 + "\n")
        for key, value in aggregate_metrics.items():
            if isinstance(value, float):
                f.write(f"{key:.<40} {value:>10.4f}\n")
            else:
                f.write(f"{key:.<40} {value:>10}\n")

    logger.info(f"Metrics saved to: {metrics_file}")
    logger.info("=" * 80)
    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()
