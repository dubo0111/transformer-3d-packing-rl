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

# Optional MIP solver import
try:
    from src.utils.mip_solver import MIPSolver, format_mip_solution
    from src.utils.mip_to_container import mip_solution_to_container
    MIP_AVAILABLE = True
except ImportError:
    MIP_AVAILABLE = False


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

    # MIP Baseline Comparison
    parser.add_argument(
        "--compare-mip",
        action="store_true",
        help="Compare RL performance with MIP optimal baseline",
    )
    parser.add_argument(
        "--mip-timeout",
        type=int,
        default=300,
        help="MIP solver timeout in seconds (default: 300)",
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

    # Initialize MIP solver if requested
    mip_solver = None
    if args.compare_mip:
        if not MIP_AVAILABLE:
            logger.warning("MIP comparison requested but Gurobi not available. Skipping MIP comparison.")
            args.compare_mip = False
        else:
            logger.info(f"Initializing MIP solver (timeout={args.mip_timeout}s)...")
            mip_solver = MIPSolver(timeout=args.mip_timeout, verbose=False)

    # Run evaluation
    logger.info(f"Evaluating over {args.n_episodes} episodes...")

    episode_metrics_list = []
    mip_metrics_list = []
    containers_for_viz = []
    mip_containers_for_viz = []

    for episode_idx in range(args.n_episodes):
        obs, info = env.reset()

        # Store initial items for MIP comparison
        if args.compare_mip:
            episode_items = [(item.length, item.width, item.height) for item in env.items]
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

        # Run MIP solver on same items if requested
        if args.compare_mip:
            logger.info(f"  Running MIP solver for episode {episode_idx + 1}...")
            mip_solution = mip_solver.solve(
                container_size=env.container_size,
                boxes=episode_items
            )
            mip_metrics_list.append(mip_solution)

            logger.info(f"  MIP: Utilization={mip_solution['utilization']:.2%}, "
                       f"Packed={mip_solution['num_packed']}/{mip_solution['num_total']}, "
                       f"Time={mip_solution['solve_time']:.2f}s, "
                       f"Optimal={'Yes' if mip_solution['optimal'] else 'No'}")

            # Convert MIP solution to container for visualization
            if args.visualize and episode_idx < 5:
                mip_container = mip_solution_to_container(
                    mip_solution,
                    container_size=env.container_size,
                    grid_size=env.grid_size
                )
                mip_containers_for_viz.append(mip_container)

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
    MetricsCalculator.print_metrics(aggregate_metrics, "RL Agent Metrics")

    # Print MIP comparison if available
    if args.compare_mip and mip_metrics_list:
        logger.info("\n" + "=" * 80)
        logger.info("MIP BASELINE COMPARISON")
        logger.info("=" * 80)

        # Aggregate MIP metrics
        mip_utilizations = [m['utilization'] for m in mip_metrics_list]
        mip_solve_times = [m['solve_time'] for m in mip_metrics_list]
        mip_optimal_count = sum(1 for m in mip_metrics_list if m['optimal'])

        rl_utilizations = [m['utilization'] for m in episode_metrics_list]

        # Compute comparison metrics
        import numpy as np
        mean_mip_util = np.mean(mip_utilizations)
        mean_rl_util = np.mean(rl_utilizations)

        if mean_mip_util > 0:
            optimality_gap = (mean_mip_util - mean_rl_util) / mean_mip_util * 100
            relative_performance = mean_rl_util / mean_mip_util
        else:
            optimality_gap = 0.0
            relative_performance = 1.0

        logger.info(f"MIP Utilization (mean):    {mean_mip_util:.2%}")
        logger.info(f"RL Utilization (mean):     {mean_rl_util:.2%}")
        logger.info(f"Optimality Gap:            {optimality_gap:.2f}%")
        logger.info(f"Relative Performance:      {relative_performance:.2%}")
        logger.info(f"MIP Solve Time (mean):     {np.mean(mip_solve_times):.2f}s")
        logger.info(f"MIP Optimal Solutions:     {mip_optimal_count}/{len(mip_metrics_list)}")
        logger.info("=" * 80)

    # Generate visualizations
    if args.visualize:
        logger.info("Generating visualizations...")
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        visualizer = PackingVisualizer()

        for idx, container in enumerate(containers_for_viz):
            logger.info(f"Visualizing episode {idx + 1}...")

            # Use side-by-side comparison if MIP is enabled
            if args.compare_mip and idx < len(mip_containers_for_viz):
                mip_container = mip_containers_for_viz[idx]

                # Create side-by-side RL vs MIP visualization
                fig = PackingVisualizer.compare_rl_vs_mip(
                    rl_container=container,
                    mip_container=mip_container,
                    episode_num=idx + 1
                )

                if args.save_html:
                    html_path = output_dir / f"episode_{idx + 1}_comparison.html"
                    fig.write_html(str(html_path))
                    logger.info(f"  Saved comparison to: {html_path}")
                else:
                    fig.show()
            else:
                # Standard single visualization (RL only)
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

        f.write("RL Agent Metrics:\n")
        f.write("-" * 80 + "\n")
        for key, value in aggregate_metrics.items():
            if isinstance(value, float):
                f.write(f"{key:.<40} {value:>10.4f}\n")
            else:
                f.write(f"{key:.<40} {value:>10}\n")

        # Add MIP comparison to file
        if args.compare_mip and mip_metrics_list:
            f.write("\n" + "=" * 80 + "\n")
            f.write("MIP Baseline Comparison\n")
            f.write("=" * 80 + "\n\n")

            import numpy as np
            mip_utilizations = [m['utilization'] for m in mip_metrics_list]
            mip_solve_times = [m['solve_time'] for m in mip_metrics_list]
            mip_optimal_count = sum(1 for m in mip_metrics_list if m['optimal'])
            rl_utilizations = [m['utilization'] for m in episode_metrics_list]

            mean_mip_util = np.mean(mip_utilizations)
            mean_rl_util = np.mean(rl_utilizations)

            if mean_mip_util > 0:
                optimality_gap = (mean_mip_util - mean_rl_util) / mean_mip_util * 100
                relative_performance = mean_rl_util / mean_mip_util
            else:
                optimality_gap = 0.0
                relative_performance = 1.0

            f.write(f"MIP Utilization (mean):      {mean_mip_util:.4f}\n")
            f.write(f"RL Utilization (mean):       {mean_rl_util:.4f}\n")
            f.write(f"Optimality Gap:              {optimality_gap:.4f}%\n")
            f.write(f"Relative Performance:        {relative_performance:.4f}\n")
            f.write(f"MIP Solve Time (mean):       {np.mean(mip_solve_times):.4f}s\n")
            f.write(f"MIP Solve Time (std):        {np.std(mip_solve_times):.4f}s\n")
            f.write(f"MIP Optimal Solutions:       {mip_optimal_count}/{len(mip_metrics_list)}\n")

    logger.info(f"Metrics saved to: {metrics_file}")
    logger.info("=" * 80)
    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()
