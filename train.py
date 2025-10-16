"""
Main Training Script for TAP-Net

Train the Transformer-based Actor-Critic Packing Network using PPO.

Usage:
    python train.py --config config/default.yaml
    python train.py --resume --checkpoint checkpoints/latest.pt
    python train.py --total-timesteps 2000000 --device cuda
"""

import argparse
import torch
from pathlib import Path

from src.environment.packing_env import PackingEnv
from src.models.tap_net import TAPNet
from src.training.ppo_trainer import PPOTrainer
from src.utils.config import load_config, save_config
from src.utils.logger import setup_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train TAP-Net for 3D bin packing")

    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to configuration file",
    )

    # Training
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Total training timesteps (overrides config)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Device to train on (overrides config)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)",
    )

    # Checkpointing
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from latest checkpoint",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Specific checkpoint to resume from",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Checkpoint directory (overrides config)",
    )

    # Logging
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="TensorBoard log directory (overrides config)",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Experiment name for logging",
    )

    return parser.parse_args()


def main():
    """Main training loop."""
    args = parse_args()

    # Setup logger
    logger = setup_logger("training", log_file="logs/training.log")
    logger.info("=" * 80)
    logger.info("TAP-Net Training")
    logger.info("=" * 80)

    # Load configuration
    logger.info(f"Loading configuration from: {args.config}")
    config = load_config(args.config)

    # Override config with command-line arguments
    if args.total_timesteps is not None:
        config["training"]["total_timesteps"] = args.total_timesteps
    if args.device is not None:
        config["training"]["device"] = args.device
    if args.seed is not None:
        config["experiment"]["seed"] = args.seed
    if args.checkpoint_dir is not None:
        config["checkpoint"]["checkpoint_dir"] = args.checkpoint_dir
    if args.log_dir is not None:
        config["logging"]["log_dir"] = args.log_dir
    if args.experiment_name is not None:
        config["experiment"]["name"] = args.experiment_name

    # Set random seeds for reproducibility
    seed = config["experiment"]["seed"]
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For all GPUs

    # Note: For full determinism, set torch.backends.cudnn.deterministic=True
    # and torch.backends.cudnn.benchmark=False, but this may reduce performance.

    logger.info(f"Random seed: {seed}")

    # Determine device
    device = config["training"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"
    elif device == "mps" and not torch.backends.mps.is_available():
        logger.warning("MPS not available, falling back to CPU")
        device = "cpu"

    logger.info(f"Using device: {device}")

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
        seed=seed,
    )
    logger.info(f"Environment created: {env}")

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

    model_info = model.get_model_info()
    logger.info(f"Model created with {model_info['total_parameters']:,} parameters")
    logger.info(f"  Actor: {model_info['actor_parameters']:,} parameters")
    logger.info(f"  Critic: {model_info['critic_parameters']:,} parameters")

    # Create trainer
    logger.info("Creating PPO trainer...")
    trainer = PPOTrainer(
        model=model,
        env=env,
        lr=config["training"]["learning_rate"],
        gamma=config["training"]["gamma"],
        gae_lambda=config["training"]["gae_lambda"],
        clip_epsilon=config["training"]["clip_epsilon"],
        value_coef=config["training"]["value_coef"],
        entropy_coef=config["training"]["entropy_coef"],
        max_grad_norm=config["training"]["max_grad_norm"],
        n_epochs=config["training"]["n_epochs"],
        batch_size=config["training"]["batch_size"],
        buffer_size=config["training"]["buffer_size"],
        device=device,
        checkpoint_dir=config["checkpoint"]["checkpoint_dir"],
        log_dir=config["logging"]["log_dir"],
    )

    # Resume from checkpoint if requested
    if args.resume or (args.checkpoint is not None):
        logger.info("Resuming from checkpoint...")
        checkpoint_info = trainer.load_checkpoint(args.checkpoint)
        logger.info(f"Resumed from epoch {checkpoint_info['epoch']}")
        logger.info(f"Previous metrics: {checkpoint_info['metrics']}")

    # Save configuration
    config_save_path = Path(config["checkpoint"]["checkpoint_dir"]) / "config.yaml"
    save_config(config, str(config_save_path))
    logger.info(f"Configuration saved to: {config_save_path}")

    # Start training
    logger.info("Starting training...")
    logger.info(f"Total timesteps: {config['training']['total_timesteps']:,}")

    try:
        trainer.train(
            total_timesteps=config["training"]["total_timesteps"],
            eval_frequency=config["checkpoint"]["eval_frequency"],
            save_frequency=config["checkpoint"]["save_frequency"],
        )
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")

    logger.info("Training completed!")

    # Final evaluation
    logger.info("Running final evaluation...")
    final_metrics = trainer.evaluate(n_episodes=20, deterministic=True)

    logger.info("Final Evaluation Results:")
    for key, value in final_metrics.items():
        logger.info(f"  {key}: {value:.4f}")

    logger.info("=" * 80)


if __name__ == "__main__":
    main()
