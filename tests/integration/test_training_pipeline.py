"""Integration tests for training pipeline."""

import pytest
import torch
from pathlib import Path

from src.environment.packing_env import PackingEnv
from src.models.tap_net import TAPNet
from src.training.ppo_trainer import PPOTrainer


class TestTrainingPipeline:
    """Test complete training pipeline."""

    def test_short_training_run(self, model, env, tmp_path):
        """Test a short training run completes without errors."""
        trainer = PPOTrainer(
            model=model,
            env=env,
            lr=3e-4,
            device='cpu',
            buffer_size=64,
            batch_size=16,
            checkpoint_dir=str(tmp_path / "checkpoints"),
            log_dir=str(tmp_path / "logs"),
        )

        # Train for a small number of timesteps
        try:
            trainer.train(
                total_timesteps=128,  # Just 2 updates
                eval_frequency=1,
                save_frequency=1,
            )
        except Exception as e:
            pytest.fail(f"Training failed: {e}")

    def test_training_produces_checkpoints(self, model, env, tmp_path):
        """Test that training produces checkpoint files."""
        checkpoint_dir = tmp_path / "checkpoints"
        trainer = PPOTrainer(
            model=model,
            env=env,
            lr=3e-4,
            device='cpu',
            buffer_size=64,
            batch_size=16,
            checkpoint_dir=str(checkpoint_dir),
            log_dir=str(tmp_path / "logs"),
        )

        trainer.train(
            total_timesteps=128,
            eval_frequency=1,
            save_frequency=1,
        )

        # Check that checkpoints were created
        checkpoints = list(checkpoint_dir.glob("*.pt"))
        assert len(checkpoints) > 0

    def test_resume_training(self, model, env, test_config, tmp_path):
        """Test resuming training from checkpoint."""
        checkpoint_dir = tmp_path / "checkpoints"
        log_dir = tmp_path / "logs"

        # Initial training
        trainer1 = PPOTrainer(
            model=model,
            env=env,
            lr=3e-4,
            device='cpu',
            buffer_size=64,
            batch_size=16,
            checkpoint_dir=str(checkpoint_dir),
            log_dir=str(log_dir),
        )

        trainer1.train(total_timesteps=64, save_frequency=1)

        # Resume training with a NEW model using the SAME config as model1
        model2 = TAPNet(
            grid_size=test_config["environment"]["grid_size"],
            item_feature_dim=5,
            d_model=test_config["model"]["d_model"],
            nhead=test_config["model"]["nhead"],
            num_layers=test_config["model"]["num_layers"],
            dim_feedforward=test_config["model"]["dim_feedforward"],
            dropout=test_config["model"]["dropout"],
            share_encoder=test_config["model"]["share_encoder"],
        )

        trainer2 = PPOTrainer(
            model=model2,
            env=env,
            lr=3e-4,
            device='cpu',
            buffer_size=64,
            batch_size=16,
            checkpoint_dir=str(checkpoint_dir),
            log_dir=str(log_dir),
        )

        # Load checkpoint
        checkpoint_path = list(checkpoint_dir.glob("*.pt"))[0]
        try:
            trainer2.load_checkpoint(str(checkpoint_path))
        except Exception as e:
            pytest.fail(f"Failed to resume training: {e}")

    def test_evaluation_after_training(self, model, env, tmp_path):
        """Test evaluation after training."""
        trainer = PPOTrainer(
            model=model,
            env=env,
            lr=3e-4,
            device='cpu',
            buffer_size=64,
            batch_size=16,
            checkpoint_dir=str(tmp_path / "checkpoints"),
            log_dir=str(tmp_path / "logs"),
        )

        # Quick training
        trainer.train(total_timesteps=64)

        # Evaluate
        eval_metrics = trainer.evaluate(n_episodes=3, deterministic=True)

        assert 'mean_reward' in eval_metrics
        assert 'mean_utilization' in eval_metrics
        assert eval_metrics['mean_utilization'] >= 0.0
        assert eval_metrics['mean_utilization'] <= 1.0