"""Unit tests for training components."""

import pytest
import torch
import numpy as np
from src.training.replay_buffer import RolloutBuffer
from src.training.ppo_trainer import PPOTrainer


class TestRolloutBuffer:
    """Test RolloutBuffer class."""

    def test_buffer_initialization(self):
        """Test buffer initializes correctly."""
        buffer = RolloutBuffer(
            buffer_size=128,
            grid_size=10,
            item_feature_dim=5,
            device='cpu'
        )

        assert buffer.buffer_size == 128
        assert buffer.pos == 0

    def test_buffer_add(self):
        """Test adding data to buffer."""
        buffer = RolloutBuffer(
            buffer_size=128,
            grid_size=10,
            item_feature_dim=5,
            device='cpu'
        )

        height_map = np.random.randn(10, 10)
        item_feature = np.random.randn(5)

        buffer.add(
            height_map=height_map,
            item_feature=item_feature,
            action=0,
            reward=1.0,
            done=False,
            value=0.5,
            log_prob=-1.0,
            action_mask=np.ones((10, 10, 6))  # Correct shape
        )

        assert buffer.pos == 1

    def test_buffer_reset(self):
        """Test buffer reset."""
        buffer = RolloutBuffer(
            buffer_size=128,
            grid_size=10,
            item_feature_dim=5,
            device='cpu'
        )

        # Add some data
        for _ in range(10):
            buffer.add(
                height_map=np.random.randn(10, 10),
                item_feature=np.random.randn(5),
                action=0,
                reward=1.0,
                done=False,
                value=0.5,
                log_prob=-1.0,
                action_mask=np.ones((10, 10, 6))  # Correct shape
            )

        buffer.reset()
        assert buffer.pos == 0

    def test_buffer_full(self):
        """Test buffer full detection."""
        buffer_size = 10
        buffer = RolloutBuffer(
            buffer_size=buffer_size,
            grid_size=10,
            item_feature_dim=5,
            device='cpu'
        )

        for i in range(buffer_size):
            buffer.add(
                height_map=np.random.randn(10, 10),
                item_feature=np.random.randn(5),
                action=0,
                reward=1.0,
                done=False,
                value=0.5,
                log_prob=-1.0,
                action_mask=np.ones((10, 10, 6))  # Correct shape
            )

        # Buffer wraps around when full, so pos resets to 0
        assert buffer.full is True
        assert buffer.pos == 0

    def test_compute_returns_and_advantages(self):
        """Test GAE computation."""
        buffer = RolloutBuffer(
            buffer_size=10,
            grid_size=10,
            item_feature_dim=5,
            device='cpu'
        )

        # Add trajectory
        for i in range(10):
            buffer.add(
                height_map=np.random.randn(10, 10),
                item_feature=np.random.randn(5),
                action=0,
                reward=1.0,
                done=(i == 9),
                value=0.5,
                log_prob=-1.0,
                action_mask=np.ones((10, 10, 6))  # Correct shape
            )

        returns, advantages = buffer.compute_returns_and_advantages(
            last_value=0.0,
            gamma=0.99,
            gae_lambda=0.95
        )

        assert returns.shape == (10,)
        assert advantages.shape == (10,)
        assert torch.isfinite(returns).all()
        assert torch.isfinite(advantages).all()


class TestPPOTrainer:
    """Test PPOTrainer class."""

    def test_trainer_initialization(self, model, env):
        """Test trainer initializes correctly."""
        trainer = PPOTrainer(
            model=model,
            env=env,
            lr=3e-4,
            device='cpu',
            buffer_size=128,
            batch_size=32,
        )

        assert trainer.model is not None
        assert trainer.env is not None
        assert trainer.device == 'cpu'

    def test_collect_rollouts(self, model, env):
        """Test rollout collection."""
        trainer = PPOTrainer(
            model=model,
            env=env,
            lr=3e-4,
            device='cpu',
            buffer_size=128,
            batch_size=32,
        )

        stats = trainer.collect_rollouts(n_steps=50)

        assert 'mean_episode_reward' in stats
        assert 'mean_utilization' in stats
        assert isinstance(stats['mean_episode_reward'], (int, float))

    def test_parse_observation(self, model, env):
        """Test observation parsing."""
        trainer = PPOTrainer(
            model=model,
            env=env,
            lr=3e-4,
            device='cpu',
        )

        obs, _ = env.reset()
        height_map, item_features = trainer._parse_observation(obs)

        assert height_map.shape == (env.grid_size, env.grid_size)
        assert item_features.shape == (5,)

    def test_checkpoint_save_load(self, model, env, tmp_path):
        """Test checkpoint saving and loading."""
        trainer = PPOTrainer(
            model=model,
            env=env,
            lr=3e-4,
            device='cpu',
            checkpoint_dir=str(tmp_path),
        )

        # Create a checkpoint
        metrics = {"utilization": 0.5, "reward": 1.0}
        trainer.checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=trainer.optimizer,
            epoch=1,
            metrics=metrics,
            is_best=True,
        )

        # Check that checkpoint was saved
        checkpoints = list(tmp_path.glob("*.pt"))
        assert len(checkpoints) > 0

    def test_evaluation(self, model, env):
        """Test model evaluation."""
        trainer = PPOTrainer(
            model=model,
            env=env,
            lr=3e-4,
            device='cpu',
        )

        eval_metrics = trainer.evaluate(n_episodes=2, deterministic=True)

        assert 'mean_reward' in eval_metrics
        assert 'mean_utilization' in eval_metrics
        assert 'mean_length' in eval_metrics
        assert isinstance(eval_metrics['mean_reward'], (int, float))