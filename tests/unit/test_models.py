"""Unit tests for neural network models."""

import pytest
import torch
import numpy as np


class TestTAPNet:
    """Test TAP-Net model."""

    def test_model_initialization(self, model):
        """Test model initializes correctly."""
        assert model is not None
        assert hasattr(model, 'actor')
        assert hasattr(model, 'critic')

    def test_model_forward_pass(self, model, device):
        """Test forward pass through model."""
        batch_size = 2
        grid_size = 10

        # Create dummy inputs
        height_map = torch.randn(batch_size, grid_size, grid_size)
        item_features = torch.randn(batch_size, 5)
        action_mask = torch.ones(batch_size, grid_size, grid_size, 6)  # (batch, grid, grid, 6)

        # Forward pass should not raise errors
        try:
            action, log_prob, value = model.get_action(
                height_map, item_features, action_mask, deterministic=True
            )
            assert action is not None
            assert log_prob is not None
            assert value is not None
        except Exception as e:
            pytest.fail(f"Forward pass failed: {e}")

    def test_get_action_shape(self, model):
        """Test that get_action returns correct shapes."""
        batch_size = 4
        grid_size = 10

        height_map = torch.randn(batch_size, grid_size, grid_size)
        item_features = torch.randn(batch_size, 5)
        action_mask = torch.ones(batch_size, grid_size, grid_size, 6)  # (batch, grid, grid, 6)

        action, log_prob, value = model.get_action(
            height_map, item_features, action_mask, deterministic=False
        )

        assert action.shape == (batch_size,)
        assert log_prob.shape == (batch_size,)
        assert value.shape == (batch_size, 1)

    def test_evaluate_actions(self, model):
        """Test evaluate_actions method."""
        batch_size = 4
        grid_size = 10

        height_map = torch.randn(batch_size, grid_size, grid_size)
        item_features = torch.randn(batch_size, 5)
        actions = torch.randint(0, grid_size ** 2 * 6, (batch_size,))  # Actions in range [0, 600)
        action_mask = torch.ones(batch_size, grid_size, grid_size, 6)  # (batch, grid, grid, 6)

        log_probs, entropy, values = model.evaluate_actions(
            height_map, item_features, actions, action_mask
        )

        assert log_probs.shape == (batch_size,)
        assert entropy.shape == (batch_size,)
        assert values.shape == (batch_size, 1)

    def test_model_device_transfer(self, model):
        """Test model can be moved to different devices."""
        model_cpu = model.to('cpu')
        assert next(model_cpu.parameters()).device.type == 'cpu'

        if torch.cuda.is_available():
            model_cuda = model.to('cuda')
            assert next(model_cuda.parameters()).device.type == 'cuda'

    def test_model_save_load(self, model, tmp_path):
        """Test model save and load."""
        save_path = tmp_path / "model.pt"

        # Save model
        model.save(str(save_path))
        assert save_path.exists()

        # Load model
        model.load(str(save_path), device='cpu')

        # Model should still work
        height_map = torch.randn(1, 10, 10)
        item_features = torch.randn(1, 5)
        action_mask = torch.ones(1, 10, 10, 6)  # (batch, grid, grid, 6)

        action, _, _ = model.get_action(
            height_map, item_features, action_mask, deterministic=True
        )
        assert action is not None

    def test_model_gradient_flow(self, model):
        """Test that gradients flow through the model."""
        model.train()

        height_map = torch.randn(2, 10, 10, requires_grad=True)
        item_features = torch.randn(2, 5, requires_grad=True)
        actions = torch.randint(0, 600, (2,))  # Actions in range [0, 600)
        action_mask = torch.ones(2, 10, 10, 6)  # (batch, grid, grid, 6)

        log_probs, entropy, values = model.evaluate_actions(
            height_map, item_features, actions, action_mask
        )

        loss = log_probs.mean() + values.mean()
        loss.backward()

        # Check that gradients exist
        assert height_map.grad is not None
        assert item_features.grad is not None

    def test_deterministic_vs_stochastic(self, model):
        """Test deterministic vs stochastic action sampling."""
        model.eval()

        height_map = torch.randn(1, 10, 10)
        item_features = torch.randn(1, 5)
        action_mask = torch.ones(1, 10, 10, 6)  # (batch, grid, grid, 6)

        # Deterministic should give same result
        with torch.no_grad():
            action1, _, _ = model.get_action(
                height_map, item_features, action_mask, deterministic=True
            )
            action2, _, _ = model.get_action(
                height_map, item_features, action_mask, deterministic=True
            )
            assert action1.item() == action2.item()

    def test_model_info(self, model):
        """Test get_model_info method."""
        info = model.get_model_info()

        assert 'total_parameters' in info
        assert 'actor_parameters' in info
        assert 'critic_parameters' in info
        assert info['total_parameters'] > 0
        assert info['actor_parameters'] > 0
        assert info['critic_parameters'] > 0


class TestActorCriticComponents:
    """Test individual actor and critic components."""

    def test_actor_output_range(self, model):
        """Test that actor outputs valid probabilities."""
        model.eval()

        with torch.no_grad():
            height_map = torch.randn(1, 10, 10)
            item_features = torch.randn(1, 5)
            action_mask = torch.ones(1, 10, 10, 6)  # (batch, grid, grid, 6)

            action, log_prob, _ = model.get_action(
                height_map, item_features, action_mask, deterministic=False
            )

            # Action should be valid index [0, 600)
            assert 0 <= action.item() < 600

            # Log prob should be negative (log of probability)
            assert log_prob.item() <= 0

    def test_critic_value_estimation(self, model):
        """Test that critic produces reasonable value estimates."""
        model.eval()

        with torch.no_grad():
            height_map = torch.randn(1, 10, 10)
            item_features = torch.randn(1, 5)

            value = model.get_value(height_map, item_features)

            # Value should be a scalar
            assert value.shape == (1, 1)
            # Value should be finite
            assert torch.isfinite(value).all()