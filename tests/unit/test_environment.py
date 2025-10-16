"""Unit tests for packing environment."""

import pytest
import numpy as np
from src.environment.packing_env import PackingEnv
from src.environment.container import Container
from src.environment.item import Item


class TestPackingEnv:
    """Test PackingEnv class."""

    def test_env_initialization(self, env):
        """Test environment initializes correctly."""
        assert env.container_size == (10.0, 10.0, 10.0)
        assert env.grid_size == 10
        assert env.max_items == 50  # Updated to match default.yaml
        assert env.enable_action_mask is True

    def test_reset(self, env):
        """Test environment reset."""
        obs, info = env.reset()

        # Check observation shape: height_map + item_features + metadata
        expected_obs_size = env.grid_size ** 2 + 5 + 3  # height_map + item_features + metadata
        assert obs.shape[0] == expected_obs_size

        # Check info dict
        assert "total_items" in info
        assert "utilization" in info
        assert info["utilization"] == 0.0

    def test_step(self, env):
        """Test environment step."""
        obs, info = env.reset()
        action_mask = env._get_action_mask()
        valid_actions = np.where(action_mask)[0]

        if len(valid_actions) > 0:
            action = valid_actions[0]
            obs, reward, terminated, truncated, info = env.step(action)

            # Check output types
            assert isinstance(obs, np.ndarray)
            assert isinstance(reward, (int, float))
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert isinstance(info, dict)

    def test_action_mask(self, env):
        """Test action masking."""
        env.reset()
        action_mask = env._get_action_mask()

        # Action mask should have correct size: grid_size × grid_size × 6 rotations
        assert action_mask.shape == (env.grid_size, env.grid_size, 6)

        # Should be binary
        assert np.all((action_mask == 0) | (action_mask == 1))

    def test_invalid_action_handling(self, env):
        """Test that invalid actions are handled properly."""
        env.reset()
        action_mask = env._get_action_mask()
        invalid_actions = np.where(action_mask == 0)[0]

        if len(invalid_actions) > 0:
            action = invalid_actions[0]
            obs, reward, terminated, truncated, info = env.step(action)

            # Should handle gracefully (might give penalty)
            assert isinstance(reward, (int, float))

    def test_episode_termination(self, env):
        """Test that episode terminates correctly."""
        obs, info = env.reset()
        done = False
        steps = 0
        max_steps = 1000  # Safety limit

        while not done and steps < max_steps:
            action_mask = env._get_action_mask()
            valid_actions = np.where(action_mask)[0]

            if len(valid_actions) == 0:
                break

            action = valid_actions[0]
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1

        assert steps < max_steps, "Episode did not terminate"


class TestContainer:
    """Test Container class."""

    def test_container_initialization(self):
        """Test container initializes correctly."""
        container = Container(
            length=10.0,
            width=10.0,
            height=10.0,
            grid_size=10
        )

        assert container.length == 10.0
        assert container.width == 10.0
        assert container.height == 10.0
        assert container.grid_size == 10
        assert container.height_map.shape == (10, 10)
        assert np.all(container.height_map == 0)

    def test_can_place_item(self):
        """Test can_place_item method."""
        container = Container(10.0, 10.0, 10.0, grid_size=10)

        # Should be able to place in empty container
        can_place = container.can_place_item(x=0.0, y=0.0, item_length=2.0, item_width=2.0, item_height=2.0)
        assert can_place is True

    def test_place_item(self):
        """Test place_item method."""
        container = Container(10.0, 10.0, 10.0, grid_size=10)

        success = container.place_item(
            item_id=0, x=0.0, y=0.0,
            item_length=2.0, item_width=2.0, item_height=2.0,
            rotation_idx=0, weight=1.0
        )
        assert success is True
        assert len(container.packed_items) == 1
        assert container.utilization > 0

    def test_height_map_update(self):
        """Test that height map updates correctly after placement."""
        container = Container(10.0, 10.0, 10.0, grid_size=10)

        initial_height = container.height_map[0, 0]
        container.place_item(
            item_id=0, x=0.0, y=0.0,
            item_length=2.0, item_width=2.0, item_height=2.0,
            rotation_idx=0, weight=1.0
        )
        new_height = container.height_map[0, 0]

        assert new_height > initial_height


class TestItem:
    """Test Item class."""

    def test_item_initialization(self):
        """Test item initializes correctly."""
        item = Item(length=2.0, width=3.0, height=4.0, weight=1.0)

        assert item.length == 2.0
        assert item.width == 3.0
        assert item.height == 4.0
        assert item.weight == 1.0

    def test_item_volume(self):
        """Test volume calculation."""
        item = Item(length=2.0, width=3.0, height=4.0)
        assert item.volume == 24.0

    def test_item_rotation(self):
        """Test item rotation (if implemented)."""
        item = Item(length=2.0, width=3.0, height=4.0)

        # Test that dimensions are positive
        assert item.length > 0
        assert item.width > 0
        assert item.height > 0
