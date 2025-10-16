"""Pytest configuration and shared fixtures."""

import pytest
import torch
import numpy as np
from pathlib import Path

from src.environment.packing_env import PackingEnv
from src.models.tap_net import TAPNet
from src.utils.config import load_config


@pytest.fixture(scope="session")
def test_config():
    """Load test configuration."""
    config_path = Path("config/default.yaml")
    if not config_path.exists():
        # Return minimal config for testing (matches production default.yaml)
        return {
            "environment": {
                "container_size": [10.0, 10.0, 10.0],
                "grid_size": 10,
                "max_items": 10,
                "item_size_range": [0.1, 0.5],
                "enable_action_mask": True,
                "reward_type": "utilization",
                "normalize_state": True,
            },
            "model": {
                "d_model": 256,  # Match production config
                "nhead": 8,      # Match production config
                "num_layers": 4, # Match production config
                "dim_feedforward": 1024,  # Match production config
                "dropout": 0.1,
                "share_encoder": False,
            },
        }
    return load_config(str(config_path))


@pytest.fixture
def env(test_config):
    """Create test environment."""
    return PackingEnv(
        container_size=tuple(test_config["environment"]["container_size"]),
        grid_size=test_config["environment"]["grid_size"],
        max_items=test_config["environment"]["max_items"],
        item_size_range=tuple(test_config["environment"]["item_size_range"]),
        enable_action_mask=test_config["environment"]["enable_action_mask"],
        reward_type=test_config["environment"]["reward_type"],
        normalize_state=test_config["environment"]["normalize_state"],
        seed=42,
    )


@pytest.fixture
def model(test_config):
    """Create test model."""
    return TAPNet(
        grid_size=test_config["environment"]["grid_size"],
        item_feature_dim=5,
        d_model=test_config["model"]["d_model"],
        nhead=test_config["model"]["nhead"],
        num_layers=test_config["model"]["num_layers"],
        dim_feedforward=test_config["model"]["dim_feedforward"],
        dropout=test_config["model"]["dropout"],
        share_encoder=test_config["model"]["share_encoder"],
    )


@pytest.fixture
def device():
    """Get test device (CPU for consistency)."""
    return "cpu"


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seeds for reproducible tests."""
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
