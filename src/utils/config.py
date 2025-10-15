"""
Configuration Management

Load, save, and validate configuration files for training and evaluation.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import copy


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Configuration dictionary

    Example:
        >>> config = load_config("config/default.yaml")
        >>> print(config["model"]["d_model"])
        256
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Validate configuration
    _validate_config(config)

    return config


def save_config(config: Dict[str, Any], save_path: str):
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        save_path: Path to save YAML file
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Configuration saved to: {save_path}")


def merge_configs(base_config: Dict[str, Any],
                  override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configurations (override takes precedence).

    Args:
        base_config: Base configuration
        override_config: Override configuration

    Returns:
        Merged configuration
    """
    merged = copy.deepcopy(base_config)

    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value

    return merged


def _validate_config(config: Dict[str, Any]):
    """
    Validate configuration structure and values.

    Args:
        config: Configuration dictionary

    Raises:
        ValueError: If configuration is invalid
    """
    required_sections = ["environment", "model", "training"]

    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section: {section}")

    # Validate environment
    env_config = config["environment"]
    if len(env_config["container_size"]) != 3:
        raise ValueError("container_size must have 3 dimensions")

    if env_config["grid_size"] < 1:
        raise ValueError("grid_size must be at least 1")

    # Validate model
    model_config = config["model"]
    if model_config["d_model"] % model_config["nhead"] != 0:
        raise ValueError("d_model must be divisible by nhead")

    # Validate training
    train_config = config["training"]
    if train_config["gamma"] < 0 or train_config["gamma"] > 1:
        raise ValueError("gamma must be in [0, 1]")

    if train_config["gae_lambda"] < 0 or train_config["gae_lambda"] > 1:
        raise ValueError("gae_lambda must be in [0, 1]")

    print("Configuration validated successfully")


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration.

    Returns:
        Default configuration dictionary
    """
    return load_config("config/default.yaml")


def update_config_from_args(config: Dict[str, Any],
                            args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update configuration from command-line arguments.

    Args:
        config: Base configuration
        args: Command-line arguments

    Returns:
        Updated configuration
    """
    updated_config = copy.deepcopy(config)

    # Map common CLI args to config keys
    arg_mapping = {
        "lr": ("training", "learning_rate"),
        "batch_size": ("training", "batch_size"),
        "total_timesteps": ("training", "total_timesteps"),
        "device": ("training", "device"),
        "seed": ("experiment", "seed"),
    }

    for arg_key, (section, config_key) in arg_mapping.items():
        if arg_key in args and args[arg_key] is not None:
            if section not in updated_config:
                updated_config[section] = {}
            updated_config[section][config_key] = args[arg_key]

    return updated_config
