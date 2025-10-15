"""
TAP-Net: Transformer-based Actor-critic Packing Network

Main model that combines Actor and Critic networks for the 3D bin packing
problem using Proximal Policy Optimization (PPO).

Paper Reference: Section 3.3 - Network Architecture
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict

from .actor import Actor
from .critic import Critic


class TAPNet(nn.Module):
    """
    Transformer-based Actor-Critic Packing Network.

    This is the main model architecture from the paper that combines:
    1. Actor network: Policy that outputs action probabilities
    2. Critic network: Value function estimator

    Both networks share similar architectures but have different heads.
    In this implementation, they have separate parameters, but parameter
    sharing is possible for efficiency.

    Paper Reference: Section 3.3
    "We propose TAP-Net, a Transformer-based actor-critic network that
    learns to pack items efficiently through deep reinforcement learning."

    Args:
        grid_size: Height map resolution (default: 10)
        item_feature_dim: Item feature dimension (default: 5)
        d_model: Transformer hidden size (default: 256)
        nhead: Number of attention heads (default: 8)
        num_layers: Number of Transformer layers (default: 4)
        dim_feedforward: FFN dimension (default: 1024)
        dropout: Dropout rate (default: 0.1)
        share_encoder: Whether to share encoders between actor/critic
    """

    def __init__(
        self,
        grid_size: int = 10,
        item_feature_dim: int = 5,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        share_encoder: bool = False,
    ):
        super().__init__()

        self.grid_size = grid_size
        self.item_feature_dim = item_feature_dim
        self.d_model = d_model

        # Actor network (policy)
        self.actor = Actor(
            grid_size=grid_size,
            item_feature_dim=item_feature_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        # Critic network (value function)
        self.critic = Critic(
            grid_size=grid_size,
            item_feature_dim=item_feature_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        # Optional: Share encoder parameters
        if share_encoder:
            self.critic.height_map_encoder = self.actor.height_map_encoder
            self.critic.item_encoder = self.actor.item_encoder
            self.critic.pos_encoder = self.actor.pos_encoder

    def forward(
        self,
        height_map: torch.Tensor,
        item_features: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through both actor and critic.

        Args:
            height_map: Height map, shape (batch, grid_size, grid_size)
            item_features: Item features, shape (batch, item_feature_dim)
            action_mask: Action mask, shape (batch, grid, grid, 6)

        Returns:
            action_logits: Raw action logits, shape (batch, grid*grid*6)
            action_probs: Masked action probabilities, shape (batch, grid*grid*6)
            state_value: Estimated state value, shape (batch, 1)
        """
        # Get action distribution from actor
        action_logits, action_probs = self.actor(height_map, item_features, action_mask)

        # Get state value from critic
        state_value = self.critic(height_map, item_features)

        return action_logits, action_probs, state_value

    def get_action(
        self,
        height_map: torch.Tensor,
        item_features: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action and get value.

        Args:
            height_map: Height map
            item_features: Item features
            action_mask: Action mask
            deterministic: If True, use greedy action selection

        Returns:
            actions: Selected actions, shape (batch,)
            log_probs: Log probabilities, shape (batch,)
            values: State values, shape (batch, 1)
        """
        # Sample action from actor
        actions, log_probs = self.actor.get_action(
            height_map, item_features, action_mask, deterministic
        )

        # Get value from critic
        values = self.critic(height_map, item_features)

        return actions, log_probs, values

    def evaluate_actions(
        self,
        height_map: torch.Tensor,
        item_features: torch.Tensor,
        actions: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO training.

        Args:
            height_map: Height map
            item_features: Item features
            actions: Actions to evaluate
            action_mask: Action mask

        Returns:
            log_probs: Log probabilities of actions, shape (batch,)
            entropy: Policy entropy, shape (batch,)
            values: State values, shape (batch, 1)
        """
        # Evaluate actions with actor
        log_probs, entropy = self.actor.evaluate_actions(
            height_map, item_features, actions, action_mask
        )

        # Get values from critic
        values = self.critic(height_map, item_features)

        return log_probs, entropy, values

    def get_value(
        self,
        height_map: torch.Tensor,
        item_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get state value only (used for advantage computation).

        Args:
            height_map: Height map
            item_features: Item features

        Returns:
            State value, shape (batch, 1)
        """
        return self.critic(height_map, item_features)

    def save(self, path: str):
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "config": {
                "grid_size": self.grid_size,
                "item_feature_dim": self.item_feature_dim,
                "d_model": self.d_model,
            },
        }
        torch.save(checkpoint, path)

    def load(self, path: str, device: str = "cpu"):
        """
        Load model checkpoint.

        Args:
            path: Path to checkpoint
            device: Device to load model on
        """
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        # Handle two checkpoint formats:
        # 1. Direct save from TAPNet.save() - has actor_state_dict and critic_state_dict
        # 2. CheckpointManager save - has model_state_dict
        if "actor_state_dict" in checkpoint and "critic_state_dict" in checkpoint:
            self.actor.load_state_dict(checkpoint["actor_state_dict"])
            self.critic.load_state_dict(checkpoint["critic_state_dict"])
        elif "model_state_dict" in checkpoint:
            self.load_state_dict(checkpoint["model_state_dict"])
        else:
            raise KeyError("Checkpoint must contain either 'actor_state_dict' and 'critic_state_dict', or 'model_state_dict'")

    def get_model_info(self) -> Dict[str, int]:
        """
        Get model information.

        Returns:
            Dictionary with model statistics
        """
        actor_params = sum(p.numel() for p in self.actor.parameters())
        critic_params = sum(p.numel() for p in self.critic.parameters())
        total_params = sum(p.numel() for p in self.parameters())

        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "actor_parameters": actor_params,
            "critic_parameters": critic_params,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
        }
