"""
Critic Network for TAP-Net

Implements the value network (critic) that estimates the state value function.
This is used in the actor-critic framework for advantage estimation.

Paper Reference: Section 3.3.3 - Critic Network
"""

import torch
import torch.nn as nn
from typing import Tuple


class Critic(nn.Module):
    """
    Critic network for value function estimation.

    The critic estimates the expected return (value) from a given state.
    It shares the same state encoder as the actor but outputs a scalar value.

    Paper Reference: Section 3.3.3
    "The critic network estimates the state value function V(s) which is used
    to compute the advantage function in the PPO algorithm."

    Architecture:
    1. Encode height map using CNN (shared with Actor)
    2. Encode item features (shared with Actor)
    3. Combine with Transformer encoder
    4. Output scalar state value

    Args:
        grid_size: Height map grid resolution
        item_feature_dim: Dimension of item features
        d_model: Transformer hidden dimension
        nhead: Number of attention heads
        num_layers: Number of Transformer encoder layers
        dim_feedforward: FFN dimension
        dropout: Dropout rate
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
    ):
        super().__init__()

        self.grid_size = grid_size
        self.d_model = d_model

        # Import encoder modules from actor
        # In practice, we can share encoders between actor and critic
        from .actor import HeightMapEncoder, ItemEncoder, PositionalEncoding

        # Encoders (can be shared with Actor in full implementation)
        self.height_map_encoder = HeightMapEncoder(grid_size, d_model)
        self.item_encoder = ItemEncoder(item_feature_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=grid_size * grid_size + 1)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        # Value head - outputs scalar value
        self.value_head = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),  # Single scalar output
        )

    def forward(
        self,
        height_map: torch.Tensor,
        item_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through critic network.

        Args:
            height_map: Container height map, shape (batch, grid_size, grid_size)
            item_features: Current item features, shape (batch, item_feature_dim)

        Returns:
            state_value: Estimated value, shape (batch, 1)

        Paper Algorithm:
        1. Encode height map → spatial features
        2. Encode item → item embedding
        3. Concatenate as sequence [item_emb, spatial_features]
        4. Process with Transformer encoder
        5. Decode to scalar value
        """
        batch_size = height_map.size(0)

        # 1. Encode height map: (batch, grid*grid, d_model)
        height_features = self.height_map_encoder(height_map)

        # 2. Encode item: (batch, d_model)
        item_embedding = self.item_encoder(item_features)

        # 3. Concatenate: (batch, 1+grid*grid, d_model)
        item_embedding = item_embedding.unsqueeze(1)  # (batch, 1, d_model)
        sequence = torch.cat([item_embedding, height_features], dim=1)

        # 4. Add positional encoding
        sequence = self.pos_encoder(sequence)

        # 5. Process with Transformer: (batch, seq_len, d_model)
        encoded = self.transformer_encoder(sequence)

        # 6. Use [CLS] token (first token) for value prediction
        cls_token = encoded[:, 0, :]  # (batch, d_model)

        # 7. Compute state value: (batch, 1)
        state_value = self.value_head(cls_token)

        return state_value

    def get_value(
        self,
        height_map: torch.Tensor,
        item_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get state value (convenience method).

        Args:
            height_map: Height map
            item_features: Item features

        Returns:
            state_value: Shape (batch, 1)
        """
        return self.forward(height_map, item_features)
