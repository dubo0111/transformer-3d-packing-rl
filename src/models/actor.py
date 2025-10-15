"""
Actor Network for TAP-Net

Implements the policy network (actor) that outputs action probabilities
for item placement. Uses Transformer encoder to process container state
and outputs distribution over valid placements.

Paper Reference: Section 3.3.2 - Actor Network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer to inject spatial information.

    Paper Reference: Section 3.3
    Positional encodings help the Transformer understand spatial relationships
    in the height map, which is crucial for packing decisions.
    """

    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            x + positional encoding
        """
        return x + self.pe[:, : x.size(1), :]


class HeightMapEncoder(nn.Module):
    """
    Encodes height map using CNN layers.

    The height map is a 2D grid representing container surface heights.
    We use CNN to extract spatial features before feeding to Transformer.

    Paper Reference: Section 3.3.1 - State Encoding
    """

    def __init__(self, grid_size: int, d_model: int):
        super().__init__()

        self.grid_size = grid_size
        self.d_model = d_model

        # CNN layers to extract spatial features
        # Input: (batch, 1, grid_size, grid_size)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Project to d_model dimensions
        self.projection = nn.Linear(128, d_model)

    def forward(self, height_map: torch.Tensor) -> torch.Tensor:
        """
        Encode height map to feature vectors.

        Args:
            height_map: Shape (batch, grid_size, grid_size)

        Returns:
            Encoded features: (batch, grid_size*grid_size, d_model)
        """
        batch_size = height_map.size(0)

        # Add channel dimension: (batch, 1, grid, grid)
        x = height_map.unsqueeze(1)

        # Apply CNN layers: (batch, 128, grid, grid)
        x = self.conv_layers(x)

        # Reshape to sequence: (batch, grid*grid, 128)
        x = x.view(batch_size, 128, -1).permute(0, 2, 1)

        # Project to d_model: (batch, grid*grid, d_model)
        x = self.projection(x)

        return x


class ItemEncoder(nn.Module):
    """
    Encodes current item features.

    Paper Reference: Section 3.3.1
    Item features (dimensions, volume, weight) are encoded and
    concatenated with container state.
    """

    def __init__(self, input_dim: int, d_model: int):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model),
        )

    def forward(self, item_features: torch.Tensor) -> torch.Tensor:
        """
        Encode item features.

        Args:
            item_features: Shape (batch, input_dim)

        Returns:
            Encoded features: (batch, d_model)
        """
        return self.encoder(item_features)


class Actor(nn.Module):
    """
    Actor network for policy learning.

    Architecture:
    1. Encode height map using CNN
    2. Encode current item features
    3. Combine and process with Transformer encoder
    4. Decode to action probabilities (position × rotation)

    Paper Reference: Section 3.3.2 - Actor Network
    "The actor network takes the state as input and outputs a probability
    distribution over the action space using a Transformer encoder."

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
        self.num_rotations = 6

        # Encoders
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

        # Action decoder
        # Outputs logits for each (position, rotation) combination
        action_dim = grid_size * grid_size * self.num_rotations

        self.action_decoder = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, action_dim),
        )

    def forward(
        self,
        height_map: torch.Tensor,
        item_features: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through actor network.

        Args:
            height_map: Container height map, shape (batch, grid_size, grid_size)
            item_features: Current item features, shape (batch, item_feature_dim)
            action_mask: Optional binary mask, shape (batch, grid, grid, 6)

        Returns:
            action_logits: Raw logits, shape (batch, grid*grid*6)
            action_probs: Masked probabilities, shape (batch, grid*grid*6)

        Paper Algorithm:
        1. Encode height map → spatial features
        2. Encode item → item embedding
        3. Concatenate as sequence [item_emb, spatial_features]
        4. Process with Transformer encoder
        5. Decode to action distribution
        6. Apply action mask and normalize
        """
        batch_size = height_map.size(0)

        # 1. Encode height map: (batch, grid*grid, d_model)
        height_features = self.height_map_encoder(height_map)

        # 2. Encode item: (batch, d_model)
        item_embedding = self.item_encoder(item_features)

        # 3. Concatenate: (batch, 1+grid*grid, d_model)
        # Item embedding as first token (like [CLS] token in BERT)
        item_embedding = item_embedding.unsqueeze(1)  # (batch, 1, d_model)
        sequence = torch.cat([item_embedding, height_features], dim=1)

        # 4. Add positional encoding
        sequence = self.pos_encoder(sequence)

        # 5. Process with Transformer: (batch, seq_len, d_model)
        encoded = self.transformer_encoder(sequence)

        # 6. Use [CLS] token (first token) for action prediction
        # Paper uses global features for decision making
        cls_token = encoded[:, 0, :]  # (batch, d_model)

        # 7. Decode to action logits: (batch, grid*grid*6)
        action_logits = self.action_decoder(cls_token)

        # 8. Apply action mask and compute probabilities
        if action_mask is not None:
            # Flatten mask: (batch, grid*grid*6)
            action_mask_flat = action_mask.view(batch_size, -1)

            # Mask invalid actions with large negative value
            action_logits_masked = action_logits.clone()
            action_logits_masked[action_mask_flat == 0] = -1e9

            # Compute probabilities
            action_probs = F.softmax(action_logits_masked, dim=-1)
        else:
            # No masking
            action_probs = F.softmax(action_logits, dim=-1)

        return action_logits, action_probs

    def get_action(
        self,
        height_map: torch.Tensor,
        item_features: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.

        Args:
            height_map: Height map
            item_features: Item features
            action_mask: Action mask
            deterministic: If True, select argmax; else sample

        Returns:
            actions: Selected actions, shape (batch,)
            log_probs: Log probabilities, shape (batch,)
        """
        _, action_probs = self.forward(height_map, item_features, action_mask)

        if deterministic:
            # Greedy action
            actions = torch.argmax(action_probs, dim=-1)
        else:
            # Sample from distribution
            dist = torch.distributions.Categorical(action_probs)
            actions = dist.sample()

        # Get log probabilities
        log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)).squeeze(1) + 1e-8)

        return actions, log_probs

    def evaluate_actions(
        self,
        height_map: torch.Tensor,
        item_features: torch.Tensor,
        actions: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate log probabilities and entropy for given actions.

        Used during PPO training to compute policy loss.

        Args:
            height_map: Height map
            item_features: Item features
            actions: Actions taken, shape (batch,)
            action_mask: Action mask

        Returns:
            log_probs: Log probabilities of actions, shape (batch,)
            entropy: Policy entropy, shape (batch,)
        """
        _, action_probs = self.forward(height_map, item_features, action_mask)

        # Log probabilities of selected actions
        log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)).squeeze(1) + 1e-8)

        # Entropy of the distribution
        entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=-1)

        return log_probs, entropy
