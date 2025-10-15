"""
Rollout Buffer for PPO

Stores trajectories (state, action, reward, etc.) collected during
environment interaction for batch training.

Paper Reference: Section 4 - Training Algorithm
"""

import torch
import numpy as np
from typing import List, Tuple, Generator


class RolloutBuffer:
    """
    Stores rollout data for PPO training.

    In PPO, we collect a batch of trajectories and then update the policy
    using multiple epochs over this data. The buffer stores:
    - States (height maps, item features)
    - Actions taken
    - Log probabilities of actions
    - Rewards received
    - Episode done flags
    - Values estimated by critic
    - Action masks

    Paper Reference: Section 4.1
    "We collect N steps of experience using the current policy and
    store them in a replay buffer for training."
    """

    def __init__(
        self,
        buffer_size: int,
        grid_size: int,
        item_feature_dim: int,
        device: str = "cpu",
    ):
        """
        Initialize rollout buffer.

        Args:
            buffer_size: Maximum number of transitions to store
            grid_size: Height map grid size
            item_feature_dim: Item feature dimension
            device: Device to store tensors on
        """
        self.buffer_size = buffer_size
        self.grid_size = grid_size
        self.item_feature_dim = item_feature_dim
        self.device = device

        # Initialize storage
        self.height_maps = torch.zeros(
            (buffer_size, grid_size, grid_size), dtype=torch.float32, device=device
        )
        self.item_features = torch.zeros(
            (buffer_size, item_feature_dim), dtype=torch.float32, device=device
        )
        self.actions = torch.zeros(buffer_size, dtype=torch.long, device=device)
        self.rewards = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.dones = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.values = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.log_probs = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.action_masks = torch.zeros(
            (buffer_size, grid_size, grid_size, 6), dtype=torch.float32, device=device
        )

        # Buffer state
        self.pos = 0
        self.full = False

    def add(
        self,
        height_map: np.ndarray,
        item_feature: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        value: float,
        log_prob: float,
        action_mask: np.ndarray,
    ):
        """
        Add a transition to the buffer.

        Args:
            height_map: Height map state
            item_feature: Item features
            action: Action taken
            reward: Reward received
            done: Episode termination flag
            value: Value estimate
            log_prob: Log probability of action
            action_mask: Action mask
        """
        self.height_maps[self.pos] = torch.from_numpy(height_map).to(self.device)
        self.item_features[self.pos] = torch.from_numpy(item_feature).to(self.device)
        self.actions[self.pos] = int(action)
        self.rewards[self.pos] = float(reward)
        self.dones[self.pos] = float(done)
        self.values[self.pos] = float(value)
        self.log_probs[self.pos] = float(log_prob)
        self.action_masks[self.pos] = torch.from_numpy(action_mask).to(self.device)

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def get_size(self) -> int:
        """Get current buffer size."""
        return self.buffer_size if self.full else self.pos

    def compute_returns_and_advantages(
        self, last_value: float, gamma: float = 0.99, gae_lambda: float = 0.95
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute returns and advantages using Generalized Advantage Estimation (GAE).

        Paper Reference: Section 4.2 - Advantage Estimation
        GAE provides a bias-variance tradeoff for advantage estimation:

        δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
        A_t = Σ (γλ)^l * δ_{t+l}

        Where:
        - γ (gamma): discount factor
        - λ (lambda): GAE parameter (controls bias-variance)
        - δ_t: temporal difference error
        - A_t: advantage estimate

        Args:
            last_value: Value of the last state (for bootstrap)
            gamma: Discount factor
            gae_lambda: GAE lambda parameter

        Returns:
            returns: Discounted returns, shape (buffer_size,)
            advantages: Advantage estimates, shape (buffer_size,)
        """
        size = self.get_size()

        advantages = torch.zeros(size, dtype=torch.float32, device=self.device)
        returns = torch.zeros(size, dtype=torch.float32, device=self.device)

        # Start from the end and work backwards
        last_gae_lambda = 0
        last_value_tensor = torch.tensor(last_value, dtype=torch.float32, device=self.device)

        for t in reversed(range(size)):
            if t == size - 1:
                next_non_terminal = 1.0 - self.dones[t]
                next_value = last_value_tensor
            else:
                next_non_terminal = 1.0 - self.dones[t]
                next_value = self.values[t + 1]

            # Temporal difference error
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]

            # GAE calculation
            last_gae_lambda = delta + gamma * gae_lambda * next_non_terminal * last_gae_lambda
            advantages[t] = last_gae_lambda

        # Returns = advantages + values
        returns = advantages + self.values[:size]

        return returns, advantages

    def get(
        self, batch_size: int
    ) -> Generator[Tuple[torch.Tensor, ...], None, None]:
        """
        Generate batches of data.

        Args:
            batch_size: Batch size for training

        Yields:
            Batches of (height_maps, item_features, actions, old_log_probs,
                       advantages, returns, action_masks)
        """
        size = self.get_size()
        indices = np.random.permutation(size)

        for start_idx in range(0, size, batch_size):
            end_idx = min(start_idx + batch_size, size)
            batch_indices = indices[start_idx:end_idx]

            # Note: advantages and returns need to be computed first
            # using compute_returns_and_advantages()

            yield (
                self.height_maps[batch_indices],
                self.item_features[batch_indices],
                self.actions[batch_indices],
                self.log_probs[batch_indices],
                self.action_masks[batch_indices],
            )

    def get_all(self) -> Tuple[torch.Tensor, ...]:
        """
        Get all data in buffer.

        Returns:
            All stored data as tensors
        """
        size = self.get_size()

        return (
            self.height_maps[:size],
            self.item_features[:size],
            self.actions[:size],
            self.rewards[:size],
            self.dones[:size],
            self.values[:size],
            self.log_probs[:size],
            self.action_masks[:size],
        )

    def reset(self):
        """Reset buffer to empty state."""
        self.pos = 0
        self.full = False

    def __len__(self) -> int:
        return self.get_size()
