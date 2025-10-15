"""
PPO Trainer for TAP-Net

Implements Proximal Policy Optimization algorithm for training the
Transformer-based packing network.

Paper Reference: Section 4 - Training Algorithm
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Optional, Callable, Any
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from ..models.tap_net import TAPNet
from ..environment.packing_env import PackingEnv
from .replay_buffer import RolloutBuffer
from .checkpoint_manager import CheckpointManager


class PPOTrainer:
    """
    Proximal Policy Optimization Trainer.

    PPO is a policy gradient method that uses a clipped surrogate objective
    to prevent excessively large policy updates, leading to more stable training.

    Paper Reference: Section 4 - PPO Algorithm
    "We employ PPO to train our TAP-Net, which balances exploration and
    exploitation while maintaining training stability."

    Key Components:
    1. Clipped surrogate objective for policy loss
    2. Value function loss (MSE)
    3. Entropy bonus for exploration
    4. Generalized Advantage Estimation (GAE)
    5. Multiple epochs per batch of data

    PPO Objective:
    L^CLIP(θ) = E_t[min(r_t(θ)Â_t, clip(r_t(θ), 1-ε, 1+ε)Â_t)]

    Where:
    - r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t) (probability ratio)
    - Â_t = advantage estimate
    - ε = clip parameter (typically 0.2)

    Args:
        model: TAP-Net model
        env: Packing environment
        lr: Learning rate
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        clip_epsilon: PPO clip parameter
        value_coef: Value loss coefficient
        entropy_coef: Entropy bonus coefficient
        max_grad_norm: Gradient clipping threshold
        n_epochs: Number of epochs per update
        batch_size: Minibatch size
        buffer_size: Rollout buffer size
        device: Device to train on
        checkpoint_dir: Directory for checkpoints
        log_dir: Directory for TensorBoard logs
    """

    def __init__(
        self,
        model: TAPNet,
        env: PackingEnv,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_epochs: int = 10,
        batch_size: int = 256,
        buffer_size: int = 2048,
        device: str = "cpu",
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs",
    ):
        self.model = model.to(device)
        self.env = env
        self.device = device

        # PPO hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=lr)

        # Learning rate scheduler (optional)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=100, gamma=0.95
        )

        # Rollout buffer
        self.buffer = RolloutBuffer(
            buffer_size=buffer_size,
            grid_size=env.grid_size,
            item_feature_dim=5,  # [l, w, h, vol, weight]
            device=device,
        )

        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            keep_best_n=5,
            metric_name="utilization",
            higher_is_better=True,
        )

        # Logging
        self.writer = SummaryWriter(log_dir=log_dir)
        self.global_step = 0
        self.episode_count = 0

        # Training statistics
        self.best_utilization = 0.0

    def collect_rollouts(self, n_steps: int) -> Dict[str, float]:
        """
        Collect rollouts (trajectories) from environment.

        Paper Reference: Section 4.1 - Data Collection
        "We collect N steps of experience using the current policy π_θ_old"

        Args:
            n_steps: Number of steps to collect

        Returns:
            Dictionary of collection statistics
        """
        self.model.eval()
        self.buffer.reset()

        episode_rewards = []
        episode_utilizations = []
        episode_lengths = []

        current_episode_reward = 0.0
        current_episode_length = 0

        obs, info = self.env.reset()
        episode_rewards_list = []

        with torch.no_grad():
            for step in range(n_steps):
                # Parse observation
                height_map, item_features = self._parse_observation(obs)

                # Get action mask
                action_mask = self.env._get_action_mask()

                # Convert to tensors
                height_map_tensor = torch.from_numpy(height_map).unsqueeze(0).to(self.device)
                item_features_tensor = torch.from_numpy(item_features).unsqueeze(0).to(self.device)
                action_mask_tensor = torch.from_numpy(action_mask).unsqueeze(0).to(self.device)

                # Get action and value
                action, log_prob, value = self.model.get_action(
                    height_map_tensor,
                    item_features_tensor,
                    action_mask_tensor,
                    deterministic=False,
                )

                action = action.item()
                log_prob = log_prob.item()
                value = value.item()

                # Step environment
                next_obs, reward, terminated, truncated, next_info = self.env.step(action)
                done = terminated or truncated

                # Store transition
                self.buffer.add(
                    height_map=height_map,
                    item_feature=item_features,
                    action=action,
                    reward=reward,
                    done=done,
                    value=value,
                    log_prob=log_prob,
                    action_mask=action_mask,
                )

                current_episode_reward += reward
                current_episode_length += 1

                # Handle episode termination
                if done:
                    episode_rewards.append(current_episode_reward)
                    episode_utilizations.append(info.get("utilization", 0.0))
                    episode_lengths.append(current_episode_length)

                    current_episode_reward = 0.0
                    current_episode_length = 0

                    obs, info = self.env.reset()
                    self.episode_count += 1
                else:
                    obs = next_obs
                    info = next_info

        # Compute statistics
        stats = {
            "mean_episode_reward": np.mean(episode_rewards) if episode_rewards else 0.0,
            "mean_utilization": np.mean(episode_utilizations) if episode_utilizations else 0.0,
            "mean_episode_length": np.mean(episode_lengths) if episode_lengths else 0.0,
            "num_episodes": len(episode_rewards),
        }

        return stats

    def train_step(self, returns: torch.Tensor, advantages: torch.Tensor) -> Dict[str, float]:
        """
        Perform one PPO training step (multiple epochs over collected data).

        Paper Reference: Section 4.2 - Policy Update
        "We update the policy using the PPO objective over K epochs"

        Args:
            returns: Computed returns for each transition
            advantages: Computed advantages for each transition

        Returns:
            Dictionary of training metrics
        """
        self.model.train()

        # Normalize advantages (improves stability)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Get all data from buffer
        (
            height_maps,
            item_features,
            actions,
            rewards,
            dones,
            values,
            old_log_probs,
            action_masks,
        ) = self.buffer.get_all()

        # Training metrics
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        total_loss = 0.0
        n_updates = 0

        # Multiple epochs over the data
        for epoch in range(self.n_epochs):
            # Generate random minibatches
            buffer_size = len(self.buffer)
            indices = torch.randperm(buffer_size, device=self.device)

            for start_idx in range(0, buffer_size, self.batch_size):
                end_idx = min(start_idx + self.batch_size, buffer_size)
                batch_indices = indices[start_idx:end_idx]

                # Get batch data
                batch_height_maps = height_maps[batch_indices]
                batch_item_features = item_features[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_action_masks = action_masks[batch_indices]

                # Evaluate actions with current policy
                log_probs, entropy, new_values = self.model.evaluate_actions(
                    batch_height_maps,
                    batch_item_features,
                    batch_actions,
                    batch_action_masks,
                )

                # Compute ratio: π_θ(a|s) / π_θ_old(a|s)
                ratio = torch.exp(log_probs - batch_old_log_probs)

                # Compute PPO clipped objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (MSE between predicted and target returns)
                new_values = new_values.squeeze()
                value_loss = nn.functional.mse_loss(new_values, batch_returns)

                # Entropy bonus (encourages exploration)
                entropy_loss = entropy.mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy_loss
                )

                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Accumulate metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
                total_loss += loss.item()
                n_updates += 1

        # Average metrics
        metrics = {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy_loss / n_updates,
            "total_loss": total_loss / n_updates,
        }

        return metrics

    def train(
        self,
        total_timesteps: int,
        eval_frequency: int = 10,
        save_frequency: int = 100,
        callback: Optional[Callable] = None,
    ):
        """
        Main training loop.

        Args:
            total_timesteps: Total number of environment steps
            eval_frequency: Evaluate every N updates
            save_frequency: Save checkpoint every N updates
            callback: Optional callback function called after each update
        """
        n_steps = self.buffer.buffer_size
        n_updates = total_timesteps // n_steps

        print(f"Starting training for {total_timesteps} timesteps ({n_updates} updates)")
        print(f"Rollout buffer size: {n_steps}")
        print(f"Batch size: {self.batch_size}")
        print(f"Epochs per update: {self.n_epochs}")
        print(f"Device: {self.device}")

        for update in tqdm(range(n_updates), desc="Training"):
            # Collect rollouts
            collection_stats = self.collect_rollouts(n_steps)

            # Compute returns and advantages
            # Get last value for bootstrapping
            obs, _ = self.env.reset()
            height_map, item_features = self._parse_observation(obs)
            with torch.no_grad():
                height_map_tensor = torch.from_numpy(height_map).unsqueeze(0).to(self.device)
                item_features_tensor = torch.from_numpy(item_features).unsqueeze(0).to(self.device)
                last_value = self.model.get_value(height_map_tensor, item_features_tensor).item()

            returns, advantages = self.buffer.compute_returns_and_advantages(
                last_value, self.gamma, self.gae_lambda
            )

            # Train on collected data
            train_metrics = self.train_step(returns, advantages)

            # Update learning rate
            self.scheduler.step()

            # Logging
            self.global_step = (update + 1) * n_steps

            for key, value in collection_stats.items():
                self.writer.add_scalar(f"rollout/{key}", value, self.global_step)

            for key, value in train_metrics.items():
                self.writer.add_scalar(f"train/{key}", value, self.global_step)

            self.writer.add_scalar("train/learning_rate", self.optimizer.param_groups[0]["lr"], self.global_step)

            # Evaluation
            if (update + 1) % eval_frequency == 0:
                eval_metrics = self.evaluate(n_episodes=5)
                for key, value in eval_metrics.items():
                    self.writer.add_scalar(f"eval/{key}", value, self.global_step)

                # Check if best model
                is_best = eval_metrics["mean_utilization"] > self.best_utilization
                if is_best:
                    self.best_utilization = eval_metrics["mean_utilization"]

                # Print progress
                print(f"\nUpdate {update + 1}/{n_updates}")
                print(f"  Rollout: reward={collection_stats['mean_episode_reward']:.3f}, "
                      f"util={collection_stats['mean_utilization']:.3f}")
                print(f"  Train: policy_loss={train_metrics['policy_loss']:.3f}, "
                      f"value_loss={train_metrics['value_loss']:.3f}")
                print(f"  Eval: util={eval_metrics['mean_utilization']:.3f} "
                      f"{'(BEST)' if is_best else ''}")

            # Save checkpoint
            if (update + 1) % save_frequency == 0 or (update + 1) == n_updates:
                metrics = {
                    "utilization": collection_stats["mean_utilization"],
                    "reward": collection_stats["mean_episode_reward"],
                }
                self.checkpoint_manager.save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=update + 1,
                    metrics=metrics,
                    extra_state={"global_step": self.global_step},
                    is_best=is_best if (update + 1) % eval_frequency == 0 else False,
                )

            # Callback
            if callback is not None:
                callback(self, update)

        print("\nTraining completed!")
        print(f"Best utilization: {self.best_utilization:.3f}")

        self.writer.close()

    def evaluate(self, n_episodes: int = 10, deterministic: bool = True) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            n_episodes: Number of episodes to evaluate
            deterministic: Whether to use deterministic actions

        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()

        episode_rewards = []
        episode_utilizations = []
        episode_lengths = []
        episode_packed_items = []

        with torch.no_grad():
            for _ in range(n_episodes):
                obs, info = self.env.reset()
                episode_reward = 0.0
                episode_length = 0
                done = False

                while not done:
                    height_map, item_features = self._parse_observation(obs)
                    action_mask = self.env._get_action_mask()

                    height_map_tensor = torch.from_numpy(height_map).unsqueeze(0).to(self.device)
                    item_features_tensor = torch.from_numpy(item_features).unsqueeze(0).to(self.device)
                    action_mask_tensor = torch.from_numpy(action_mask).unsqueeze(0).to(self.device)

                    action, _, _ = self.model.get_action(
                        height_map_tensor,
                        item_features_tensor,
                        action_mask_tensor,
                        deterministic=deterministic,
                    )

                    obs, reward, terminated, truncated, info = self.env.step(action.item())
                    episode_reward += reward
                    episode_length += 1
                    done = terminated or truncated

                episode_rewards.append(episode_reward)
                episode_utilizations.append(info.get("utilization", 0.0))
                episode_lengths.append(episode_length)
                episode_packed_items.append(info.get("num_packed", 0))

        return {
            "mean_reward": np.mean(episode_rewards),
            "mean_utilization": np.mean(episode_utilizations),
            "mean_length": np.mean(episode_lengths),
            "mean_packed_items": np.mean(episode_packed_items),
            "std_utilization": np.std(episode_utilizations),
        }

    def _parse_observation(self, obs: np.ndarray) -> tuple:
        """
        Parse flattened observation into height map and item features.

        Args:
            obs: Flattened observation vector

        Returns:
            height_map: (grid_size, grid_size)
            item_features: (5,)
        """
        grid_size = self.env.grid_size
        height_map_size = grid_size * grid_size

        height_map = obs[:height_map_size].reshape(grid_size, grid_size)
        item_features = obs[height_map_size:height_map_size + 5]

        return height_map, item_features

    def load_checkpoint(self, checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load checkpoint and resume training.

        Args:
            checkpoint_path: Path to checkpoint (None = latest)

        Returns:
            Checkpoint information
        """
        return self.checkpoint_manager.load_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            checkpoint_path=checkpoint_path,
            device=self.device,
        )
