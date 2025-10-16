"""
3D Bin Packing Environment (Gymnasium-compatible)

Implements a Gymnasium-style environment for the 3D bin packing problem
with online item arrival. The agent learns to sequentially place items
into a container to maximize space utilization.

Paper Reference: Section 3 - Problem Formulation and Method
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, List, Optional, Any

from .container import Container
from .item import Item, generate_random_items
from .action_mask import ActionMasker


class PackingEnv(gym.Env):
    """
    3D Bin Packing Environment.

    The environment simulates online bin packing where items arrive sequentially
    and must be placed into a container. The goal is to maximize space utilization
    while maintaining stability.

    State Space:
        - Height map: (grid_size, grid_size) normalized heights
        - Current item features: (5,) [length, width, height, volume, weight]
        - Packed ratio: scalar indicating packing progress

    Action Space:
        - Discrete: grid_size × grid_size × 6 (position + rotation)

    Reward:
        - Hybrid reward combining space utilization, compactness, and stability
        - Large penalty for invalid actions

    Paper Reference: Section 3.1-3.2
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        container_size: Tuple[float, float, float] = (10.0, 10.0, 10.0),
        grid_size: int = 10,
        max_items: int = 50,
        item_size_range: Tuple[float, float] = (0.1, 0.5),
        enable_action_mask: bool = True,
        reward_type: str = "hybrid",
        normalize_state: bool = True,
        seed: Optional[int] = None,
    ):
        """
        Initialize packing environment.

        Args:
            container_size: Container dimensions (L, W, H)
            grid_size: Height map resolution
            max_items: Maximum number of items per episode
            item_size_range: Item size as fraction of container (min, max)
            enable_action_mask: Whether to use action masking
            reward_type: Reward function type ("hybrid", "utilization", "dense")
            normalize_state: Whether to normalize state features
            seed: Random seed
        """
        super().__init__()

        self.container_size = container_size
        self.grid_size = grid_size
        self.max_items = max_items
        self.item_size_range = item_size_range
        self.enable_action_mask = enable_action_mask
        self.reward_type = reward_type
        self.normalize_state = normalize_state

        # Initialize container
        self.container = Container(*container_size, grid_size=grid_size)

        # Initialize action masker
        self.action_masker = ActionMasker(
            enable_corner_heuristic=True,
            enable_stability_check=True
        )

        # Define action space: position (grid_x, grid_y) × rotation (0-5)
        self.action_space = spaces.Discrete(grid_size * grid_size * 6)

        # Define observation space
        # State = [height_map, item_features, metadata]
        height_map_dim = grid_size * grid_size
        item_features_dim = 5  # [l, w, h, volume, weight]
        metadata_dim = 3  # [packed_ratio, num_packed, remaining_items]

        obs_dim = height_map_dim + item_features_dim + metadata_dim
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # Episode state
        self.items: List[Item] = []
        self.current_item_idx = 0
        self.episode_reward = 0.0
        self.episode_step = 0
        self.invalid_action_count = 0

        # Random number generator
        self.np_random = None
        self.seed(seed)

    def seed(self, seed: Optional[int] = None):
        """Set random seed."""
        self.np_random = np.random.RandomState(seed)
        return [seed]

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment to initial state.

        Args:
            seed: Random seed
            options: Additional options (e.g., custom item list)

        Returns:
            observation: Initial state
            info: Additional information
        """
        if seed is not None:
            self.seed(seed)

        # Reset container
        self.container.reset()

        # Generate items
        if options and "items" in options:
            self.items = options["items"]
        else:
            self.items = generate_random_items(
                n_items=self.max_items,
                container_size=self.container_size,
                size_range=self.item_size_range,
                seed=seed
            )

        # Reset episode state
        self.current_item_idx = 0
        self.episode_reward = 0.0
        self.episode_step = 0
        self.invalid_action_count = 0

        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: Action index (flattened grid_x, grid_y, rotation)

        Returns:
            observation: Next state
            reward: Reward for this step
            terminated: Whether episode ended successfully
            truncated: Whether episode was cut off
            info: Additional information

        Paper Reference: Section 3.2 - Reward Function
        """
        self.episode_step += 1

        # Decode action
        grid_x, grid_y, rot_idx = self.action_masker.get_action_from_index(
            action, self.grid_size
        )

        # Get current item
        if self.current_item_idx >= len(self.items):
            # No more items
            terminated = True
            truncated = False
            reward = 0.0
            obs = self._get_observation()
            info = self._get_info()
            info["termination_reason"] = "no_more_items"
            return obs, reward, terminated, truncated, info

        current_item = self.items[self.current_item_idx]

        # Get rotated dimensions
        item_dims = current_item.get_rotation(rot_idx)
        item_l, item_w, item_h = item_dims

        # Convert grid position to continuous coordinates
        x = grid_x * self.container.cell_length
        y = grid_y * self.container.cell_width

        # Try to place item
        success = self.container.place_item(
            item_id=current_item.item_id,
            x=x,
            y=y,
            item_length=item_l,
            item_width=item_w,
            item_height=item_h,
            rotation_idx=rot_idx,
            weight=current_item.weight
        )

        # Calculate reward
        reward = self._compute_reward(success, current_item)

        # Update state
        if success:
            self.current_item_idx += 1
            self.invalid_action_count = 0  # Reset counter on successful placement
        else:
            self.invalid_action_count += 1

        # Check termination conditions
        terminated = False
        truncated = False

        # Episode ends when all items are processed or no valid actions remain
        if self.current_item_idx >= len(self.items):
            terminated = True
        elif self.invalid_action_count >= 10:  # Too many invalid actions
            truncated = True
        elif not success:
            # Check if any valid actions remain for current item
            mask = self._get_action_mask()
            if not self.action_masker.has_valid_action(mask):
                # Cannot place current item, skip to next
                self.current_item_idx += 1
                if self.current_item_idx >= len(self.items):
                    terminated = True

        # Get next observation
        obs = self._get_observation()
        info = self._get_info()
        info["action_success"] = success

        self.episode_reward += reward

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """
        Get current observation.

        Returns:
            Flattened state vector

        State components:
            1. Height map (flattened): (grid_size × grid_size,)
            2. Current item features: (5,)
            3. Metadata: (3,) [packed_ratio, num_packed_norm, remaining_norm]
        """
        # Height map (normalized)
        height_map = self.container.get_height_map_features(normalize=True)
        height_map_flat = height_map.flatten()

        # Current item features
        if self.current_item_idx < len(self.items):
            item = self.items[self.current_item_idx]
            item_features = item.to_feature_vector(
                normalize=self.normalize_state,
                max_dims=self.container_size
            )
        else:
            # No current item (episode ending)
            item_features = np.zeros(5, dtype=np.float32)

        # Metadata
        packed_ratio = self.container.num_packed / len(self.items) if len(self.items) > 0 else 0
        num_packed_norm = self.container.num_packed / self.max_items
        remaining_norm = (len(self.items) - self.current_item_idx) / self.max_items

        metadata = np.array([packed_ratio, num_packed_norm, remaining_norm], dtype=np.float32)

        # Concatenate all features
        obs = np.concatenate([height_map_flat, item_features, metadata])

        return obs.astype(np.float32)

    def _get_action_mask(self) -> np.ndarray:
        """
        Get action mask for current state.

        Returns:
            Binary mask of shape (grid_size, grid_size, 6)
        """
        if self.current_item_idx >= len(self.items):
            # No current item
            return np.zeros((self.grid_size, self.grid_size, 6), dtype=np.float32)

        current_item = self.items[self.current_item_idx]

        if self.enable_action_mask:
            mask = self.action_masker.get_valid_actions(self.container, current_item)
        else:
            # All actions potentially valid
            mask = np.ones((self.grid_size, self.grid_size, 6), dtype=np.float32)

        return mask

    def get_action_mask_flat(self) -> np.ndarray:
        """
        Get flattened action mask.

        Returns:
            Binary mask of shape (action_space_size,)
        """
        mask_3d = self._get_action_mask()
        return mask_3d.flatten()

    def _compute_reward(self, success: bool, item: Item) -> float:
        """
        Compute reward for current step.

        Paper Reference: Section 3.2 - Reward Function

        The reward consists of:
        1. Placement reward: change in space utilization
        2. Compactness bonus: reward for dense packing
        3. Stability bonus: reward for stable configurations
        4. Invalid action penalty: large negative reward

        Args:
            success: Whether placement was successful
            item: Item that was attempted to be placed

        Returns:
            Reward value
        """
        if not success:
            # Invalid action penalty
            return -1.0

        if self.reward_type == "utilization":
            # Legacy: Absolute utilization (for backward compatibility)
            # NOTE: This returns cumulative utilization, not delta. Actions later in
            # the episode receive higher rewards even for same volume contribution.
            # For new experiments, prefer "delta_utilization" or "dense".
            return self.container.utilization

        elif self.reward_type == "delta_utilization":
            # Delta utilization reward (recommended)
            # Each action is rewarded proportionally to the utilization improvement it provides.
            # This gives fair credit assignment regardless of when the action occurs.
            delta_utilization = item.volume / self.container.volume
            return delta_utilization

        elif self.reward_type == "dense":
            # Dense reward: immediate feedback (same as delta_utilization)
            delta_volume = item.volume
            reward = delta_volume / self.container.volume
            return reward

        elif self.reward_type == "hybrid":
            # Hybrid reward (from paper)
            # Components:
            # 1. Space utilization
            utilization_reward = self.container.utilization

            # 2. Compactness (lower height is better)
            max_height = self.container.get_max_height()
            compactness_penalty = max_height / self.container.height
            compactness_reward = 1.0 - compactness_penalty

            # 3. Stability
            stability_reward = self.container.get_stability_score()

            # Weighted combination
            reward = (
                0.6 * utilization_reward +
                0.2 * compactness_reward +
                0.2 * stability_reward
            )

            return reward

        else:
            raise ValueError(f"Unknown reward type: {self.reward_type}")

    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about current state."""
        info = {
            "num_packed": self.container.num_packed,
            "utilization": self.container.utilization,
            "max_height": self.container.get_max_height(),
            "compactness": self.container.get_compactness(),
            "stability": self.container.get_stability_score(),
            "current_item_idx": self.current_item_idx,
            "total_items": len(self.items),
            "episode_step": self.episode_step,
            "invalid_action_count": self.invalid_action_count,
        }

        # Add action mask info
        mask = self._get_action_mask()
        info["num_valid_actions"] = self.action_masker.count_valid_actions(mask)

        return info

    def render(self, mode: str = "human"):
        """
        Render current state.

        For now, print text representation.
        3D visualization will be in separate module.
        """
        if mode == "human":
            print(f"\n{'='*50}")
            print(f"Episode Step: {self.episode_step}")
            print(f"Container: {self.container}")
            print(f"Current Item: {self.current_item_idx + 1}/{len(self.items)}")
            if self.current_item_idx < len(self.items):
                print(f"  {self.items[self.current_item_idx]}")
            print(f"Utilization: {self.container.utilization:.2%}")
            print(f"Stability: {self.container.get_stability_score():.3f}")
            print(f"Invalid Actions: {self.invalid_action_count}")
            print(f"{'='*50}\n")

    def close(self):
        """Clean up resources."""
        pass

    def get_container_state(self) -> Container:
        """Get current container state (for visualization)."""
        return self.container

    def get_packed_items(self) -> List:
        """Get list of packed items (for visualization)."""
        return self.container.packed_items
