"""
Heuristic-based Action Masking for 3D Bin Packing

Implements action masking strategies to reduce the action space and improve
learning efficiency by filtering out obviously invalid or suboptimal actions.

Paper Reference: Section 3.4 - Action Space Reduction
The paper uses heuristic rules to mask invalid actions, significantly
reducing the search space and accelerating training convergence.
"""

import numpy as np
from typing import List, Tuple, Optional
from .container import Container
from .item import Item


class ActionMasker:
    """
    Generates action masks for valid item placements.

    The action space is: position (grid_x, grid_y) × rotation (0-5)
    Action masking eliminates invalid actions based on:
    1. Boundary constraints (item must fit in container)
    2. Height constraints (total height must not exceed container)
    3. Stability heuristics (prefer supported placements)
    4. Corner-occupying strategy (prefer corner/edge placements)

    Paper Reference: Section 3.4
    "To reduce the action space, we employ heuristic-based action masking
    that filters out infeasible actions before the agent makes decisions."
    """

    def __init__(self, enable_corner_heuristic: bool = True,
                 enable_stability_check: bool = True,
                 stability_threshold: float = 0.7):
        """
        Initialize action masker.

        Args:
            enable_corner_heuristic: If True, prioritize corner/edge positions
            enable_stability_check: If True, check for stable placements
            stability_threshold: Minimum support ratio for stable placement
        """
        self.enable_corner_heuristic = enable_corner_heuristic
        self.enable_stability_check = enable_stability_check
        self.stability_threshold = stability_threshold

    def get_valid_actions(self, container: Container, item: Item) -> np.ndarray:
        """
        Get binary mask for valid actions.

        Args:
            container: Current container state
            item: Item to be placed

        Returns:
            Binary mask array of shape (grid_size, grid_size, 6)
            where 1 indicates valid action, 0 indicates invalid

        Example:
            >>> masker = ActionMasker()
            >>> mask = masker.get_valid_actions(container, item)
            >>> valid_action_indices = np.argwhere(mask == 1)
        """
        grid_size = container.grid_size
        mask = np.zeros((grid_size, grid_size, 6), dtype=np.float32)

        # Check each rotation
        for rot_idx in range(6):
            item_dims = item.get_rotation(rot_idx)
            item_l, item_w, item_h = item_dims

            # Check each grid position
            for grid_x in range(grid_size):
                for grid_y in range(grid_size):
                    # Convert grid to continuous position
                    x = grid_x * container.cell_length
                    y = grid_y * container.cell_width

                    # Basic feasibility check
                    if container.can_place_item(x, y, item_l, item_w, item_h):
                        # Additional heuristic checks
                        if self._passes_heuristics(container, x, y, item_l,
                                                  item_w, item_h, grid_x, grid_y):
                            mask[grid_x, grid_y, rot_idx] = 1.0

        return mask

    def _passes_heuristics(self, container: Container,
                          x: float, y: float,
                          item_l: float, item_w: float, item_h: float,
                          grid_x: int, grid_y: int) -> bool:
        """
        Check if placement passes heuristic rules.

        Args:
            container: Container state
            x, y: Continuous position
            item_l, item_w, item_h: Item dimensions
            grid_x, grid_y: Grid coordinates

        Returns:
            True if passes heuristics, False otherwise
        """
        # Stability check: ensure adequate support
        if self.enable_stability_check and len(container.packed_items) > 0:
            if not self._check_stability(container, x, y, item_l, item_w):
                return False

        # Corner heuristic: prefer corners and edges early in packing
        if self.enable_corner_heuristic and len(container.packed_items) < 3:
            if not self._is_corner_or_edge(grid_x, grid_y, container.grid_size):
                return False

        return True

    def _check_stability(self, container: Container,
                        x: float, y: float,
                        item_l: float, item_w: float) -> bool:
        """
        Check if item placement is stable (has adequate support).

        An item is considered stable if a sufficient percentage of its
        bottom area is supported by items below or the container floor.

        Args:
            container: Container state
            x, y: Item position
            item_l, item_w: Item footprint dimensions

        Returns:
            True if placement is stable

        Paper Reference: Section 3.2 - Stability Constraint
        Items should be well-supported to prevent unrealistic configurations.
        """
        # Get grid cells covered by item
        x_start, y_start = container._position_to_grid(x, y)
        x_end, y_end = container._position_to_grid(x + item_l, y + item_w)

        # Ensure valid range
        x_start = max(0, min(x_start, container.grid_size - 1))
        x_end = max(0, min(x_end, container.grid_size - 1))
        y_start = max(0, min(y_start, container.grid_size - 1))
        y_end = max(0, min(y_end, container.grid_size - 1))

        if x_end <= x_start or y_end <= y_start:
            return True  # Edge case: very small item

        # Get heights in footprint area
        footprint_heights = container.height_map[x_start:x_end+1, y_start:y_end+1]

        # Placement height (where item bottom will be)
        placement_height = np.max(footprint_heights)

        # Count cells that provide support (within small tolerance)
        tolerance = 1e-3
        supported_cells = np.sum(footprint_heights >= placement_height - tolerance)
        total_cells = footprint_heights.size

        # Calculate support ratio
        support_ratio = supported_cells / total_cells if total_cells > 0 else 0

        return support_ratio >= self.stability_threshold

    def _is_corner_or_edge(self, grid_x: int, grid_y: int, grid_size: int) -> bool:
        """
        Check if position is at corner or edge of container.

        Corner-occupying strategy: prioritize corners and edges to
        create a compact base for subsequent items.

        Args:
            grid_x, grid_y: Grid coordinates
            grid_size: Grid dimension

        Returns:
            True if position is corner or edge

        Paper Reference: Section 3.4
        "We employ a corner-occupying strategy that prioritizes placements
        at corners and edges of the container."
        """
        # Check if on any edge
        is_edge_x = (grid_x == 0 or grid_x == grid_size - 1)
        is_edge_y = (grid_y == 0 or grid_y == grid_size - 1)

        return is_edge_x or is_edge_y

    def get_masked_probabilities(self, action_probs: np.ndarray,
                                action_mask: np.ndarray,
                                epsilon: float = 1e-8) -> np.ndarray:
        """
        Apply action mask to probability distribution.

        Args:
            action_probs: Action probabilities from policy, shape (grid, grid, 6)
            action_mask: Binary mask, shape (grid, grid, 6)
            epsilon: Small constant for numerical stability

        Returns:
            Masked and re-normalized probabilities

        Paper Reference: Section 3.3.2 - Actor Network
        "Invalid actions are masked out and the probability distribution
        is renormalized over valid actions only."
        """
        # Mask invalid actions
        masked_probs = action_probs * action_mask

        # Add small epsilon to avoid division by zero
        masked_probs = masked_probs + epsilon * action_mask

        # Renormalize
        prob_sum = np.sum(masked_probs)
        if prob_sum > epsilon:
            masked_probs = masked_probs / prob_sum
        else:
            # If all actions are invalid, uniform over valid actions
            num_valid = np.sum(action_mask)
            if num_valid > 0:
                masked_probs = action_mask / num_valid
            else:
                # No valid actions (shouldn't happen, but handle gracefully)
                masked_probs = np.ones_like(action_mask) / action_mask.size

        return masked_probs

    def count_valid_actions(self, action_mask: np.ndarray) -> int:
        """
        Count number of valid actions.

        Args:
            action_mask: Binary mask array

        Returns:
            Number of valid (non-zero) actions
        """
        return int(np.sum(action_mask))

    def has_valid_action(self, action_mask: np.ndarray) -> bool:
        """
        Check if there are any valid actions.

        Args:
            action_mask: Binary mask array

        Returns:
            True if at least one valid action exists
        """
        return np.any(action_mask > 0)

    def get_corner_priority_mask(self, container: Container) -> np.ndarray:
        """
        Get priority weights for corner/edge positions.

        Returns higher weights for corners and edges, which can be
        multiplied with action probabilities to bias selection.

        Args:
            container: Container state

        Returns:
            Priority weights, shape (grid_size, grid_size)

        Usage:
            This can be used to softly encourage corner placements
            without hard masking.
        """
        grid_size = container.grid_size
        priority = np.ones((grid_size, grid_size), dtype=np.float32)

        for grid_x in range(grid_size):
            for grid_y in range(grid_size):
                # Corners get highest priority
                if (grid_x in [0, grid_size-1]) and (grid_y in [0, grid_size-1]):
                    priority[grid_x, grid_y] = 3.0
                # Edges get medium priority
                elif grid_x in [0, grid_size-1] or grid_y in [0, grid_size-1]:
                    priority[grid_x, grid_y] = 2.0
                # Interior gets base priority
                else:
                    priority[grid_x, grid_y] = 1.0

        return priority

    def get_action_from_index(self, action_idx: int, grid_size: int) -> Tuple[int, int, int]:
        """
        Convert flattened action index to (grid_x, grid_y, rotation).

        Args:
            action_idx: Flattened action index
            grid_size: Grid dimension

        Returns:
            Tuple of (grid_x, grid_y, rotation_idx)
        """
        rotations_per_pos = 6
        pos_idx = action_idx // rotations_per_pos
        rot_idx = action_idx % rotations_per_pos

        grid_x = pos_idx // grid_size
        grid_y = pos_idx % grid_size

        return grid_x, grid_y, rot_idx

    def get_index_from_action(self, grid_x: int, grid_y: int, rot_idx: int,
                             grid_size: int) -> int:
        """
        Convert (grid_x, grid_y, rotation) to flattened action index.

        Args:
            grid_x: X grid coordinate
            grid_y: Y grid coordinate
            rot_idx: Rotation index (0-5)
            grid_size: Grid dimension

        Returns:
            Flattened action index
        """
        pos_idx = grid_x * grid_size + grid_y
        action_idx = pos_idx * 6 + rot_idx
        return action_idx
