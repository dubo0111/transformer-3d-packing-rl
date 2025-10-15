"""
Container Class with Height Map Representation

Implements the container (bin) for 3D packing using a height map representation.
This is the key data structure from the paper that efficiently represents the
packing state and enables fast collision detection.

Paper Reference: Section 3.1 - Problem Formulation & State Representation
"""

import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass, field


@dataclass
class PackedItem:
    """Represents a packed item with its position and rotation."""
    item_id: int
    position: Tuple[float, float, float]  # (x, y, z) bottom-left-front corner
    dimensions: Tuple[float, float, float]  # (l, w, h) after rotation
    rotation_idx: int
    weight: float


class Container:
    """
    3D Container with Height Map representation.

    The height map is a 2D grid where each cell represents the maximum height
    of items at that (x, y) position. This enables O(1) collision detection
    and efficient placement validation.

    Paper Reference: Section 3.2 - Height Map
    The height map discretizes the container floor into a grid and tracks
    the surface height at each grid cell, allowing fast feasibility checks.

    Attributes:
        length (float): Container length (X-dimension)
        width (float): Container width (Y-dimension)
        height (float): Container height (Z-dimension)
        grid_size (int): Resolution of height map grid
        height_map (np.ndarray): 2D array of heights [grid_size x grid_size]
        packed_items (List[PackedItem]): List of successfully packed items
    """

    def __init__(self, length: float, width: float, height: float, grid_size: int = 10):
        """
        Initialize container.

        Args:
            length: X-dimension
            width: Y-dimension
            height: Z-dimension (maximum stacking height)
            grid_size: Height map resolution (paper uses 10x10)
        """
        if length <= 0 or width <= 0 or height <= 0:
            raise ValueError("All dimensions must be positive")
        if grid_size < 1:
            raise ValueError("Grid size must be at least 1")

        self.length = length
        self.width = width
        self.height = height
        self.grid_size = grid_size

        # Height map: tracks maximum height at each grid cell
        # Shape: (grid_size, grid_size)
        self.height_map = np.zeros((grid_size, grid_size), dtype=np.float32)

        # List of packed items with their positions
        self.packed_items: List[PackedItem] = []

        # Grid cell dimensions
        self.cell_length = length / grid_size
        self.cell_width = width / grid_size

    def reset(self):
        """Reset container to empty state."""
        self.height_map.fill(0.0)
        self.packed_items.clear()

    @property
    def volume(self) -> float:
        """Total container volume."""
        return self.length * self.width * self.height

    @property
    def packed_volume(self) -> float:
        """Volume of packed items."""
        return sum(
            item.dimensions[0] * item.dimensions[1] * item.dimensions[2]
            for item in self.packed_items
        )

    @property
    def utilization(self) -> float:
        """
        Space utilization ratio.

        Returns:
            Packed volume / Container volume

        Paper Reference: This is the primary metric (Equation 8)
        """
        return self.packed_volume / self.volume if self.volume > 0 else 0.0

    @property
    def num_packed(self) -> int:
        """Number of successfully packed items."""
        return len(self.packed_items)

    def _position_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """
        Convert continuous position to grid coordinates.

        Args:
            x: X position in container
            y: Y position in container

        Returns:
            (grid_x, grid_y) indices
        """
        grid_x = int(x / self.cell_length)
        grid_y = int(y / self.cell_width)

        # Clamp to valid range
        grid_x = max(0, min(grid_x, self.grid_size - 1))
        grid_y = max(0, min(grid_y, self.grid_size - 1))

        return grid_x, grid_y

    def _grid_to_position(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        """
        Convert grid coordinates to continuous position.

        Args:
            grid_x: X grid index
            grid_y: Y grid index

        Returns:
            (x, y) continuous position (center of grid cell)
        """
        x = (grid_x + 0.5) * self.cell_length
        y = (grid_y + 0.5) * self.cell_width
        return x, y

    def get_placement_height(self, x: float, y: float,
                            item_length: float, item_width: float) -> float:
        """
        Get the height at which an item would be placed at position (x, y).

        This is the maximum height across all grid cells the item would occupy.

        Args:
            x: X position (bottom-left corner)
            y: Y position (bottom-left corner)
            item_length: Item length (X-dimension)
            item_width: Item width (Y-dimension)

        Returns:
            Z-height where item bottom would be placed

        Paper Reference: Section 3.2
        The placement height is determined by the maximum height map value
        in the footprint area of the item.
        """
        # Find grid cells covered by item
        x_start, y_start = self._position_to_grid(x, y)
        x_end, y_end = self._position_to_grid(x + item_length, y + item_width)

        # Get maximum height in this region
        max_height = np.max(self.height_map[x_start:x_end+1, y_start:y_end+1])

        return max_height

    def can_place_item(self, x: float, y: float,
                      item_length: float, item_width: float, item_height: float,
                      tolerance: float = 1e-6) -> bool:
        """
        Check if item can be placed at given position.

        Args:
            x: X position
            y: Y position
            item_length: Item length after rotation
            item_width: Item width after rotation
            item_height: Item height after rotation
            tolerance: Numerical tolerance for boundary checks

        Returns:
            True if placement is valid, False otherwise

        Validity checks:
            1. Item must fit within container boundaries
            2. Item top must not exceed container height
            3. Position must be non-negative
        """
        # Check boundaries
        if x < -tolerance or y < -tolerance:
            return False

        if x + item_length > self.length + tolerance:
            return False

        if y + item_width > self.width + tolerance:
            return False

        # Check height constraint
        placement_z = self.get_placement_height(x, y, item_length, item_width)
        if placement_z + item_height > self.height + tolerance:
            return False

        return True

    def place_item(self, item_id: int, x: float, y: float,
                  item_length: float, item_width: float, item_height: float,
                  rotation_idx: int, weight: float = 1.0) -> bool:
        """
        Place item in container and update height map.

        Args:
            item_id: Unique item identifier
            x: X position
            y: Y position
            item_length: Item length after rotation
            item_width: Item width after rotation
            item_height: Item height after rotation
            rotation_idx: Rotation index (0-5)
            weight: Item weight

        Returns:
            True if successfully placed, False otherwise
        """
        # Check validity
        if not self.can_place_item(x, y, item_length, item_width, item_height):
            return False

        # Get placement height
        z = self.get_placement_height(x, y, item_length, item_width)

        # Update height map
        x_start, y_start = self._position_to_grid(x, y)
        x_end, y_end = self._position_to_grid(x + item_length, y + item_width)

        new_height = z + item_height
        self.height_map[x_start:x_end+1, y_start:y_end+1] = new_height

        # Record packed item
        packed_item = PackedItem(
            item_id=item_id,
            position=(x, y, z),
            dimensions=(item_length, item_width, item_height),
            rotation_idx=rotation_idx,
            weight=weight
        )
        self.packed_items.append(packed_item)

        return True

    def get_height_map_features(self, normalize: bool = True) -> np.ndarray:
        """
        Get height map as feature array for neural network.

        Args:
            normalize: If True, normalize heights to [0, 1]

        Returns:
            Height map array, shape (grid_size, grid_size)

        Paper Reference: Section 3.3.1 - State Representation
        The height map is a key component of the state fed to the actor network.
        """
        if normalize:
            return self.height_map / self.height if self.height > 0 else self.height_map
        return self.height_map.copy()

    def get_compactness(self) -> float:
        """
        Calculate packing compactness (lower is better).

        Compactness measures the vertical spread and gaps in packing.
        Lower values indicate more compact packing with fewer gaps.

        Returns:
            Average height of the height map

        Paper Reference: Section 3.2 - Reward Function
        Compactness is part of the reward to encourage dense packing.
        """
        return np.mean(self.height_map)

    def get_max_height(self) -> float:
        """Get maximum height of packed items."""
        return np.max(self.height_map) if self.height_map.size > 0 else 0.0

    def get_stability_score(self) -> float:
        """
        Calculate packing stability score.

        Stability is based on the center of gravity height. Lower center
        of gravity indicates more stable packing.

        Returns:
            Normalized stability score (0-1, higher is more stable)

        Paper Reference: Section 3.2 - Reward Function
        Stability is important for real-world packing applications.
        """
        if not self.packed_items:
            return 1.0

        total_weight = 0.0
        weighted_height_sum = 0.0

        for item in self.packed_items:
            x, y, z = item.position
            l, w, h = item.dimensions
            weight = item.weight

            # Center of gravity of this item
            cog_z = z + h / 2

            weighted_height_sum += cog_z * weight
            total_weight += weight

        avg_cog_height = weighted_height_sum / total_weight if total_weight > 0 else 0

        # Normalize (lower is better, so invert)
        stability = 1.0 - (avg_cog_height / self.height)
        return max(0.0, min(1.0, stability))

    def get_available_positions(self, item_length: float, item_width: float,
                               item_height: float) -> List[Tuple[int, int]]:
        """
        Get all valid grid positions where item can be placed.

        Args:
            item_length: Item length
            item_width: Item width
            item_height: Item height

        Returns:
            List of valid (grid_x, grid_y) positions
        """
        valid_positions = []

        for grid_x in range(self.grid_size):
            for grid_y in range(self.grid_size):
                x, y = self._grid_to_position(grid_x, grid_y)

                if self.can_place_item(x, y, item_length, item_width, item_height):
                    valid_positions.append((grid_x, grid_y))

        return valid_positions

    def __repr__(self) -> str:
        return (f"Container(L={self.length}, W={self.width}, H={self.height}, "
                f"grid={self.grid_size}x{self.grid_size}, "
                f"packed={self.num_packed}, util={self.utilization:.2%})")
