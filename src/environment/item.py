"""
Item Class for 3D Bin Packing

Represents a 3D rectangular item with support for 6 rotation orientations.
Each item has dimensions (length, width, height) and can be rotated to fit
optimally in the container.

Paper Reference: Section 3.1 - Problem Formulation
"""

import numpy as np
from typing import Tuple, List
from dataclasses import dataclass


@dataclass
class Item:
    """
    3D rectangular item for bin packing.

    Attributes:
        length (float): X-dimension (depth)
        width (float): Y-dimension (width)
        height (float): Z-dimension (height)
        item_id (int): Unique identifier for the item
        weight (float): Weight of the item (for stability calculation)
    """

    length: float
    width: float
    height: float
    item_id: int = 0
    weight: float = 1.0

    def __post_init__(self):
        """Validate dimensions are positive."""
        if self.length <= 0 or self.width <= 0 or self.height <= 0:
            raise ValueError("All dimensions must be positive")
        if self.weight <= 0:
            raise ValueError("Weight must be positive")

    @property
    def dimensions(self) -> Tuple[float, float, float]:
        """Get item dimensions as tuple (l, w, h)."""
        return (self.length, self.width, self.height)

    @property
    def volume(self) -> float:
        """Calculate item volume."""
        return self.length * self.width * self.height

    def get_rotations(self) -> List[Tuple[float, float, float]]:
        """
        Get all 6 unique rotation orientations.

        For a rectangular box, there are 6 distinct orientations
        corresponding to which face is placed on the bottom:

        1. (l, w, h) - Original orientation
        2. (l, h, w) - Rotate 90° around X-axis
        3. (w, l, h) - Rotate 90° around Z-axis
        4. (w, h, l) - Rotate 90° around Y-axis
        5. (h, l, w) - Rotate to place smallest face down
        6. (h, w, l) - Rotate to place largest face down

        Returns:
            List of 6 tuples representing (length, width, height) for each rotation

        Note:
            In the paper (Section 3.2), rotations are part of the action space
            to maximize space utilization and stability.
        """
        l, w, h = self.length, self.width, self.height

        # All permutations of dimensions
        rotations = [
            (l, w, h),  # Rotation 0: Original
            (l, h, w),  # Rotation 1: 90° around X
            (w, l, h),  # Rotation 2: 90° around Z
            (w, h, l),  # Rotation 3: 90° around Y
            (h, l, w),  # Rotation 4: Complex rotation 1
            (h, w, l),  # Rotation 5: Complex rotation 2
        ]

        return rotations

    def get_rotation(self, rotation_idx: int) -> Tuple[float, float, float]:
        """
        Get specific rotation by index.

        Args:
            rotation_idx: Index from 0 to 5

        Returns:
            Tuple of (length, width, height) for specified rotation
        """
        if not 0 <= rotation_idx < 6:
            raise ValueError(f"Rotation index must be 0-5, got {rotation_idx}")

        return self.get_rotations()[rotation_idx]

    def get_rotated_volume(self, rotation_idx: int) -> float:
        """
        Get volume for specific rotation (always same, included for completeness).

        Args:
            rotation_idx: Rotation index

        Returns:
            Volume (constant regardless of rotation)
        """
        return self.volume

    def to_feature_vector(self, normalize: bool = False,
                         max_dims: Tuple[float, float, float] = None) -> np.ndarray:
        """
        Convert item to feature vector for neural network input.

        Args:
            normalize: Whether to normalize dimensions
            max_dims: Maximum dimensions for normalization (container size)

        Returns:
            Feature vector [length, width, height, volume, weight]

        Paper Reference: Section 3.3.1 - State Representation
        The item features are encoded and fed into the actor network.
        """
        features = np.array([
            self.length,
            self.width,
            self.height,
            self.volume,
            self.weight
        ], dtype=np.float32)

        if normalize and max_dims is not None:
            max_l, max_w, max_h = max_dims
            max_vol = max_l * max_w * max_h

            features[0] /= max_l
            features[1] /= max_w
            features[2] /= max_h
            features[3] /= max_vol
            # Weight normalization depends on your use case
            # features[4] /= max_weight  # Uncomment if you have max_weight

        return features

    def __repr__(self) -> str:
        return f"Item(id={self.item_id}, l={self.length:.2f}, w={self.width:.2f}, h={self.height:.2f}, vol={self.volume:.2f})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Item):
            return False
        return (self.item_id == other.item_id and
                np.isclose(self.length, other.length) and
                np.isclose(self.width, other.width) and
                np.isclose(self.height, other.height))


def generate_random_items(n_items: int,
                         container_size: Tuple[float, float, float],
                         size_range: Tuple[float, float] = (0.1, 0.5),
                         seed: int = None) -> List[Item]:
    """
    Generate random items for testing and training.

    Args:
        n_items: Number of items to generate
        container_size: Container dimensions (L, W, H)
        size_range: Fraction of container size for item dimensions (min, max)
        seed: Random seed for reproducibility

    Returns:
        List of randomly generated items

    Example:
        >>> items = generate_random_items(10, (100, 100, 100), (0.1, 0.5))
        >>> print(f"Generated {len(items)} items")
    """
    if seed is not None:
        np.random.seed(seed)

    min_frac, max_frac = size_range
    container_l, container_w, container_h = container_size

    items = []
    for i in range(n_items):
        # Generate random dimensions as fraction of container
        l = np.random.uniform(min_frac, max_frac) * container_l
        w = np.random.uniform(min_frac, max_frac) * container_w
        h = np.random.uniform(min_frac, max_frac) * container_h

        # Weight proportional to volume (density = 1)
        volume = l * w * h
        weight = volume

        items.append(Item(
            length=l,
            width=w,
            height=h,
            item_id=i,
            weight=weight
        ))

    return items
