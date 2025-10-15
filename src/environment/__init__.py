"""
3D Bin Packing Environment

This module implements the packing environment with:
- Container representation using height maps
- Item with multiple rotation orientations
- Heuristic-based action masking
- Gymnasium-compatible interface
"""

from .container import Container
from .item import Item
from .packing_env import PackingEnv
from .action_mask import ActionMasker

__all__ = ["Container", "Item", "PackingEnv", "ActionMasker"]
