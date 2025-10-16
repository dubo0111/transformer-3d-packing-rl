"""
Utility to convert MIP solution to Container format for visualization.

Converts the output from MIP optimizer into a Container object that can
be visualized using the existing PackingVisualizer.
"""

from typing import Dict, Tuple
from ..environment.container import Container, PackedItem


def mip_solution_to_container(
    mip_result: Dict,
    container_size: Tuple[float, float, float],
    grid_size: int = 10
) -> Container:
    """
    Convert MIP optimization result to Container object.

    Args:
        mip_result: Result dictionary from MIPOptimizer.optimize()
        container_size: Container dimensions (L, W, H)
        grid_size: Grid size for height map (default: 10)

    Returns:
        Container object with packed items from MIP solution
    """
    # Create empty container
    container = Container(*container_size, grid_size=grid_size)

    # Add each loaded box from MIP solution
    for box_info in mip_result['loaded_boxes']:
        box_idx = box_info['index']
        dimensions = box_info['dimensions']
        position = box_info['position']

        # Create PackedItem
        packed_item = PackedItem(
            item_id=box_idx,
            position=position,
            dimensions=dimensions,
            rotation_idx=box_info['orientation_idx'],
            weight=1.0  # MIP doesn't track weight
        )

        # Add to container
        container.packed_items.append(packed_item)

        # Update height map (approximate - using grid cells)
        x_pos, y_pos, z_pos = position
        l, w, h = dimensions

        # Convert continuous coordinates to grid indices
        x_start = int(x_pos / container.cell_length)
        x_end = min(int((x_pos + l) / container.cell_length) + 1, grid_size)
        y_start = int(y_pos / container.cell_width)
        y_end = min(int((y_pos + w) / container.cell_width) + 1, grid_size)

        # Update height map cells covered by this box
        for i in range(x_start, x_end):
            for j in range(y_start, y_end):
                if i < grid_size and j < grid_size:
                    container.height_map[i, j] = max(
                        container.height_map[i, j],
                        z_pos + h
                    )

    return container
