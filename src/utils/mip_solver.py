"""
MIP Solver Wrapper for Optimal 3D Bin Packing

Provides interface between the RL environment and Gurobi MIP solver
for computing optimal packing solutions as a baseline comparison.

The MIP model maximizes volume utilization subject to:
- Non-overlapping constraints
- Container boundary constraints
- Orientation selection (6 rotations per box)
"""

from typing import List, Tuple, Dict, Optional
import numpy as np

from .mip_optimizer import solve_bin_packing


class MIPSolver:
    """
    Wrapper for Gurobi MIP solver to compute optimal packing solutions.

    Converts between RL environment format and MIP solver format,
    runs optimization, and extracts metrics for comparison.
    """

    def __init__(self, timeout: int = 300, verbose: bool = False):
        """
        Initialize MIP solver.

        Args:
            timeout: Maximum solve time in seconds (default 300s = 5min)
            verbose: Whether to print Gurobi solver output
        """
        self.timeout = timeout
        self.verbose = verbose

    def solve(
        self,
        container_size: Tuple[float, float, float],
        boxes: List[Tuple[float, float, float]],
    ) -> Dict[str, any]:
        """
        Solve 3D bin packing problem using MIP.

        Args:
            container_size: Container dimensions (L, W, H)
            boxes: List of box dimensions [(l1, w1, h1), (l2, w2, h2), ...]

        Returns:
            Dictionary containing:
                - 'loaded_boxes': List of packed boxes with positions/orientations
                - 'unloaded_boxes': List of box indices that couldn't be packed
                - 'utilization': Volume utilization ratio
                - 'num_packed': Number of boxes successfully packed
                - 'solve_time': Solver runtime in seconds
                - 'optimal': Whether optimal solution was found
                - 'gap': MIP gap at termination
        """
        # Call the MIP optimizer
        result = solve_bin_packing(
            container_size=container_size,
            boxes=boxes,
            timeout=self.timeout,
            verbose=self.verbose
        )

        return result

    def solve_from_env(self, env) -> Dict[str, any]:
        """
        Solve packing problem from RL environment state.

        Args:
            env: PackingEnv instance with current items

        Returns:
            Solution dictionary (same format as solve())
        """
        # Extract container size
        container_size = (env.container.length, env.container.width, env.container.height)

        # Extract box dimensions from environment items
        boxes = [(item.length, item.width, item.height) for item in env.items]

        return self.solve(container_size, boxes)

    def compare_with_rl(
        self,
        rl_utilization: float,
        mip_utilization: float,
    ) -> Dict[str, float]:
        """
        Calculate comparison metrics between RL and MIP solutions.

        Args:
            rl_utilization: Utilization achieved by RL agent
            mip_utilization: Utilization achieved by MIP solver (optimal)

        Returns:
            Dictionary with comparison metrics:
                - 'optimality_gap': Percentage gap from optimal
                - 'relative_performance': RL performance as ratio of MIP
        """
        if mip_utilization > 0:
            optimality_gap = (mip_utilization - rl_utilization) / mip_utilization * 100
            relative_performance = rl_utilization / mip_utilization
        else:
            optimality_gap = 0.0
            relative_performance = 1.0 if rl_utilization == 0 else float('inf')

        return {
            'optimality_gap': optimality_gap,
            'relative_performance': relative_performance,
        }


def format_mip_solution(solution: Dict) -> str:
    """
    Format MIP solution for display.

    Args:
        solution: Solution dictionary from MIPSolver.solve()

    Returns:
        Formatted string representation
    """
    lines = []
    lines.append("=" * 60)
    lines.append("MIP Solver Results")
    lines.append("=" * 60)
    lines.append(f"Status: {'OPTIMAL' if solution['optimal'] else 'FEASIBLE/TIMEOUT'}")
    lines.append(f"MIP Gap: {solution['gap']:.2%}")
    lines.append(f"Solve Time: {solution['solve_time']:.2f}s")
    lines.append(f"Boxes Packed: {solution['num_packed']}/{solution['num_total']}")
    lines.append(f"Utilization: {solution['utilization']:.2%}")
    lines.append(f"Packed Volume: {solution['packed_volume']:.2f}")
    lines.append(f"Total Volume: {solution['total_volume']:.2f}")
    lines.append("=" * 60)

    return "\n".join(lines)
