"""
MIP-based 3D Bin Packing Optimizer

Standalone Mixed-Integer Programming model for 3D bin packing using Gurobi.
This implementation uses the same mathematical formulation as the reference model
but with volume maximization objective for better utilization comparison.

Mathematical Model:
- Decision variables: box positions (x, y, z), orientations, loaded status
- Constraints: non-overlapping, container boundaries, orientation selection
- Objective: Maximize total packed volume
"""

from typing import List, Tuple, Dict, Optional
from itertools import permutations
from gurobipy import Model, GRB, quicksum, Env
import time
from tqdm import tqdm
import threading
import sys
import os
from contextlib import contextmanager


class Box:
    """Box with multiple orientations."""

    def __init__(self, index: int, length: float, width: float, height: float):
        self.index = index
        self.length = length
        self.width = width
        self.height = height
        self.volume = length * width * height

        # Decision variables (set during optimization)
        self.var_loaded = None
        self.var_x = None
        self.var_y = None
        self.var_z = None

        # Position after optimization
        self.position = (0.0, 0.0, 0.0)

        # Generate all 6 orientations
        self.orientations: List[Orientation] = []
        self._generate_orientations()

    def _generate_orientations(self):
        """Generate all unique orientations (permutations of dimensions)."""
        perms = list(permutations([self.length, self.width, self.height]))
        for idx, (x, y, z) in enumerate(perms):
            self.orientations.append(Orientation(idx, x, y, z))


class Orientation:
    """Single orientation of a box."""

    def __init__(self, idx: int, x: float, y: float, z: float):
        self.idx = idx
        self.x = x  # length in this orientation
        self.y = y  # width in this orientation
        self.z = z  # height in this orientation
        self.var_selection = None


class BoxRelation:
    """Spatial relation between two boxes (for non-overlapping constraints)."""

    def __init__(self, box1: Box, box2: Box):
        self.first_box = box1
        self.second_box = box2
        self.var_relation_x = None
        self.var_relation_y = None
        self.var_relation_z = None


class Container:
    """Container with fixed dimensions."""

    def __init__(self, length: float, width: float, height: float):
        self.length = length
        self.width = width
        self.height = height
        self.volume = length * width * height


class MIPOptimizer:
    """
    MIP-based 3D bin packing optimizer.

    Solves the bin packing problem to optimality (or best feasible solution
    within time limit) using Mixed-Integer Programming.
    """

    def __init__(
        self,
        container_size: Tuple[float, float, float],
        boxes: List[Tuple[float, float, float]],
        timeout: int = 300,
        verbose: bool = False,
    ):
        """
        Initialize optimizer.

        Args:
            container_size: Container dimensions (L, W, H)
            boxes: List of box dimensions [(l1, w1, h1), ...]
            timeout: Maximum solve time in seconds
            verbose: Whether to print solver output
        """
        self.container = Container(*container_size)
        self.boxes = [Box(i, l, w, h) for i, (l, w, h) in enumerate(boxes)]
        self.timeout = timeout
        self.verbose = verbose

        # Box relations for non-overlap constraints
        self.box_relations: List[BoxRelation] = []
        self.box_map: Dict[int, Box] = {}
        self.box_relation_map: Dict[Tuple[int, int], BoxRelation] = {}

        self._init_relations()

        # Gurobi model - create environment with output suppression if needed
        if not verbose:
            # Create silent environment
            env = Env(empty=True)
            env.setParam('OutputFlag', 0)
            env.start()
            self.model = Model("3D_BinPacking", env=env)
        else:
            self.model = Model("3D_BinPacking")

        self.model.setParam('TimeLimit', timeout)

        # Solution
        self.loaded_boxes = []
        self.unloaded_boxes = []
        self.solve_time = 0.0
        self.optimal = False
        self.mip_gap = 1.0

    def _init_relations(self):
        """Initialize box relationships for pairwise constraints."""
        for box in self.boxes:
            self.box_map[box.index] = box
            for other_box in self.boxes:
                if box.index != other_box.index:
                    relation = BoxRelation(box, other_box)
                    self.box_relations.append(relation)
                    self.box_relation_map[(box.index, other_box.index)] = relation

    def _add_variables(self):
        """Add decision variables to model."""
        # Box position and loading variables
        for box in self.boxes:
            box.var_x = self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"x_{box.index}")
            box.var_y = self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"y_{box.index}")
            box.var_z = self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"z_{box.index}")
            box.var_loaded = self.model.addVar(vtype=GRB.BINARY, name=f"loaded_{box.index}")

            # Orientation selection variables
            for orientation in box.orientations:
                orientation.var_selection = self.model.addVar(
                    vtype=GRB.BINARY,
                    name=f"orient_{box.index}_{orientation.idx}"
                )

        # Box relation variables (for non-overlap constraints)
        for relation in self.box_relations:
            relation.var_relation_x = self.model.addVar(
                vtype=GRB.BINARY,
                name=f"rel_x_{relation.first_box.index}_{relation.second_box.index}"
            )
            relation.var_relation_y = self.model.addVar(
                vtype=GRB.BINARY,
                name=f"rel_y_{relation.first_box.index}_{relation.second_box.index}"
            )
            relation.var_relation_z = self.model.addVar(
                vtype=GRB.BINARY,
                name=f"rel_z_{relation.first_box.index}_{relation.second_box.index}"
            )

    def _add_orientation_constraints(self):
        """Each loaded box must have exactly one orientation selected."""
        for box in self.boxes:
            # Sum of orientation selections equals loaded status
            # If var_loaded=1, exactly one orientation is selected
            # If var_loaded=0, no orientation is selected (box not packed)
            self.model.addConstr(
                quicksum([o.var_selection for o in box.orientations]) == box.var_loaded,
                name=f"orient_sum_{box.index}"
            )

    def _add_geometry_constraints(self):
        """Boxes must fit within container boundaries."""
        for box in self.boxes:
            # X-dimension: position + dimension <= container length
            self.model.addConstr(
                box.var_x + quicksum([o.var_selection * o.x for o in box.orientations])
                <= self.container.length * box.var_loaded,
                name=f"geom_x_{box.index}"
            )

            # Y-dimension: position + dimension <= container width
            self.model.addConstr(
                box.var_y + quicksum([o.var_selection * o.y for o in box.orientations])
                <= self.container.width * box.var_loaded,
                name=f"geom_y_{box.index}"
            )

            # Z-dimension: position + dimension <= container height
            self.model.addConstr(
                box.var_z + quicksum([o.var_selection * o.z for o in box.orientations])
                <= self.container.height * box.var_loaded,
                name=f"geom_z_{box.index}"
            )

    def _add_non_overlap_constraints(self):
        """Boxes must not overlap (pairwise separation constraints)."""
        indices = list(self.box_map.keys())

        for i in indices:
            for j in indices:
                if i < j:
                    # Mutual exclusion: both boxes can't claim same space
                    self.model.addConstr(
                        self.box_relation_map[(i, j)].var_relation_x +
                        self.box_relation_map[(j, i)].var_relation_x
                        <= self.box_map[i].var_loaded,
                        name=f"mutex_x_{i}_{j}"
                    )
                    self.model.addConstr(
                        self.box_relation_map[(i, j)].var_relation_y +
                        self.box_relation_map[(j, i)].var_relation_y
                        <= self.box_map[i].var_loaded,
                        name=f"mutex_y_{i}_{j}"
                    )
                    self.model.addConstr(
                        self.box_relation_map[(i, j)].var_relation_z +
                        self.box_relation_map[(j, i)].var_relation_z
                        <= self.box_map[i].var_loaded,
                        name=f"mutex_z_{i}_{j}"
                    )

                    # If both boxes loaded, they must be separated in at least one dimension
                    self.model.addConstr(
                        self.box_relation_map[(i, j)].var_relation_x +
                        self.box_relation_map[(i, j)].var_relation_y +
                        self.box_relation_map[(i, j)].var_relation_z +
                        self.box_relation_map[(j, i)].var_relation_x +
                        self.box_relation_map[(j, i)].var_relation_y +
                        self.box_relation_map[(j, i)].var_relation_z
                        >= self.box_map[i].var_loaded + self.box_map[j].var_loaded - 1,
                        name=f"sep_{i}_{j}"
                    )

                if i != j:
                    box1 = self.box_map[i]
                    box2 = self.box_map[j]

                    # Box i must be to the left of box j (in x-direction)
                    self.model.addConstr(
                        box1.var_x + quicksum([o.var_selection * o.x for o in box1.orientations])
                        <= box2.var_x + self.container.length *
                           (1 - self.box_relation_map[(i, j)].var_relation_x),
                        name=f"pos_x_{i}_{j}"
                    )

                    # Box i must be in front of box j (in y-direction)
                    self.model.addConstr(
                        box1.var_y + quicksum([o.var_selection * o.y for o in box1.orientations])
                        <= box2.var_y + self.container.width *
                           (1 - self.box_relation_map[(i, j)].var_relation_y),
                        name=f"pos_y_{i}_{j}"
                    )

                    # Box i must be below box j (in z-direction)
                    self.model.addConstr(
                        box1.var_z + quicksum([o.var_selection * o.z for o in box1.orientations])
                        <= box2.var_z + self.container.height *
                           (1 - self.box_relation_map[(i, j)].var_relation_z),
                        name=f"pos_z_{i}_{j}"
                    )

    def _set_objective(self):
        """Maximize total volume of packed boxes."""
        self.model.setObjective(
            quicksum([box.var_loaded * box.volume for box in self.boxes]),
            GRB.MAXIMIZE
        )

    def optimize(self) -> Dict:
        """
        Solve the optimization problem.

        Returns:
            Dictionary containing solution and metrics
        """
        start_time = time.time()

        # Build model
        self._add_variables()
        self._add_orientation_constraints()
        self._add_geometry_constraints()
        self._add_non_overlap_constraints()
        self._set_objective()

        # Solve with progress bar if not verbose
        if not self.verbose:
            pbar = tqdm(
                total=self.timeout,
                desc="MIP Solving",
                unit="s",
                ncols=100,
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n:.1f}/{total:.0f}s [{elapsed}<{remaining}]"
            )
            pbar_stop = threading.Event()

            def update_pbar():
                """Update progress bar based on elapsed time."""
                while not pbar_stop.is_set():
                    elapsed = time.time() - start_time
                    pbar.n = min(elapsed, self.timeout)
                    pbar.refresh()
                    time.sleep(0.1)

            # Start progress bar in background thread
            pbar_thread = threading.Thread(target=update_pbar, daemon=True)
            pbar_thread.start()

            try:
                self.model.optimize()
            finally:
                # Stop progress bar
                pbar_stop.set()
                pbar_thread.join(timeout=1.0)
                # Update to final time
                final_time = time.time() - start_time
                pbar.n = min(final_time, self.timeout)

                # Update description based on solver status
                if self.model.status == GRB.OPTIMAL:
                    pbar.set_description("MIP Solved (Optimal)")
                elif self.model.status == GRB.TIME_LIMIT:
                    pbar.set_description("MIP Timeout")
                elif self.model.SolCount > 0:
                    pbar.set_description("MIP Solved (Feasible)")
                else:
                    pbar.set_description("MIP Failed")

                pbar.refresh()
                pbar.close()
        else:
            self.model.optimize()

        self.solve_time = time.time() - start_time

        # Extract solution
        self._extract_solution()

        # Calculate metrics
        return self._compute_metrics()

    def _extract_solution(self):
        """Extract solution from optimized model."""
        self.loaded_boxes = []
        self.unloaded_boxes = []

        if self.model.status == GRB.OPTIMAL:
            self.optimal = True
            self.mip_gap = 0.0
        elif self.model.SolCount > 0:
            self.optimal = False
            self.mip_gap = self.model.MIPGap
        else:
            self.optimal = False
            self.mip_gap = 1.0
            self.unloaded_boxes = [box.index for box in self.boxes]
            return

        for box in self.boxes:
            if box.var_loaded.x > 0.5:
                # Box is loaded - extract position and orientation
                position = (box.var_x.x, box.var_y.x, box.var_z.x)
                box.position = position

                # Find selected orientation
                selected_orientation = None
                for orientation in box.orientations:
                    if orientation.var_selection.x > 0.5:
                        selected_orientation = orientation
                        break

                if selected_orientation:
                    self.loaded_boxes.append({
                        'index': box.index,
                        'dimensions': (selected_orientation.x, selected_orientation.y, selected_orientation.z),
                        'position': position,
                        'orientation_idx': selected_orientation.idx,
                        'volume': box.volume,
                    })
            else:
                self.unloaded_boxes.append(box.index)

    def _compute_metrics(self) -> Dict:
        """Compute solution metrics."""
        packed_volume = sum(box['volume'] for box in self.loaded_boxes)
        utilization = packed_volume / self.container.volume if self.container.volume > 0 else 0.0

        return {
            'loaded_boxes': self.loaded_boxes,
            'unloaded_boxes': self.unloaded_boxes,
            'num_packed': len(self.loaded_boxes),
            'num_total': len(self.boxes),
            'packed_volume': packed_volume,
            'total_volume': self.container.volume,
            'utilization': utilization,
            'solve_time': self.solve_time,
            'optimal': self.optimal,
            'gap': self.mip_gap,
            'container_size': (self.container.length, self.container.width, self.container.height),
        }


def solve_bin_packing(
    container_size: Tuple[float, float, float],
    boxes: List[Tuple[float, float, float]],
    timeout: int = 300,
    verbose: bool = False,
) -> Dict:
    """
    Solve 3D bin packing problem using MIP.

    Args:
        container_size: Container dimensions (L, W, H)
        boxes: List of box dimensions [(l1, w1, h1), ...]
        timeout: Maximum solve time in seconds
        verbose: Whether to print solver output

    Returns:
        Solution dictionary with metrics
    """
    optimizer = MIPOptimizer(container_size, boxes, timeout, verbose)
    return optimizer.optimize()
