# MIP Baseline Integration

This document describes the MIP (Mixed-Integer Programming) optimizer integration for computing optimal packing solutions as a baseline comparison with the RL agent.

## Overview

The MIP optimizer provides exact or near-optimal solutions to the 3D bin packing problem using the Gurobi solver. This allows for rigorous performance comparison between the learned RL policy and the mathematical optimum.

## Files Created

### 1. `src/utils/mip_optimizer.py`
Standalone MIP model implementation with the following components:

**Mathematical Formulation:**
- **Decision Variables:**
  - `x_i, y_i, z_i`: Position of box i (continuous)
  - `loaded_i`: Whether box i is packed (binary)
  - `orient_i_k`: Whether box i uses orientation k (binary, 6 orientations)
  - `rel_x_ij, rel_y_ij, rel_z_ij`: Spatial relations between boxes i and j (binary)

- **Objective Function:**
  ```
  maximize Σ(loaded_i × volume_i)
  ```
  Maximizes total packed volume for better utilization comparison.

- **Constraints:**
  1. **Orientation constraint:** Each packed box must have exactly one orientation selected
  2. **Geometry constraints:** Boxes must fit within container boundaries
  3. **Non-overlapping constraints:** Boxes cannot overlap (pairwise separation)

**Key Classes:**
- `Box`: Box with 6 rotation orientations
- `Orientation`: Single orientation of a box
- `BoxRelation`: Spatial relation between two boxes
- `Container`: Container with fixed dimensions
- `MIPOptimizer`: Main optimizer class

**Usage:**
```python
from src.utils.mip_optimizer import solve_bin_packing

result = solve_bin_packing(
    container_size=(100, 100, 100),
    boxes=[(10, 10, 10), (20, 20, 20), ...],
    timeout=300,
    verbose=False
)

print(f"Utilization: {result['utilization']:.2%}")
print(f"Packed: {result['num_packed']}/{result['num_total']}")
print(f"Optimal: {result['optimal']}")
```

### 2. `src/utils/mip_solver.py`
High-level wrapper providing interface between RL environment and MIP optimizer:

**Key Methods:**
- `solve()`: Solve for given container and boxes
- `solve_from_env()`: Solve directly from PackingEnv instance
- `compare_with_rl()`: Compute comparison metrics

**Metrics Returned:**
- `utilization`: Volume utilization ratio
- `num_packed`: Number of boxes successfully packed
- `solve_time`: Solver runtime in seconds
- `optimal`: Whether optimal solution was found
- `gap`: MIP optimality gap at termination
- `loaded_boxes`: List of packed boxes with positions/orientations
- `unloaded_boxes`: List of box indices that couldn't be packed

### 3. `src/utils/mip_to_container.py`
Utility to convert MIP solutions to Container format for visualization:

**Function:**
- `mip_solution_to_container()`: Converts MIP result dict to Container object
- Reconstructs packed items from MIP solution
- Updates height map based on box placements

### 4. `src/visualization/plotly_3d.py` (Enhanced)
Added side-by-side comparison visualization:

**New Method:**
- `compare_rl_vs_mip()`: Creates side-by-side RL vs MIP visualization
  - Shows both solutions in same figure
  - Displays metrics (utilization, optimality gap) in title
  - Synchronized camera views for easy comparison
  - Interactive 3D rotation and zoom

### 5. `evaluate.py` (Modified)
Enhanced with optional MIP baseline comparison:

**New Arguments:**
- `--compare-mip`: Enable MIP baseline comparison
- `--mip-timeout`: Solver timeout in seconds (default: 300)

**Output Enhancements:**
- Side-by-side RL vs MIP utilization per episode
- Aggregate comparison metrics:
  - Mean MIP utilization
  - Mean RL utilization
  - Optimality gap (%)
  - Relative performance ratio
  - MIP solve time statistics
  - Number of optimal solutions found
- **Side-by-side HTML visualizations** when `--visualize --save-html` used with `--compare-mip`

## Usage Examples

### Basic Evaluation (RL only)
```bash
python evaluate.py --checkpoint checkpoints/best.pt --n-episodes 10
```

### Evaluation with MIP Baseline
```bash
python evaluate.py \
    --checkpoint checkpoints/best.pt \
    --n-episodes 10 \
    --compare-mip \
    --mip-timeout 300
```

### Evaluation with Side-by-Side Visualization
```bash
python evaluate.py \
    --checkpoint checkpoints/latest.pt \
    --n-episodes 1 \
    --compare-mip \
    --visualize \
    --save-html \
    --mip-timeout 300
```
This will generate HTML files with interactive side-by-side comparisons of RL vs MIP solutions.

### Output Example
```
Episode 1/10: Utilization=75.23%, Packed=48, Reward=125.342
  Running MIP solver for episode 1...
  MIP: Utilization=82.45%, Packed=50/50, Time=45.32s, Optimal=Yes

================================================================================
MIP BASELINE COMPARISON
================================================================================
MIP Utilization (mean):    82.45%
RL Utilization (mean):     75.23%
Optimality Gap:            8.76%
Relative Performance:      91.24%
MIP Solve Time (mean):     45.32s
MIP Optimal Solutions:     10/10
================================================================================
```

### Metrics File Output
Results saved to `outputs/evaluation_metrics.txt`:
```
================================================================================
TAP-Net Evaluation Results
================================================================================

Checkpoint: checkpoints/best.pt
Episodes: 10
Deterministic: False

RL Agent Metrics:
--------------------------------------------------------------------------------
mean_utilization........................      0.7523
std_utilization.........................      0.0234
...

================================================================================
MIP Baseline Comparison
================================================================================

MIP Utilization (mean):          0.8245
RL Utilization (mean):           0.7523
Optimality Gap:                  8.7600%
Relative Performance:            0.9124
MIP Solve Time (mean):           45.3200s
MIP Solve Time (std):            12.4500s
MIP Optimal Solutions:           10/10
```

## Implementation Notes

### Mathematical Model Integrity
The mathematical formulation in `src/utils/mip_optimizer.py` is **identical** to the reference implementation in `reference/mip/packer.py`, with the following key difference:

**Objective Function Change:**
- **Reference:** Maximize number of boxes packed
- **New:** Maximize total volume packed

This change better aligns with the utilization metric used for RL evaluation, making the comparison more meaningful.

### Performance Considerations

**Solver Complexity:**
- The MIP model has O(n²) binary variables for n boxes (due to pairwise relations)
- For 50 boxes: ~5000 binary variables, ~15000 constraints
- Typical solve time: 30-300 seconds per instance
- Timeout recommended: 300s (5 minutes) per episode

**Scalability:**
- Works well for 20-50 boxes (paper range)
- May timeout for 100+ boxes without finding optimal solution
- Still provides best feasible solution within timeout

### Dependencies

**Required:**
- `gurobipy` >= 10.0.0 (Gurobi Python API)
- Valid Gurobi license (free academic license available)

**Installation:**
```bash
# Install gurobipy
pip install gurobipy

# Activate academic license (if applicable)
grbgetkey YOUR-LICENSE-KEY
```

**Graceful Degradation:**
If Gurobi is not available, `evaluate.py` will skip MIP comparison with a warning message.

## Interpreting Results

### Optimality Gap
```
Optimality Gap = (MIP_util - RL_util) / MIP_util × 100%
```
- **< 5%**: Excellent RL performance, near-optimal
- **5-15%**: Good RL performance, room for improvement
- **15-30%**: Moderate RL performance, significant gap
- **> 30%**: Poor RL performance, needs investigation

### Relative Performance
```
Relative Performance = RL_util / MIP_util
```
- **> 0.95**: Excellent (within 5% of optimal)
- **0.85-0.95**: Good (within 15% of optimal)
- **0.70-0.85**: Moderate (15-30% gap)
- **< 0.70**: Needs improvement

### MIP Status Interpretation
- **Optimal=Yes, Gap=0.0**: Proven optimal solution
- **Optimal=No, Gap<0.01**: Near-optimal (within 1%)
- **Optimal=No, Gap>0.1**: Timeout, best feasible solution

## Troubleshooting

### Issue: ImportError for gurobipy
**Solution:** Install Gurobi and activate license
```bash
pip install gurobipy
grbgetkey YOUR-LICENSE-KEY
```

### Issue: MIP solver timeout on every episode
**Solution:** Increase timeout or reduce problem size
```bash
python evaluate.py --compare-mip --mip-timeout 600  # 10 minutes
```

### Issue: Out of memory
**Solution:** Reduce number of boxes or run episodes sequentially
- MIP solver is memory-intensive for large problems
- Consider evaluating fewer episodes or smaller instances

### Issue: Inconsistent results between RL and MIP
**Check:**
1. Same items used for both methods (verified by implementation)
2. Container dimensions match (verified by implementation)
3. Coordinate systems align (both use bottom-left-front origin)

## Visualization

When using `--compare-mip --visualize --save-html`, the output HTML files show side-by-side comparison:

**Layout:**
```
┌─────────────────────────────────────────────────────────────┐
│  Episode 1: RL vs MIP Comparison                            │
│  Optimality Gap: 8.76% | Relative Performance: 91.24%       │
├──────────────────────────┬──────────────────────────────────┤
│     RL Agent             │      MIP Optimal                 │
│  Utilization: 75.23%     │   Utilization: 82.45%            │
│  Packed: 48 items        │   Packed: 50 items               │
│                          │                                  │
│  [3D visualization]      │   [3D visualization]             │
│                          │                                  │
└──────────────────────────┴──────────────────────────────────┘
```

**Features:**
- Interactive 3D rotation (synchronized or independent)
- Zoom and pan controls
- Hover to see box details
- Color-coded boxes for easy identification
- Container boundary wireframe
- Same camera angle for fair comparison

**Output Files:**
- `outputs/episode_1_comparison.html` - Side-by-side RL vs MIP
- `outputs/episode_2_comparison.html` - etc.
- `outputs/evaluation_metrics.txt` - Text metrics summary

## Future Enhancements

Potential improvements:
1. **Parallel MIP solving:** Run multiple episodes in parallel
2. **Warm start:** Use RL solution to initialize MIP
3. **Constraint relaxation:** Speed up MIP with heuristic constraints
4. ✅ **Visualization:** Show both RL and MIP solutions side-by-side (DONE)
5. **Per-episode comparison:** Detailed breakdown for each episode

## References

- **Paper:** "Solving 3D packing problem using Transformer network and reinforcement learning" (Que et al., 2023)
- **Gurobi Documentation:** https://www.gurobi.com/documentation/
- **3D Bin Packing Survey:** Zhao et al. (2016), "A comparative review of 3D container loading algorithms"
