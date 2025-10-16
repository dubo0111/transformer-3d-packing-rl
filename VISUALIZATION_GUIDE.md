# Side-by-Side RL vs MIP Visualization Guide

This guide explains how to use the enhanced visualization feature that displays RL and MIP solutions side-by-side for direct comparison.

## Quick Start

### Command to Generate Side-by-Side Visualizations

```bash
python evaluate.py \
    --checkpoint checkpoints/best.pt \
    --n-episodes 5 \
    --compare-mip \
    --visualize \
    --save-html \
    --mip-timeout 300
```

This will create HTML files in `outputs/` directory:
- `episode_1_comparison.html` - Interactive side-by-side comparison
- `episode_2_comparison.html` - etc.
- `evaluation_metrics.txt` - Text summary with metrics

## Visualization Features

### Layout Structure

```
┌──────────────────────────────────────────────────────────────────┐
│           Episode 1: RL vs MIP Comparison                        │
│  Optimality Gap: 13.24% | Relative Performance: 86.76%           │
├─────────────────────────────┬────────────────────────────────────┤
│      RL Agent               │         MIP Optimal                │
│   Utilization: 75.23%       │      Utilization: 86.76%           │
│   Packed: 48 items          │      Packed: 50 items              │
│                             │                                    │
│   ┌─────────────────────┐   │   ┌─────────────────────┐          │
│   │                     │   │   │                     │          │
│   │  3D Visualization   │   │   │  3D Visualization   │          │
│   │  (Interactive)      │   │   │  (Interactive)      │          │
│   │                     │   │   │                     │          │
│   └─────────────────────┘   │   └─────────────────────┘          │
│                             │                                    │
└─────────────────────────────┴────────────────────────────────────┘
```

### Interactive Controls

**Mouse Controls:**
- **Left Click + Drag:** Rotate view
- **Right Click + Drag:** Pan view
- **Scroll Wheel:** Zoom in/out
- **Hover:** Show box details (ID, position, dimensions)

**Camera:**
- Both views start with same camera angle for fair comparison
- Can rotate each view independently
- Camera positioned at (1.5, 1.5, 1.2) relative to container center

**Visual Elements:**
- Color-coded boxes (different colors for each box)
- Black wireframe for container boundaries
- Semi-transparent boxes (80% opacity) to see overlaps
- Hover tooltips with detailed information

## Metrics Displayed

### Title Metrics

**Optimality Gap:**
```
Gap = (MIP_util - RL_util) / MIP_util × 100%
```
- Shows how far RL is from optimal
- Lower is better (closer to optimal)

**Relative Performance:**
```
Performance = RL_util / MIP_util × 100%
```
- Shows RL performance as percentage of optimal
- Higher is better (closer to 100%)

### Subplot Titles

Each subplot shows:
- **Method:** "RL Agent" or "MIP Optimal"
- **Utilization:** Volume utilization percentage
- **Packed:** Number of items successfully packed

## Interpreting the Visualization

### What to Look For

**1. Space Efficiency:**
- Compare how tightly boxes are packed
- Look for gaps and wasted space
- MIP typically has fewer gaps

**2. Packing Strategy:**
- **RL:** May show learned heuristics (e.g., corner placement, layering)
- **MIP:** Shows mathematically optimal arrangement
- Compare placement patterns

**3. Box Count:**
- If MIP packed more boxes → RL missed opportunities
- If same count but different utilization → RL has worse orientations

**4. Common Patterns:**
- RL often learns "greedy" strategies (fill corners first)
- MIP can make non-obvious choices for global optimality
- RL may struggle with tetris-like configurations

### Performance Interpretation

| Optimality Gap | Quality      | Interpretation                           |
|----------------|--------------|------------------------------------------|
| < 5%           | Excellent    | RL nearly optimal, publication-worthy    |
| 5-10%          | Very Good    | RL competitive, minor improvements left  |
| 10-20%         | Good         | RL functional, room for improvement      |
| 20-30%         | Moderate     | RL works but significant gap remains     |
| > 30%          | Poor         | RL needs significant improvements        |

## Example Analysis Workflow

### Step 1: Generate Visualizations
```bash
python evaluate.py \
    --checkpoint checkpoints/best.pt \
    --n-episodes 10 \
    --compare-mip \
    --visualize \
    --save-html
```

### Step 2: Review Aggregate Metrics
Check console output or `outputs/evaluation_metrics.txt`:
```
MIP Utilization (mean):    82.45%
RL Utilization (mean):     75.23%
Optimality Gap:            8.76%
Relative Performance:      91.24%
```

### Step 3: Analyze Individual Episodes
Open HTML files in browser:
1. Start with worst-performing episodes (highest gap)
2. Look for patterns in RL mistakes
3. Compare with best-performing episodes

### Step 4: Identify Improvement Opportunities
Common issues to look for:
- **Poor orientation choices:** Box rotated sub-optimally
- **Corner wasting:** Not utilizing container corners
- **Height inefficiency:** Poor vertical stacking
- **Fragmentation:** Many small gaps between boxes

## Technical Details

### File Structure

**Input:**
- Checkpoint: Trained RL model weights
- Environment config: Container size, item ranges

**Processing:**
1. RL agent packs items → Container object
2. MIP solver packs same items → MIP result dict
3. Convert MIP result → Container object (via `mip_to_container`)
4. Generate comparison figure → HTML file

**Output Files:**
```
outputs/
├── episode_1_comparison.html  # Interactive visualization
├── episode_2_comparison.html
├── ...
└── evaluation_metrics.txt     # Text summary
```

### Visualization Pipeline

```
RL Container ──┐
               ├──→ PackingVisualizer.compare_rl_vs_mip() ──→ HTML
MIP Container ─┘
```

**Key Components:**
1. `PackingVisualizer.compare_rl_vs_mip()` - Creates figure
2. `mip_solution_to_container()` - Converts MIP format
3. Plotly `make_subplots()` - Side-by-side layout
4. Plotly `Mesh3d` - 3D box rendering

### Customization Options

**Change Figure Size:**
Edit `plotly_3d.py:498-499`:
```python
height=800,  # Default: 600
width=1600,  # Default: 1400
```

**Change Camera Angle:**
Edit `plotly_3d.py:509-511`:
```python
eye=dict(x=2.0, y=2.0, z=1.5),  # Further away
```

**Change Colors:**
Edit `plotly_3d.py:36`:
```python
color_scheme = "Plasma"  # Default: "Viridis"
# Options: Viridis, Plasma, Inferno, Magma, Cividis
```

## Troubleshooting

### Issue: HTML files won't open
**Solution:** Use a modern browser (Chrome, Firefox, Edge)
- Plotly visualizations require JavaScript
- Some browsers block local file scripts

### Issue: Visualizations look different
**Possible causes:**
1. Different item sets (check that same items were used)
2. Different container sizes (verify config)
3. Rendering differences across browsers

### Issue: MIP solution looks wrong
**Check:**
1. MIP solver didn't timeout (check "Optimal=Yes")
2. All items were packed (check packed count)
3. No overlapping boxes (rotate view to check)

### Issue: Performance is slow
**Solutions:**
1. Reduce number of visualized episodes (default: first 5)
2. Simplify boxes (fewer items per episode)
3. Disable labels: `show_labels=False`

## Advanced Usage

### Compare Multiple Checkpoints

```bash
# Evaluate checkpoint 1
python evaluate.py --checkpoint checkpoints/epoch_100.pt \
    --compare-mip --visualize --save-html --output-dir outputs/epoch_100

# Evaluate checkpoint 2
python evaluate.py --checkpoint checkpoints/epoch_200.pt \
    --compare-mip --visualize --save-html --output-dir outputs/epoch_200

# Compare the outputs/ directories
```

### Export for Presentations

HTML files can be:
1. **Embedded in web pages** (use `<iframe>`)
2. **Converted to images** (screenshot or use Plotly export)
3. **Shared as links** (host on web server)

### Batch Processing

```bash
# Evaluate all checkpoints
for ckpt in checkpoints/*.pt; do
    python evaluate.py --checkpoint "$ckpt" \
        --compare-mip --visualize --save-html \
        --output-dir "outputs/$(basename $ckpt .pt)"
done
```

## Best Practices

1. **Always use same seed** for reproducible comparisons
2. **Run multiple episodes** (10+) for statistical significance
3. **Check MIP optimal status** before concluding about RL performance
4. **Visualize both best and worst cases** to understand variance
5. **Document findings** with screenshots and metrics

## References

- **Main Documentation:** `MIP_INTEGRATION.md`
- **Paper:** Que et al., 2023 - "Solving 3D packing problem using Transformer network and reinforcement learning"
- **Plotly Docs:** https://plotly.com/python/3d-charts/
