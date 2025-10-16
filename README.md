# 3D Bin Packing with Deep Reinforcement Learning

Transformer-based neural network trained with PPO to solve 3D bin packing problems.

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Training

```bash
# Basic training
python train.py

# Train on GPU
python train.py --device cuda --total-timesteps 2000000

# Resume training
python train.py --resume
```

### Monitoring

```bash
tensorboard --logdir logs/
```

Open http://localhost:6006

### Evaluation

```bash
# Evaluate model
python evaluate.py --checkpoint checkpoints/best.pt --n-episodes 50

# With 3D visualization
python evaluate.py --checkpoint checkpoints/best.pt --visualize --save-html

# Compare with MIP optimal solution (requires Gurobi)
python evaluate.py --checkpoint checkpoints/best.pt --compare-mip --mip-timeout 300
```

## Features

- **Transformer-based Policy**: Separate encoders for container state and unpacked boxes
- **Sequential Actions**: Selects position, then box, then orientation
- **Height Map + Plane Features**: Enhanced spatial representation for better decision-making
- **PPO Training**: Stable policy gradient training with GAE
- **MIP Baseline**: Compare against optimal solutions using Gurobi (optional)
- **Interactive Visualization**: 3D plots with side-by-side RL vs MIP comparison

## Project Structure

```
src/
├── environment/      # 3D packing environment
├── models/          # Transformer actor-critic networks
├── training/        # PPO trainer
├── visualization/   # 3D plots
└── utils/          # Config, logging, MIP optimizer

config/default.yaml  # Hyperparameters
train.py            # Training script
evaluate.py         # Evaluation script
```

## Configuration

Edit `config/default.yaml`:

- **Environment**: Container size, item count/dimensions
- **Model**: Hidden dim, layers, attention heads
- **Training**: Learning rates, batch size, buffer size
- **PPO**: Discount, GAE lambda, clip epsilon

Or use CLI:
```bash
python train.py --total-timesteps 1000000 --max-items 30 --device cuda
```

## MIP Baseline

Requires Gurobi:

```bash
python evaluate.py --checkpoint checkpoints/best.pt \
  --compare-mip --mip-timeout 300 \
  --visualize --save-html --n-episodes 10
```

Generates side-by-side visualizations and optimality gap analysis.

## Security Note

**⚠️ Checkpoint Loading Security**: This project uses `weights_only=False` when loading PyTorch checkpoints to support full model state restoration. Only load checkpoints from trusted sources, as malicious checkpoints could execute arbitrary code. Do not load checkpoints from untrusted or unknown sources.

If you need to load checkpoints from untrusted sources, consider:
1. Inspecting the checkpoint contents first
2. Running in an isolated environment (container/VM)
3. Using `weights_only=True` (may require code modifications)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
