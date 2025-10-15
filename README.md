# 3D Bin Packing with Deep Reinforcement Learning

> ⚠️ **Development Warning**: This project is currently under development and may not be fully stable or feature-complete.

Production-ready implementation of **TAP-Net** (Transformer-based Actor-critic Packing Network) for solving the 3D bin packing problem using Deep Reinforcement Learning and PPO.

## Quick Start

### Training

```bash
# Train with default configuration
python train.py

# Train on GPU
python train.py --device cuda

# Train with custom timesteps
python train.py --total-timesteps 2000000 --device cuda

# Resume from checkpoint
python train.py --resume
```

### Monitoring

```bash
# Launch TensorBoard (in a separate terminal)
tensorboard --logdir logs/

# Open browser to http://localhost:6006
```

### Evaluation

```bash
# Evaluate trained model
python evaluate.py --checkpoint checkpoints/best.pt

# Evaluate with 3D visualization
python evaluate.py --checkpoint checkpoints/best.pt --visualize --save-html

# Evaluate 50 episodes
python evaluate.py --checkpoint checkpoints/best.pt --n-episodes 50
```

## Project Structure

```
transformer-3d-packing-rl/
├── src/
│   ├── environment/        # 3D packing environment
│   │   ├── container.py    # Container with height map
│   │   ├── item.py         # Items with 6 rotations
│   │   ├── action_mask.py  # Heuristic action masking
│   │   └── packing_env.py  # Gymnasium environment
│   ├── models/             # Neural network models
│   │   ├── tap_net.py      # Main TAP-Net model
│   │   ├── actor.py        # Transformer actor network
│   │   └── critic.py       # Value network
│   ├── training/           # Training infrastructure
│   │   ├── ppo_trainer.py  # PPO algorithm
│   │   ├── replay_buffer.py # Rollout buffer with GAE
│   │   └── checkpoint_manager.py # Checkpoint management
│   ├── visualization/      # Visualization tools
│   │   ├── plotly_3d.py    # 3D interactive plots
│   │   └── training_plots.py # Training curves
│   └── utils/              # Utilities
│       ├── config.py       # Configuration management
│       ├── logger.py       # Logging setup
│       └── metrics.py      # Evaluation metrics
├── config/
│   └── default.yaml        # Default hyperparameters
├── reference/              # Paper reference code
├── train.py                # Main training script
├── evaluate.py             # Evaluation script
├── requirements.txt        # Python dependencies
├── CLAUDE.md               # Development guide
└── README.md               # This file
```


**Happy Packing!** 📦
