"""
Checkpoint Manager for Saving and Loading Model States

Handles saving, loading, and managing checkpoints during training
with support for automatic resume and keeping best models.
"""

import os
import glob
import torch
from typing import Dict, Optional, Any, List
from pathlib import Path
import json


class CheckpointManager:
    """
    Manages model checkpoints during training.

    Features:
    - Save checkpoints periodically
    - Load checkpoints to resume training
    - Keep best N models based on metrics
    - Automatically detect latest checkpoint
    - Save training state (optimizer, scheduler, etc.)

    Args:
        checkpoint_dir: Directory to save checkpoints
        keep_best_n: Number of best checkpoints to keep
        metric_name: Metric to use for ranking (e.g., "utilization")
        higher_is_better: Whether higher metric values are better
    """

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        keep_best_n: int = 5,
        metric_name: str = "utilization",
        higher_is_better: bool = True,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.keep_best_n = keep_best_n
        self.metric_name = metric_name
        self.higher_is_better = higher_is_better

        # Track saved checkpoints with their metrics
        self.checkpoint_history: List[Dict[str, Any]] = []

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        config: Optional[Dict[str, Any]] = None,
        extra_state: Optional[Dict[str, Any]] = None,
        is_best: bool = False,
    ) -> str:
        """
        Save a checkpoint.

        Args:
            model: Model to save
            optimizer: Optimizer to save
            epoch: Current epoch/iteration
            metrics: Dictionary of metrics (e.g., {"utilization": 0.85})
            config: Model configuration
            extra_state: Additional state to save (e.g., scheduler)
            is_best: Whether this is the best model so far

        Returns:
            Path to saved checkpoint
        """
        # Create checkpoint dictionary
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "config": config or {},
        }

        # Add extra state if provided
        if extra_state:
            checkpoint.update(extra_state)

        # Generate filename
        metric_value = metrics.get(self.metric_name, 0.0)
        filename = f"checkpoint_epoch_{epoch:05d}_{self.metric_name}_{metric_value:.4f}.pt"
        filepath = self.checkpoint_dir / filename

        # Save checkpoint
        print(f"Saving checkpoint: {filepath}") # debug
        torch.save(checkpoint, filepath)

        # Save as "latest" for easy resume
        latest_path = self.checkpoint_dir / "latest.pt"
        torch.save(checkpoint, latest_path)

        # Save as "best" if applicable
        if is_best:
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(checkpoint, best_path)

        # Update checkpoint history
        self.checkpoint_history.append({
            "epoch": epoch,
            "path": str(filepath),
            "metric": metric_value,
            "metrics": metrics,
        })

        # Clean up old checkpoints
        self._cleanup_old_checkpoints()

        print(f"Checkpoint saved: {filepath}")
        if is_best:
            print(f"  New best {self.metric_name}: {metric_value:.4f}")

        return str(filepath)

    def load_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        checkpoint_path: Optional[str] = None,
        device: str = "cpu",
        load_optimizer: bool = True,
    ) -> Dict[str, Any]:
        """
        Load a checkpoint.

        Args:
            model: Model to load state into
            optimizer: Optimizer to load state into (optional)
            checkpoint_path: Path to checkpoint (if None, loads latest)
            device: Device to load checkpoint on
            load_optimizer: Whether to load optimizer state

        Returns:
            Dictionary containing epoch, metrics, and extra state
        """
        # Determine checkpoint path
        if checkpoint_path is None:
            checkpoint_path = self.get_latest_checkpoint()
            if checkpoint_path is None:
                raise FileNotFoundError("No checkpoint found")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Load model state
        model.load_state_dict(checkpoint["model_state_dict"])

        # Load optimizer state
        if optimizer is not None and load_optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        print(f"Checkpoint loaded from: {checkpoint_path}")
        print(f"  Epoch: {checkpoint['epoch']}")
        print(f"  Metrics: {checkpoint['metrics']}")

        return {
            "epoch": checkpoint["epoch"],
            "metrics": checkpoint["metrics"],
            "config": checkpoint.get("config", {}),
            **{k: v for k, v in checkpoint.items()
               if k not in ["model_state_dict", "optimizer_state_dict", "epoch", "metrics", "config"]},
        }

    def get_latest_checkpoint(self) -> Optional[str]:
        """
        Get path to latest checkpoint.

        Returns:
            Path to latest checkpoint or None if no checkpoints exist
        """
        latest_path = self.checkpoint_dir / "latest.pt"
        if latest_path.exists():
            return str(latest_path)

        # Fallback: find most recent checkpoint by filename
        checkpoints = glob.glob(str(self.checkpoint_dir / "checkpoint_epoch_*.pt"))
        if not checkpoints:
            return None

        # Sort by epoch number
        checkpoints.sort(key=lambda x: int(x.split("_epoch_")[1].split("_")[0]))
        return checkpoints[-1] if checkpoints else None

    def get_best_checkpoint(self) -> Optional[str]:
        """
        Get path to best checkpoint.

        Returns:
            Path to best checkpoint or None if doesn't exist
        """
        best_path = self.checkpoint_dir / "best.pt"
        return str(best_path) if best_path.exists() else None

    def _cleanup_old_checkpoints(self):
        """
        Remove old checkpoints, keeping only the best N.
        """
        if self.keep_best_n <= 0:
            return  # Keep all checkpoints

        # Get all checkpoint files (excluding latest.pt and best.pt)
        checkpoints = glob.glob(str(self.checkpoint_dir / "checkpoint_epoch_*.pt"))

        if len(checkpoints) <= self.keep_best_n:
            return  # Not enough checkpoints to clean up

        # Sort by metric value
        checkpoint_metrics = []
        for ckpt_path in checkpoints:
            try:
                checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                metric = checkpoint["metrics"].get(self.metric_name, 0.0)
                checkpoint_metrics.append((ckpt_path, metric))
            except Exception:
                continue

        # Sort by metric
        checkpoint_metrics.sort(key=lambda x: x[1], reverse=self.higher_is_better)

        # Keep best N, remove the rest
        to_remove = checkpoint_metrics[self.keep_best_n:]
        for ckpt_path, _ in to_remove:
            try:
                os.remove(ckpt_path)
                print(f"Removed old checkpoint: {ckpt_path}")
            except Exception as e:
                print(f"Failed to remove checkpoint {ckpt_path}: {e}")

    def save_training_info(self, info: Dict[str, Any]):
        """
        Save training information (hyperparameters, config, etc.).

        Args:
            info: Dictionary of training information
        """
        info_path = self.checkpoint_dir / "training_info.json"
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)

    def load_training_info(self) -> Optional[Dict[str, Any]]:
        """
        Load training information.

        Returns:
            Dictionary of training info or None if doesn't exist
        """
        info_path = self.checkpoint_dir / "training_info.json"
        if not info_path.exists():
            return None

        with open(info_path, "r") as f:
            return json.load(f)

    def has_checkpoint(self) -> bool:
        """
        Check if any checkpoint exists.

        Returns:
            True if checkpoint exists
        """
        return self.get_latest_checkpoint() is not None
