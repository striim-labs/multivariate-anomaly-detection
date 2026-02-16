"""
TranAD Multi-Machine Model Registry

Manages per-machine models, scalers, scorers, and thresholds.
Each SMD machine maps to a physical store deployment.

Filesystem layout:
    models/tranad/{machine_id}/
        model.ckpt          -- PyTorch checkpoint (from training)
        scorer_state.json   -- calibrated thresholds + POT params (from evaluation)
        eval_results.json   -- evaluation metrics (from evaluation)
"""

import json
import logging
from pathlib import Path

import numpy as np
import torch

from tranad_model import TranADConfig, TranADNet

logger = logging.getLogger(__name__)


class TranADRegistry:
    """Registry managing per-machine TranAD models.

    Provides unified load/save for model checkpoints, normalization
    parameters, and scoring state (thresholds).

    Args:
        base_dir: Root directory for model artifacts.
    """

    def __init__(self, base_dir: str | Path = "models/tranad"):
        self.base_dir = Path(base_dir)
        self._model_cache: dict[str, tuple[TranADNet, TranADConfig]] = {}

    def machine_dir(self, machine_id: str) -> Path:
        """Return the directory for a specific machine."""
        return self.base_dir / machine_id

    def list_machines(self) -> list[str]:
        """List all registered machine IDs (those with a model.ckpt)."""
        machines = []
        if self.base_dir.exists():
            for d in sorted(self.base_dir.iterdir()):
                if d.is_dir() and (d / "model.ckpt").exists():
                    machines.append(d.name)
        return machines

    def get_model(
        self,
        machine_id: str,
        device: str | torch.device = "cpu",
    ) -> tuple[TranADNet, TranADConfig]:
        """Load a trained TranAD model for the given machine.

        Returns cached model if already loaded. The model is set to
        eval mode and moved to the specified device.

        Args:
            machine_id: Machine identifier.
            device: Target device.

        Returns:
            (model, config) tuple.

        Raises:
            FileNotFoundError: If no checkpoint exists for this machine.
        """
        device = torch.device(device) if isinstance(device, str) else device
        cache_key = f"{machine_id}_{device}"

        if cache_key in self._model_cache:
            return self._model_cache[cache_key]

        ckpt_path = self.machine_dir(machine_id) / "model.ckpt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

        config = checkpoint["config"]
        model = TranADNet(config).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        self._model_cache[cache_key] = (model, config)
        logger.info("Loaded model for %s on %s", machine_id, device)
        return model, config

    def get_norm_params(
        self,
        machine_id: str,
        data_dir: str | Path = "data/smd/processed",
    ) -> np.ndarray | None:
        """Load normalization parameters for a machine.

        Looks for {machine_id}_norm_params.npy in the data directory.

        Args:
            machine_id: Machine identifier.
            data_dir: Directory containing preprocessed data files.

        Returns:
            Array of shape (2, n_features) with [min_vals, max_vals],
            or None if not found.
        """
        path = Path(data_dir) / f"{machine_id}_norm_params.npy"
        if path.exists():
            return np.load(path)
        return None

    def get_scorer_state(self, machine_id: str) -> dict | None:
        """Load saved scoring state (thresholds, method, params).

        Args:
            machine_id: Machine identifier.

        Returns:
            Dict with scoring state, or None if not saved yet.
        """
        path = self.machine_dir(machine_id) / "scorer_state.json"
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)

    def save_scorer_state(self, machine_id: str, state: dict) -> Path:
        """Save scoring state (thresholds, method, params) for a machine.

        Args:
            machine_id: Machine identifier.
            state: Dict containing at minimum 'threshold' and 'method'.

        Returns:
            Path to the saved scorer_state.json file.
        """
        mdir = self.machine_dir(machine_id)
        mdir.mkdir(parents=True, exist_ok=True)
        path = mdir / "scorer_state.json"

        # Convert numpy/dataclass types for JSON serialization
        serializable = {}
        for k, v in state.items():
            if isinstance(v, np.ndarray):
                serializable[k] = v.tolist()
            elif isinstance(v, (np.floating, float)):
                serializable[k] = float(v)
            elif isinstance(v, (np.integer, int)):
                serializable[k] = int(v)
            elif isinstance(v, dict):
                serializable[k] = {
                    sk: float(sv) if isinstance(sv, (np.floating, float)) else sv
                    for sk, sv in v.items()
                }
            else:
                serializable[k] = v

        with open(path, "w") as f:
            json.dump(serializable, f, indent=2)
        logger.info("Saved scorer state for %s to %s", machine_id, path)
        return path

    def clear_cache(self, machine_id: str | None = None) -> None:
        """Remove cached models from memory.

        Args:
            machine_id: If provided, clear only that machine's cache.
                       If None, clear all.
        """
        if machine_id is None:
            self._model_cache.clear()
        else:
            keys_to_remove = [k for k in self._model_cache if k.startswith(machine_id)]
            for k in keys_to_remove:
                del self._model_cache[k]
