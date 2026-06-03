"""General utility helpers used by the training and evaluation scripts."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping, MutableMapping

import numpy as np
import torch


def to_tensor_raw(image) -> torch.Tensor:
    """Convert an image-like object to an int64 torch tensor without scaling."""
    return torch.from_numpy(np.asarray(image, dtype=np.int64))


def config_logging(logfile: str | None = None, level: int = logging.INFO) -> None:
    """Configure console/file logging with a compact, readable format."""
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if logfile:
        Path(logfile).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(logfile))
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def safe_load_state_dict(model: torch.nn.Module, state_dict: Mapping[str, torch.Tensor]) -> None:
    """Load matching checkpoint tensors and skip incompatible/missing layers.

    This is useful when fine-tuning from a checkpoint whose classifier head has a
    different number of output channels.
    """
    model_state: MutableMapping[str, torch.Tensor] = model.state_dict()
    compatible = {
        key: value
        for key, value in state_dict.items()
        if key in model_state and tuple(model_state[key].shape) == tuple(value.shape)
    }
    skipped = sorted(set(state_dict) - set(compatible))
    model_state.update(compatible)
    model.load_state_dict(model_state)
    if skipped:
        logging.warning("Skipped %d incompatible checkpoint tensors.", len(skipped))
