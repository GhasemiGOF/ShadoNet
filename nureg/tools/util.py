"""Tool-specific tensor helpers."""

from __future__ import annotations

import torch


def make_variable(tensor: torch.Tensor, volatile: bool = False, requires_grad: bool = True, device=None):
    """Move a tensor to the selected device and set ``requires_grad``.

    ``volatile`` is accepted for compatibility with older training scripts; it
    now simply disables gradients.
    """
    if device is None and torch.cuda.is_available():
        device = torch.device("cuda")
    if device is not None:
        tensor = tensor.to(device, non_blocking=True)
    tensor.requires_grad_(False if volatile else requires_grad)
    return tensor
