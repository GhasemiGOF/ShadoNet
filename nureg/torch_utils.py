"""Small PyTorch helper functions."""

from __future__ import annotations

import numpy as np
import torch


def Indexflow(Totalnum, batch_size, random=True):
    """Yield index batches covering ``range(Totalnum)`` once."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    numberofchunk = int(Totalnum + batch_size - 1) // int(batch_size)
    totalIndx = np.arange(Totalnum).astype(int)
    if random is True:
        totalIndx = np.random.permutation(totalIndx)

    chunkstart = 0
    for chunkidx in range(numberofchunk):
        thisnum = min(batch_size, Totalnum - chunkidx * batch_size)
        thisInd = totalIndx[chunkstart : chunkstart + thisnum]
        chunkstart += thisnum
        yield thisInd


def to_variable(x, requires_grad=False, cuda=False, var=True):
    """Convert numpy arrays to tensors and optionally move to CUDA.

    ``var`` is kept for backward compatibility with older call sites where
    ``torch.autograd.Variable`` was used explicitly.
    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x.astype(np.float32))
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Expected numpy.ndarray or torch.Tensor, got {type(x)!r}")
    x.requires_grad_(requires_grad)
    if cuda:
        return x.cuda()
    return x


def to_device(src, ref=None, var=True, requires_grad=False):
    """Move ``src`` to the same device as ``ref`` when possible."""
    src = to_variable(src, requires_grad=requires_grad, var=var)
    if ref is None:
        return src
    if isinstance(ref, torch.device):
        return src.to(ref)
    if isinstance(ref, torch.Tensor):
        return src.to(ref.device)
    if hasattr(ref, "parameters"):
        try:
            return src.to(next(ref.parameters()).device)
        except StopIteration:
            return src
    return src
