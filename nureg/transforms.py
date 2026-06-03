"""Batch transforms that apply the same random operation to image/label tensors.

These transforms are useful for dense prediction tasks where an input image and
its target masks must receive identical crops and flips.
"""

from __future__ import annotations

import numbers
import random
from typing import Iterable, Sequence

import torch


def _is_tensor(value) -> bool:
    return isinstance(value, torch.Tensor)


class RandomCrop:
    """Crop every tensor in a sample to the same random spatial window."""

    def __init__(self, size: int | Sequence[int]):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            if len(size) != 2:
                raise ValueError("size must be an int or a (height, width) pair")
            self.size = (int(size[0]), int(size[1]))

    def __call__(self, tensors: Iterable):
        tensors = list(tensors)
        th, tw = self.size
        h = w = None

        for tensor in tensors:
            if not _is_tensor(tensor):
                continue
            if tensor.dim() < 2:
                raise ValueError("Tensor inputs must have at least 2 spatial dimensions")
            current_h, current_w = tensor.size(-2), tensor.size(-1)
            if h is None:
                h, w = current_h, current_w
            elif (current_h, current_w) != (h, w):
                raise ValueError(
                    f"All tensor inputs must share spatial size; got {(current_h, current_w)} and {(h, w)}"
                )

        if h is None or w is None:
            return tensors
        if h < th or w < tw:
            raise ValueError(f"Crop size {(th, tw)} is larger than tensor size {(h, w)}")
        if h == th and w == tw:
            return tensors

        y1 = random.randint(0, h - th)
        x1 = random.randint(0, w - tw)
        return [
            tensor[..., y1 : y1 + th, x1 : x1 + tw].contiguous()
            if _is_tensor(tensor)
            else tensor
            for tensor in tensors
        ]


class HalfCrop:
    """Randomly keep the left or right half of every tensor in a sample."""

    def __init__(self, size: int | Sequence[int] | None = None):
        if size is None:
            self.size = None
        elif isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            if len(size) != 2:
                raise ValueError("size must be None, an int, or a (height, width) pair")
            self.size = (int(size[0]), int(size[1]))

    def __call__(self, tensors: Iterable):
        tensors = list(tensors)
        first_tensor = next((tensor for tensor in tensors if _is_tensor(tensor)), None)
        if first_tensor is None:
            return tensors

        h, w = first_tensor.size(-2), first_tensor.size(-1)
        _, tw = self.size if self.size is not None else (h, w)
        tw_half = tw // 2
        if tw_half <= 0 or w < tw_half:
            raise ValueError(f"Cannot half-crop width {w} with target width {tw}")

        left_side = random.randint(0, 1)
        x1 = left_side * tw_half
        if x1 + tw_half > w:
            x1 = max(0, w - tw_half)

        return [
            tensor[..., x1 : x1 + tw_half].contiguous() if _is_tensor(tensor) else tensor
            for tensor in tensors
        ]


class RandomHorizontalFlip:
    """Flip every tensor in a sample horizontally with probability 0.5."""

    def __call__(self, tensors: Iterable):
        tensors = list(tensors)
        if random.random() >= 0.5:
            return tensors
        output = []
        for tensor in tensors:
            if not _is_tensor(tensor):
                output.append(tensor)
                continue
            indices = torch.arange(tensor.size(-1) - 1, -1, -1, device=tensor.device).long()
            output.append(tensor.index_select(-1, indices))
        return output


def augment_collate(batch, crop=None, halfcrop=None, flip=True):
    """Collate function for DataLoader with synchronized augmentation."""
    transforms = []
    if crop is not None:
        transforms.append(RandomCrop(crop))
    if halfcrop is not None:
        transforms.append(HalfCrop(halfcrop))
    if flip:
        transforms.append(RandomHorizontalFlip())
    def apply_transforms(sample):
        for transform in transforms:
            sample = transform(sample)
        return sample

    batch = [apply_transforms(sample) for sample in batch]
    return torch.utils.data.dataloader.default_collate(batch)
