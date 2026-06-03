"""
Generate SAM-shaped proximity labels (labels_postm / labels_negtm / labels_other).

Filtering is controlled by FilterConfig. Predefined strategies: no_sam, raw_sam,
sam_all, sam_area, sam_geom, sam_full, sam_cell_p20, sam_cell_p40,
sam_cell_p60, sam_cell_p80.

sam_all: every SAM mask is painted; no human centers/labels or fallback circles.

Example:
  python Gen_refactored.py datasets/NETnewClass_sam_full cuda:0 --strategy sam_full
  python Gen_refactored.py datasets/NETnewClass_sam_cell_p20 cuda:0 --strategy sam_cell_p20
  python Gen_refactored.py datasets/NETnewClass cuda:0 --use-area-filter --use-nesting-filter
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from scipy.io import loadmat, savemat


@dataclass
class FilterConfig:
    use_sam: bool = True
    use_area_filter: bool = False
    use_nesting_filter: bool = False
    use_center_filter: bool = False
    use_fallback_circles: bool = True
    use_annotations: bool = True
    max_area_frac: float = 0.02
    shape_fraction: float = 1.0
    shape_seed: int = 0


FILTER_STRATEGIES: dict[str, FilterConfig] = {
    "no_sam": FilterConfig(
        use_sam=False,
        use_fallback_circles=True,
    ),
    "raw_sam": FilterConfig(
        use_area_filter=False,
        use_nesting_filter=False,
        use_center_filter=False,
        use_fallback_circles=True,
    ),
    "sam_all": FilterConfig(
        use_area_filter=False,
        use_nesting_filter=False,
        use_center_filter=False,
        use_fallback_circles=False,
        use_annotations=False,
    ),
    "sam_area": FilterConfig(
        use_area_filter=True,
        use_nesting_filter=False,
        use_center_filter=False,
        use_fallback_circles=True,
    ),
    "sam_geom": FilterConfig(
        use_area_filter=True,
        use_nesting_filter=True,
        use_center_filter=False,
        use_fallback_circles=True,
    ),
    "sam_full": FilterConfig(
        use_area_filter=True,
        use_nesting_filter=True,
        use_center_filter=True,
        use_fallback_circles=True,
    ),
    "sam_cell_p20": FilterConfig(
        use_area_filter=True,
        use_nesting_filter=True,
        use_center_filter=True,
        use_fallback_circles=True,
        shape_fraction=0.20,
    ),
    "sam_cell_p40": FilterConfig(
        use_area_filter=True,
        use_nesting_filter=True,
        use_center_filter=True,
        use_fallback_circles=True,
        shape_fraction=0.40,
    ),
    "sam_cell_p60": FilterConfig(
        use_area_filter=True,
        use_nesting_filter=True,
        use_center_filter=True,
        use_fallback_circles=True,
        shape_fraction=0.60,
    ),
    "sam_cell_p80": FilterConfig(
        use_area_filter=True,
        use_nesting_filter=True,
        use_center_filter=True,
        use_fallback_circles=True,
        shape_fraction=0.80,
    ),
}


def extract_labels_and_centers(mat_path: str) -> Tuple[np.ndarray, np.ndarray]:
    mat_data = loadmat(mat_path)
    if "Labels" not in mat_data or "Centers" not in mat_data:
        raise KeyError(f"'Labels' or 'Centers' not found in {mat_path}")
    return np.array(mat_data["Labels"]), np.array(mat_data["Centers"])


def centers_as_nx2(centers: np.ndarray) -> np.ndarray:
    """Return centers as an ``N x 2`` array in ``(x, y)`` order."""
    centers = np.asarray(centers)
    if centers.size == 0:
        return np.empty((0, 2), dtype=float)
    if centers.ndim == 1:
        if centers.size != 2:
            raise ValueError(f"Expected center array with pairs, got shape {centers.shape}")
        return centers.reshape(1, 2)
    if centers.shape[0] == 2:
        return centers.T
    if centers.shape[1] != 2:
        raise ValueError(f"Expected centers with shape (N, 2) or (2, N), got {centers.shape}")
    return centers


def label_for_index(labels: np.ndarray, idx: int) -> int:
    """Return a scalar class label from MATLAB-style label arrays."""
    labels = np.asarray(labels)
    if labels.ndim == 1:
        return int(labels[idx])
    if labels.shape[0] == 1:
        return int(labels[0, idx])
    if labels.shape[1] == 1:
        return int(labels[idx, 0])
    return int(labels.reshape(-1)[idx])


def initialize_sam_model(
    checkpoint_path: str, model_type: str = "vit_h", device: str = "cuda:1"
):
    try:
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    except ImportError as exc:
        raise ImportError(
            "segment-anything is required when SAM generation is enabled. "
            "Install it or run with --strategy no_sam/--no-sam."
        ) from exc

    if model_type not in sam_model_registry:
        raise ValueError(f"Unknown SAM model_type {model_type!r}; choose from {list(sam_model_registry)}")
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    return SamAutomaticMaskGenerator(sam)


def run_sam_on_image(image_path: str, mask_generator) -> List[dict]:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return mask_generator.generate(image)


def _max_area_threshold(image_shape: Tuple[int, int], max_area_frac: float) -> float:
    if image_shape is not None:
        return max_area_frac * image_shape[0] * image_shape[1]
    return 5000.0


def _mask_is_contained_in(inner: np.ndarray, outer: np.ndarray) -> bool:
    """True if inner segmentation is fully covered by outer (and non-empty)."""
    inner_sum = int(inner.sum())
    if inner_sum == 0:
        return False
    return int(np.logical_and(inner, outer).sum()) == inner_sum


def _count_centers_in_segmentation(
    segmentation: np.ndarray, centers: np.ndarray
) -> int:
    count = 0
    h, w = segmentation.shape
    for center in centers:
        x = int(np.clip(center[0], 0, w - 1))
        y = int(np.clip(center[1], 0, h - 1))
        if segmentation[y, x]:
            count += 1
    return count


def _remove_nested_masks_numpy(masks: List[dict]) -> List[dict]:
    """Drop outer masks when another mask's segmentation is nested inside them."""
    n = len(masks)
    if n <= 1:
        return masks
    segs = [m["segmentation"].astype(bool) for m in masks]
    to_remove = set()
    for i in range(n):
        if i in to_remove:
            continue
        for j in range(n):
            if i == j or j in to_remove:
                continue
            if _mask_is_contained_in(segs[j], segs[i]) and segs[j].sum() < segs[i].sum():
                to_remove.add(i)
                break
    return [m for idx, m in enumerate(masks) if idx not in to_remove]


def refine_segments(
    masks: List[dict],
    centers: np.ndarray,
    labels: np.ndarray,
    image_shape: Optional[Tuple[int, int]] = None,
    config: Optional[FilterConfig] = None,
) -> List[dict]:
    """
    Refine SAM masks using optional area, nesting (segmentation overlap), and
    center-count filters.
    """
    cfg = config or FilterConfig()
    c = centers_as_nx2(centers)

    if cfg.use_area_filter:
        max_area = _max_area_threshold(image_shape, cfg.max_area_frac)
        masks = [m for m in masks if m["area"] <= max_area]

    if cfg.use_nesting_filter:
        masks = _remove_nested_masks_numpy(masks)

    if cfg.use_center_filter:
        refined = []
        for mask in masks:
            n_centers = _count_centers_in_segmentation(mask["segmentation"], c)
            if n_centers == 1:
                refined.append(mask)
        return refined

    return masks


def _pick_mask_for_center(masks: List[dict], center: np.ndarray) -> Optional[dict]:
    """Smallest-area mask whose segmentation contains the center (raw / geom modes)."""
    x = int(center[0])
    y = int(center[1])
    best = None
    best_area = None
    for mask in masks:
        seg = mask["segmentation"]
        if y < 0 or y >= seg.shape[0] or x < 0 or x >= seg.shape[1]:
            continue
        if not seg[y, x]:
            continue
        area = mask["area"]
        if best is None or area < best_area:
            best = mask
            best_area = area
    return best


def _stable_image_seed(base_seed: int, image_path: str) -> int:
    image_key = os.path.join(
        os.path.basename(os.path.dirname(image_path)), os.path.basename(image_path)
    )
    digest = hashlib.sha256(f"{base_seed}:{image_key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def _selected_shape_positions(
    n_candidates: int, shape_fraction: float, seed: int
) -> set[int]:
    if n_candidates <= 0 or shape_fraction <= 0:
        return set()
    if shape_fraction >= 1:
        return set(range(n_candidates))
    n_shape = int(np.floor(shape_fraction * n_candidates + 0.5))
    n_shape = max(0, min(n_candidates, n_shape))
    if n_shape == 0:
        return set()
    rng = np.random.default_rng(seed)
    return set(int(i) for i in rng.choice(n_candidates, size=n_shape, replace=False))


def _collect_shape_candidates(
    refined_masks: List[dict],
    centers: np.ndarray,
    labels: np.ndarray,
    image_shape: Tuple[int, int],
    config: FilterConfig,
) -> List[dict]:
    label_map = {1: "positive", 2: "negative", 3: "other"}
    c = centers_as_nx2(centers)
    candidates = []
    assigned_indices = set()

    if config.use_center_filter:
        for mask in refined_masks:
            segmentation = mask["segmentation"]
            for idx, center in enumerate(c):
                if idx in assigned_indices:
                    continue
                x = int(np.clip(center[0], 0, image_shape[1] - 1))
                y = int(np.clip(center[1], 0, image_shape[0] - 1))
                if not (0 <= y < segmentation.shape[0] and 0 <= x < segmentation.shape[1]):
                    continue
                if not segmentation[y, x]:
                    continue
                map_key = label_map.get(label_for_index(labels, idx))
                if map_key is None:
                    continue
                cX, cY = _segmentation_centroid(segmentation)
                candidates.append(
                    {
                        "idx": idx,
                        "map_key": map_key,
                        "segmentation": segmentation,
                        "cX": cX,
                        "cY": cY,
                    }
                )
                assigned_indices.add(idx)
                break
    else:
        for idx, center in enumerate(c):
            mask = _pick_mask_for_center(refined_masks, center)
            if mask is None:
                continue
            map_key = label_map.get(label_for_index(labels, idx))
            if map_key is None:
                continue
            segmentation = mask["segmentation"]
            cX, cY = _segmentation_centroid(segmentation)
            candidates.append(
                {
                    "idx": idx,
                    "map_key": map_key,
                    "segmentation": segmentation,
                    "cX": cX,
                    "cY": cY,
                }
            )

    return candidates


def regression_decay(distance, d=15, alpha=3, min_border_value=30):
    if distance <= d:
        value = (np.exp(alpha * (1 - (distance / d))) - 1) / (np.exp(alpha) - 1)
        scaled_value = value * (255 - min_border_value) + min_border_value
        return scaled_value
    return min_border_value


def _regression_decay_array(distances, d=15, alpha=3, min_border_value=30):
    values = np.full(distances.shape, min_border_value, dtype=np.float32)
    inside = distances <= d
    if np.any(inside):
        values[inside] = (
            (np.exp(alpha * (1 - (distances[inside] / d))) - 1)
            / (np.exp(alpha) - 1)
            * (255 - min_border_value)
            + min_border_value
        )
    return values


def _scaled_proximity_params(image_shape: Tuple[int, int], d=15, min_border_value=30):
    reference_size = 500
    min_dim = min(image_shape[0], image_shape[1])
    scale_factor = min_dim / reference_size
    d_scaled = max(1, int(round(d * scale_factor)))
    d_missed_scaled = max(1, int(round(8 * scale_factor)))
    min_border_value_scaled = max(0, int(round(min_border_value * scale_factor)))
    min_border_value_missed_scaled = max(0, int(round(50 * scale_factor)))
    return (
        scale_factor,
        d_scaled,
        d_missed_scaled,
        min_border_value_scaled,
        min_border_value_missed_scaled,
    )


def _paint_segmentation_proximity(
    proximity_maps: dict,
    map_key: str,
    segmentation: np.ndarray,
    cX: int,
    cY: int,
    d_found: int,
    alpha_found: int,
    min_border_value_found: int,
):
    ys, xs = np.nonzero(segmentation)
    if ys.size == 0:
        return
    distances = np.sqrt((xs - cX) ** 2 + (ys - cY) ** 2)
    values = _regression_decay_array(
        distances, d_found, alpha_found, min_border_value_found
    )
    target = proximity_maps[map_key]
    target[ys, xs] = np.maximum(target[ys, xs], values)


def _paint_fallback_circle(
    proximity_maps: dict,
    map_key: str,
    cX: int,
    cY: int,
    image_shape: Tuple[int, int],
    d_missed: int,
    alpha_missed: int,
    min_border_value_missed: int,
):
    for dy in range(-d_missed, d_missed + 1):
        for dx in range(-d_missed, d_missed + 1):
            distance = np.sqrt(dx ** 2 + dy ** 2)
            if distance <= d_missed:
                ny, nx = cY + dy, cX + dx
                if 0 <= ny < image_shape[0] and 0 <= nx < image_shape[1]:
                    value = regression_decay(
                        distance, d_missed, alpha_missed, min_border_value_missed
                    )
                    proximity_maps[map_key][ny, nx] = max(
                        proximity_maps[map_key][ny, nx], value
                    )


def create_proximity_maps_sam_all(
    masks: List[dict],
    image_shape: Tuple[int, int],
    d=15,
    alpha=3,
    min_border_value=30,
) -> dict:
    """
    Paint every SAM mask using proximity from its segmentation centroid.
    No human centers/labels; the same combined map is written to all three channels.
    """
    (
        scale_factor,
        d_found,
        _,
        min_border_value_found,
        _,
    ) = _scaled_proximity_params(image_shape, d, min_border_value)

    print(f"  Image size: {image_shape}, Scale factor: {scale_factor:.2f}")
    print(f"  sam_all: painting {len(masks)} masks (no annotation filtering)")

    combined = np.zeros(image_shape, dtype=np.float32)
    for mask in masks:
        segmentation = mask["segmentation"]
        cX, cY = _segmentation_centroid(segmentation)
        _paint_segmentation_proximity(
            {"_": combined},
            "_",
            segmentation,
            cX,
            cY,
            d_found,
            alpha,
            min_border_value_found,
        )

    combined = np.clip(combined, 0, 255).astype(np.uint8)
    return {
        "positive": combined.copy(),
        "negative": combined.copy(),
        "other": combined.copy(),
    }


def _sam_mask_centers_array(masks: List[dict]) -> Tuple[np.ndarray, np.ndarray]:
    """Centers (2, N) and Labels (1, N) from SAM mask centroids."""
    if not masks:
        return np.zeros((2, 0), dtype=np.float64), np.zeros((1, 0), dtype=np.uint8)
    xs, ys = [], []
    for mask in masks:
        cX, cY = _segmentation_centroid(mask["segmentation"])
        xs.append(float(cX))
        ys.append(float(cY))
    centers = np.vstack([np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)])
    labels = np.ones((1, centers.shape[1]), dtype=np.uint8)
    return centers, labels


def create_proximity_maps_no_sam(
    centers: np.ndarray,
    labels: np.ndarray,
    image_shape: Tuple[int, int],
    d=15,
    alpha=3,
    min_border_value=30,
) -> dict:
    _, d_scaled, _, min_border_scaled, _ = _scaled_proximity_params(
        image_shape, d, min_border_value
    )
    proximity_maps = {
        "positive": np.zeros(image_shape, dtype=np.float32),
        "negative": np.zeros(image_shape, dtype=np.float32),
        "other": np.zeros(image_shape, dtype=np.float32),
    }
    label_map = {1: "positive", 2: "negative", 3: "other"}
    c = centers_as_nx2(centers)
    for idx, center in enumerate(c):
        map_key = label_map.get(label_for_index(labels, idx))
        if map_key is None:
            continue
        _paint_fallback_circle(
            proximity_maps,
            map_key,
            int(center[0]),
            int(center[1]),
            image_shape,
            d_scaled,
            alpha,
            min_border_scaled,
        )
    for key in proximity_maps:
        proximity_maps[key] = np.clip(proximity_maps[key], 0, 255).astype(np.uint8)
    return proximity_maps


def create_proximity_maps(
    refined_masks: List[dict],
    centers: np.ndarray,
    labels: np.ndarray,
    image_shape: Tuple[int, int],
    config: Optional[FilterConfig] = None,
    shape_seed: int = 0,
    d=15,
    alpha=3,
    min_border_value=30,
) -> dict:
    cfg = config or FilterConfig()
    (
        scale_factor,
        d_found,
        d_missed,
        min_border_value_found,
        min_border_value_missed,
    ) = _scaled_proximity_params(image_shape, d, min_border_value)

    print(f"  Image size: {image_shape}, Scale factor: {scale_factor:.2f}")
    print(f"  Scaled parameters - d_found: {d_found}, d_missed: {d_missed}")

    proximity_maps = {
        "positive": np.zeros(image_shape, dtype=np.float32),
        "negative": np.zeros(image_shape, dtype=np.float32),
        "other": np.zeros(image_shape, dtype=np.float32),
    }
    label_map = {1: "positive", 2: "negative", 3: "other"}
    c = centers_as_nx2(centers)
    assigned_centers = set()

    shape_candidates = _collect_shape_candidates(
        refined_masks, centers, labels, image_shape, cfg
    )
    selected_shape_positions = _selected_shape_positions(
        len(shape_candidates), cfg.shape_fraction, shape_seed
    )
    if cfg.shape_fraction < 1:
        print(
            f"  Shape fraction: {cfg.shape_fraction:.0%} "
            f"({len(selected_shape_positions)}/{len(shape_candidates)} filtered cells)"
        )

    for pos, candidate in enumerate(shape_candidates):
        if pos not in selected_shape_positions:
            continue
        _paint_segmentation_proximity(
            proximity_maps,
            candidate["map_key"],
            candidate["segmentation"],
            candidate["cX"],
            candidate["cY"],
            d_found,
            alpha,
            min_border_value_found,
        )
        assigned_centers.add(candidate["idx"])

    if cfg.use_fallback_circles:
        circle_radius = d_found if cfg.shape_fraction < 1 else d_missed
        circle_min_border = (
            min_border_value_found
            if cfg.shape_fraction < 1
            else min_border_value_missed
        )
        for idx, center in enumerate(c):
            if idx in assigned_centers:
                continue
            label = label_for_index(labels, idx)
            map_key = label_map.get(label)
            if map_key is None:
                continue
            x = int(np.clip(center[0], 0, image_shape[1] - 1))
            y = int(np.clip(center[1], 0, image_shape[0] - 1))
            _paint_fallback_circle(
                proximity_maps,
                map_key,
                x,
                y,
                image_shape,
                circle_radius,
                alpha,
                circle_min_border,
            )

    for key in proximity_maps:
        proximity_maps[key] = np.clip(proximity_maps[key], 0, 255).astype(np.uint8)
    return proximity_maps


def save_proximity_map(proximity_map: np.ndarray, output_folder: str, label_name: str):
    os.makedirs(output_folder, exist_ok=True)
    out_path = os.path.join(output_folder, label_name)
    print(f"  Saving proximity map: {out_path}")
    cv2.imwrite(out_path, proximity_map)


def _segmentation_centroid(segmentation: np.ndarray) -> Tuple[int, int]:
    M = cv2.moments(segmentation.astype(np.uint8))
    if M["m00"] != 0:
        return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
    ys, xs = np.where(segmentation)
    if len(xs) == 0:
        return 0, 0
    return int(xs.mean()), int(ys.mean())


def _update_center_in_array(
    centers: np.ndarray, idx: int, cX: int, cY: int
) -> np.ndarray:
    new_centers = centers.copy()
    if centers.shape[0] == 2:
        new_centers[0, idx] = cX
        new_centers[1, idx] = cY
    else:
        new_centers[idx, 0] = cX
        new_centers[idx, 1] = cY
    return new_centers


def process_image_and_mat(
    image_path: str,
    mat_path: str,
    dataset_root: str,
    config: FilterConfig,
    mask_generator=None,
    update_mat_centers: bool = True,
):
    print(f"\nProcessing image: {image_path}")
    print(f"  Mat: {mat_path}")
    print(f"  Filters: {asdict(config)}")

    image = cv2.imread(image_path)
    if image is None:
        print(f"  Could not load image: {image_path}")
        return

    image_shape = image.shape[:2]
    shape_seed = _stable_image_seed(config.shape_seed, image_path)
    labels = None
    centers = None
    if config.use_annotations or not config.use_sam:
        labels, centers = extract_labels_and_centers(mat_path)

    if not config.use_sam:
        print("  no_sam: circle proximity maps only")
        proximity_maps = create_proximity_maps_no_sam(centers, labels, image_shape)
        refined_masks = []
    else:
        if mask_generator is None:
            raise ValueError("mask_generator required when use_sam=True")
        print("  Running SAM...")
        masks = run_sam_on_image(image_path, mask_generator)
        if config.use_annotations:
            print("  Refining segments...")
            refined_masks = refine_segments(
                masks, centers, labels, image_shape=image_shape, config=config
            )
            print(f"  Masks after refine: {len(refined_masks)}")
            proximity_maps = create_proximity_maps(
                refined_masks,
                centers,
                labels,
                image_shape,
                config=config,
                shape_seed=shape_seed,
            )
        else:
            refined_masks = masks
            print(f"  sam_all: {len(refined_masks)} masks (unfiltered, no annotations)")
            proximity_maps = create_proximity_maps_sam_all(
                refined_masks, image_shape
            )

    subfolder = os.path.basename(os.path.dirname(image_path))
    label_folders = {
        "positive": os.path.join(dataset_root, "labels_postm", subfolder),
        "negative": os.path.join(dataset_root, "labels_negtm", subfolder),
        "other": os.path.join(dataset_root, "labels_other", subfolder),
    }
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    label_name = f"{base_name}_label.png"

    for key, prox_map in proximity_maps.items():
        save_proximity_map(prox_map, label_folders[key], label_name)

    if update_mat_centers and config.use_sam:
        print(f"  Updating mat centers: {mat_path}")
        mat_data = loadmat(mat_path)
        if config.use_annotations:
            new_centers = centers.copy()
            shape_candidates = _collect_shape_candidates(
                refined_masks, centers, labels, image_shape, config
            )
            selected_shape_positions = _selected_shape_positions(
                len(shape_candidates), config.shape_fraction, shape_seed
            )
            for pos in selected_shape_positions:
                candidate = shape_candidates[pos]
                new_centers = _update_center_in_array(
                    new_centers,
                    candidate["idx"],
                    candidate["cX"],
                    candidate["cY"],
                )
            mat_data["Centers"] = new_centers
        else:
            sam_centers, sam_labels = _sam_mask_centers_array(refined_masks)
            mat_data["Centers"] = sam_centers
            mat_data["Labels"] = sam_labels
        savemat(mat_path, mat_data)

    print(f"  Done: {label_name}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def process_dataset(
    dataset_root: str,
    config: FilterConfig,
    checkpoint_path: str,
    device: str,
    splits: Sequence[str] = ("train", "val", "test"),
    update_mat_centers: bool = True,
):
    dataset_root = os.path.abspath(dataset_root)
    images_root = os.path.join(dataset_root, "images")
    mats_root = os.path.join(dataset_root, "mats")

    mask_generator = None
    if config.use_sam:
        mask_generator = initialize_sam_model(checkpoint_path, device=device)

    for split in splits:
        images_dir = os.path.join(images_root, split)
        mats_dir = os.path.join(mats_root, split)
        if not os.path.isdir(images_dir):
            print(f"Skip split (no images): {images_dir}")
            continue
        if not os.path.isdir(mats_dir):
            print(f"Skip split (no mats): {mats_dir}")
            continue

        image_files = sorted(
            f
            for f in os.listdir(images_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
        )
        for image_file in image_files:
            image_path = os.path.join(images_dir, image_file)
            mat_path = os.path.join(
                mats_dir, f"{os.path.splitext(image_file)[0]}_withcontour.mat"
            )
            if not os.path.isfile(mat_path):
                print(f"Missing mat: {mat_path}")
                continue
            process_image_and_mat(
                image_path,
                mat_path,
                dataset_root,
                config,
                mask_generator=mask_generator,
                update_mat_centers=update_mat_centers,
            )


def config_from_args(args: argparse.Namespace) -> FilterConfig:
    if args.strategy:
        if args.strategy not in FILTER_STRATEGIES:
            raise ValueError(
                f"Unknown strategy {args.strategy!r}. "
                f"Choose from: {list(FILTER_STRATEGIES)}"
            )
        cfg = replace(FILTER_STRATEGIES[args.strategy])
    else:
        cfg = FilterConfig()

    if args.use_area_filter:
        cfg.use_area_filter = True
    if args.no_area_filter:
        cfg.use_area_filter = False
    if args.use_nesting_filter:
        cfg.use_nesting_filter = True
    if args.no_nesting_filter:
        cfg.use_nesting_filter = False
    if args.use_center_filter:
        cfg.use_center_filter = True
    if args.no_center_filter:
        cfg.use_center_filter = False
    if args.use_fallback_circles:
        cfg.use_fallback_circles = True
    if args.no_fallback_circles:
        cfg.use_fallback_circles = False
    if args.no_sam:
        cfg.use_sam = False
    if args.use_annotations:
        cfg.use_annotations = True
    if args.no_annotations:
        cfg.use_annotations = False
    if args.max_area_frac is not None:
        cfg.max_area_frac = args.max_area_frac
    if args.shape_fraction is not None:
        if not 0 <= args.shape_fraction <= 1:
            raise ValueError("--shape-fraction must be between 0 and 1")
        cfg.shape_fraction = args.shape_fraction
    cfg.shape_seed = args.shape_seed
    return cfg


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate SAM proximity label datasets.")
    p.add_argument(
        "dataset_root",
        help="Dataset root (images/, mats/, labels_* written here)",
    )
    p.add_argument(
        "device",
        nargs="?",
        default="cuda:0",
        help="Torch device for SAM (ignored for no_sam)",
    )
    p.add_argument(
        "--strategy",
        choices=list(FILTER_STRATEGIES),
        help="Predefined filter preset",
    )
    p.add_argument(
        "--checkpoint",
        default="sam_vit_h_4b8939.pth",
        help="SAM ViT-H checkpoint path",
    )
    p.add_argument("--splits", nargs="*", default=["train", "val", "test"])
    p.add_argument("--max-area-frac", type=float, default=None)
    p.add_argument(
        "--shape-fraction",
        type=float,
        default=None,
        help="Fraction of filtered matched cells to paint with SAM shapes; remaining cells use circles.",
    )
    p.add_argument(
        "--shape-seed",
        type=int,
        default=0,
        help="Base seed for deterministic per-image shaped-cell selection.",
    )

    g = p.add_mutually_exclusive_group()
    g.add_argument("--use-area-filter", action="store_true")
    g.add_argument("--no-area-filter", action="store_true")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--use-nesting-filter", action="store_true")
    g.add_argument("--no-nesting-filter", action="store_true")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--use-center-filter", action="store_true")
    g.add_argument("--no-center-filter", action="store_true")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--use-fallback-circles", action="store_true")
    g.add_argument("--no-fallback-circles", action="store_true")
    p.add_argument("--no-sam", action="store_true", help="Circle labels only")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--use-annotations", action="store_true")
    g.add_argument(
        "--no-annotations",
        action="store_true",
        help="Paint all SAM masks; ignore human centers/labels",
    )
    p.add_argument(
        "--no-update-mat-centers",
        action="store_true",
        help="Do not write Centers back to .mat files",
    )
    return p


def main(argv: Optional[Sequence[str]] = None):
    args = build_arg_parser().parse_args(argv)
    cfg = config_from_args(args)

    if cfg.use_sam and not os.path.isfile(args.checkpoint):
        print(f"SAM checkpoint not found: {args.checkpoint}", file=sys.stderr)
        sys.exit(1)

    print(f"Dataset: {args.dataset_root}")
    print(f"Config: {json.dumps(asdict(cfg), indent=2)}")

    process_dataset(
        args.dataset_root,
        cfg,
        checkpoint_path=args.checkpoint,
        device=args.device,
        splits=args.splits,
        update_mat_centers=not args.no_update_mat_centers,
    )


if __name__ == "__main__":
    main()
