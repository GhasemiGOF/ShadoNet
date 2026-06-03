"""Train a fully convolutional nucleus classification model."""

from __future__ import annotations

import logging
import os
import random
from collections import deque

import click
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy.ndimage import distance_transform_edt as edt
from shapely.geometry import Polygon

from nureg.tools.util import make_variable
from nureg.transforms import augment_collate
from nureg.util import safe_load_state_dict

try:
    from tensorboardX import SummaryWriter
except ImportError:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        class SummaryWriter:  # type: ignore[no-redef]
            """No-op fallback when TensorBoard is not installed."""

            def __init__(self, *args, **kwargs):
                pass

            def add_scalar(self, *args, **kwargs):
                pass

            def close(self):
                pass


BCD_DATASETS = {
    'BCD', 'BCDSam', 'BCDSam_fold_1', 'BCDSam_fold_2', 'BCDSam_fold_3',
    'BCDSam_fold_4', 'BCDSam_fold_5', 'BCD_fold_1', 'BCD_fold_2',
    'BCD_fold_3', 'BCD_fold_4', 'BCD_fold_5',
}
NET_DATASETS = {
    'NETnewClass', 'NETnewClassSam', 'NETnewClass_mix_p20', 'NETnewClass_mix_p40',
    'NETnewClass_mix_p60', 'NETnewClass_mix_p80', 'NETnewClass_sam_cell_p20',
    'NETnewClass_sam_cell_p40', 'NETnewClass_sam_cell_p60', 'NETnewClass_sam_cell_p80',
    'NETnewClass_no_sam', 'NETnewClass_raw_sam', 'NETnewClass_sam_area',
    'NETnewClass_sam_geom', 'NETnewClass_sam_full', 'NETnewClass_sam_unfiltered',
    'NETnewClass_sam_overlap', 'NETnewClass256', 'NETnewClassSam256',
    'NETnewClassSam_fold_1', 'NETnewClassSam_fold_2', 'NETnewClassSam_fold_3',
    'NETnewClassSam_fold_4', 'NETnewClassSam_fold_5', 'NETnewClassSam_fold_6',
    'NETnewClass_fold_1', 'NETnewClass_fold_2', 'NETnewClass_fold_3',
    'NETnewClass_fold_4', 'NETnewClass_fold_5', 'NETnewClass_fold_6',
}
PNET_DATASETS = {
    'PNET', 'PNET11', 'PNETSam', 'PNETSam_fold_1', 'PNETSam_fold_2',
    'PNETSam_fold_3', 'PNETSam_fold_4', 'PNETSam_fold_5', 'PNET_fold_1',
    'PNET_fold_2', 'PNET_fold_3', 'PNET_fold_4', 'PNET_fold_5',
}
SHIDC_BARE_DATASETS = {'SHIDC_bare', 'SHIDC_bare_SAM'}
SHIDC_DATASETS = {
    'SHIDC500', 'SHIDCSam', 'SHIDC500Sam', 'SHIDC256', 'SHIDC256Sam',
    'SHIDCSam_fold_1', 'SHIDCSam_fold_2', 'SHIDCSam_fold_3',
    'SHIDCSam_fold_4', 'SHIDCSam_fold_5', 'SHIDC_fold_1', 'SHIDC_fold_2',
    'SHIDC_fold_3', 'SHIDC_fold_4', 'SHIDC_fold_5',
}
PANNUKE_DATASETS = {
    'PanNuke', 'PanNukeSam', 'PanNukeBreast', 'PanNukeBreastSam',
    'PanNukeBreast256', 'PanNukeBreast256Sam',
}


def _load_project_components(model_name: str):
    """Load project-specific data/model registries only when training starts."""
    try:
        from nureg.data.data_loader import get_fcn_dataset as get_dataset
        from nureg.models import get_model
        from nureg.models.models import models
    except ImportError as exc:
        raise click.ClickException(
            "Missing nureg.data or nureg.models modules. Add the project data/model "
            "package to this repository before running training."
        ) from exc

    if model_name not in models:
        raise click.ClickException(
            f"Unknown model {model_name!r}. Available models: {sorted(models.keys())}"
        )
    return get_dataset, get_model


class HausdorffDTLoss(nn.Module):
    def __init__(self, alpha=2.0, **kwargs):
        super(HausdorffDTLoss, self).__init__()
        self.alpha = alpha

    @torch.no_grad()
    def distance_field(self, img: np.ndarray) -> np.ndarray:
        field = np.zeros_like(img)
        for batch in range(len(img)):
            fg_mask = img[batch] > 0.5
            if fg_mask.any():
                bg_mask = ~fg_mask
                fg_dist = edt(fg_mask)
                bg_dist = edt(bg_mask)
                field[batch] = fg_dist + bg_dist
        return field

    def forward(self, pred: torch.Tensor, target: torch.Tensor, debug=False) -> torch.Tensor:
        assert pred.dim() in (4, 5), "Only 2D and 3D supported"
        assert pred.dim() == target.dim(), "Prediction and target must have the same dimensions"

        pred_dt = torch.from_numpy(
            self.distance_field(pred.detach().cpu().numpy())).float().to(pred.device)
        target_dt = torch.from_numpy(
            self.distance_field(target.detach().cpu().numpy())).float().to(target.device)

        pred_error = (pred - target) ** 2
        distance = pred_dt ** self.alpha + target_dt ** self.alpha

        dt_field = pred_error * distance
        loss = dt_field.mean()

        if debug:
            return (
                loss.cpu().numpy(),
                (
                    dt_field.cpu().numpy()[0, 0],
                    pred_error.cpu().numpy()[0, 0],
                    distance.cpu().numpy()[0, 0],
                    pred_dt.cpu().numpy()[0, 0],
                    target_dt.cpu().numpy()[0, 0],
                ),
            )
        else:
            return loss


def create_binary_mask_from_single_contour(shape, contour, offset=(0, 0)):
    mask = np.zeros(shape, dtype=np.uint8)

    contour = np.array(contour, dtype=np.int32)
    if contour.ndim == 2:
        contour = contour.reshape((-1, 1, 2))

    # Offset the contour so it fits in the canvas
    contour -= np.array(offset, dtype=np.int32).reshape(1, 1, 2)

    cv2.drawContours(mask, [contour], -1, color=1, thickness=cv2.FILLED)
    return mask


def get_optimal_shape(corners):
    x1 = min(c[0] for c in corners)
    y1 = min(c[1] for c in corners)
    x2 = max(c[2] for c in corners)
    y2 = max(c[3] for c in corners)
    width = int(np.ceil(x2 - x1))
    height = int(np.ceil(y2 - y1))
    return (height, width), (int(x1), int(y1))  # Shape + offset


def enclosing_box(corners1, corners2, enclosing_type="smallest"):
    """ Compute the enclosing box size for IoU calculations """
    min_x = min(np.min(corners1[:, 0]), np.min(corners2[:, 0]))
    max_x = max(np.max(corners1[:, 0]), np.max(corners2[:, 0]))
    min_y = min(np.min(corners1[:, 1]), np.min(corners2[:, 1]))
    max_y = max(np.max(corners1[:, 1]), np.max(corners2[:, 1]))

    w = max_x - min_x
    h = max_y - min_y
    return w, h


def find_bounding_box_and_center(contour):
    try:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        # box = np.int0(box)
        box = box.astype(np.intp)
        angle = rect[2] * np.pi / 180  # Convert angle to radians
        # Use the contour's moments to calculate the centroid
        M = cv2.moments(contour)
        if M['m00'] != 0:
            center = (M['m10'] / M['m00'], M['m01'] / M['m00'])
        else:
            # Fallback to the rectangle center if moments fail
            center = rect[0]
        # Compute the bounding box in SIoU format (x1, y1, x2, y2)
        x1, y1 = np.min(box[:, 0]), np.min(box[:, 1])  # Top-left corner
        x2, y2 = np.max(box[:, 0]), np.max(box[:, 1])  # Bottom-right corner
        x, y = rect[0]  # Extract center
        w, h = rect[1]  # Extract width & height
        diou_input = torch.tensor([x, y, w, h, angle], dtype=torch.float32)
        siou_input = torch.tensor([x1, y1, x2, y2], dtype=torch.float32)
        return box, center, siou_input, diou_input
    except ValueError as e:
        print(f"Error in shape loss calculation: {e}")
        return None, None, None, None


def tensor_to_image(tensor):
    """Convert a tensor/array channel to an 8-bit BGR image for OpenCV."""
    tensor = np.asarray(tensor).squeeze()
    if tensor.ndim == 3:
        tensor = np.transpose(tensor, (1, 2, 0))
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-10)
    tensor = (tensor * 255).astype(np.uint8)
    if tensor.ndim == 2:
        return cv2.merge([tensor] * 3)
    return tensor


def polygon_area(corners):
    """ Calculate the area of a polygon given its corners. """
    x = corners[:, 0]
    y = corners[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def intersection_area(box1_corners, box2_corners):
    """
    Compute the intersection area of two rotated rectangles using a convex hull algorithm.
    """

    poly1 = Polygon(box1_corners)
    poly2 = Polygon(box2_corners)

    if not poly1.is_valid or not poly2.is_valid:
        return 0.0

    intersection = poly1.intersection(poly2)

    if not intersection.is_empty:
        return intersection.area
    return 0.0


def box2corners(x, y, w, h, alpha):
    """
    Convert box parameters (x, y, w, h, alpha) to four box corners.
    """
    x4 = np.array([0.5, -0.5, -0.5, 0.5]) * w
    y4 = np.array([0.5, 0.5, -0.5, -0.5]) * h
    corners = np.stack([x4, y4], axis=1)
    sin = np.sin(alpha)
    cos = np.cos(alpha)
    R = np.array([[cos, -sin], [sin, cos]])
    rotated = corners @ R.T
    rotated[:, 0] += x
    rotated[:, 1] += y
    return rotated


class SIoU(nn.Module):
    def __init__(self, x1y1x2y2=True, eps=1e-7):
        super(SIoU, self).__init__()
        self.x1y1x2y2 = x1y1x2y2
        self.eps = eps

    def forward(self, box3, box4):
        box3 = torch.as_tensor(box3, dtype=torch.float32)
        box4 = torch.as_tensor(box4, dtype=torch.float32, device=box3.device)

        box1_np = box3.detach().cpu().numpy()
        box2_np = box4.detach().cpu().numpy()

        box1_corners = box2corners(*box1_np)
        box2_corners = box2corners(*box2_np)

        area1 = polygon_area(box1_corners)
        area2 = polygon_area(box2_corners)
        inter_area = intersection_area(box1_corners, box2_corners)
        union_area = area1 + area2 - inter_area

        iou = torch.as_tensor(inter_area / (union_area + 1e-6), dtype=box3.dtype, device=box3.device)

        ###
        # Extract center, width, and height
        x1, y1, w1, h1, angle1 = box3
        x2, y2, w2, h2, angle2 = box4
        b1_x1 = x1 - w1 / 2
        b1_x2 = x1 + w1 / 2
        b1_y1 = y1 - h1 / 2
        b1_y2 = y1 + h1 / 2

        b2_x1 = x2 - w2 / 2
        b2_x2 = x2 + w2 / 2
        b2_y1 = y2 - h2 / 2
        b2_y2 = y2 + h2 / 2

        # Enclosing box width and height (axis-aligned)
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)

        s_cw = x2 - x1
        s_ch = y2 - y1

        sigma = torch.pow(s_cw ** 2 + s_ch ** 2, 0.5) + self.eps
        sin_alpha_1 = torch.abs(s_cw) / sigma
        sin_alpha_2 = torch.abs(s_ch) / sigma
        threshold = pow(2, 0.5) / 2
        sin_alpha = torch.where(sin_alpha_1 > threshold,
                                sin_alpha_2, sin_alpha_1)

        angle_cost = 1 - 2 * \
            torch.pow(torch.sin(torch.arcsin(sin_alpha) - np.pi / 4), 2)
        rho_x = (s_cw / (cw + self.eps)) ** 2
        rho_y = (s_ch / (ch + self.eps)) ** 2
        gamma = 2 - angle_cost

        distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)
        # here i can use enclosing
        omiga_w = torch.abs(w1 - w2) / (torch.max(w1, w2) + self.eps)
        omiga_h = torch.abs(h1 - h2) / (torch.max(h1, h2)+self.eps)

        shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + \
            torch.pow(1 - torch.exp(-1 * omiga_h), 4)
        return 1 - (iou + 0.5 * (distance_cost + shape_cost))


def process_image(image_path, min_contour_area=100, max_contour_area_ratio=0.8):
    try:
        image = image_path
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh_img = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(
            thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        height, width = image.shape[:2]
        max_contour_area = height * width * max_contour_area_ratio
        results = []
        for contour in contours:
            contour_area = cv2.contourArea(contour)
            if contour_area < min_contour_area or contour_area > max_contour_area:
                continue
            box, center, siou_input, diou_input = find_bounding_box_and_center(
                contour)
            if box is None or center is None:
                continue

            results.append((center, siou_input, diou_input, contour))
        return results
    except ValueError as e:
        print(f"Error in image processing: {e}")
        return []


def lossall(gt_results, pred_results, threshold=7):
    siou_loss_fn = SIoU()
    HD_dt = HausdorffDTLoss()  # Initialize Hausdorff Distance
    total_loss, matches = 0, 0
    total_loss_H = 0
    i = 0

    for gt_center, gt_siou_input, gt_diou_input, gt_countor in gt_results:
        best_pred = None
        best_pred2 = None
        best_pred3 = None

        min_distance = float("inf")

        for pred_center, pred_siou_input, pred_diou_input, pred_countor in pred_results:
            distance = np.linalg.norm(
                np.array(gt_center) - np.array(pred_center))

            if distance <= threshold and distance < min_distance:
                min_distance = distance
                best_pred = pred_siou_input
                best_pred2 = pred_diou_input
                best_pred3 = pred_countor

        if best_pred is not None:
            gt_box_tensor = gt_siou_input.clone().detach() if isinstance(gt_siou_input,
                                                                         torch.Tensor) else torch.as_tensor(gt_diou_input, dtype=torch.float32).clone().detach()
            pred_box_tensor = best_pred.clone().detach() if isinstance(best_pred,
                                                                       torch.Tensor) else torch.as_tensor(best_pred, dtype=torch.float32).clone().detach()
            gt_box_tensor2 = gt_diou_input.clone().detach() if isinstance(gt_diou_input,
                                                                          torch.Tensor) else torch.as_tensor(gt_diou_input, dtype=torch.float32).clone().detach()
            pred_box_tensor2 = best_pred2.clone().detach() if isinstance(best_pred2,
                                                                         torch.Tensor) else torch.as_tensor(best_pred2, dtype=torch.float32).clone().detach()

            shape, offset = get_optimal_shape(
                [gt_siou_input.tolist(), best_pred.tolist()])
            gt_mask = create_binary_mask_from_single_contour(
                shape, gt_countor, offset)
            pred_mask = create_binary_mask_from_single_contour(
                shape, best_pred3, offset)

            # pred_mask_img = (pred_mask * 255).astype(np.uint8)
            # gt_mask_img = (gt_mask * 255).astype(np.uint8)

            # Image.fromarray(pred_mask_img).save(f"./G50pred1/mask_pred_{i}.png")
            # Image.fromarray(gt_mask_img).save(f"./G50true1/mask_true_{i}.png")
            # i=i+1

            # Convert masks to PyTorch tensors
            gt_tensor = torch.from_numpy(np.array([[gt_mask]])).float()
            pred_tensor = torch.from_numpy(np.array([[pred_mask]])).float()

            total_loss += siou_loss_fn(gt_box_tensor2, pred_box_tensor2).item()
            if gt_tensor is not None and torch.sum(gt_tensor) > 0:
                loss_dt, fields = HD_dt.forward(
                    gt_tensor, pred_tensor, debug=True)
            else:
                loss_dt = 0

            loss_dt = min(loss_dt, 100)  # Clip huge spikes
            total_loss_H += loss_dt
            matches += 1

    if matches == 0:
        return 0, 0
    return total_loss/matches, total_loss_H/matches  # Average loss


def shape_loss(y_true, y_pred, itr):

    aspect = 0
    angle = 0
    diameter = 0
    siou_loss_value = 0
    siou_loss_value_H = 0
    try:
        if y_true.dim() == 3:
            y_true = y_true.unsqueeze(1)

        y_true = y_true.cpu().detach().numpy()
        y_pred = y_pred.cpu().detach().numpy()
        num_channels = y_true.shape[1] if y_true.ndim == 4 else y_true.shape[0]

        # for true, pred in zip(y_true, y_pred): #it has 4 images
        for i, (true, pred) in enumerate(zip(y_true, y_pred), start=1):  # Iterates over images

            for channel in range(num_channels):

                true_channel = tensor_to_image(true[channel])
                pred_channel = tensor_to_image(pred[channel])

                # Process the images
                results1 = process_image(true_channel)
                results2 = process_image(pred_channel)

                # aspect_loss, angle_loss, diameter_loss = lossall(results1, results2)
                a, b = lossall(results1, results2)
                siou_loss_value += a
                siou_loss_value_H += b

                # aspect += aspect_loss
                # angle += angle_loss
                # diameter += diameter_loss

    except ValueError as e:
        print(f"Error in shape loss calculation: {e}")

    # return aspect, angle, diameter
    return siou_loss_value, siou_loss_value_H


def to_tensor_raw(im):
    return torch.from_numpy(np.asarray(im, dtype=np.int64))


def _random_crop_params(image: Image.Image, output_size):
    """Return torchvision-compatible random crop params ``(i, j, h, w)``."""
    th, tw = output_size
    width, height = image.size
    if height < th or width < tw:
        raise ValueError(f"Crop size {(th, tw)} is larger than image size {(height, width)}")
    if height == th and width == tw:
        return 0, 0, th, tw
    i = random.randint(0, height - th)
    j = random.randint(0, width - tw)
    return i, j, th, tw


def _pil_crop(image: Image.Image, i: int, j: int, h: int, w: int) -> Image.Image:
    return image.crop((j, i, j + w, i + h))


def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 2:
        array = array[:, :, None]
    array = array.transpose((2, 0, 1)) / 255.0
    return torch.from_numpy(array)


def _normalize_tensor(tensor: torch.Tensor, mean, std) -> torch.Tensor:
    mean_t = torch.as_tensor(mean, dtype=tensor.dtype).view(-1, 1, 1)
    std_t = torch.as_tensor(std, dtype=tensor.dtype).view(-1, 1, 1)
    return (tensor - mean_t) / std_t


def roundrobin_infinite(*loaders):
    if not loaders:
        return
    iters = [iter(loader) for loader in loaders]
    while True:
        for i in range(len(iters)):
            it = iters[i]
            try:
                yield next(it)
            except StopIteration:
                iters[i] = iter(loaders[i])
                yield next(iters[i])


def get_weight_mask(y_true, params=None):
    if params is not None:
        y_true = y_true.float() / 255.0 * params['scale']
        mean_label = torch.mean(torch.mean(
            y_true, dim=-1, keepdim=True), dim=-2, keepdim=True)
        y_mask = y_true / params['scale'] + \
            params['alpha'] * mean_label / params['scale']
    else:
        y_mask = torch.ones(y_true.size())
    return y_true, y_mask


def mean_squared_error(y_true, y_pred, y_mask):
    diff = y_pred - y_true
    naive_loss = diff*diff
    masked = naive_loss * y_mask
    last_dim = len(y_pred.size()) - 1
    return torch.mean(masked, dim=last_dim)


def weighted_loss(y_true, y_pred, y_mask):
    '''
    y_true and y_pred are (batch, channel, row, col), we need to permite the dimensio first
    '''
    assert y_pred.dim() == 4, 'dimension is not matched!!'
    if y_true.dim() == 3:
        y_true = y_true.unsqueeze(1)
    y_true = y_true.permute(0, 2, 3, 1)
    y_pred = y_pred.permute(0, 2, 3, 1)
    y_mask = y_mask.permute(0, 2, 3, 1)
    masked_loss = mean_squared_error(y_true, y_pred, y_mask)
    return torch.mean(masked_loss)


def get_validation(datapath, patchsize, dataset, imgext='.png'):
    im_dir = os.path.join(datapath, 'images', 'val')
    if imgext:
        candidate_exts = [imgext.lower()]
    else:
        candidate_exts = []
    for ext in ('.png', '.jpg', '.jpeg'):
        if ext not in candidate_exts:
            candidate_exts.append(ext)

    ids = []
    id_to_ext = {}
    for filename in os.listdir(im_dir):
        stem, ext = os.path.splitext(filename)
        ext = ext.lower()
        if ext in candidate_exts:
            ids.append(stem)
            id_to_ext[stem] = ext

    valid_data = {}
    valid_data['image'] = torch.zeros(len(ids), 3, patchsize, patchsize)

    # Set label channels and normalization parameters
    if dataset in BCD_DATASETS:
        label_channels = 2
        norm_mean = [0.641, 0.616, 0.586]
        norm_std = [0.170, 0.183, 0.183]
        mask_dirs = ['labels_postm', 'labels_negtm']
    elif dataset in (NET_DATASETS | PNET_DATASETS):
        label_channels = 3
        norm_mean = [0.485, 0.456, 0.406]
        norm_std = [0.229, 0.224, 0.225]
        mask_dirs = ['labels_postm', 'labels_negtm', 'labels_other']
    elif dataset in SHIDC_BARE_DATASETS:
        label_channels = 3
        # Match nureg/data/SHIDC_bare.py joint_transform
        norm_mean = [0.845555, 0.805666, 0.706896]
        norm_std = [0.096997, 0.128095, 0.124529]
        mask_dirs = ['labels_postm', 'labels_negtm', 'labels_other']
    elif dataset in SHIDC_DATASETS:
        label_channels = 3
        norm_mean = [0.845, 0.805, 0.706]
        norm_std = [0.088, 0.118, 0.104]
        mask_dirs = ['labels_postm', 'labels_negtm', 'labels_other']
    elif dataset in PANNUKE_DATASETS:
        label_channels = 5
        norm_mean = [0.741, 0.573, 0.704]
        norm_std = [0.149, 0.165, 0.125]
        mask_dirs = ['labels_mask1', 'labels_mask2',
                     'labels_mask3', 'labels_mask4', 'labels_mask5']
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    valid_data['label'] = torch.zeros(
        len(ids), label_channels, patchsize, patchsize)

    for idx, im_id in enumerate(ids):
        # Load and crop image
        image_ext = id_to_ext.get(im_id, candidate_exts[0])
        imagename = os.path.join(datapath, 'images', 'val', im_id + image_ext)
        image = Image.open(imagename).convert('RGB')
        i, j, h, w = _random_crop_params(image, output_size=(patchsize, patchsize))
        image = _pil_crop(image, i, j, h, w)
        image = _pil_to_tensor(image)
        image = _normalize_tensor(image, mean=norm_mean, std=norm_std)

        label_list = []
        for mask_dir in mask_dirs:
            labelname = os.path.join(
                datapath, mask_dir, 'val', im_id + '_label.png')
            label_mask = Image.open(labelname).convert('L')
            label_mask = _pil_crop(label_mask, i, j, h, w)
            label_tensor = torch.from_numpy(
                np.asarray(label_mask, dtype=np.int64)).unsqueeze(0)
            label_list.append(label_tensor)

        label = torch.cat(label_list, dim=0)

        valid_data['image'][idx] = image
        valid_data['label'][idx] = label

    return valid_data


def display_loss(steps, values, plot=None, name='default', legend=None):
    if plot is None:
        try:
            from visdom import Visdom
        except ImportError as exc:
            raise RuntimeError("display_loss requires visdom; install it or pass a plot object.") from exc
        plot = Visdom(use_incoming_socket=False)
    if type(steps) is not list:
        steps = [steps]
    assert type(values) is list, 'values have to be list'
    if type(values[0]) is not list:
        values = [values]

    n_lines = len(values)
    repeat_steps = [steps]*n_lines
    steps = np.array(repeat_steps).transpose()
    values = np.array(values).transpose()
    # values = np.array(values)

    win = name + '_loss'
    if n_lines == 1:
        # have to do this otherwise visdom will have some bug to make the plot
        steps = steps.reshape(steps.shape[0])
        values = values.reshape(values.shape[0])

    res = plot.line(
        X=steps,
        Y=values,
        win=win,
        update='replace',
        opts=dict(title=win)
    )
    if res != win:
        plot.line(
            X=steps,
            Y=values,
            win=win,
            opts=dict(title=win)
        )


@click.command()
@click.argument('output')
@click.option('--dataset', required=True, multiple=True)
@click.option('--datadir', default="", type=click.Path(exists=True))
@click.option('--batch_size', '-b', default=1)
@click.option('--lr', '-l', default=0.001)
@click.option('--step', type=int)
@click.option('--iterations', '-i', default=100000)
@click.option('--momentum', '-m', default=0.9)
@click.option('--snapshot', '-s', default=5000)
@click.option('--downscale', type=int)
@click.option('--augmentation/--no-augmentation', default=False)
@click.option('--fyu/--torch', default=False)
@click.option('--crop_size', default=720)
@click.option('--weights', type=click.Path(exists=True))
@click.option('--model', default='frcn', type=str)
@click.option('--num_cls', default=1, type=int)
@click.option('--gpu', default='0')
@click.option('--use_validation/--no-use_validation', default=False)
@click.option('--alpha', default=1.0, type=float)
@click.option('--beta', default=1.0, type=float)
@click.option('--gamma', default=1.0, type=float)
@click.option('--use_shape_loss/--no-use_shape_loss', default=True)
def main(output, dataset, datadir, batch_size, lr, step, iterations,
         momentum, snapshot, downscale, augmentation, use_validation, fyu, crop_size,
         weights, model, gpu, num_cls, alpha, beta, gamma, use_shape_loss):

    get_dataset, get_model = _load_project_components(model)

    if torch.cuda.is_available():
        try:
            torch.cuda.set_device(int(gpu))
        except (TypeError, ValueError):
            logging.warning("Could not set CUDA device from --gpu=%r; using current device.", gpu)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        logging.warning('CUDA is not available; training on CPU.')

    logdir = 'runs/{:s}/{:s}'.format(model, '-'.join(dataset))
    writer = SummaryWriter(log_dir=logdir)
    net = get_model(model, num_cls=num_cls, finetune=True)

    if weights is not None:
        print(f'Loading pretrained weights from: {weights}')
        weights_dict = torch.load(
            weights, map_location='cuda' if torch.cuda.is_available() else 'cpu')
        safe_load_state_dict(net, weights_dict)
        print('Pretrained weights loaded (mismatched layers skipped).')

    net.to(device)

    transform = []
    target_transform = []
    datasets = [get_dataset(name, os.path.join(datadir, name))
                for name in dataset]
    # if weights is not None:
    #     weights = np.loadtxt(weights)
    opt = torch.optim.SGD(net.parameters(), lr=lr,
                          momentum=momentum, nesterov=True, weight_decay=1e-06)
    if augmentation:
        def collate_fn(batch): return augment_collate(
            batch, crop=crop_size, flip=True)
    else:
        collate_fn = torch.utils.data.dataloader.default_collate
    loaders = [torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                           shuffle=True, num_workers=2,
                                           collate_fn=collate_fn,
                                           pin_memory=torch.cuda.is_available())
               for dataset in datasets]

    if use_validation:
        imgext = '.png'
        valid_data = get_validation(os.path.join(
            datadir, dataset[0]), crop_size, dataset[0], imgext)
    validfreq = 200
    showlossfreq = 200
    savefre = 500
    best_score = 10000.0
    count_ = 0
    tolerance = 3000
    iteration = 0

    dataset_name = dataset[0]
    if dataset_name in BCD_DATASETS:
        mean_m, std_m = 0.03, 0.01
        mean_s, std_s = 2.42, 0.47
        mean_h, std_h = 26.58, 17.33
    elif dataset_name in {'PanNuke', 'PanNukeSam'}:
        mean_m, std_m = 0.01, 0.01
        mean_s, std_s = 1.29, 0.61
        mean_h, std_h = 43.98, 41.24
    elif dataset_name in {'PanNukeBreast', 'PanNukeBreastSam', 'PanNukeBreast256', 'PanNukeBreast256Sam'}:
        mean_m, std_m = 0.0126, 0.0075
        mean_s, std_s = 1.33, 0.59
        mean_h, std_h = 44.69, 41.13
    elif dataset_name in (SHIDC_BARE_DATASETS | SHIDC_DATASETS):
        mean_m, std_m = 0.06, 0.045
        mean_s, std_s = 0.42, 0.37
        mean_h, std_h = 11.14, 21.03
    elif dataset_name in (NET_DATASETS | PNET_DATASETS):
        mean_m, std_m = 0.04, 0.02
        mean_s, std_s = 2.27, 0.67
        mean_h, std_h = 8.58, 6.61
    else:
        raise click.ClickException(f"Unsupported dataset for normalization statistics: {dataset_name}")

    def normalized_loss(m_loss, s_loss, h_loss, alpha=1.0, beta=1.0, gamma=1.0):
        norm_m = (m_loss - mean_m) / std_m
        norm_s = (s_loss - mean_s) / std_s
        norm_h = (h_loss - mean_h) / std_h
        return alpha * norm_m + beta * norm_s + gamma * norm_h

    def normalized_loss1(m_loss, s_loss, h_loss, alpha=1.0, beta=1.0, gamma=1.0):
        norm_m = m_loss
        norm_s = s_loss
        norm_h = h_loss
        return alpha * norm_m + beta * norm_s + gamma * norm_h

    ####

    data = []
    losses = deque(maxlen=10)
    params = dict()
    params['scale'] = 5.0
    params['alpha'] = 5.0
    steps, vals = [], []
    valid_steps, valid_vals = [], []

    for im, label in roundrobin_infinite(*loaders):
        net.train()
        opt.zero_grad()

        im_v = make_variable(im, requires_grad=False, device=device)
        label_scale, label_mask = get_weight_mask(label, params)
        label_v = make_variable(label_scale, requires_grad=False, device=device)
        label_mask_v = make_variable(label_mask, requires_grad=False, device=device)

        pred = net(im_v)
        m_loss = weighted_loss(label_v, pred, label_mask_v)

        if use_shape_loss:
            # Add shape-aware components
            s_loss, s_loss_H = shape_loss(label_v, pred, iteration)
            loss = normalized_loss(
                m_loss, s_loss, s_loss_H, alpha, beta, gamma)
        else:
            loss = m_loss  # Just pixel-wise loss

        loss_val = float(loss.detach().cpu().item())
        assert not np.isnan(loss_val), "nan error"

        steps.append(iteration)
        vals.append(loss_val)

        '''
        # Prepare row for logging
        log_entry = {
            "iteration": iteration,
            "m_loss": m_loss.item()
        }

        if use_shape_loss:
            log_entry["s_loss"] = s_loss
            log_entry["h_loss"] = s_loss_H

        data.append(log_entry)

        # Save to CSV named after dataset
        csv_name = f"train_{dataset[0]}.csv"
        df = pd.DataFrame(data)
        df.to_csv(csv_name, index=False)
        '''

        if np.mod(iteration, savefre) == 0:
            output_path = '{}.pth'.format(output)
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            torch.save(net.state_dict(), output_path)
            print('Save weights to: ', output_path)
        if iteration == 0:
            output_path = '{}-iter{}.pth'.format(output, iteration)
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            torch.save(net.state_dict(), output_path)

        if use_validation and iteration % validfreq == 0:

            with torch.no_grad():
                valid_images = valid_data['image'].to(device, non_blocking=True)
                valid_pred = net.predict(valid_images, batch_size=batch_size)
                if not isinstance(valid_pred, torch.Tensor):
                    valid_pred = torch.as_tensor(valid_pred)
                valid_pred = valid_pred.to(device)

                valid_label_scale, valid_label_mask = get_weight_mask(
                    valid_data['label'].to(device, non_blocking=True), params)

                valid_loss_m = weighted_loss(
                    valid_label_scale,
                    valid_pred,
                    valid_label_mask
                )

            if use_shape_loss:
                valid_loss_s, valid_loss_s_H = shape_loss(
                    valid_label_scale,
                    valid_pred,
                    iteration
                )
                valid_loss = normalized_loss(
                    valid_loss_m, valid_loss_s, valid_loss_s_H, alpha, beta, gamma)
            else:
                valid_loss = valid_loss_m

            valid_loss_val = float(valid_loss.detach().cpu().item())
            valid_steps.append(iteration)
            valid_vals.append(valid_loss_val)
            print('\nValidation loss: {}, best_score: {}'.format(
                valid_loss_val, best_score))
            if valid_loss_val <= best_score:
                best_score = valid_loss_val
                print('update to new best_score: {}'.format(best_score))
                output_path = '{}-best.pth'.format(output)
                output_dir = os.path.dirname(output_path)
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                torch.save(net.state_dict(), output_path)
                print('Save best weights to: ', output_path)
                count_ = 0
            else:
                count_ = count_ + 1
            if count_ >= tolerance:
                assert 0, 'performance not imporoved for so long'

        loss.backward()
        losses.append(loss_val)

        opt.step()

        # log results
        if iteration % 100 == 0:
            logging.info('Iteration {}:\t{}'
                         .format(iteration, np.mean(losses)))
            writer.add_scalar('loss', np.mean(losses), iteration)
        iteration += 1
        if step is not None and iteration % step == 0:
            logging.info('Decreasing learning rate by 0.1.')
            for param_group in opt.param_groups:
                param_group['lr'] *= 0.1
        if iteration % snapshot == 0:
            output_path = '{}-iter{}.pth'.format(output, iteration)
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            torch.save(net.state_dict(), output_path)
        if iteration >= iterations:
            logging.info('Optimization complete.')
            break

    writer.close()


if __name__ == '__main__':
    main()
