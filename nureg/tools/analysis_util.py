"""Visualization and file-list helpers for detection/classification outputs."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.io import loadmat
from skimage import color


def get_seed_name(threshhold, min_len):
    """Return the historical result-key name for a threshold/radius pair."""
    name = ("t_" + "{:01.02f}".format(threshhold) + "_r_" + "{:02.02f}".format(min_len)).replace(".", "_")
    return name


def getfilelist(Imagefolder, inputext):
    """Return files under ``Imagefolder`` whose extension is in ``inputext``."""
    if not isinstance(inputext, list):
        inputext = [inputext]
    inputext = {ext.lower() for ext in inputext}
    filelist = []
    filenames = []
    for filename in sorted(os.listdir(Imagefolder)):
        stem, ext = os.path.splitext(filename)
        path = os.path.join(Imagefolder, filename)
        if ext.lower() in inputext and os.path.isfile(path):
            filelist.append(path)
            filenames.append(stem)
    return filelist, filenames


def imread(imgfile):
    """Read an image as RGB."""
    if not os.path.exists(imgfile):
        raise FileNotFoundError(f"{imgfile} does not exist")
    src_bgr = cv2.imread(imgfile)
    if src_bgr is None:
        raise ValueError(f"Could not read image: {imgfile}")
    return cv2.cvtColor(src_bgr, cv2.COLOR_BGR2RGB)


def overlayImg(img, mask, print_color=(5, 119, 72), linewidth=1, alpha=0.618, savepath=None):
    """Overlay a binary mask on an RGB image."""
    del linewidth  # kept for backward-compatible signature
    rows, cols = img.shape[0:2]
    color_mask = np.zeros((rows, cols, 3), dtype=np.float32)
    if len(mask.shape) != 2:
        raise ValueError("mask should be 2-dimensional")

    if len(img.shape) == 2:
        img_color = np.dstack((img, img, img))
    else:
        img_color = img

    color_mask[mask == 1] = print_color
    color_mask[mask == 0] = img_color[mask == 0]

    img_hsv = color.rgb2hsv(img_color)
    color_mask_hsv = color.rgb2hsv(color_mask)
    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha

    img_masked = color.hsv2rgb(img_hsv)
    max_value = np.max(img_masked)
    if max_value > 0:
        img_masked = img_masked / max_value
    img_masked = np.asarray(img_masked * 255, dtype=np.uint8)
    img_masked[mask == 1] = print_color

    if savepath is not None:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img_masked).save(savepath)
    return img_masked


def printCoordClass(Img=None, coordinates=None, labels=None, num_cls=3, savepath=None, alpha=0.85):
    """Draw detected coordinates over an image.

    ``coordinates`` should be an ``N x 2`` array in ``(row, col)`` order.
    """
    if Img is None or coordinates is None:
        raise ValueError("Img and coordinates are required")

    marker_linewidth = 7
    marker_color = [
        [255, 0, 255],
        [0, 255, 255],
        [255, 255, 0],
        [255, 128, 0],
        [128, 255, 0],
        [0, 128, 255],
    ]
    overlaid_res = Img.copy()

    for icls in range(1, num_cls + 1):
        dot_mask = np.zeros(Img.shape[0:2], dtype=np.uint8)
        if isinstance(coordinates, dict):
            iSeed = coordinates.get("class" + str(icls), np.asarray([]))
        else:
            if labels is None:
                raise ValueError("labels are required when coordinates is not a dict")
            iClass = np.where(labels == icls)
            iSeed = coordinates[iClass]

        if getattr(iSeed, "size", 0) != 0:
            rows = np.clip(iSeed[:, 0].astype(int), 0, dot_mask.shape[0] - 1)
            cols = np.clip(iSeed[:, 1].astype(int), 0, dot_mask.shape[1] - 1)
            dot_mask[rows, cols] = 1

        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (marker_linewidth, marker_linewidth))
        dot_mask = cv2.dilate(dot_mask, se)
        overlaid_res = overlayImg(
            overlaid_res,
            dot_mask,
            print_color=marker_color[(icls - 1) % len(marker_color)],
            linewidth=1,
            alpha=alpha,
        )

    if savepath:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(overlaid_res).save(savepath)
    return overlaid_res


def printCoordsClass(savefolder, imgdir, imgext, threshhold=0.50, min_len=5, num_cls=3):
    """Overlay saved prediction coordinates on the source images."""
    imglist_, imagenamelist_ = getfilelist(imgdir, imgext)
    imglist = list(imglist_)
    imagenamelist = list(imagenamelist_)

    ol_folder = os.path.join(savefolder, get_seed_name(threshhold, min_len))
    if os.path.exists(ol_folder):
        shutil.rmtree(ol_folder)
    os.makedirs(ol_folder, exist_ok=True)

    for imgindx, imgpath in enumerate(imglist):
        print("overlay image {ind}".format(ind=imgindx))
        thisimg = imread(imgpath)
        imgname = imagenamelist[imgindx]
        savepath = os.path.join(ol_folder, imgname + "_ol.png")

        resultDictPath = os.path.join(savefolder, imgname + ".mat")
        if not os.path.isfile(resultDictPath):
            print(f"Missing result file: {resultDictPath}")
            continue
        resultsDict = loadmat(resultDictPath)
        localseedname = get_seed_name(threshhold, min_len)
        coordinates = resultsDict[localseedname]
        labels = resultsDict[localseedname + "_label"]
        labels = np.squeeze(labels)
        printCoordClass(Img=thisimg, coordinates=coordinates, labels=labels, num_cls=num_cls, savepath=savepath, alpha=1)
