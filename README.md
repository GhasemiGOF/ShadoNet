# ShadoNet: A Nucleus Detection and Classification Framework for Ki-67 Pathology Images

<p align="center">
  <img src="assets/shadonet_architecture.png" width="950">
</p>

<p align="center">
  <em>
  Overview of ShadoNet. Given a Ki-67 pathology image, the model predicts class-specific proximity maps for nucleus detection and classification while incorporating morphology-aware supervision through rotation-aware SIoU and Hausdorff Distance Transform (HDT) losses.
  </em>
</p>

---

## Overview

**ShadoNet** is a deep learning framework for nucleus detection and classification in **Ki-67-stained histopathology images**.

Unlike conventional nucleus analysis pipelines that separate detection and classification into multiple stages, ShadoNet formulates the problem as a **single-stage structured regression task**. The model predicts class-specific proximity maps that simultaneously encode:

- nucleus location,
- nucleus class identity,
- local morphological structure.

To further improve recognition performance, ShadoNet integrates shape-aware supervision derived from the **Segment Anything Model (SAM)** and introduces geometry-sensitive training objectives based on:

- Rotation-aware Scaled Intersection over Union (**SIoU**)
- Hausdorff Distance Transform (**HDT**)

This design enables morphology-guided learning without requiring exhaustive instance-level boundary annotations.

---

## Highlights

✅ Single-stage nucleus detection and classification

✅ Shape-aware learning using SAM-generated pseudo masks

✅ Center-based supervision instead of manual instance segmentation

✅ Rotation-aware SIoU loss for geometric alignment

✅ Hausdorff Distance Transform loss for boundary-aware regularization

✅ Designed specifically for Ki-67 pathology analysis

✅ Competitive performance across multiple Ki-67 datasets

---

## Method Overview

ShadoNet learns to predict **class-specific proximity maps** centered at annotated nuclei.

### Training Label Generation

1. Human experts annotate nucleus centers and cell classes.
2. SAM generates candidate cellular masks.
3. SAM masks are filtered using the expert-provided nucleus centers.
4. Refined masks are converted into morphology-aware proximity maps.
5. These proximity maps are used as training targets.

### Shape-Aware Learning

In addition to standard pixel-wise supervision, ShadoNet incorporates:

### MSE Loss

Provides pixel-level supervision for proximity map regression.

### Rotation-aware SIoU Loss

Encourages geometric consistency between predicted and reference structures while accounting for orientation and shape.

### Hausdorff Distance Transform Loss

Penalizes boundary discrepancies through distance-transform representations, improving structural alignment.

---

## Repository Structure

```text
ShadoNet/
│
├── train_fcn_cell_class.py
├── eval_fcn_cell_class.py
├── Gen_refactored.py
│
├── train_fcn_cell_class.sh
├── eval_fcn_cell_class.sh
│
├── requirements.txt
├── requirements-sam.txt
│
├── README.md
│
├── assets/
│   └── shadonet_architecture.png
│
└── nureg/
    ├── data/
    ├── models/
    ├── tools/
    ├── transforms.py
    ├── util.py
    └── torch_utils.py
```

---

## Datasets

The framework supports experiments on multiple Ki-67 pathology datasets, including:

### NETnewClass

Pancreatic neuroendocrine tumor (PanNET) Ki-67 dataset with:

- Positive tumor nuclei
- Negative tumor nuclei
- Non-tumor nuclei

### BCD

Breast cancer Ki-67 dataset containing:

- Ki-67 positive tumor cells
- Ki-67 negative tumor cells

### PNET

Whole-slide PanNET Ki-67 dataset used for fine-tuning and external evaluation.

---

## Installation

### Clone Repository

```bash
git clone https://github.com/GhasemiGOF/ShadoNet.git
cd ShadoNet
```

### Create Environment

Using Conda:

```bash
conda create -n shadonet python=3.10 -y
conda activate shadonet
```

or using virtual environments:

```bash
python -m venv .venv
source .venv/bin/activate
```

### Install Core Dependencies

```bash
pip install -r requirements.txt
```

---

## Optional: SAM-Based Label Generation

The SAM pipeline is only required if you want to generate morphology-aware labels using `Gen_refactored.py`.

### Install Additional Dependencies

Create a file named:

```text
requirements-sam.txt
```

with the following content:

```text
# Optional: Segment Anything pseudo-label pipeline (Gen_refactored.py)
# Install core dependencies first, then this file:
#
# pip install -r requirements.txt
# pip install -r requirements-sam.txt
#
# Download SAM checkpoints (e.g. sam_vit_h_4b8939.pth)
# and set paths in Gen_refactored.py.

segment-anything @ git+https://github.com/facebookresearch/segment-anything.git
```

Install:

```bash
pip install -r requirements-sam.txt
```

### Download SAM Checkpoint

Download a checkpoint such as:

```text
sam_vit_h_4b8939.pth
```

and configure the path in:

```python
Gen_refactored.py
```

---

## Data Preparation

A typical dataset structure is:

```text
datasets/
├── NETnewClass/
│   ├── train/
│   ├── val/
│   └── test/
│
├── BCD/
│   ├── train/
│   ├── val/
│   └── test/
│
└── PNET/
    ├── train/
    ├── val/
    └── test/
```

> Datasets are not distributed with this repository.

---

## Generate Shape-Aware Labels

Example:

```bash
python Gen_refactored.py datasets/NETnewClass cuda:0 --strategy sam_full
```

Available strategies:

```text
no_sam
raw_sam
sam_all
sam_area
sam_geom
sam_full
sam_cell_p20
sam_cell_p40
sam_cell_p60
sam_cell_p80
```

---

## Training

### Shell Script

```bash
bash train_fcn_cell_class.sh
```

### Python

Example:

```bash
python train_fcn_cell_class.py \
    --data NETnewClassSam \
    --cuda 0 \
    --use_shape true
```

---

## Evaluation

### Shell Script

```bash
bash eval_fcn_cell_class.sh
```

### Python

Example:

```bash
python eval_fcn_cell_class.py \
    /path/to/checkpoint.pth \
    --dataset NETnewClass \
    --datadir datasets \
    --model ki67net \
    --num_cls 3
```

Evaluation reports:

- Detection Precision
- Detection Recall
- Detection F1
- Classification Precision
- Classification Recall
- Classification F1
- Per-class metrics

---

## Citation

If you use ShadoNet in your work, please cite:

```bibtex
@article{ghasemi2025shadonet,
  title={ShadoNet: A Nucleus Detection and Classification Framework for Ki-67 Pathology Images},
  author={Ghasemi, Mahsa and Xing, Fuyong and Cornish, Toby C and Ghosh, Debashis and Bian, Jiang and Zhang, Xuhong},
  journal={Bioinformatics},
  year={2025}
}
```

---

## Contact

**Xuhong Zhang**

📧 zhangxuh@iu.edu

---

## License

Please add your preferred license before public release.

Recommended options:

- MIT License
- Apache License 2.0
- BSD 3-Clause License

---

## Acknowledgments

This work was developed through collaboration between:

- Indiana University
- University of Colorado Anschutz Medical Campus
- Medical College of Wisconsin
- Indiana University School of Medicine
- Regenstrief Institute

We thank the pathologists, annotators, and collaborators who contributed to the Ki-67 pathology datasets and study design.
