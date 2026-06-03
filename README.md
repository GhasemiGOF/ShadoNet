# ShadoNet: A Nucleus Detection and Classification Framework for Ki-67 Pathology Images

[![Paper](https://img.shields.io/badge/paper-Bioinformatics-blue)](https://github.com/GhasemiGOF/ShadoNet)
[![Code](https://img.shields.io/badge/code-Python-green)](https://github.com/GhasemiGOF/ShadoNet)
[![License](https://img.shields.io/badge/license-TBD-lightgrey)](LICENSE)

**ShadoNet** is a deep learning framework for nucleus detection and classification in Ki-67-stained histopathology images.  
It uses center-based supervision with morphology-aware guidance to learn class-specific proximity maps, enabling accurate nucleus localization and subtype prediction without requiring full manual instance segmentation masks.

---

## Overview

Ki-67 immunohistochemistry is widely used for proliferation assessment and tumor grading, but manual nucleus counting and classification are time-consuming and error-prone in dense tissue regions.

ShadoNet addresses this challenge with a single-stage, end-to-end regression framework that:

- predicts **class-specific proximity maps** for nucleus centers,
- leverages **SAM-generated shape priors** refined by human point annotations,
- incorporates **rotation-aware SIoU** and **Hausdorff Distance Transform (HDT)** losses for morphology-sensitive supervision,
- supports **nucleus detection and classification** in a unified pipeline.

This repository contains the code for training, evaluation, and SAM-based label generation.

---

## Key Features

- **Single-stage nucleus detection + classification**
- **Center-based supervision** rather than full instance masks
- **Shape-aware training** using SAM-derived priors
- **Rotation-aware SIoU loss** for geometric alignment
- **HDT loss** for boundary-sensitive regularization
- **Cross-dataset evaluation** on Ki-67 histopathology data
- **Optional SAM pseudo-label pipeline** for generating refined supervision maps

---

## Repository Structure

A typical repository layout is:

```text
ShadoNet/
├── train_fcn_cell_class.py
├── eval_fcn_cell_class.py
├── Gen_refactored.py
├── train_fcn_cell_class.sh
├── eval_fcn_cell_class.sh
├── requirements.txt
├── requirements-sam.txt
├── README.md
└── nureg/
    ├── data/
    ├── models/
    ├── tools/
    ├── transforms.py
    ├── util.py
    └── torch_utils.py
```

Depending on your local setup, you may also have:

- `datasets/` — input data and annotations
- `learned_models/` — checkpoints and outputs
- `experiments/` — evaluation results
- `Fig/` — figures used in the paper

---

## Method Summary

ShadoNet learns to predict smooth proximity maps centered at annotated nuclei. Each map encodes:

- the **location** of the nucleus center,
- the **class identity** of the nucleus,
- **morphological structure** through shape-guided refinement.

During supervision generation:

1. Human experts annotate nucleus centers with class labels.
2. SAM generates candidate cellular masks.
3. SAM masks are filtered and refined using the human center annotations.
4. Refined masks are used to create proximity-based training labels.

During training, the model is optimized with:

- **MSE loss** on proximity maps,
- **rotation-aware SIoU loss**,
- **HDT loss**.

---

## Datasets

This repository supports Ki-67 histopathology experiments on multiple datasets, including:

- **NETnewClass**
- **BCD**
- **PNET**

The codebase may also include dataset variants used for ablation and SAM-based supervision, such as:

- `NETnewClassSam`
- `NETnewClass_no_sam`
- `NETnewClass_raw_sam`
- `NETnewClass_sam_area`
- `NETnewClass_sam_geom`
- `NETnewClass_sam_full`
- `NETnewClass_sam_overlap`
- `NETnewClass_sam_cell_p20/p40/p60/p80`
- `BCDSam`
- `PNETSam`

> The datasets are **not included** in this repository.

### Expected data organization

Your local dataset directory should follow the folder structure expected by the training and evaluation scripts. A common layout is:

```text
datasets/
├── NETnewClass/
│   ├── train/
│   ├── val/
│   └── test/
├── BCD/
│   ├── train/
│   ├── val/
│   └── test/
└── PNET/
    ├── train/
    ├── val/
    └── test/
```

The exact names used by your scripts may differ depending on the dataset variant. Keep the directory names consistent with the `--data` / `--dataset` options used in the scripts.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/GhasemiGOF/ShadoNet.git
cd ShadoNet
```

### 2. Create a Python environment

Using `conda`:

```bash
conda create -n shadonet python=3.10 -y
conda activate shadonet
```

Or using `venv`:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install core dependencies

```bash
pip install -r requirements.txt
```

---

## Optional: Segment Anything pseudo-label pipeline

The SAM-based pseudo-label pipeline is only needed if you want to generate refined proximity labels with `Gen_refactored.py`.

### Install optional SAM dependencies

```bash
pip install -r requirements-sam.txt
```

### Install Segment Anything

```bash
pip install segment-anything @ git+https://github.com/facebookresearch/segment-anything.git
```

### Download SAM checkpoints

Download a SAM checkpoint such as:

- `sam_vit_h_4b8939.pth`

Then set the checkpoint path inside `Gen_refactored.py` or in your own configuration file.

---

## Data Preparation

Before training, prepare:

- pathology image patches,
- nucleus center annotations,
- class labels,
- optional SAM-generated auxiliary masks.

The label-generation script can be used to create refined supervision maps from annotated data.

### Example label generation

```bash
python Gen_refactored.py datasets/NETnewClass cuda:0 --strategy sam_full
```

Other useful strategies include:

- `no_sam`
- `raw_sam`
- `sam_all`
- `sam_area`
- `sam_geom`
- `sam_full`
- `sam_cell_p20`
- `sam_cell_p40`
- `sam_cell_p60`
- `sam_cell_p80`

The strategy you choose controls how SAM masks are filtered and how strongly the generated labels follow shape priors.

---

## Training

### Shell launcher

```bash
bash train_fcn_cell_class.sh
```

### Direct Python call

Example:

```bash
python train_fcn_cell_class.py \
    --data NETnewClassSam \
    --cuda 0 \
    --use_shape true
```

Typical training options may include:

- dataset name,
- CUDA device,
- shape-aware training toggle,
- loss weights,
- learning rate,
- batch size,
- number of iterations.

The exact command-line arguments may vary depending on the dataset and experiment configuration in your scripts.

---

## Evaluation

### Shell launcher

```bash
bash eval_fcn_cell_class.sh
```

### Direct Python call

Example:

```bash
python eval_fcn_cell_class.py \
    /path/to/checkpoint.pth \
    --dataset NETnewClass \
    --datadir datasets \
    --model ki67net \
    --num_cls 3
```

The evaluation pipeline reports:

- detection precision, recall, and F1,
- classification precision, recall, and F1,
- per-class performance,
- image-level prediction outputs.

Results are saved in the configured evaluation directory.

---

## Metrics

Nucleus detection and classification are evaluated using standard matching-based protocols, including:

- precision,
- recall,
- F1-score,
- per-class F1-score,
- weighted classification scores.

A predicted nucleus is counted as correct when its center lies within the matching radius of a ground-truth nucleus.

---

## Loss Functions

ShadoNet is trained with a combination of:

- **MSE** on class-specific proximity maps,
- **rotation-aware SIoU** for geometric alignment,
- **HDT** for boundary-sensitive supervision.

These losses are designed to improve robustness in dense and morphologically heterogeneous tissue regions.

---

## Example Results

ShadoNet shows consistent gains over the baseline in Ki-67 nucleus detection and classification, especially in dense regions and morphologically diverse tissue.

For exact numbers, please refer to the results reported in the paper and the figures/tables in the manuscript.

---

## Citation

If you use this code in your research, please cite:

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

For questions about the project, please contact:

**Xuhong Zhang**  
`zhangxuh@iu.edu`

---

## License

Add a license file before public release. Common options include:

- MIT
- Apache 2.0
- BSD 3-Clause

If you have not selected a license yet, the repository should be treated as “all rights reserved” by default.

---

## Acknowledgments

This project was developed at Indiana University and in collaboration with partner institutions.  
We thank the annotators and domain experts who contributed to the Ki-67 pathology data.
