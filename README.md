# Nucleus Regression and SAM Label Utilities

This repository contains the cleaned training, evaluation, and SAM-label-generation code for nucleus/cell classification experiments.

## What is included

- `Gen_refactored.py` - generates proximity label maps from manual centers, SAM masks, or both.
- `train_fcn_cell_class.py` / `train_fcn_cell_class.sh` - trains the FCN classifier.
- `eval_fcn_cell_class.py` / `eval_fcn_cell_class.sh` - evaluates trained checkpoints and writes detection/classification metrics.
- `nureg/tools/*` - evaluation and visualization helpers.
- `nureg/transforms.py`, `nureg/torch_utils.py`, `nureg/util.py` - reusable PyTorch/data helpers.

The original training/evaluation scripts depend on project-specific modules that were not part of the uploaded files: `nureg.data` and `nureg.models`. Add those folders back under `nureg/` before running training or evaluation, or install the full package that provides them. The scripts now fail with a clear message when those modules are missing instead of crashing during import.

## Expected dataset layout

```text
datasets/<DATASET>/
  images/
    train/
    val/
    test/
  mats/
    train/
    val/
    test/
  labels_postm/
    train/
    val/
    test/
  labels_negtm/
    train/
    val/
    test/
  labels_other/
    train/
    val/
    test/
```

MAT files are expected to contain at least `Centers` and `Labels`. For SAM label generation, images are matched to MAT files named `<image_stem>_withcontour.mat`.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For SAM-based label generation, install Meta Segment Anything and download the SAM checkpoint you plan to use. Put the checkpoint path in `--checkpoint`.

## Generate labels

Circle-only labels:

```bash
python Gen_refactored.py datasets/NETnewClass cpu --strategy no_sam --splits train val test
```

SAM-filtered labels:

```bash
python Gen_refactored.py datasets/NETnewClass_sam_full cuda:0 \
  --strategy sam_full \
  --checkpoint sam_vit_h_4b8939.pth
```

Available presets are `no_sam`, `raw_sam`, `sam_all`, `sam_area`, `sam_geom`, `sam_full`, `sam_cell_p20`, `sam_cell_p40`, `sam_cell_p60`, and `sam_cell_p80`.

## Train

```bash
./train_fcn_cell_class.sh 0 NETnewClassSam false 0.8 0.1 0.1
```

Arguments are:

```text
./train_fcn_cell_class.sh [cuda_id] [dataset] [use_shape_loss] [alpha] [beta] [gamma]
```

Environment variables can override defaults: `DATA_DIR`, `RESULT_DIR`, `BATCH_SIZE`, `CROP_SIZE`, `ITERATIONS`, `LR`, `MOMENTUM`, and `SNAPSHOT`.

## Evaluate

```bash
./eval_fcn_cell_class.sh SHIDC_bare 0 SHIDC_bare
```

Arguments are:

```text
./eval_fcn_cell_class.sh [test_dataset] [cuda_id] [train_dataset]
```

The evaluator looks for a checkpoint at:

```text
learned_models/<train_dataset>-<model>_<train_dataset>/<model>-best.pth
```

Override paths with `DATA_DIR`, `RESULT_DIR`, `ITERATION`, and `EVAL_RESULT_FOLDER`.

## Important cleanup notes

- The shell scripts now call the correct Python files and use `CUDA_VISIBLE_DEVICES` consistently.
- Heavy optional dependencies such as `segment-anything` and TensorBoard are loaded lazily where possible.
- `torchvision` is no longer required by the included utility code, avoiding import-time failures from mismatched Torch/Torchvision builds.
- Generated datasets, checkpoints, TensorBoard runs, and evaluation folders are ignored by Git.

## Before public release

Choose and add the license you want for this code. I did not add a license file because that is a legal/publishing choice for the repository owner.
