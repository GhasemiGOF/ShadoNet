# Release notes

## 0.1.0 cleanup

Fixed and prepared the uploaded code for public sharing:

- Corrected `train_fcn_cell_class.sh` so it calls `train_fcn_cell_class.py` instead of the missing `train_fcn_cell_class2.py`.
- Reworked train/eval shell scripts with safer argument handling, dataset-family selection, and clear checkpoint errors.
- Made `nureg.data` and `nureg.models` imports lazy, so `--help` and utility imports work even when the private project modules are not present.
- Replaced deprecated/brittle tensor helper code while keeping backward-compatible function names.
- Fixed `HalfCrop` in `nureg/transforms.py` and removed the dependency on `torchvision` from the included transforms.
- Fixed the shape-loss channel indexing bug that used `channel - 1` inside a zero-based loop.
- Fixed `find_bounding_box_and_center` to always return the expected four values.
- Fixed the evaluation accuracy denominator in `detectclassify_eval`.
- Made evaluation MAT-file lookup path-safe for relative and absolute dataset paths.
- Added safe handling for empty distance arrays in metric summaries.
- Made SAM imports lazy and converted OpenCV BGR images to RGB before calling SAM.
- Added `README.md`, `requirements.txt`, `pyproject.toml`, and `.gitignore`.
- Verified syntax with `python -m compileall` and smoke-tested `Gen_refactored.py --strategy no_sam` on a tiny synthetic dataset.
