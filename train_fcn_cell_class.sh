#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./train_fcn_cell_class.sh [cuda_id] [dataset] [use_shape_loss] [alpha] [beta] [gamma]
# Example:
#   ./train_fcn_cell_class.sh 0 NETnewClassSam false 0.8 0.1 0.1

cuda=${1:-0}
data=${2:-NETnewClassSam}
use_shape=${3:-false}
alpha=${4:-0.8}
beta=${5:-0.1}
gamma=${6:-0.1}
export CUDA_VISIBLE_DEVICES="$cuda"

crop=${CROP_SIZE:-250}
datadir=${DATA_DIR:-datasets}
batch=${BATCH_SIZE:-4}
iterations=${ITERATIONS:-100000}
lr=${LR:-1e-3}
momentum=${MOMENTUM:-0.99}
snapshot=${SNAPSHOT:-10000}
result=${RESULT_DIR:-learned_models}

case "$data" in
  BCD|BCDSam|BCDSam_fold_*|BCD_fold_*)
    model="ki67netBcd"
    num_cls=2
    ;;
  PanNuke|PanNukeSam|PanNukeBreast|PanNukeBreastSam|PanNukeBreast256|PanNukeBreast256Sam)
    model="ki67netPan"
    num_cls=5
    ;;
  SHIDC_bare|SHIDC_bare_SAM|SHIDC500|SHIDCSam|SHIDC500Sam|SHIDC256Sam|SHIDCSam_fold_*|SHIDC_fold_*)
    model="ki67net"
    num_cls=3
    ;;
  NETnewClass|NETnewClassSam|NETnewClass_mix_p20|NETnewClass_mix_p40|NETnewClass_mix_p60|NETnewClass_mix_p80|NETnewClass_sam_cell_p20|NETnewClass_sam_cell_p40|NETnewClass_sam_cell_p60|NETnewClass_sam_cell_p80|NETnewClass_no_sam|NETnewClass_raw_sam|NETnewClass_sam_area|NETnewClass_sam_geom|NETnewClass_sam_full|NETnewClass_sam_unfiltered|NETnewClass_sam_overlap|NETnewClass256|NETnewClassSam256|NETnewClassSam_fold_*|NETnewClass_fold_*|PNET|PNET11|PNETSam|PNETSam_fold_*|PNET_fold_*)
    model="ki67net"
    num_cls=3
    ;;
  *)
    echo "Unknown dataset: $data" >&2
    exit 1
    ;;
esac

case "${use_shape,,}" in
  true|1|yes|y)
    shape_flag="--use_shape_loss"
    ;;
  false|0|no|n)
    shape_flag="--no-use_shape_loss"
    ;;
  *)
    echo "use_shape_loss must be true/false, got: $use_shape" >&2
    exit 1
    ;;
esac

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

run_dir="${result}/${data}-${model}_${data}"
outdir="${run_dir}/${model}"
mkdir -p "$run_dir"

echo "Training dataset: $data"
echo "Model: $model | classes: $num_cls | CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Output prefix: $outdir"

python train_fcn_cell_class.py "$outdir" --model "$model" \
  --num_cls "$num_cls" \
  --gpu 0 \
  --lr "$lr" -b "$batch" -m "$momentum" \
  --crop_size "$crop" --iterations "$iterations" \
  --augmentation \
  --snapshot "$snapshot" \
  --datadir "$datadir" \
  --dataset "$data" \
  --use_validation \
  --alpha "$alpha" \
  --beta "$beta" \
  --gamma "$gamma" \
  "$shape_flag"
