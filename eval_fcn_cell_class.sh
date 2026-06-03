#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./eval_fcn_cell_class.sh [test_dataset] [cuda_id] [train_dataset]
# Example:
#   ./eval_fcn_cell_class.sh SHIDC_bare 0 SHIDC_bare
# You can also set TRAINDATA, ITERATION, DATA_DIR, RESULT_DIR, or EVAL_RESULT_FOLDER.

testdata=${1:-SHIDC_bare}
cuda=${2:-0}
traindata=${3:-${TRAINDATA:-$testdata}}
export CUDA_VISIBLE_DEVICES="$cuda"

iteration=${ITERATION:-best}
datadir=${DATA_DIR:-datasets}
result=${RESULT_DIR:-learned_models}

case "$traindata" in
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
    echo "Unknown training dataset: $traindata" >&2
    exit 1
    ;;
esac

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

eval_result_folder=${EVAL_RESULT_FOLDER:-"eval_${traindata}_on_${testdata}"}
mkdir -p "$eval_result_folder"

load_model="${result}/${traindata}-${model}_${traindata}/${model}-${iteration}.pth"
if [[ ! -f "$load_model" ]]; then
  echo "Model checkpoint not found: $load_model" >&2
  exit 1
fi

echo "Evaluating test dataset: $testdata"
echo "Training dataset/checkpoint: $traindata"
echo "Model: $model | classes: $num_cls | CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Checkpoint: $load_model"

python eval_fcn_cell_class.py "$load_model" --model "$model" \
  --num_cls "$num_cls" \
  --gpu 0 \
  --datadir "$datadir" \
  --dataset "$testdata" \
  --eval_result_folder "$eval_result_folder"
