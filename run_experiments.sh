#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="STL10"
OUTPUT_DIR="outputs"
EPOCHS="30"
BATCH_SIZE="64"
NUM_WORKERS="4"
DEVICE="cuda:0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --num-workers)
      NUM_WORKERS="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

COMMON_ARGS=(
  --data-root "$DATA_ROOT"
  --output-dir "$OUTPUT_DIR"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --device "$DEVICE"
)

run_train() {
  echo "Running: $*"
  uv run python -m stl10_project.train "$@"
}

run_train \
  "${COMMON_ARGS[@]}" \
  --run-name baseline_simple_cnn \
  --model simple_cnn \
  --augmentation none \
  --optimizer sgd \
  --lr 0.03 \
  --no-batchnorm \
  --gradcam-samples 0

run_train \
  "${COMMON_ARGS[@]}" \
  --run-name aug_simple_cnn \
  --model simple_cnn \
  --augmentation basic \
  --optimizer sgd \
  --lr 0.03 \
  --no-batchnorm \
  --gradcam-samples 0

run_train \
  "${COMMON_ARGS[@]}" \
  --run-name regularized_simple_cnn \
  --model simple_cnn \
  --augmentation basic \
  --optimizer adamw \
  --lr 0.001 \
  --dropout 0.3 \
  --gradcam-samples 0

run_train \
  "${COMMON_ARGS[@]}" \
  --run-name resnet18_adamw \
  --model resnet18 \
  --augmentation basic \
  --optimizer adamw \
  --lr 0.001 \
  --dropout 0.2 \
  --gradcam-samples 10

OUTPUT_DIR="$OUTPUT_DIR" uv run python - <<'PY'
import csv
import json
import os
from pathlib import Path

output_dir = Path(os.environ["OUTPUT_DIR"])
run_names = [
    "baseline_simple_cnn",
    "aug_simple_cnn",
    "regularized_simple_cnn",
    "resnet18_adamw",
]
rows = []
for run_name in run_names:
    with (output_dir / run_name / "summary.json").open("r", encoding="utf-8") as f:
        summary = json.load(f)
    rows.append(
        {
            "run_name": run_name,
            "best_epoch": summary["best_epoch"],
            "best_valid_acc": summary["best_valid_acc"],
            "test_acc": summary["test_acc"],
            "test_macro_f1": summary["test_macro_f1"],
            "test_weighted_f1": summary["test_weighted_f1"],
            "run_dir": summary["run_dir"],
        }
    )

summary_path = output_dir / "experiment_summary.csv"
with summary_path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(f"Experiment summary saved to {summary_path}")
PY

