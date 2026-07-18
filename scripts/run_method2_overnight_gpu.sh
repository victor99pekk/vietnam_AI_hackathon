#!/usr/bin/env bash
# GPU-optimized overnight runner for Method 2 evaluation
# Uses 7B+ models with mixed precision training on NVIDIA H200+ GPUs
# Requires CUDA and eval-model dependencies

set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

KG_PATH="${1:-${KG_PATH:-}}"
DRY_RUN="${DRY_RUN:-0}"
CONFIG_PATH="${CONFIG_PATH:-configs/evaluation_overnight_vi.yaml}"
RUN_ROOT="${RUN_ROOT:-output_eval/overnight_gpu}"
RUN_ID="${RUN_ID:-$(date '+%Y%m%d_%H%M%S')}"
RUN_DIR="$RUN_ROOT/$RUN_ID"

if [[ -z "$KG_PATH" ]]; then
  echo "Usage: $0 PATH/TO/knowledge_graph.json"
  echo "Optional: RUN_ROOT=output_eval/overnight_gpu nohup $0 ... &"
  echo "Requires: CUDA GPU with sufficient VRAM (H200: 141GB recommended for 7B models)"
  exit 2
fi

if [[ ! -f "$KG_PATH" ]]; then
  echo "Knowledge graph not found: $KG_PATH"
  exit 2
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Evaluation config not found: $CONFIG_PATH"
  exit 2
fi

if [[ -e "$RUN_DIR" ]]; then
  echo "Refusing to overwrite existing experiment: $RUN_DIR"
  echo "Choose another RUN_ID or omit RUN_ID for a new timestamp."
  exit 2
fi

# Check CUDA availability
if ! python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" 2>/dev/null; then
  echo "Error: CUDA GPU not detected. GPU fine-tuning requires CUDA."
  echo "Use run_method2_overnight.sh for CPU-based fine-tuning instead."
  exit 2
fi

COUNTS="$({
  python3 - "$KG_PATH" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    data = json.load(handle)

structural = {"NEXT", "PART_OF", "MENTIONS"}
domain = 0
for triple in data.get("triples", []):
    predicate = (
        triple.get("predicate", "")
        if isinstance(triple, dict)
        else triple[1] if len(triple) > 1 else ""
    )
    if str(predicate).upper() not in structural:
        domain += 1

chunks = sum(
    1 for node in data.get("graph", {}).get("nodes", [])
    if node.get("type") == "Chunk" and str(node.get("text", "")).strip()
)
print(domain, chunks)
PY
})"
read -r DOMAIN_TRIPLES RAW_CHUNKS <<< "$COUNTS"

if (( DOMAIN_TRIPLES == 0 )); then
  echo "KG has no domain triples after structural edges are excluded: $KG_PATH"
  echo "Rebuild it with relation extraction before starting the overnight run."
  exit 2
fi

if (( RAW_CHUNKS == 0 )); then
  echo "KG has no embedded source chunks for fair Model C training: $KG_PATH"
  exit 2
fi

mkdir -p "$RUN_ROOT"
mkdir "$RUN_DIR"

{
  echo "run_id=$RUN_ID"
  echo "started_at=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "kg_path=$KG_PATH"
  echo "config_path=$CONFIG_PATH"
  echo "device=gpu"
  echo "dry_run=$DRY_RUN"
  echo "domain_triples=$DOMAIN_TRIPLES"
  echo "raw_chunks=$RAW_CHUNKS"
  echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  
  # GPU info
  python3 -c "import torch; print('gpu_device=' + torch.cuda.get_device_name(0)); print('gpu_memory_gb=' + str(torch.cuda.get_device_properties(0).total_mem / 1e9))" || true
} > "$RUN_DIR/experiment.env"

cp "$CONFIG_PATH" "$RUN_DIR/evaluation_config.yaml"

# GPU-optimized models: 3B and 7B for high-VRAM GPUs (H200, A100)
MODEL_NAMES=(
  "Qwen/Qwen2.5-3B-Instruct"
  "Qwen/Qwen2.5-7B-Instruct"
)
MODEL_SLUGS=(
  "qwen2.5-3b-gpu"
  "qwen2.5-7b-gpu"
)

all_ok=true
for index in "${!MODEL_NAMES[@]}"; do
  model="${MODEL_NAMES[$index]}"
  slug="${MODEL_SLUGS[$index]}"
  output_dir="$RUN_DIR/$slug"
  log_path="$RUN_DIR/$slug.log"
  running_marker="$RUN_DIR/$slug.running"

  mkdir "$output_dir"
  printf 'model=%s\nstarted_at=%s\n' "$model" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" > "$running_marker"

  echo "[$(date '+%F %T')] Starting GPU-optimized A/B/C ablation: $model (mixed precision, large batches)"
  command=(
    uv run --extra eval-model python evaluation/run_eval.py
    --method 2
    --kg "$KG_PATH"
    --config "$CONFIG_PATH"
    --output "$output_dir"
    --fine-tune-target both
    --model "$model"
    --gpu
  )

  if [[ "$DRY_RUN" == "1" ]]; then
    printf '%q ' "${command[@]}" | tee "$log_path"
    printf '\n' | tee -a "$log_path"
    mv "$running_marker" "$RUN_DIR/$slug.planned"
    continue
  fi

  if "${command[@]}" 2>&1 | tee "$log_path"; then
    mv "$running_marker" "$RUN_DIR/$slug.complete"
    echo "[$(date '+%F %T')] Completed: $model"
  else
    status=$?
    printf 'exit_status=%s\nfailed_at=%s\n' "$status" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" >> "$running_marker"
    mv "$running_marker" "$RUN_DIR/$slug.failed"
    echo "[$(date '+%F %T')] Failed ($status): $model. Continuing to next model."
    all_ok=false
  fi
done

if [[ "$DRY_RUN" == "1" ]]; then
  printf 'planned_at=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" > "$RUN_DIR/PLANNED"
  echo "Dry run complete. Commands recorded in: $RUN_DIR"
  exit 0
fi

if [[ "$all_ok" == true ]]; then
  printf 'completed_at=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" > "$RUN_DIR/COMPLETE"
  echo "All GPU overnight experiments completed: $RUN_DIR"
  exit 0
fi

printf 'finished_with_failures_at=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" > "$RUN_DIR/FAILED"
echo "One or more experiments failed. Inspect logs in: $RUN_DIR"
exit 1
