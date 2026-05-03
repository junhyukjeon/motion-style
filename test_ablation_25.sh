#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

DICT_PATH="${DICT_PATH:-$SCRIPT_DIR/dataset/100style/100STYLE_name_dict.txt}"
CONTENT_JSON_PATH="${CONTENT_JSON_PATH:-$SCRIPT_DIR/dataset/100style/10content_smoodi.json}"
MOTION_DIR="${MOTION_DIR:-$SCRIPT_DIR/dataset/100style/new_joint_vecs}"
CAPTION="${CAPTION:-a person moves sideways}"

mkdir -p "$TMP_DIR/configs/final_final"

if [[ ! -f "$DICT_PATH" ]]; then
  echo "Dictionary file not found: $DICT_PATH" >&2
  exit 1
fi

if [[ ! -f "$CONTENT_JSON_PATH" ]]; then
  echo "Content JSON file not found: $CONTENT_JSON_PATH" >&2
  exit 1
fi

if [[ ! -d "$MOTION_DIR" ]]; then
  echo "Motion directory not found: $MOTION_DIR" >&2
  exit 1
fi

mapfile -t STYLE_REFS < <(
  DICT_PATH="$DICT_PATH" CONTENT_JSON_PATH="$CONTENT_JSON_PATH" MOTION_DIR="$MOTION_DIR" python - <<'PY'
import json
import os
from collections import OrderedDict

dict_path = os.environ["DICT_PATH"]
content_json_path = os.environ["CONTENT_JSON_PATH"]
motion_dir = os.environ["MOTION_DIR"]

with open(content_json_path, "r", encoding="utf-8") as f:
    content_map = json.load(f)

valid_fw_ids = set(content_map.get("FW", []))
styles = OrderedDict()

with open(dict_path, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 3:
            continue

        motion_id, filename, style_idx = parts[:3]
        if motion_id.startswith("M") or "_FW_" not in filename:
            continue

        style_name = filename.split("_FW_", 1)[0]
        motion_path = os.path.join(motion_dir, f"{motion_id}.npy")
        if motion_id not in valid_fw_ids or not os.path.exists(motion_path):
            continue

        key = (style_idx, style_name)
        styles.setdefault(key, []).append(motion_id)

for (style_idx, style_name), motion_ids in styles.items():
    print(f"{style_idx}\t{style_name}\t{','.join(motion_ids)}")
PY
)

if [[ "${#STYLE_REFS[@]}" -eq 0 ]]; then
  echo "Failed to find any FW reference motions in $DICT_PATH" >&2
  exit 1
fi

echo "Running ablation for ${#STYLE_REFS[@]} styles"
echo "Caption: $CAPTION"
echo

run_case() {
  local base_config="$1"
  local temp_config="$2"
  local ref_motion_id="$3"
  local use_zero_guidance="$4"
  local log_file="$TMP_DIR/run_${ref_motion_id}_$(basename "$temp_config")_${use_zero_guidance}.log"

  cp "$base_config" "$temp_config"
  if [[ "$use_zero_guidance" == "1" ]]; then
    sed -i 's/style_guidance: 0.75/style_guidance: 0.0/' "$temp_config"
  fi

  if python get_teaser_current.py \
    --config "$temp_config" \
    --ref_motion_id "$ref_motion_id" \
    --caption "$CAPTION" >"$log_file" 2>&1; then
    return 0
  fi

  cat "$log_file" >&2
  if grep -q "not found in dataset" "$log_file"; then
    return 2
  fi

  return 1
}

run_style_with_ref() {
  local ref_motion_id="$1"

  run_case \
    "$SCRIPT_DIR/configs/final_final/25.yaml" \
    "$TMP_DIR/configs/final_final/25.yaml" \
    "$ref_motion_id" \
    "0" || return $?

  run_case \
    "$SCRIPT_DIR/configs/final_final/25.yaml" \
    "$TMP_DIR/configs/final_final/25.yaml" \
    "$ref_motion_id" \
    "1" || return $?

  run_case \
    "$SCRIPT_DIR/configs/final_final/25_supcon.yaml" \
    "$TMP_DIR/configs/final_final/25_supcon.yaml" \
    "$ref_motion_id" \
    "0" || return $?

  run_case \
    "$SCRIPT_DIR/configs/final_final/25_supcon.yaml" \
    "$TMP_DIR/configs/final_final/25_supcon.yaml" \
    "$ref_motion_id" \
    "1" || return $?
}

for i in "${!STYLE_REFS[@]}"; do
  IFS=$'\t' read -r STYLE_IDX STYLE_NAME REF_MOTION_IDS_CSV <<< "${STYLE_REFS[$i]}"
  RUN_NUM=$((i + 1))

  IFS=',' read -r -a REF_MOTION_IDS <<< "$REF_MOTION_IDS_CSV"

  echo "[$RUN_NUM/${#STYLE_REFS[@]}] style_idx=$STYLE_IDX style=$STYLE_NAME candidates=${#REF_MOTION_IDS[@]}"

  STYLE_DONE=0
  for REF_MOTION_ID in "${REF_MOTION_IDS[@]}"; do
    echo "  trying ref_motion_id=$REF_MOTION_ID"

    if run_style_with_ref "$REF_MOTION_ID"; then
      echo "  selected ref_motion_id=$REF_MOTION_ID"
      STYLE_DONE=1
      break
    fi

    STATUS=$?
    if [[ "$STATUS" -eq 2 ]]; then
      echo "  ref_motion_id=$REF_MOTION_ID is unavailable in the runtime dataset, trying the next FW clip"
      continue
    fi

    echo "  ref_motion_id=$REF_MOTION_ID failed for another reason; stopping." >&2
    exit "$STATUS"
  done

  if [[ "$STYLE_DONE" -ne 1 ]]; then
    echo "  skipping style=$STYLE_NAME because no usable FW reference motion was found"
  fi

  echo
done
