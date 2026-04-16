#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_ROOT="${1:-$SCRIPT_DIR/../style-salad_}"

if [[ ! -d "$SOURCE_ROOT" ]]; then
  echo "Source repo not found: $SOURCE_ROOT" >&2
  exit 1
fi

link_path() {
  local source_path="$1"
  local target_path="$2"

  mkdir -p "$(dirname "$target_path")"

  if [[ -L "$target_path" ]]; then
    local current_target
    current_target="$(readlink "$target_path")"
    if [[ "$current_target" == "$source_path" ]]; then
      echo "ok   $target_path -> $source_path"
      return 0
    fi
    rm "$target_path"
  elif [[ -e "$target_path" ]]; then
    echo "skip $target_path already exists"
    return 0
  fi

  ln -s "$source_path" "$target_path"
  echo "link $target_path -> $source_path"
}

link_path "../style-salad_/checkpoints" "$SCRIPT_DIR/checkpoints"
link_path "../style-salad_/glove" "$SCRIPT_DIR/glove"
link_path "../style-salad_/kit" "$SCRIPT_DIR/kit"
link_path "../style-salad_/save" "$SCRIPT_DIR/save"
link_path "../style-salad_/t2m" "$SCRIPT_DIR/t2m"

link_path "../../style-salad_/dataset/100style" "$SCRIPT_DIR/dataset/100style"
link_path "../../style-salad_/dataset/kit-ml" "$SCRIPT_DIR/dataset/kit-ml"
link_path "../../style-salad_/dataset/smoodi" "$SCRIPT_DIR/dataset/smoodi"

link_path "../../style-salad_/salad/__init__.py" "$SCRIPT_DIR/salad/__init__.py"
link_path "../../style-salad_/salad/assets" "$SCRIPT_DIR/salad/assets"
link_path "../../style-salad_/salad/common" "$SCRIPT_DIR/salad/common"
link_path "../../style-salad_/salad/data" "$SCRIPT_DIR/salad/data"
link_path "../../style-salad_/salad/glove" "$SCRIPT_DIR/salad/glove"
link_path "../../style-salad_/salad/models" "$SCRIPT_DIR/salad/models"
link_path "../../style-salad_/salad/motion_loaders" "$SCRIPT_DIR/salad/motion_loaders"
link_path "../../style-salad_/salad/options" "$SCRIPT_DIR/salad/options"
link_path "../../style-salad_/salad/prepare" "$SCRIPT_DIR/salad/prepare"
link_path "../../style-salad_/salad/setup.py" "$SCRIPT_DIR/salad/setup.py"
link_path "../../style-salad_/salad/utils" "$SCRIPT_DIR/salad/utils"
link_path "../../style-salad_/salad/visualization" "$SCRIPT_DIR/salad/visualization"

link_path "../../style-salad_/motion-diffusion-model/assets" "$SCRIPT_DIR/motion-diffusion-model/assets"
link_path "../../style-salad_/motion-diffusion-model/body_models" "$SCRIPT_DIR/motion-diffusion-model/body_models"
link_path "../../style-salad_/motion-diffusion-model/data_loaders" "$SCRIPT_DIR/motion-diffusion-model/data_loaders"
link_path "../../style-salad_/motion-diffusion-model/dataset" "$SCRIPT_DIR/motion-diffusion-model/dataset"
link_path "../../style-salad_/motion-diffusion-model/diffusion" "$SCRIPT_DIR/motion-diffusion-model/diffusion"
link_path "../../style-salad_/motion-diffusion-model/eval" "$SCRIPT_DIR/motion-diffusion-model/eval"
link_path "../../style-salad_/motion-diffusion-model/model" "$SCRIPT_DIR/motion-diffusion-model/model"
link_path "../../style-salad_/motion-diffusion-model/prepare" "$SCRIPT_DIR/motion-diffusion-model/prepare"
link_path "../../style-salad_/motion-diffusion-model/sample" "$SCRIPT_DIR/motion-diffusion-model/sample"
link_path "../../style-salad_/motion-diffusion-model/train" "$SCRIPT_DIR/motion-diffusion-model/train"
link_path "../../style-salad_/motion-diffusion-model/utils" "$SCRIPT_DIR/motion-diffusion-model/utils"
link_path "../../style-salad_/motion-diffusion-model/visualize" "$SCRIPT_DIR/motion-diffusion-model/visualize"
