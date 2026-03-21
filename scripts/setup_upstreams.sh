#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

clone_if_missing() {
  local url="$1"
  local target="$2"

  if [[ -e "$repo_root/$target" ]]; then
    echo "Skipping $target because it already exists at $repo_root/$target"
    return 0
  fi

  echo "Cloning $url into $target"
  git clone "$url" "$repo_root/$target"
}

clone_if_missing "https://github.com/MIAGroupUT/IDIR.git" "IDIR"
clone_if_missing "https://github.com/jqmcginnis/multi_contrast_inr.git" "multi_contrast_inr"

echo "Upstream repositories are available."
