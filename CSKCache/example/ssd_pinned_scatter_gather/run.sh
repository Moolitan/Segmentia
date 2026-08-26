#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source /home/wsh/miniconda3/etc/profile.d/conda.sh
conda activate opencode
set -u

cd "${SCRIPT_DIR}"
python run.py
