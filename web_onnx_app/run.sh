#!/usr/bin/env bash
set -euo pipefail

PORT="${1:-8080}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_ROOT="${ROOT}/export_onnx"

# if [[ ! -d "${MODEL_ROOT}" ]]; then
#   echo "Missing model directory: ${MODEL_ROOT}" >&2
#   echo "Run the ONNX export first, or update web_onnx_app/app.js to point at the exported directory." >&2
#   exit 1
# fi

echo "Serving AFTER ONNX web app at http://localhost:${PORT}/web_onnx_app/"
echo "Repository root: ${ROOT}"
echo "Model root:      http://localhost:${PORT}/export_onnx/"
python3 -m http.server "${PORT}" --directory "${ROOT}"
