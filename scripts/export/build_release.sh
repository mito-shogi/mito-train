#!/usr/bin/env bash
# Build a release tarball from ./models/*.onnx.
#
# Produces (in ./dist/):
#   models-<version>.tar.gz         all *.onnx bundled
#   models-<version>-MANIFEST.json  model -> size / sha256 / realistic accuracy
#   models-<version>-SHA256SUMS     integrity check list
#
# The MANIFEST.json bakes in realistic-eval results
# (see docs/w384-sweep-analysis.md), so a downstream consumer can pick a
# backbone by (accuracy, size) without re-running eval.
#
# Usage:
#   ./scripts/export/build_release.sh v0.1.0
#   VERSION=v0.1.0 ./scripts/export/build_release.sh
#
# Prerequisites: run scripts/export/export_all.sh first to populate ./models/.

set -euo pipefail

VERSION="${1:-${VERSION:-}}"
MODELS_DIR="${MODELS_DIR:-./models}"
DIST_DIR="${DIST_DIR:-./dist}"

if [[ -z "${VERSION}" ]]; then
    echo "usage: $0 <version>  (e.g. $0 v0.1.0)" >&2
    exit 2
fi
if [[ ! "${VERSION}" =~ ^v[0-9]+\.[0-9]+\.[0-9]+ ]]; then
    echo "[build_release] warning: version '${VERSION}' doesn't match vX.Y.Z" >&2
fi

# Files we expect (skip missing so partial releases still bundle what's there).
FILES=(
    board-detector-mnv3s-w384-fp32.onnx
    board-detector-mnv3s-w384-fp16.onnx
    board-detector-mnv3s-w384-int8.onnx
    board-ocr-mobilenet_v3_large-w384-fp32.onnx
    board-ocr-mobilenet_v3_large-w384-fp16.onnx
    board-ocr-efficientnet_b1-w384-fp32.onnx
    board-ocr-efficientnet_b1-w384-fp16.onnx
    board-ocr-convnext_nano-w384-fp32.onnx
)

mkdir -p "${DIST_DIR}"

# Sanity: fail loud if nothing to release.
existing=()
for f in "${FILES[@]}"; do
    if [[ -f "${MODELS_DIR}/${f}" ]]; then
        existing+=("${f}")
    else
        echo "[build_release] missing (skipping): ${f}" >&2
    fi
done
if [[ ${#existing[@]} -eq 0 ]]; then
    echo "[build_release] no models found in ${MODELS_DIR}" >&2
    exit 1
fi
echo "[build_release] bundling ${#existing[@]} files"

# Emit MANIFEST via Python so the accuracy table stays alongside the code
# that describes it (docs/w384-sweep-analysis.md).
uv run python scripts/export/write_manifest.py \
    --version "${VERSION}" \
    --models-dir "${MODELS_DIR}" \
    --out "${DIST_DIR}/models-${VERSION}-MANIFEST.json" \
    "${existing[@]}"

# Human-readable release notes generated FROM the manifest — same source of
# truth for accuracy numbers as the JSON, so they can't drift.
uv run python scripts/export/write_release_notes.py \
    --manifest "${DIST_DIR}/models-${VERSION}-MANIFEST.json" \
    --out "${DIST_DIR}/release-notes-${VERSION}.md"

# SHA256SUMS in the format `sha256sum` verifies against directly.
(cd "${MODELS_DIR}" && sha256sum "${existing[@]}") \
    > "${DIST_DIR}/models-${VERSION}-SHA256SUMS"
echo "[build_release] wrote ${DIST_DIR}/models-${VERSION}-SHA256SUMS"

# tar.gz: relative paths from models/ so extraction lands at $CWD/*.onnx.
tar -czf "${DIST_DIR}/models-${VERSION}.tar.gz" -C "${MODELS_DIR}" "${existing[@]}"
sz=$(du -b "${DIST_DIR}/models-${VERSION}.tar.gz" | cut -f1)
echo "[build_release] wrote ${DIST_DIR}/models-${VERSION}.tar.gz ($((sz / 1024 / 1024)) MB)"

echo
echo "[build_release] === release assets ==="
ls -lh "${DIST_DIR}"/models-${VERSION}* | awk '{printf "  %-8s  %s\n", $5, $9}'

cat <<EOF

Next step — create GitHub Release:
  gh release create models-${VERSION} \\
    --title "Models ${VERSION}" \\
    --notes-file ${DIST_DIR}/release-notes-${VERSION}.md \\
    ${DIST_DIR}/models-${VERSION}.tar.gz \\
    ${DIST_DIR}/models-${VERSION}-MANIFEST.json \\
    ${DIST_DIR}/models-${VERSION}-SHA256SUMS
EOF
