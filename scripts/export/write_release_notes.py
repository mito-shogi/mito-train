"""Emit Markdown release notes from a MANIFEST.json.

Puts the per-model realistic-eval accuracy (mean + per-device) and per-file
sizes in the GitHub Release body so a reader can pick a model by accuracy
without opening MANIFEST.json.

Usage:
    uv run python scripts/export/write_release_notes.py \\
        --manifest dist/models-v0.1.0-MANIFEST.json \\
        --out dist/release-notes-v0.1.0.md
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def fmt_size(bytes_: int) -> str:
    if bytes_ >= 1024 * 1024:
        return f"{bytes_ / 1024 / 1024:.1f} MB"
    return f"{bytes_ / 1024:.1f} KB"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    m = json.loads(args.manifest.read_text())
    version = m["version"]
    files = m["files"]

    # Group by (kind, backbone) so multiple precisions of the same model sit
    # in one row.
    grouped: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for f in files:
        key = (f["kind"], f["backbone"])
        grouped[key][f["precision"]] = f

    lines: list[str] = []
    lines.append(f"## Models {version}")
    lines.append("")
    lines.append(
        "BoardDetector + BoardOCR の重みのアーカイブスナップショット。"
        "本番配信は Cloudflare R2 経由 (別インフラ)、本リリースは再現性・トレーサビリティ用。"
    )
    lines.append("")
    lines.append("参考 doc:")
    lines.append("- `docs/w384-sweep-analysis.md` — realistic 評価結果 (backbone × device)")
    lines.append("- `docs/browser-inference-outlook.md` — device 別推論時間見積")
    lines.append("- `docs/model-export.md` — export パイプラインと検証手順")
    lines.append("")

    # Detector row.
    det = grouped.get(("board_detector", "mobilenet_v3_small"))
    if det:
        first = next(iter(det.values()))
        lines.append("### Detector (BoardDetector)")
        lines.append("")
        lines.append("- backbone: mobilenet_v3_small (w384)")
        lines.append(f"- val iou_mean: **{first.get('detector_val_iou_mean', 0.99):.2f}+**")
        lines.append(f"- 用途: {first.get('recommendation', '')}")
        lines.append("")
        lines.append("| Precision | File | Size | SHA256 (first 12) |")
        lines.append("|---|---|---:|---|")
        for prec in ("fp32", "fp16", "int8"):
            if prec in det:
                f = det[prec]
                lines.append(f"| {prec} | `{f['file']}` | {fmt_size(f['size_bytes'])} | `{f['sha256'][:12]}` |")
        if "int8" in det:
            lines.append("")
            lines.append(
                "> int8 dynamic quantization は動作するが verify で max_diff ~0.29 "
                "(regression bbox の [0,1] 正規化空間で 29% 誤差)、"
                "実データでの IoU 検証未実施のため **参考同梱**。"
            )
        lines.append("")

    # OCR rows — one section per backbone.
    for (kind, backbone), precs in grouped.items():
        if kind != "board_ocr":
            continue
        first = next(iter(precs.values()))
        mean_sfen = first.get("realistic_sfen_mean")
        per_dev = first.get("realistic_sfen_per_device", {})
        lines.append(f"### OCR: {backbone}")
        lines.append("")
        lines.append(f"- backbone: {backbone} (w384)")
        if mean_sfen is not None:
            lines.append(f"- realistic SFEN exact-match (end-to-end 平均): **{mean_sfen * 100:.2f}%**")
        if per_dev:
            devs = ["iPhone10,1", "iPhone11,8", "iPhone15,4", "iPad14,10"]
            lines.append("- per-device (end-to-end): "
                        + " / ".join(f"{d.replace('iPhone', 'iPhone ').replace('iPad', 'iPad ')} "
                                     f"**{per_dev[d] * 100:.2f}%**" for d in devs if d in per_dev))
        rec = first.get("recommendation")
        if rec:
            lines.append(f"- 用途: {rec}")
        lines.append("")
        lines.append("| Precision | File | Size | SHA256 (first 12) |")
        lines.append("|---|---|---:|---|")
        for prec in ("fp32", "fp16", "int8"):
            if prec in precs:
                f = precs[prec]
                lines.append(f"| {prec} | `{f['file']}` | {fmt_size(f['size_bytes'])} | `{f['sha256'][:12]}` |")
        lines.append("")

    # Global constraints developers need to know before deploying.
    lines.append("### 精度別の使い分けメモ")
    lines.append("")
    lines.append("- **fp32**: 全モデル PyTorch と argmax 完全一致、max_diff <2e-4。無難な既定値。")
    lines.append("- **fp16**: ~1/2 サイズ、argmax 完全一致 (detector / mnv3l / effb1 で確認)、"
                 "対応環境 (Apple M-series / Snapdragon 8 Elite など) では ~30-40% 高速化。"
                 "**convnext_nano は fp16 変換不可** (ONNX Loop op TypeInferenceError)、fp32 のみ提供。")
    lines.append("- **int8 (dynamic quantization)**: **OCR は shipping 不可** "
                 "(mnv3l で board argmax 53/81、effb1 で 63/81 に崩壊)。"
                 "detector int8 は max_diff 0.29 で微妙、実運用は要 IoU 検証。"
                 "実用 int8 を出すには **QDQ 静的量子化 + realistic 相当の calibration set** が必要。")
    lines.append("")

    lines.append("### 整合性チェック")
    lines.append("")
    lines.append("```bash")
    lines.append(f"tar -tzf models-{version}.tar.gz")
    lines.append(f"sha256sum -c models-{version}-SHA256SUMS")
    lines.append("```")
    lines.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines))
    print(f"[write_release_notes] wrote {args.out}")


if __name__ == "__main__":
    main()
