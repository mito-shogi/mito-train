from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from mito_train.datasets import PiyoDataset


def _write_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "piyo-train"
    root.mkdir()
    (root / "manifest-train.jsonl").write_text(
        json.dumps({"file": "d0/00.png", "sfen_normalized": "startpos"}) + "\n"
    )
    Image.new("RGB", (32, 32), (255, 0, 0)).save(root / "00.png")
    return root


def test_reads_manifest_and_image(tmp_path: Path) -> None:
    root = _write_fixture(tmp_path)
    ds = PiyoDataset(root, split="train")
    assert len(ds) == 1
    img, sfen = ds[0]
    assert sfen == "startpos"
    assert img.size == (32, 32)


def test_missing_manifest_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        PiyoDataset(tmp_path, split="train")
