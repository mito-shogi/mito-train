"""ぴよ将棋クリーン画像用の PyTorch Dataset."""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import torch
from PIL import Image
from torch.utils.data import Dataset


class PiyoDataset(Dataset):
    """`manifest-{split}.jsonl` を読んで `(image, sfen_normalized)` を返す Dataset."""

    def __init__(
        self,
        data_root: str | Path,
        split: Literal["train", "valid"],
        transform: Callable | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform

        manifest_path = self.data_root / f"manifest-{split}.jsonl"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"manifest not found: {manifest_path}. "
                "run scripts/download-data.sh first."
            )

        self.entries: list[dict] = []
        with open(manifest_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.entries.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor | Image.Image, str]:
        entry = self.entries[idx]
        file_basename = Path(entry["file"]).name
        image_path = self.data_root / file_basename
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, entry["sfen_normalized"]
