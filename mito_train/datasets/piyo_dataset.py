"""PyTorch Dataset for clean piyo-shogi images."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Callable, Literal

import torch
from torch.utils.data import Dataset
from PIL import Image


class PiyoDataset(Dataset):
    """
    Read manifest-train.jsonl / manifest-valid.jsonl and return
    (image_tensor, sfen_str).

    Preprocessing differs across board / piece / hand, so a transform function is
    injected.
    """

    def __init__(
        self,
        data_root: Path,
        split: Literal['train', 'valid'],
        transform: Callable | None = None,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform

        manifest_path = self.data_root / f'manifest-{split}.jsonl'
        self.entries: list[dict] = []
        with open(manifest_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.entries.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        entry = self.entries[idx]
        # The "file" field in the manifest is like "d0/xxx.png", but the actual
        # layout is flat -> take the basename and read from data_root directly.
        file_basename = Path(entry['file']).name
        image_path = self.data_root / file_basename
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, entry['sfen_normalized']
