"""Dataset for the piece-classifier: iterates pieces/{theme}/{theme}_XXX_Normal.png.

Filenames follow the `{theme}_{code3}_Normal.png` format. code3 is the piyo-shogi
piece ID (3 digits: [side][promoted-flag][piece-type]).

For the current smoke-test usage:
- Overfit a single theme (k1) to confirm the pipeline works
- The contract (docs/ocr-model-interface.md) is not yet synced, so instead of
  pinning to 29 classes, sort the code3 values that actually appear in the data
  and map them to 0..N-1
- Once the contract lands, swap this in for a code3 -> contract class ID map
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import Dataset

FILENAME_RE = re.compile(r"^(?P<theme>[^_]+)_(?P<code>\d{3})_Normal\.png$")


class PiecesDataset(Dataset):
    """In-memory Dataset that loads every image under `data/pieces/{theme}/*.png`.

    Assumes small data (about 30 images per theme). All images are loaded into
    PIL at __init__ time; the transform is applied in __getitem__ on each access.

    Parameters
    ----------
    root:
        Directory pointing to `data/pieces`.
    themes:
        List of theme names to use (e.g. ["k1"]). When None, all subdirectories
        directly under root are used.
    transform:
        Function to convert PIL.Image -> torch.Tensor. When None, uses the
        equivalent of `torchvision.transforms.ToTensor()` internally.
    label_map:
        Mapping from code3 (str) to class ID (int). When None, the code3 values
        that appear in the data are sorted ascending and assigned 0..N-1.
        Provided as a hook for later injecting a contract-compliant (29-class) map.
    """

    def __init__(
        self,
        root: Path,
        themes: list[str] | None = None,
        transform: Callable | None = None,
        label_map: dict[str, int] | None = None,
    ) -> None:
        self.root = Path(root)
        self.transform = transform

        theme_dirs: list[Path]
        if themes is None:
            theme_dirs = sorted(p for p in self.root.iterdir() if p.is_dir())
        else:
            theme_dirs = [self.root / t for t in themes]

        entries: list[tuple[Path, str, str]] = []
        for tdir in theme_dirs:
            if not tdir.is_dir():
                raise FileNotFoundError(f"theme directory missing: {tdir}")
            for png in sorted(tdir.glob("*.png")):
                m = FILENAME_RE.match(png.name)
                if not m:
                    continue
                entries.append((png, m["theme"], m["code"]))

        if not entries:
            raise RuntimeError(f"no files found: {self.root}")

        if label_map is None:
            codes = sorted({code for _, _, code in entries})
            label_map = {c: i for i, c in enumerate(codes)}
        self.label_map = label_map

        self.samples: list[tuple[Image.Image, int, str, str]] = []
        for path, theme, code in entries:
            if code not in label_map:
                raise KeyError(f"code {code} not in label_map ({path})")
            img = Image.open(path)
            # RGBA -> RGB (composited on white background) so transparent pixels do not turn pure black.
            if img.mode == "RGBA":
                bg = Image.new("RGB", img.size, (255, 255, 255))
                bg.paste(img, mask=img.split()[3])
                img = bg
            else:
                img = img.convert("RGB")
            self.samples.append((img, label_map[code], theme, code))

    @property
    def num_classes(self) -> int:
        return len(self.label_map)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img, label, _theme, _code = self.samples[idx]
        if self.transform is not None:
            tensor = self.transform(img)
        else:
            from torchvision.transforms.functional import to_tensor
            tensor = to_tensor(img)
        return tensor, label
