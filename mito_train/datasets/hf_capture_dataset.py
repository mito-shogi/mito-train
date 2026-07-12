"""HF Hub-backed variant of CaptureDataset.

Loads the dataset from a Hugging Face dataset repo where each row is
{"images": [<PIL image>, ...], "devices": [<str>, ...], "sfen": <str>, "hash": <str>}.
Each sfen is paired with multiple device renderings; __getitem__ picks one
device per call (random for train, fixed index for val). See HFPairedDataset
for the alternative flatten-per-device approach.

Usage (in training):
    from mito_train.datasets import HFCaptureDataset, build_transform
    train_ds = HFCaptureDataset(
        repo_id="ultemica/piyoshogi",
        split="train",
        transform=build_transform("train", image_size=288),
    )

First call downloads the parquet shards into `~/.cache/huggingface/datasets/`.
Subsequent runs are cache-hits.
"""
from __future__ import annotations
import fcntl
import os
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import albumentations as A
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from .capture_dataset import build_transform
from .sfen_utils import parse_sfen

DEFAULT_PRELOAD_CACHE_DIR = Path(
    os.environ.get("MITO_PRELOAD_CACHE", Path.home() / ".cache" / "mito-train" / "preload")
)


def _decode_resize_pad(entry_bytes: bytes, image_size: int) -> np.ndarray:
    """WebP decode + LongestMaxSize + center-pad-to-square (all deterministic).

    Returns (image_size, image_size, 3) uint8 RGB. This encapsulates the fixed
    part of the training pipeline so it can be cached once and skipped every
    epoch. cv2 ops release the GIL, so a ThreadPoolExecutor scales.
    """
    buf = np.frombuffer(entry_bytes, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR_RGB)
    h, w = img.shape[:2]
    scale = image_size / max(h, w)
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    pad_h = image_size - new_h
    pad_w = image_size - new_w
    top = pad_h // 2
    left = pad_w // 2
    return cv2.copyMakeBorder(
        resized, top, pad_h - top, left, pad_w - left,
        cv2.BORDER_CONSTANT, value=0,
    )


class HFCaptureDataset(Dataset):
    """CaptureDataset backed by a Hugging Face Hub dataset repo."""

    def __init__(
        self,
        repo_id: str,
        split: str = "train",
        transform: Callable | A.Compose | None = None,
        limit: int | None = None,
        cache_dir: str | None = None,
        streaming: bool = False,
        config_name: str = "ocr_paired",
        device_index: int | None = None,
        preload: bool = False,
        preload_image_size: int = 224,
        preload_workers: int = 16,
        preload_cache_dir: str | Path | None = None,
    ) -> None:
        """
        Parameters
        ----------
        repo_id:              Hugging Face dataset repo (e.g. "ultemica/piyoshogi")
        split:                "train" or "val"
        transform:            Albumentations Compose or callable. Defaults to val transform.
        limit:                Truncate to first N rows (smoke tests).
        cache_dir:            Override HF cache location. None = ~/.cache/huggingface/datasets
        streaming:            If True, stream from Hub (no local cache). Random access breaks;
                              __getitem__ becomes O(n). Use only for one-shot iteration.
        config_name:          HF dataset config (e.g. "ocr_paired", "detector_paired").
        device_index:         Each row holds a list of device renderings for one sfen.
                              None (default) picks one uniformly at random per __getitem__
                              call — good for training (sees all devices across epochs).
                              int uses that index (clamped) — good for stable val metrics.
        preload:              If True, decode + resize + pad every image at init time and
                              hold them in a single uint8 ndarray. Removes WebP decode
                              and the deterministic resize/pad from the hot loop entirely.
                              Cost: ~preload_image_size^2 * 3 * N * num_devices bytes of RAM
                              (~10.5 GB for 18k rows x 4 devices at 224). Fork+COW shares
                              this across DataLoader workers on Linux.
        preload_image_size:   Target square size for the preloaded arrays. Should match
                              the image_size used when building the augmentation transform.
        preload_workers:      Threads used to decode during preload. cv2.imdecode releases
                              the GIL so this scales well up to physical core count.
        preload_cache_dir:    Directory holding the on-disk preload cache. Second and
                              later runs mmap the cache and skip decode entirely.
                              None -> $MITO_PRELOAD_CACHE or ~/.cache/mito-train/preload.
        """
        from datasets import Image as HFImage
        from datasets import Sequence, load_dataset

        self.transform = transform if transform is not None else build_transform("val")
        self.streaming = streaming
        self.device_index = device_index

        ds = load_dataset(
            repo_id,
            config_name,
            split=split,
            cache_dir=cache_dir,
            streaming=streaming,
        )
        # Disable auto-decode so row access returns raw {bytes, path} per image.
        # We pick one device index first, then decode only that single WebP.
        # Cuts per-row decode cost by ~len(devices) (typically 4x).
        ds = ds.cast_column("images", Sequence(HFImage(decode=False)))
        if limit is not None and not streaming:
            ds = ds.select(range(min(limit, len(ds))))
        self.ds = ds

        if streaming:
            # Streaming datasets are one-shot iterables; expose an iterator only.
            self._length = limit
        else:
            self._length = len(ds)

        self._cache: np.ndarray | None = None
        if preload and not streaming:
            cache_dir = Path(preload_cache_dir) if preload_cache_dir is not None else DEFAULT_PRELOAD_CACHE_DIR
            self._preload(preload_image_size, preload_workers, cache_dir, repo_id, config_name, split)

    def _cache_path(
        self, cache_dir: Path, repo_id: str, config_name: str,
        split: str, image_size: int,
    ) -> Path:
        """Build a stable cache path keyed by the dataset identity + image size.

        HF's `_fingerprint` also encodes any `.select(...)` truncation, so a
        `--limit N` run has a different cache from the full run — no risk of
        pulling a truncated cache in full mode.
        """
        fingerprint = getattr(self.ds, "_fingerprint", "nofp")
        safe_repo = repo_id.replace("/", "_")
        name = f"{safe_repo}__{config_name}__{split}__{fingerprint}__sz{image_size}.npy"
        return cache_dir / name

    def _preload(
        self, image_size: int, workers: int, cache_dir: Path,
        repo_id: str, config_name: str, split: str,
    ) -> None:
        """Decode + resize + pad every device rendering into an on-disk ndarray.

        Shape: (N, num_devices, image_size, image_size, 3) uint8 RGB.

        First call writes to `cache_dir/<key>.npy`; subsequent calls memory-map
        the file and skip decode entirely (~30-60s startup collapses to <1s).

        DDP: only rank 0 builds. Other ranks wait on a barrier and mmap the same
        file, sharing the kernel page cache instead of duplicating the array.

        PARALLEL_GPU sweeps: many independent Python processes may race on the
        same cache path. An fcntl advisory lock on a sibling .lock file
        serializes them cross-process — the first process builds, the rest wait
        on the lock and then find the cache already there.
        """
        from mito_train.training.dist_utils import barrier, is_main

        path = self._cache_path(cache_dir, repo_id, config_name, split, image_size)

        if is_main():
            self._maybe_build_with_lock(path, image_size, workers)

        # All ranks meet here: rank 0 has finished writing, others were idle.
        barrier()

        arr = np.load(path, mmap_mode="r")
        gb = arr.nbytes / (1024 ** 3)
        if is_main():
            print(f"[HFCaptureDataset] preload cache ready: {path} ({gb:.2f} GB, mmap)")
        self._cache = arr

    def _maybe_build_with_lock(
        self, path: Path, image_size: int, workers: int,
    ) -> None:
        """Cross-process safe cache-or-build.

        Fast path: file exists -> return immediately without touching the lock.
        Slow path: take an exclusive fcntl lock on a sibling .lock file, then
        re-check under lock (another process may have built while we waited)
        before doing the actual decode + save.
        """
        if path.exists():
            return

        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = path.with_name(path.name + ".lock")

        with open(lock_path, "w") as lockf:
            print(f"[HFCaptureDataset] acquiring build lock: {lock_path}")
            fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
            # Re-check: while we were blocked, another process may have finished.
            if path.exists():
                print("[HFCaptureDataset] cache built by another process while waiting; skipping")
                return
            self._build_and_save_cache(path, image_size, workers)

    def _build_and_save_cache(
        self, path: Path, image_size: int, workers: int,
    ) -> None:
        """Decode all rows and persist as an atomic .npy file. Rank 0 only."""
        n = self._length
        first_row = self.ds[0]
        num_devices = len(first_row["images"])
        cache = np.zeros(
            (n, num_devices, image_size, image_size, 3), dtype=np.uint8,
        )

        # Grab the raw bytes column up-front so worker threads don't hit the
        # HF dataset object concurrently (its indexing isn't thread-safe).
        all_images = self.ds["images"]

        def process(idx: int) -> None:
            entries = all_images[idx]
            for dev_idx in range(min(num_devices, len(entries))):
                cache[idx, dev_idx] = _decode_resize_pad(
                    entries[dev_idx]["bytes"], image_size,
                )
            for dev_idx in range(len(entries), num_devices):
                cache[idx, dev_idx] = cache[idx, len(entries) - 1]

        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(tqdm(
                pool.map(process, range(n)),
                total=n, desc=f"preload {image_size}px",
            ))
        gb = cache.nbytes / (1024 ** 3)
        print(f"[HFCaptureDataset] preloaded {n} rows x {num_devices} devices = {gb:.2f} GB")

        # Atomic write: tmp then rename. Handles Ctrl-C mid-write cleanly.
        # File-object form so numpy doesn't re-append ".npy" to our tmp name.
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as f:
            np.save(f, cache)
        os.replace(tmp, path)
        print(f"[HFCaptureDataset] preload cache saved: {path}")

    @property
    def entries(self):
        """Sfen-only entries list, compatible with compute_hand_class_weights."""
        if self.streaming:
            raise RuntimeError(
                "entries is not available in streaming mode. "
                "Disable streaming to compute class weights."
            )
        # Only pulls the sfen column (fast; ~4MB for 27k rows), no image decode.
        return [{"sfen": s} for s in self.ds["sfen"]]

    def __len__(self) -> int:
        if self._length is None:
            raise RuntimeError("Streaming dataset without --limit has unknown length.")
        return self._length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.streaming:
            raise RuntimeError("Streaming datasets do not support random access; iterate instead.")

        if self._cache is not None:
            # Preloaded path: decode + resize + pad were already done at init.
            # LongestMaxSize and PadIfNeeded in the transform become no-ops
            # because the input is already the target square size.
            num_devices = self._cache.shape[1]
            if self.device_index is None:
                dev_idx = random.randrange(num_devices)
            else:
                dev_idx = min(self.device_index, num_devices - 1)
            # .copy() defends against Albumentations in-place ops mutating the cache.
            img_np = self._cache[idx, dev_idx].copy()
            sfen = self.ds[idx]["sfen"]
        else:
            row = self.ds[idx]
            images = row["images"]  # list[{"bytes": ..., "path": ...}] — undecoded
            if self.device_index is None:
                entry = random.choice(images)
            else:
                entry = images[min(self.device_index, len(images) - 1)]

            # cv2.imdecode releases the GIL (C++ libwebp), so multiple workers
            # actually decode in parallel. PIL holds the GIL during decode.
            # IMREAD_COLOR_RGB (OpenCV >= 4.10) decodes straight to RGB, saving
            # the extra full-image cvtColor copy per sample.
            buf = np.frombuffer(entry["bytes"], dtype=np.uint8)
            img_np = cv2.imdecode(buf, cv2.IMREAD_COLOR_RGB)
            sfen = row["sfen"]

        parsed = parse_sfen(sfen)
        board = torch.tensor(parsed.board, dtype=torch.long)  # (9,9)
        hand = torch.tensor(parsed.hand, dtype=torch.long)    # (14,)

        if isinstance(self.transform, A.Compose):
            out = self.transform(image=img_np)
            image_tensor = out["image"]
        else:
            image_tensor = self.transform(Image.fromarray(img_np))

        return image_tensor, board, hand
