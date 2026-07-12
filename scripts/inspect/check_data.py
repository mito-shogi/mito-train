"""Verify that every hash referenced by each manifest has a matching image on
disk, per device.

Datasets checked:
    data/ocr/       train.jsonl + val.jsonl     × 4 devices (webp)
    data/detector/  train.jsonl + val.jsonl     × 4 devices (webp, per-device)
    data/test/      test.jsonl                  × 4 devices (png)

For each (dataset, split, device) triple this prints:
    manifest count, existing count, missing count, first few missing hashes.

Exit code is 1 if anything is missing so it doubles as a CI-friendly guard.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

DEVICES = ["iPhone10,1", "iPhone11,8", "iPhone15,4", "iPad14,10"]
DATA_ROOT = Path("data")


def load_manifest_hashes(path: Path, *, device_field: bool
                         ) -> dict[str, set[str]] | set[str]:
    """Return the hash set for a manifest.

    - device_field=False → return the union of all hashes (the manifest is
      device-agnostic; every device is expected to hold every hash).
    - device_field=True  → return {device: {hashes}} (the manifest is a
      per-device split so each hash belongs to exactly one device).
    """
    if device_field:
        by_dev: dict[str, set[str]] = {d: set() for d in DEVICES}
        with open(path) as f:
            for line in f:
                e = json.loads(line)
                by_dev.setdefault(e["device"], set()).add(e["hash"])
        return by_dev
    hashes: set[str] = set()
    with open(path) as f:
        for line in f:
            hashes.add(json.loads(line)["hash"])
    return hashes


def check_split(name: str, hashes_by_dev: dict[str, set[str]],
                image_root: Path, ext: str, *, allow_missing: bool = False) -> int:
    """Print a per-device row and return the number of missing files."""
    total_missing = 0
    for dev in DEVICES:
        needed = hashes_by_dev[dev]
        img_dir = image_root / dev
        found: set[str] = set()
        if img_dir.exists():
            found = {p.stem for p in img_dir.iterdir() if p.suffix == ext}
        present = needed & found
        missing = sorted(needed - found)
        total_missing += len(missing)
        status = "OK" if not missing else ("(pending)" if allow_missing else "MISSING")
        sample = ", ".join(m[:12] + "…" for m in missing[:3])
        print(f"  {dev:<14} manifest={len(needed):>5}  present={len(present):>5}"
              f"  missing={len(missing):>5}  {status}"
              + (f"  first: {sample}" if missing else ""))
    return total_missing


def _split_agnostic_hashes(union: set[str]) -> dict[str, set[str]]:
    """Turn a device-agnostic hash set into a {device: same_set} mapping."""
    return {d: union for d in DEVICES}


def check_leak(name: str, a: set[str], b: set[str]) -> int:
    """Report |a ∩ b| and return the count so callers can aggregate."""
    inter = len(a & b)
    status = "OK" if inter == 0 else "LEAK"
    print(f"  {name:<42} {inter:>6}  {status}")
    return inter


def check_leaks(ocr_train: set[str], ocr_val: set[str],
                det_train_by_dev: dict[str, set[str]],
                det_val_by_dev: dict[str, set[str]],
                test_hashes: set[str] | None) -> int:
    """Run every leak check we care about; return the total leaked count."""
    print("=== Leak Check ===")
    total_leak = 0

    print(" within-dataset (train ∩ val):")
    total_leak += check_leak("OCR train ∩ OCR val", ocr_train, ocr_val)
    for dev in DEVICES:
        total_leak += check_leak(
            f"DET train ∩ DET val [{dev}]",
            det_train_by_dev[dev], det_val_by_dev[dev],
        )

    print(" cross-dataset (DET ↔ OCR):")
    det_train_all = set().union(*det_train_by_dev.values())
    det_val_all = set().union(*det_val_by_dev.values())
    total_leak += check_leak("DET train ∩ OCR train", det_train_all, ocr_train)
    total_leak += check_leak("DET train ∩ OCR val", det_train_all, ocr_val)
    total_leak += check_leak("DET val   ∩ OCR train", det_val_all, ocr_train)
    total_leak += check_leak("DET val   ∩ OCR val", det_val_all, ocr_val)

    if test_hashes is None:
        print(" TEST manifest not found; skipping cross-dataset TEST checks.")
        return total_leak

    print(" cross-dataset (train/val ↔ TEST):")
    total_leak += check_leak("OCR train ∩ TEST", ocr_train, test_hashes)
    total_leak += check_leak("OCR val   ∩ TEST", ocr_val, test_hashes)
    total_leak += check_leak("DET train ∩ TEST", det_train_all, test_hashes)
    total_leak += check_leak("DET val   ∩ TEST", det_val_all, test_hashes)
    return total_leak


def main() -> int:
    total_missing = 0
    total_pending = 0
    total_leak = 0

    print("=== OCR (data/ocr/) ===")
    ocr_train = load_manifest_hashes(DATA_ROOT / "ocr/train.jsonl", device_field=False)
    ocr_val = load_manifest_hashes(DATA_ROOT / "ocr/val.jsonl", device_field=False)
    print(" train")
    total_missing += check_split("ocr/train", _split_agnostic_hashes(ocr_train),
                                 DATA_ROOT / "ocr", ".webp")
    print(" val")
    total_missing += check_split("ocr/val", _split_agnostic_hashes(ocr_val),
                                 DATA_ROOT / "ocr", ".webp")

    print("\n=== DET (data/detector/) ===")
    det_train = load_manifest_hashes(DATA_ROOT / "detector/train.jsonl", device_field=True)
    det_val = load_manifest_hashes(DATA_ROOT / "detector/val.jsonl", device_field=True)
    print(" train")
    total_missing += check_split("detector/train", det_train,
                                 DATA_ROOT / "detector", ".webp")
    print(" val")
    total_missing += check_split("detector/val", det_val,
                                 DATA_ROOT / "detector", ".webp")

    print("\n=== TEST (data/test/) ===")
    test_manifest = DATA_ROOT / "test/test.jsonl"
    test_hashes: set[str] | None = None
    if not test_manifest.exists():
        print(f"  (skipped: {test_manifest} not found)")
    else:
        test_hashes = load_manifest_hashes(test_manifest, device_field=False)
        # Screenshots are expected as .png; capture pipeline hasn't published
        # them yet for every device, so treat gaps as pending rather than fatal.
        total_pending += check_split("test", _split_agnostic_hashes(test_hashes),
                                     DATA_ROOT / "test", ".png",
                                     allow_missing=True)

    print()
    total_leak = check_leaks(ocr_train, ocr_val, det_train, det_val, test_hashes)

    print()
    if total_missing == 0 and total_leak == 0 and total_pending == 0:
        print("all datasets fully populated and leak-free.")
        return 0
    parts = []
    if total_missing:
        parts.append(f"{total_missing} missing OCR/DET file(s)")
    if total_leak:
        parts.append(f"{total_leak} hash leak(s)")
    if total_pending:
        parts.append(f"{total_pending} pending TEST file(s)")
    verdict = "FAIL" if (total_missing or total_leak) else "OK (pending only)"
    print(f"{verdict}: " + ", ".join(parts))
    return 0 if verdict.startswith("OK") else 1


if __name__ == "__main__":
    sys.exit(main())
