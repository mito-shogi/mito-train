"""Build leak-free v4 manifests for OCR / DET / TEST.

Sources for fresh splits:
    assets/mate{3,5,7,9,11}.sfen        — natural game positions (~5M SFENs)
    assets/start_sfens_ply{24,32}.txt   — opening positions (~56k SFENs)
    scripts/data/generate_synthetic_sfens.py — synthetic high-hand distribution

TEST reuse:
    data/detector/<device>/*.webp × 4 machines all render the same 1000
    hashes. When --reuse-test is on (default), TEST is populated from that
    intersection so the raw screenshots already sitting on disk become the
    test set instead of asking piyo-hook for 1000 more captures. SFENs are
    looked up from `uploads/eval/data/iPhone10_1-*.parquet` (any iPhone works
    — the SFEN is device-independent).

Outputs (each row: {sfen, hash, type}):
    data/test/test.jsonl                — TEST (1000, device-agnostic)
    data/ocr/train.jsonl                — OCR train (18000, device-agnostic)
    data/ocr/val.jsonl                  — OCR val (2000, device-agnostic)
    data/detector/train.jsonl           — DET train (800 hashes × 4 devices = 3200 rows)
    data/detector/val.jsonl             — DET val (200 hashes × 4 devices = 800 rows)

All 5 splits share zero hash. Split priority: TEST first (held out or reused),
then OCR train/val, then DET train/val. `type` on each row is one of
"natural" / "opening" / "synthetic" / "existing" for provenance tracking.

Default mix ratio per split: 40% natural + 20% opening + 40% synthetic. That
gives roughly the same variety across TEST/OCR/DET so per-split distribution
shifts don't muddle downstream comparisons.

DET manifest keeps the per-device row format DetectorDataset already expects:
each shared hash is expanded into one row per device with
`{path, device, hash, sfen, type}`.

Usage:
    uv run python scripts/data/build_manifests.py               # default sizes
    uv run python scripts/data/build_manifests.py --dry-run     # print plan only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import tempfile
from itertools import combinations
from pathlib import Path

ASSETS = Path("assets")
NATURAL_FILES = [ASSETS / f"mate{n}.sfen" for n in (3, 5, 7, 9, 11)]
OPENING_FILES = [ASSETS / "start_sfens_ply24.txt", ASSETS / "start_sfens_ply32.txt"]
HIGH_HAND_SCRIPT = Path("scripts/data/generate_synthetic_sfens.py")
DETECTOR_ROOT = Path("data/detector")
PARQUET_DIR = Path("uploads/eval/data")
PARQUET_SFEN_LOOKUP_GLOB = "iPhone10_1-*-of-00010.parquet"

DEVICES = ["iPhone10,1", "iPhone11,8", "iPhone15,4", "iPad14,10"]


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def _tag_rows(sfens: list[str], src: str) -> list[dict]:
    return [{"sfen": s, "hash": _sha(s), "type": src} for s in sfens]


def sample_pool(paths: list[Path], k: int, seed: int, src: str,
                strip_prefix: str | None = None) -> list[dict]:
    """Load every non-empty line from `paths`, dedup by hash, sample `k`."""
    rng = random.Random(seed)
    lines: list[str] = []
    for p in paths:
        with p.open() as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                if strip_prefix and s.startswith(strip_prefix):
                    s = s[len(strip_prefix):]
                lines.append(s)
    # Dedup by hash to avoid dominance if the source has repeats.
    seen: set[str] = set()
    unique: list[str] = []
    for s in lines:
        h = _sha(s)
        if h in seen:
            continue
        seen.add(h)
        unique.append(s)
    print(f"[load] {src:<9} loaded={len(lines):>7}  unique={len(unique):>7}"
          f"  target={k}")
    if k >= len(unique):
        return _tag_rows(unique, src)
    return _tag_rows(rng.sample(unique, k), src)


def load_position_pool(seed: int) -> tuple[list[dict], list[dict]]:
    """Return (all4_covered, iphone_only) SFEN pools drawn from existing captures.

    `all4_covered` : hashes whose raw screenshot exists on every one of the 4
                     device dirs (data/detector/<dev>/<hash>.webp). These are
                     the intersection — safe to use for TEST because piyo-hook
                     doesn't need to re-render anything.
    `iphone_only`  : hashes covered by the iPhone parquets but not present in
                     the iPad detector dir. Reusable for OCR train/val on
                     iPhone3; iPad still needs fresh renders for these.

    Returns ([], []) if any prerequisite is missing so the caller falls back
    to the pure-fresh path.
    """
    if not all((DETECTOR_ROOT / d).exists() for d in DEVICES):
        return [], []
    per_dev = [{p.stem for p in (DETECTOR_ROOT / d).iterdir() if p.suffix == ".webp"}
               for d in DEVICES]
    all4 = per_dev[0]
    for other in per_dev[1:]:
        all4 &= other

    import pyarrow.parquet as pq  # local import; only needed on reuse path
    shards = sorted(PARQUET_DIR.glob(PARQUET_SFEN_LOOKUP_GLOB))
    if not shards:
        return [], []
    sfen_by_hash: dict[str, str] = {}
    for p in shards:
        t = pq.read_table(p, columns=["hash", "sfen"])
        for h, s in zip(t.column("hash").to_pylist(), t.column("sfen").to_pylist()):
            if h not in sfen_by_hash:
                sfen_by_hash[h] = s

    rng = random.Random(seed)
    all4_rows = [{"sfen": sfen_by_hash[h], "hash": h, "type": "existing"}
                 for h in sorted(all4) if h in sfen_by_hash]
    iphone_only_rows = [{"sfen": sfen_by_hash[h], "hash": h, "type": "existing"}
                        for h in sorted(sfen_by_hash) if h not in all4]
    rng.shuffle(all4_rows)
    rng.shuffle(iphone_only_rows)
    print(f"[reuse] position pool: 4-device coverage = {len(all4_rows)}, "
          f"iPhone-only coverage = {len(iphone_only_rows)}")
    return all4_rows, iphone_only_rows


def generate_synthetic(count: int, seed: int, workdir: Path) -> list[dict]:
    """Delegate to generate_high_hand_sfens.py and read the output back."""
    out = workdir / "synth.jsonl"
    cmd = [
        sys.executable, str(HIGH_HAND_SCRIPT),
        "--count", str(count), "--seed", str(seed), "--out", str(out),
    ]
    print(f"[synth] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    rows: list[dict] = []
    with out.open() as f:
        for line in f:
            e = json.loads(line)
            rows.append({"sfen": e["sfen"], "hash": e["hash"], "type": "synthetic"})
    return rows


def take_disjoint(pool: list[dict], seen: set[str], n: int, label: str) -> list[dict]:
    """Pop items from pool until n disjoint hashes are collected."""
    picked: list[dict] = []
    while pool and len(picked) < n:
        r = pool.pop(0)
        if r["hash"] in seen:
            continue
        seen.add(r["hash"])
        picked.append(r)
    if len(picked) < n:
        raise SystemExit(
            f"[split] {label}: pool exhausted, got {len(picked)}/{n}. "
            "Increase source oversample or reduce split size."
        )
    return picked


def compose_split(natural: list[dict], opening: list[dict], synthetic: list[dict],
                  size: int, ratio: dict[str, float], seen: set[str],
                  label: str, reuse_first: list[list[dict]] | None = None) -> list[dict]:
    """Draw a mixed-source split honoring ratio × size.

    When `reuse_first` is set, we first drain items from those priority pools
    (in order) before touching the fresh natural/opening/synthetic buckets.
    Priority pools are lists of {sfen, hash, type} rows already tagged with
    provenance; anything left over is filled by the fresh mix.
    """
    picks: list[dict] = []
    for pool in reuse_first or []:
        while pool and len(picks) < size:
            r = pool.pop(0)
            if r["hash"] in seen:
                continue
            seen.add(r["hash"])
            picks.append(r)
    remaining = size - len(picks)
    if remaining <= 0:
        return picks
    n_nat = int(round(remaining * ratio["natural"]))
    n_open = int(round(remaining * ratio["opening"]))
    n_syn = remaining - n_nat - n_open
    picks += take_disjoint(natural, seen, n_nat, f"{label}/natural")
    picks += take_disjoint(opening, seen, n_open, f"{label}/opening")
    picks += take_disjoint(synthetic, seen, n_syn, f"{label}/synthetic")
    return picks


def write_ocr_like(path: Path, rows: list[dict]) -> None:
    """Device-agnostic manifest: {sfen, hash, type} per row."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps({"sfen": r["sfen"], "hash": r["hash"],
                                "type": r["type"]}, ensure_ascii=False) + "\n")
    print(f"[write] {path}  ({len(rows)} rows)")


def write_detector(path: Path, rows: list[dict], devices: list[str]) -> None:
    """Per-device manifest expected by DetectorDataset.

    For every shared hash, emit one row per device with the resolved image
    path; that keeps `data_root / entry["path"]` semantics unchanged.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("w") as f:
        for r in rows:
            for dev in devices:
                entry = {
                    "path": f"detector/{dev}/{r['hash']}.webp",
                    "device": dev,
                    "hash": r["hash"],
                    "sfen": r["sfen"],
                    "type": r["type"],
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                written += 1
    print(f"[write] {path}  ({written} rows = {len(rows)} hashes × {len(devices)} devices)")


def verify_disjoint(splits: dict[str, list[dict]]) -> int:
    """Confirm every pair of splits has zero hash overlap. Return leak count."""
    sets = {name: {r["hash"] for r in rows} for name, rows in splits.items()}
    leaked = 0
    print("[verify] pairwise leak check:")
    for a, b in combinations(sets, 2):
        inter = len(sets[a] & sets[b])
        leaked += inter
        status = "OK" if inter == 0 else "LEAK"
        print(f"  {a:<10} ∩ {b:<10}  = {inter:>4}  {status}")
    return leaked


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--test-size", type=int, default=1000)
    p.add_argument("--ocr-train-size", type=int, default=18000)
    p.add_argument("--ocr-val-size", type=int, default=2000)
    p.add_argument("--det-train-size", type=int, default=800)
    p.add_argument("--det-val-size", type=int, default=200)
    p.add_argument("--ratio-natural", type=float, default=0.4)
    p.add_argument("--ratio-opening", type=float, default=0.2)
    p.add_argument("--ratio-synthetic", type=float, default=0.4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-root", type=Path, default=Path("data"))
    p.add_argument("--devices", nargs="+", default=DEVICES)
    p.add_argument("--reuse-test", action=argparse.BooleanOptionalAction, default=True,
                   help="Populate TEST from data/detector/* (4-device intersection) "
                        "instead of sampling fresh. Disable with --no-reuse-test.")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    ratio = {"natural": args.ratio_natural, "opening": args.ratio_opening,
             "synthetic": args.ratio_synthetic}
    if abs(sum(ratio.values()) - 1.0) > 1e-6:
        raise SystemExit(f"ratios must sum to 1.0, got {sum(ratio.values())}")

    total = (args.test_size + args.ocr_train_size + args.ocr_val_size
             + args.det_train_size + args.det_val_size)
    # 1.5× oversample per source to survive dedup collisions comfortably.
    n_nat = int(total * ratio["natural"] * 1.5) + 100
    n_open = int(total * ratio["opening"] * 1.5) + 100
    n_syn = int(total * ratio["synthetic"] * 1.5) + 100

    print(f"[plan] total unique hashes needed: {total}")
    print(f"[plan] oversample budget: natural={n_nat} opening={n_open} synthetic={n_syn}")
    if args.dry_run:
        return 0

    all4_pool: list[dict] = []
    iphone_only_pool: list[dict] = []
    if args.reuse_test:
        all4_pool, iphone_only_pool = load_position_pool(seed=args.seed + 3)
        if not all4_pool and not iphone_only_pool:
            print("[reuse] position reuse skipped (missing detector dirs or parquet).")

    # Estimate fresh budget: everything not covered by the reuse pool. The
    # 4-device pool feeds TEST; the iPhone-only pool feeds OCR val and part of
    # OCR train. DET stays fully fresh so it doesn't cannibalize either pool.
    reused_available = len(all4_pool) + len(iphone_only_pool)
    fresh_needed = max(total - reused_available,
                       args.det_train_size + args.det_val_size + 1000)
    n_nat = int(fresh_needed * ratio["natural"] * 1.5) + 100
    n_open = int(fresh_needed * ratio["opening"] * 1.5) + 100
    n_syn = int(fresh_needed * ratio["synthetic"] * 1.5) + 100
    print(f"[plan] reused pool available: {reused_available} "
          f"(all4={len(all4_pool)}, iphone_only={len(iphone_only_pool)})")
    print(f"[plan] fresh oversample: natural={n_nat} opening={n_open} synthetic={n_syn}")

    natural = sample_pool(NATURAL_FILES, n_nat, seed=args.seed, src="natural")
    opening = sample_pool(OPENING_FILES, n_open, seed=args.seed + 1, src="opening",
                          strip_prefix="sfen ")
    with tempfile.TemporaryDirectory() as td:
        synthetic = generate_synthetic(n_syn, seed=args.seed + 2, workdir=Path(td))
    print(f"[load] synthetic loaded={len(synthetic)}  target={n_syn}")

    seen: set[str] = set()
    # Split priority: TEST first (needs perfect 4-device coverage), then OCR
    # val + train (partial iPhone reuse OK; iPad gets fresh renders anyway),
    # then DET (always fresh so it stays disjoint from every reused hash).
    test = compose_split(natural, opening, synthetic, args.test_size,
                         ratio, seen, "test", reuse_first=[all4_pool])
    ocr_val = compose_split(natural, opening, synthetic, args.ocr_val_size,
                            ratio, seen, "ocr_val", reuse_first=[iphone_only_pool])
    ocr_train = compose_split(natural, opening, synthetic, args.ocr_train_size,
                              ratio, seen, "ocr_train", reuse_first=[iphone_only_pool])
    det_train = compose_split(natural, opening, synthetic, args.det_train_size,
                              ratio, seen, "det_train")
    det_val = compose_split(natural, opening, synthetic, args.det_val_size,
                            ratio, seen, "det_val")

    write_ocr_like(args.out_root / "test/test.jsonl", test)
    write_ocr_like(args.out_root / "ocr/train.jsonl", ocr_train)
    write_ocr_like(args.out_root / "ocr/val.jsonl", ocr_val)
    write_detector(args.out_root / "detector/train.jsonl", det_train, args.devices)
    write_detector(args.out_root / "detector/val.jsonl", det_val, args.devices)

    leaked = verify_disjoint({
        "test": test, "ocr_train": ocr_train, "ocr_val": ocr_val,
        "det_train": det_train, "det_val": det_val,
    })
    if leaked:
        print(f"[verify] FAIL: {leaked} total hash leaks between splits")
        return 1
    print("[verify] all splits pairwise disjoint. leak-free ✓")

    # Emit the union — the single SFEN list piyo-hook renders per device.
    union: dict[str, dict] = {}
    for group in (test, ocr_train, ocr_val, det_train, det_val):
        for r in group:
            union.setdefault(r["hash"], {"sfen": r["sfen"], "hash": r["hash"],
                                        "type": r["type"]})
    positions_path = args.out_root / "positions.jsonl"
    with positions_path.open("w") as f:
        for h in sorted(union):
            f.write(json.dumps(union[h], ensure_ascii=False) + "\n")
    print(f"[write] {positions_path}  ({len(union)} unique SFEN)")

    # Emit the two piyo-hook work lists derived from the union.
    #  - shared: fresh SFENs → every device must capture these
    #  - ipad_only: existing SFENs that iPhone parquets already cover but the
    #    iPad detector dir does not — piyo-hook renders only on iPad
    ipad_covered: set[str] = {
        p.stem for p in (DETECTOR_ROOT / "iPad14,10").iterdir()
        if p.suffix == ".webp"
    } if (DETECTOR_ROOT / "iPad14,10").exists() else set()

    shared_rows = [r for r in union.values() if r["type"] != "existing"]
    ipad_only_rows = [
        r for r in union.values()
        if r["type"] == "existing" and r["hash"] not in ipad_covered
    ]
    shared_path = args.out_root / "positions_shared.jsonl"
    ipad_path = args.out_root / "positions_ipad_only.jsonl"
    with shared_path.open("w") as f:
        for r in sorted(shared_rows, key=lambda x: x["hash"]):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with ipad_path.open("w") as f:
        for r in sorted(ipad_only_rows, key=lambda x: x["hash"]):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[write] {shared_path}  ({len(shared_rows)} SFEN — every device captures)")
    print(f"[write] {ipad_path}  ({len(ipad_only_rows)} SFEN — iPad only)")

    fully_reused = len(union) - len(shared_rows) - len(ipad_only_rows)
    total_captures = (
        len(shared_rows) * len(args.devices) + len(ipad_only_rows)
    )
    print(f"[plan] {fully_reused} SFEN 全機種流用のみ (move existing webp to captures/)")
    print(f"[plan] piyo-hook 新規撮影: {total_captures} 枚 = "
          f"{len(shared_rows)}×{len(args.devices)} shared + {len(ipad_only_rows)} iPad-only")

    print()
    print("next steps:")
    print(f" 1) 全4機種 → {shared_path.name} の SFEN を撮影 (各 {len(shared_rows)} 枚)")
    print(f" 2) iPad14,10 → {ipad_path.name} も追加で撮影 ({len(ipad_only_rows)} 枚)")
    print(f" 3) TEST の既存 webp {fully_reused} × 4 = "
          f"{fully_reused * len(args.devices)} 枚を captures/ にコピー")
    print(" 4) uv run python scripts/inspect/check_data.py で leak 0 / missing 0 を確認")
    return 0


if __name__ == "__main__":
    sys.exit(main())
