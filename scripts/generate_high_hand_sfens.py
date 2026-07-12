"""Generate SFEN positions with rich hand distributions across all piece types.

Motivation: `data/ocr/train.jsonl` is drawn from natural game positions, so both
pawn *and* non-pawn hand counts skew low. On eval, positions with 7+ hand
pawns fail catastrophically (up to 100%). This script fills that gap and, per
user guidance, also widens the distribution of the other hand pieces
(L, N, S, G, B, R) so the model sees the full 0..max range for each.

Approach:
    1. Play a random legal game with `python-shogi` for a random number of
       plies. Sample positions along the trajectory (legal middle- and
       end-game states).
    2. For each of the 7 hand piece types, sample independently:
           total_in_hand ~ Uniform[0, max_count_for_type]
           sente_share   ~ Uniform[0, total_in_hand]
           gote_share    = total_in_hand - sente_share
       Piece caps: P=18, L=4, N=4, S=4, G=4, B=2, R=2.
    3. Transfer board pieces of the required type into the hand until we hit
       the target (or run out of that piece on the board — accept partial).
       Removing from the board never violates nifu, and total-per-type is
       conserved because we only move pieces around, never create them.

Output: `data/synthetic_sfens.jsonl` matching the eval convention:
    {"sfen": <sfen>, "hash": <sha256(sfen)>}

Usage:
    uv run python scripts/generate_high_hand_sfens.py \
        --count 10000 --out data/synthetic_sfens.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import shogi

from mito_train.datasets.sfen_utils import encode_sfen, parse_sfen

# 7 hand piece types in slot order (0..6 for sente, 7..13 for gote)
HAND_PIECE_ORDER = ["P", "L", "N", "S", "G", "B", "R"]
PIECE_MAX = {"P": 18, "L": 4, "N": 4, "S": 4, "G": 4, "B": 2, "R": 2}


def _build_cid_to_type() -> dict[int, str]:
    """Board class-id -> piece type letter (K excluded — kings never go to hand)."""
    m: dict[int, str] = {}
    # From sfen_utils:
    #  1..8    sente base   (P,L,N,S,G,B,R,K)
    #  9..14   sente promoted (P,L,N,S,B,R)
    # 15..22   gote base
    # 23..28   gote promoted
    for cid in (1, 9, 15, 23):
        m[cid] = "P"
    for cid in (2, 10, 16, 24):
        m[cid] = "L"
    for cid in (3, 11, 17, 25):
        m[cid] = "N"
    for cid in (4, 12, 18, 26):
        m[cid] = "S"
    for cid in (5, 19):
        m[cid] = "G"
    for cid in (6, 13, 20, 27):
        m[cid] = "B"
    for cid in (7, 14, 21, 28):
        m[cid] = "R"
    return m


CID_TYPE = _build_cid_to_type()


def play_random_game(max_plies: int) -> list[str]:
    b = shogi.Board()
    plies = random.randint(20, max_plies)
    sfens: list[str] = []
    for _ in range(plies):
        legal = list(b.legal_moves)
        if not legal:
            break
        b.push(random.choice(legal))
        sfens.append(b.sfen())
    return sfens


def sample_hand_targets(max_total: int) -> list[int]:
    """Return 14-slot hand target where each piece type's total-in-hand is
    Uniform[0, max_count_for_type], then split randomly between sente and gote.

    Rejection-sampled so the sum of hand pieces stays <= `max_total`. This
    keeps the board from going near-empty (max non-king pieces = 38, so
    `max_total = 26` guarantees at least 12 non-king pieces remain on the board).
    """
    # Loop is bounded because the mean of the raw uniform is 19; rejection
    # rate at max_total=26 is small (<10%). Fallback guard just returns the
    # last sample if we somehow can't hit the cap.
    for _ in range(64):
        target = [0] * 14
        for i, pt in enumerate(HAND_PIECE_ORDER):
            total = random.randint(0, PIECE_MAX[pt])
            s = random.randint(0, total)
            g = total - s
            target[i] = s
            target[i + 7] = g
        if sum(target) <= max_total:
            return target
    return target


def transfer_to_hand(sfen: str, target_hand: list[int]) -> str:
    """Move board pieces to hand to hit `target_hand` where possible.

    Never adds to the board; only removes. Nifu and per-type total conservation
    hold automatically. If the board lacks enough pieces of a given type to
    reach the target, we take as many as we can and move on (partial fill).
    """
    p = parse_sfen(sfen)
    board = [row[:] for row in p.board]
    hand = p.hand[:]

    # Bucket board squares by underlying piece type (kings excluded)
    by_type: dict[str, list[tuple[int, int]]] = {pt: [] for pt in HAND_PIECE_ORDER}
    for r in range(9):
        for c in range(9):
            pt = CID_TYPE.get(board[r][c])
            if pt is not None:
                by_type[pt].append((r, c))

    for i, pt in enumerate(HAND_PIECE_ORDER):
        s_slot, g_slot = i, i + 7
        need_s = max(0, target_hand[s_slot] - hand[s_slot])
        need_g = max(0, target_hand[g_slot] - hand[g_slot])
        total_need = need_s + need_g
        if total_need == 0:
            continue
        avail = by_type[pt]
        can_take = min(total_need, len(avail))
        if can_take == 0:
            continue
        take = random.sample(avail, can_take)
        for r, c in take:
            board[r][c] = 0
            # Distribute to whichever side still needs pieces, weighted by remaining need.
            if need_s > 0 and (need_g == 0 or random.random() < need_s / (need_s + need_g)):
                hand[s_slot] += 1
                need_s -= 1
            else:
                hand[g_slot] += 1
                need_g -= 1

    return encode_sfen(board, hand, p.turn, move_count=1)


def _hand_summary(hand: list[int]) -> dict[str, int]:
    """Total count per piece type (sente + gote), for reporting."""
    out: dict[str, int] = {}
    for i, pt in enumerate(HAND_PIECE_ORDER):
        out[pt] = hand[i] + hand[i + 7]
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--count", type=int, default=10000)
    p.add_argument("--out", type=Path,
                   default=Path("data/synthetic_sfens.jsonl"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-plies", type=int, default=250,
                   help="Cap random game length. Longer -> more captures -> "
                        "board thins out and non-pawn transfers may run short.")
    p.add_argument("--samples-per-game", type=int, default=8,
                   help="Positions sampled along each random game.")
    p.add_argument("--max-hand-total", type=int, default=26,
                   help="Cap on total pieces across both hands. Prevents the "
                        "board from going near-empty. Max possible = 38 "
                        "(all non-king pieces in hand).")
    args = p.parse_args()

    random.seed(args.seed)

    out_rows: list[dict[str, str]] = []
    seen_hashes: set[str] = set()
    games_played = 0
    # Per-piece-type distribution over produced positions (total in hand).
    type_totals: dict[str, Counter[int]] = {pt: Counter() for pt in HAND_PIECE_ORDER}
    board_pieces_hist: Counter[int] = Counter()  # non-king board piece count

    while len(out_rows) < args.count:
        games_played += 1
        traj = play_random_game(args.max_plies)
        if not traj:
            continue
        picks = random.sample(traj, min(args.samples_per_game, len(traj)))
        for base_sfen in picks:
            target = sample_hand_targets(args.max_hand_total)
            new_sfen = transfer_to_hand(base_sfen, target)
            h = hashlib.sha256(new_sfen.encode()).hexdigest()
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            out_rows.append({"sfen": new_sfen, "hash": h})
            # Track achieved (may fall short of target if board ran out).
            parsed = parse_sfen(new_sfen)
            achieved = _hand_summary(parsed.hand)
            for pt, v in achieved.items():
                type_totals[pt][v] += 1
            # Count non-king board pieces (kings are class ids 8 and 22).
            n_pieces = sum(
                1 for r in range(9) for c in range(9)
                if parsed.board[r][c] not in (0, 8, 22)
            )
            board_pieces_hist[n_pieces] += 1
            if len(out_rows) % 2000 == 0:
                print(f"[gen] {len(out_rows)}/{args.count} "
                      f"(games={games_played})")
            if len(out_rows) >= args.count:
                break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[gen] wrote {len(out_rows)} positions to {args.out}")
    print(f"[gen] achieved distribution (total in hand, sente+gote):")
    for pt in HAND_PIECE_ORDER:
        counts = [f"{k}:{type_totals[pt][k]}" for k in sorted(type_totals[pt])]
        print(f"  {pt} (max {PIECE_MAX[pt]}):  " + "  ".join(counts))
    print(f"[gen] non-king board pieces per position:")
    for k in sorted(board_pieces_hist):
        n = board_pieces_hist[k]
        print(f"     {k:>2}  ->  {n:>5}  ({n/args.count*100:5.2f}%)")
    avg = sum(k*v for k, v in board_pieces_hist.items()) / args.count
    print(f"[gen] avg board pieces (non-king): {avg:.1f}  "
          f"(min {min(board_pieces_hist)}, max {max(board_pieces_hist)})")


if __name__ == "__main__":
    main()
