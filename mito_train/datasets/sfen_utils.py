"""Utilities for converting an sfen string into (9x9 board matrix, 14-slot hand counts).

sfen (Shogi Forsyth-Edwards Notation) format:
    <board> <turn> <hand> <move_count>

- board: nine ranks separated by '/', from the top rank (rank 1 = gote side) down.
  Each rank contains piece letters (uppercase = sente / lowercase = gote) and
  digits for empty squares. '+' is the promoted-piece prefix.
- hand: sequence of <count><letter>; `-` means empty. Count may be omitted for
  1 (e.g. "R2GN" = 1R, 2G, 1N).

Board-square class IDs (29 classes):
    0     : empty square
    1..8  : sente base (P, L, N, S, G, B, R, K)
    9..14 : sente promoted (+P, +L, +N, +S, +B, +R)  (G and K never promote)
    15..22: gote base (p, l, n, s, g, b, r, k)
    23..28: gote promoted (+p, +l, +n, +s, +b, +r)

Hand slot order (14 slots):
    0..6 : sente (P, L, N, S, G, B, R)  (K cannot be in hand)
    7..13: gote (p, l, n, s, g, b, r)

Each slot value: integer in 0..HAND_MAX_COUNT-1 (= 0..18).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

BOARD_SIZE = 9
NUM_BOARD_CLASSES = 29  # empty + 14 (sente base+promoted) + 14 (gote base+promoted)
HAND_SLOTS = 14  # 7 piece types x 2 sides
HAND_MAX_COUNT = 19  # 0..18 (pawn max is 18)

# Base pieces (sente / uppercase)
BASE_PIECES = ["P", "L", "N", "S", "G", "B", "R", "K"]
# Pieces that can promote (sente form)
PROMOTABLE = ["P", "L", "N", "S", "B", "R"]
# Pieces that can be in hand (sente form)
HAND_PIECES = ["P", "L", "N", "S", "G", "B", "R"]

# Board-square class ID map
_BOARD_CLASS: dict[str, int] = {}
_BOARD_CLASS["."] = 0  # empty square is represented internally as '.'
for i, p in enumerate(BASE_PIECES, start=1):
    _BOARD_CLASS[p] = i  # 1..8 (sente base)
for i, p in enumerate(PROMOTABLE, start=1 + len(BASE_PIECES)):
    _BOARD_CLASS["+" + p] = i  # 9..14 (sente promoted)
for i, p in enumerate(BASE_PIECES, start=1 + len(BASE_PIECES) + len(PROMOTABLE)):
    _BOARD_CLASS[p.lower()] = i  # 15..22 (gote base)
for i, p in enumerate(PROMOTABLE, start=1 + 2 * len(BASE_PIECES) + len(PROMOTABLE)):
    _BOARD_CLASS["+" + p.lower()] = i  # 23..28 (gote promoted)

# Hand slot index map
_HAND_SLOT: dict[str, int] = {}
for i, p in enumerate(HAND_PIECES):
    _HAND_SLOT[p] = i  # 0..6 sente
for i, p in enumerate(HAND_PIECES):
    _HAND_SLOT[p.lower()] = i + len(HAND_PIECES)  # 7..13 gote


@dataclass
class ParsedSfen:
    board: list[list[int]]  # 9x9, class ID
    hand: list[int]  # 14 counts
    turn: str  # 'b' or 'w'


def parse_sfen(sfen: str) -> ParsedSfen:
    """Parse an sfen string into (board, hand, turn)."""
    parts = sfen.strip().split()
    if len(parts) < 3:
        raise ValueError(f"sfen too short: {sfen!r}")
    board_str, turn, hand_str = parts[0], parts[1], parts[2]

    board = _parse_board(board_str)
    hand = _parse_hand(hand_str)
    if turn not in ("b", "w"):
        raise ValueError(f"invalid turn: {turn!r}")
    return ParsedSfen(board=board, hand=hand, turn=turn)


def _parse_board(board_str: str) -> list[list[int]]:
    ranks = board_str.split("/")
    if len(ranks) != BOARD_SIZE:
        raise ValueError(f"rank count is not 9: {len(ranks)} ({board_str!r})")
    result: list[list[int]] = []
    for rank_idx, rank in enumerate(ranks):
        row: list[int] = []
        i = 0
        while i < len(rank):
            ch = rank[i]
            if ch.isdigit():
                row.extend([0] * int(ch))  # empty squares
                i += 1
            elif ch == "+":
                if i + 1 >= len(rank):
                    raise ValueError(f"no piece after +: {rank!r}")
                key = "+" + rank[i + 1]
                row.append(_BOARD_CLASS[key])
                i += 2
            else:
                row.append(_BOARD_CLASS[ch])
                i += 1
        if len(row) != BOARD_SIZE:
            raise ValueError(f"rank {rank_idx} square count is not 9: {len(row)} ({rank!r})")
        result.append(row)
    return result


_HAND_TOKEN_RE = re.compile(r"(\d*)([PLNSGBRKplnsgbrk])")


def _parse_hand(hand_str: str) -> list[int]:
    hand = [0] * HAND_SLOTS
    if hand_str == "-":
        return hand
    pos = 0
    while pos < len(hand_str):
        m = _HAND_TOKEN_RE.match(hand_str, pos)
        if not m:
            raise ValueError(f"hand parse failed: {hand_str!r} (pos={pos})")
        count = int(m.group(1)) if m.group(1) else 1
        letter = m.group(2)
        if letter not in _HAND_SLOT:
            raise ValueError(f"invalid piece in hand: {letter!r} (K cannot be in hand)")
        slot = _HAND_SLOT[letter]
        hand[slot] += count
        pos = m.end()
    # Overflow guard for training: clip to HAND_MAX_COUNT-1 (should never trigger in practice)
    hand = [min(c, HAND_MAX_COUNT - 1) for c in hand]
    return hand


_ID_TO_SYMBOL: dict[int, str] = {v: k for k, v in _BOARD_CLASS.items()}


def encode_board(board: list[list[int]]) -> str:
    ranks = []
    for row in board:
        rank = ""
        empty = 0
        for cid in row:
            sym = _ID_TO_SYMBOL[cid]
            if sym == ".":
                empty += 1
                continue
            if empty:
                rank += str(empty)
                empty = 0
            rank += sym
        if empty:
            rank += str(empty)
        ranks.append(rank)
    return "/".join(ranks)


def encode_hand(hand: list[int]) -> str:
    if sum(hand) == 0:
        return "-"
    out = ""
    # SFEN convention: sente first (R,B,G,S,N,L,P), then gote same order.
    order_sente = ["R", "B", "G", "S", "N", "L", "P"]
    order_gote = [p.lower() for p in order_sente]
    for p in order_sente + order_gote:
        slot = _HAND_SLOT[p]
        c = hand[slot]
        if c <= 0:
            continue
        out += (str(c) if c > 1 else "") + p
    return out


def encode_sfen(board: list[list[int]], hand: list[int], turn: str = "b",
                move_count: int | None = None) -> str:
    parts = [encode_board(board), turn, encode_hand(hand)]
    if move_count is not None:
        parts.append(str(move_count))
    return " ".join(parts)
