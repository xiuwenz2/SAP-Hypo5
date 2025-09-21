#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import string
from typing import List, Tuple, Optional, Dict, Any

import editdistance


# =========================
# Basic utilities
# =========================
def calculate_wer(ref_tokens: List[str], hyp_tokens: List[str]) -> float:
    """
    Word-level WER using edit distance / reference length.
    Inputs must be token lists (already split).
    """
    if not ref_tokens:
        return 0.0
    return editdistance.eval(hyp_tokens, ref_tokens) / len(ref_tokens)


# =========================
# Repetition detection helpers
# =========================
_WS = re.compile(r"\s+")


def _normalize_value(s: Optional[str]) -> str:
    """Lowercase, trim, collapse spaces, and strip edge punctuation."""
    if s is None:
        return ""
    s = s.lower().strip()
    s = _WS.sub(" ", s).strip(string.punctuation + " ")
    return s


def _find_min_repeat_unit_anywhere(
    words: List[str],
    max_unit_len: int = 8,
    min_unit_len: int = 1,
    min_repeats: int = 2,
) -> Optional[Tuple[List[str], int, int, int]]:
    """
    Find a minimal repeating unit that appears as a contiguous run somewhere.
    Among all runs, pick the one that covers the most tokens (run_tokens).
    Return (unit_tokens, start_index, repeats, run_tokens), or None if not found.
    """
    n = len(words)
    if n == 0:
        return None

    best: Optional[Tuple[int, int, int, int, List[str]]] = None  # (run_tokens, -unit_len, start, repeats, unit)
    i = 0
    while i < n:
        max_len_here = min(max_unit_len, (n - i) // max(1, min_repeats))
        if max_len_here < min_unit_len:
            i += 1
            continue

        for unit_len in range(min_unit_len, max_len_here + 1):
            unit = words[i : i + unit_len]
            repeats = 1
            j = i + unit_len
            while j + unit_len <= n and words[j : j + unit_len] == unit:
                repeats += 1
                j += unit_len
            if repeats >= min_repeats:
                run_tokens = repeats * unit_len
                cand = (run_tokens, -unit_len, i, repeats, unit)
                if best is None or cand > best:
                    best = cand
        i += 1

    if best is None:
        return None
    run_tokens, neg_unit_len, start_idx, repeats, unit = best
    return (unit, start_idx, repeats, run_tokens)


def _first_index(sub: List[str], arr: List[str]) -> int:
    """Return the first index where `sub` appears in `arr`, or -1 if not found."""
    n, k = len(arr), len(sub)
    if k == 0 or k > n:
        return -1
    i = 0
    while i + k <= n:
        if arr[i : i + k] == sub:
            return i
        i += 1
    return -1


def _normalize_unit_rotation_by_earliest_occurrence(unit: List[str], words: List[str]) -> List[str]:
    """
    Rotate the unit (all left rotations) and pick the rotation that occurs earliest in `words`.
    On ties, prefer the smallest rotation amount.
    """
    m = len(unit)
    if m == 0:
        return unit

    best = None  # (first_pos, rotation_amount, rotated_unit)
    for r in range(m):  # left-rotate by r
        rot = unit[r:] + unit[:r]
        pos = _first_index(rot, words)
        if pos == -1:
            continue
        cand = (pos, r, rot)
        if best is None or cand < best:
            best = cand
    return best[2] if best is not None else unit


def _strip_truncated_tail_by_prefix(words: List[str], unit: List[str], min_prefix_len: int = 2) -> List[str]:
    """
    If the sentence ends with a *prefix* of `unit` and *immediately* before that
    there is a full occurrence of `unit`, drop the trailing prefix.
    This avoids keeping an echo-like truncated tail.
    """
    n, m = len(words), len(unit)
    if n == 0 or m == 0:
        return words

    # Try the longest prefix first down to `min_prefix_len`
    for k in range(m - 1, min_prefix_len - 1, -1):
        if n >= k and words[n - k : n] == unit[:k]:
            j = n - k
            # Require a full unit right before the prefix
            if j - m >= 0 and words[j - m : j] == unit:
                return words[:j]
    return words


def _compress_consecutive_runs_of_unit_selective(
    words: List[str],
    unit: List[str],
    min_repeats: int = 2,
    # heuristics
    max_keep_repeats_non_tail: int = 3,  # keep ≤3 repeats if there is content after the run
    tail_keep_repeats: int = 2,          # keep exactly 2 repeats at sentence tail
    tail_max_unit_len: int = 2,          # only keep tail repeats if unit length ≤ this
    default_keep_repeats: int = 1,       # compress other runs to this many repeats
) -> List[str]:
    """
    For each *contiguous* run of `unit`, decide to keep it or compress it:
      - Start/middle runs with content after them: keep if repeats ≤ max_keep_repeats_non_tail.
      - Tail runs (at end) with content before them: keep if repeats == tail_keep_repeats and len(unit) ≤ tail_max_unit_len.
      - Otherwise, compress the run to `default_keep_repeats` repeats.
    Non-contiguous re-occurrences are not modified.
    """
    n, m = len(words), len(unit)
    if n == 0 or m == 0:
        return words[:]

    out: List[str] = []
    i = 0
    while i < n:
        if i + m <= n and words[i : i + m] == unit:
            # count contiguous repeats
            repeats = 1
            j = i + m
            while j + m <= n and words[j : j + m] == unit:
                repeats += 1
                j += m

            if repeats >= min_repeats:
                has_after = (j < n)
                has_before = (i > 0)
                is_tail = (j == n)

                keep = False
                if has_after and repeats <= max_keep_repeats_non_tail:
                    keep = True
                elif is_tail and has_before and repeats == tail_keep_repeats and m <= tail_max_unit_len:
                    keep = True

                out.extend(unit * (repeats if keep else default_keep_repeats))
                i = j
            else:
                out.append(words[i])
                i += 1
        else:
            out.append(words[i])
            i += 1
    return out


def remove_repeated_phrase(text: str) -> str:
    """
    Detect a minimal repeating unit in the sentence and selectively compress
    *contiguous* runs following the rules above. Also strip a truncated tail
    that is a prefix of the unit when it follows a full unit.
    """
    words = text.strip().split()
    if not words:
        return text

    found = _find_min_repeat_unit_anywhere(
        words, max_unit_len=8, min_unit_len=1, min_repeats=2
    )
    if found is None:
        return text

    unit, _start, _repeats, _run_tokens = found
    unit = _normalize_unit_rotation_by_earliest_occurrence(unit, words)
    words = _strip_truncated_tail_by_prefix(words, unit)
    words = _compress_consecutive_runs_of_unit_selective(
        words,
        unit,
        min_repeats=2,
        max_keep_repeats_non_tail=3,
        tail_keep_repeats=2,
        tail_max_unit_len=2,
        default_keep_repeats=1,
    )
    return " ".join(words)


# =========================
# Public API
# =========================
def post_processing(data: List[Dict[str, Any]], verbose: bool = False) -> List[Dict[str, Any]]:
    """
    Clean and normalize ASR transcriptions in-place with (optional) debug prints:

      - Remove "hash dis" prefix.
      - Remove tokens equal to "unk".
      - Detect and compress repeated phrases (value-level rules).
      - If the cleaned text becomes too short relative to `before`,
        duplicate it once (heuristic from your original code).
      - Recompute `after_score` as token-level WER against `answer`.

    Expected item schema:
      {
        "before": <str>, "after": <str>, "answer": <str>, "after_score": <float>, ...
      }
    """
    modified_count = 0

    for i, item in enumerate(data):
        after = str(item.get("after", ""))
        before = str(item.get("before", ""))
        answer = str(item.get("answer", ""))

        # 1) strip "hash dis" prefix
        if after.startswith("hash dis"):
            after = after[len("hash dis") :].lstrip()

        # 2) compress repeated phrases
        after = remove_repeated_phrase(after)

        # 3) drop "unk" tokens
        if "unk" in after.split():
            after = " ".join(t for t in after.split() if t != "unk")

        # 4) if changed, optionally duplicate short outputs
        if after != item.get("after", ""):
            # if BEFORE is at least ~ twice the new AFTER (minus one token), double AFTER
            if len(before.split()) >= 2 * max(0, len(after.split()) - 1):
                after = " ".join([after, after]).strip()

            if verbose:
                print(f"\nSample {modified_count}")
                print(" Before:", item.get("after", ""))
                print(" After :", after)
                print(" Answer:", answer)
                print(" Old Score:", item.get("after_score"))

            # update fields
            item["after"] = after
            item["after_score"] = calculate_wer(answer.split(), after.split())

            if verbose:
                print(" New Score:", item["after_score"])

            modified_count += 1

    return data
