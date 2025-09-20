import re
from typing import List, Tuple, Optional
import jiwer

# Minimal, English-only post-processing utilities for ASR texts

# Word-level normalization used for quick WER (optional)
_TRANSFORM = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(),
])


def post_processing(data: list[dict], verbose: bool = False, normalize: bool = False) -> list[dict]:
    """Lightly clean and de-echo model outputs in-place.

    Assumes each item has keys: "after" (pred), "before" (raw pred), and "answer" (reference).
    - Cleans punctuation and stray apostrophes
    - Removes common echo/repeated phrases with simple heuristics
    - Drops token "unk"
    - Optionally recomputes after_score using jiwer WER if the key exists
    """
    changed = 0

    for i, item in enumerate(data):
        after = item.get("after", "")

        # Trim accidental prefix like "hash dis" (rare; kept for compatibility)
        if after.startswith("hash dis"):
            after = after[len("hash dis"):].lstrip()

        # Basic cleaning
        after = clean_text(after)

        # Remove repeated phrases/echos
        after = remove_repeated_phrase(after, min_words=2, threshold=0.7)

        # Drop explicit UNK tokens
        if "unk" in after.split():
            after = " ".join(t for t in after.split() if t != "unk")

        if after != item.get("after"):
            # Optional length guard (kept from original behavior)
            if len(item.get("before", "").split()) >= 2 * max(0, len(after.split()) - 1):
                after = " ".join([after, after])

            if verbose:
                print(f"\nSample {changed}")
                print(" Before:", item.get("after", ""))
                print(" After :", after)
                print(" Answer:", item.get("answer", ""))
                if "after_score" in item:
                    print(" Old Score:", item.get("after_score"))

            item["after"] = after

            # Optionally recompute after_score if reference is available
            if "answer" in item:
                item["after_score"] = _quick_wer(item["answer"], after)

            if verbose and "after_score" in item:
                print(" New Score:", item["after_score"])

            changed += 1

    return data


def clean_text(text: str) -> str:
    # Keep word chars and spaces/apostrophes; strip punctuation; normalize stray apostrophes
    text = re.sub(r"[^\w\s']", "", text)
    text = re.sub(r"(?<!\w)'|'(?!\w)", "", text)
    return text.strip()


# --- Repetition handling helpers ------------------------------------------------

def _find_min_repeat_unit_anywhere(
    words: List[str], *, max_unit_len: int = 8, min_unit_len: int = 1, min_repeats: int = 2
) -> Optional[Tuple[List[str], int, int, int]]:
    """Find a minimal repeated unit that forms a contiguous run somewhere.
    Returns (unit, start_idx, repeats, run_tokens) or None.
    Picks the candidate covering the most tokens; tie-break by shortest unit.
    """
    n = len(words)
    best: Optional[Tuple[int, int, int, int, List[str]]] = None  # (run_tokens, -unit_len, start, repeats, unit)
    i = 0
    while i < n:
        max_len_here = min(max_unit_len, (n - i) // max(1, min_repeats))
        if max_len_here < min_unit_len:
            i += 1
            continue
        for unit_len in range(min_unit_len, max_len_here + 1):
            unit = words[i : i + unit_len]
            r, j = 1, i + unit_len
            while j + unit_len <= n and words[j : j + unit_len] == unit:
                r += 1
                j += unit_len
            if r >= min_repeats:
                run_tokens = r * unit_len
                cand = (run_tokens, -unit_len, i, r, unit)
                if best is None or cand > best:
                    best = cand
        i += 1
    if best is None:
        return None
    run_tokens, neg_unit_len, start_idx, repeats, unit = best
    return unit, start_idx, repeats, run_tokens


def _normalize_unit_rotation_by_earliest_occurrence(unit: List[str], words: List[str]) -> List[str]:
    """Rotate unit to the variant that appears earliest in the sequence."""
    m = len(unit)
    if m == 0:
        return unit

    def first_index(sub: List[str], arr: List[str]) -> int:
        n, k = len(arr), len(sub)
        i = 0
        while i + k <= n:
            if arr[i : i + k] == sub:
                return i
            i += 1
        return -1

    best = None  # (pos, rotation, rotated_unit)
    for r in range(m):
        rot = unit[r:] + unit[:r]
        pos = first_index(rot, words)
        if pos == -1:
            continue
        cand = (pos, r, rot)
        if best is None or cand < best:
            best = cand
    return best[2] if best else unit


def _strip_truncated_tail_by_prefix(words: List[str], unit: List[str], *, min_prefix_len: int = 2) -> List[str]:
    """If the sequence ends with a truncated prefix of `unit` and is preceded by a full `unit`, drop the prefix."""
    n, m = len(words), len(unit)
    for k in range(m - 1, min_prefix_len - 1, -1):
        if n >= k and words[n - k : n] == unit[:k]:
            j = n - k
            if j - m >= 0 and words[j - m : j] == unit:
                return words[:j]
    return words


def _compress_runs(
    words: List[str], unit: List[str], *, min_repeats: int = 2, non_tail_keep: int = 3, tail_keep: int = 2, tail_max_unit: int = 2, default_keep: int = 1
) -> List[str]:
    """Compress contiguous runs of `unit` using simple rules.
    - Middle runs: keep up to `non_tail_keep` repeats if followed by content.
    - Tail runs: keep exactly `tail_keep` repeats if preceded by content and len(unit) <= `tail_max_unit`.
    - Otherwise compress to `default_keep`.
    """
    out: List[str] = []
    n, m = len(words), len(unit)
    i = 0
    while i < n:
        if i + m <= n and words[i : i + m] == unit:
            r, j = 1, i + m
            while j + m <= n and words[j : j + m] == unit:
                r += 1
                j += m
            if r >= min_repeats:
                has_after = j < n
                has_before = i > 0
                is_tail = j == n
                keep = (
                    (has_after and r <= non_tail_keep) or
                    (is_tail and has_before and r == tail_keep and m <= tail_max_unit)
                )
                out.extend(unit * (r if keep else default_keep))
                i = j
                continue
        out.append(words[i])
        i += 1
    return out


def remove_repeated_phrase(text: str, *, min_words: int = 2, threshold: float = 0.7) -> str:
    """Remove echo-like repeated phrases.
    Steps: find minimal unit -> rotate to earliest occurrence -> trim truncated tail -> compress runs.
    `threshold` kept for API compatibility (not used in this condensed version).
    """
    words = text.strip().split()
    if not words:
        return text

    found = _find_min_repeat_unit_anywhere(words, max_unit_len=8, min_unit_len=1, min_repeats=2)
    if not found:
        return text

    unit, *_ = found
    unit = _normalize_unit_rotation_by_earliest_occurrence(unit, words)
    words = _strip_truncated_tail_by_prefix(words, unit)
    words = _compress_runs(words, unit, min_repeats=2, non_tail_keep=3, tail_keep=2, tail_max_unit=3, default_keep=1)
    return " ".join(words)


def _quick_wer(ref: str, hyp: str) -> float:
    """WER using jiwer with the local transform; returns a float in [0,1]."""
    out = jiwer.process_words(ref, hyp, reference_transform=_TRANSFORM, hypothesis_transform=_TRANSFORM)
    C, S, D, I = int(out.hits), int(out.substitutions), int(out.deletions), int(out.insertions)
    N = C + S + D
    return 0.0 if N == 0 else (S + D + I) / N
