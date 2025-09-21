#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
import re
import string
from collections import Counter
from typing import List, Dict, Tuple, Optional

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from utils.postprocessing import post_processing

# Default HF slot model
SLOT_MODEL_ID = "cartesinus/xlm-r-base-amazon-massive-slot"


# -----------------------------
# Robust slot extraction (HF)
# -----------------------------
def _load_slot_pipe(model_id: str = SLOT_MODEL_ID):
    """Create a token-classification pipeline (no truncation kwargs for older HF)."""
    device = 0 if torch.cuda.is_available() else -1
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForTokenClassification.from_pretrained(model_id)
    return pipeline(
        "token-classification",
        model=mdl,
        tokenizer=tok,
        device=device,
        aggregation_strategy="simple",  # span aggregation
        # Do NOT pass truncation/padding/max_length here (older transformers will error)
    )


def _extract_slots(texts: List[str], slot_pipe, batch_size: int = 64) -> List[List[Dict]]:
    """Batch extract slots for texts; return list of slot lists with name/value/start/end/score."""
    out: List[List[Dict]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        # Older transformers don't allow extra kwargs in __call__, so call bare:
        preds = slot_pipe(batch)
        # preds: List[List[Ent]]; normalize
        for ent_list in preds:
            slots = []
            for r in ent_list:
                slots.append({
                    "name":  str(r.get("entity_group", "")).lower(),
                    "value": str(r.get("word", "")).strip(),
                    "start": int(r.get("start", 0)),
                    "end":   int(r.get("end", 0)),
                    "score": float(r.get("score", 0.0)),
                })
            out.append(slots)
    return out


# -----------------------------
# Value-level normalization
# -----------------------------
_WS = re.compile(r"\s+")

def _normalize_value(s: Optional[str]) -> str:
    """Lower, strip, collapse spaces, strip leading/trailing punctuation."""
    if s is None:
        return ""
    s = s.lower().strip()
    s = _WS.sub(" ", s).strip(string.punctuation + " ")
    return s


def _type_value_multiset(text: str, slots: List[Dict]) -> Counter:
    """Counter of (slot_type, normalized_value). Fallback to [start:end] if value missing."""
    cnt = Counter()
    text = text or ""
    T = len(text)
    for s in (slots or []):
        stype = s.get("name")
        if not stype:
            continue
        val = s.get("value")
        if not isinstance(val, str) or val == "":
            try:
                st = int(s.get("start", 0)); ed = int(s.get("end", 0))
                if 0 <= st < ed <= T:
                    val = text[st:ed]
                else:
                    val = ""
            except Exception:
                val = ""
        norm = _normalize_value(val)
        if norm:
            cnt[(str(stype), norm)] += 1
    return cnt


# -----------------------------
# F1 helpers (value-level)
# -----------------------------
def _prf1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def _slot_value_f1_micro(
    gold_texts: List[str], gold_slots: List[List[Dict]],
    pred_texts: List[str], pred_slots: List[List[Dict]]
) -> float:
    TP = FP = FN = 0
    for gt, gs, pt, ps in zip(gold_texts, gold_slots, pred_texts, pred_slots):
        gset = _type_value_multiset(gt, gs)
        pset = _type_value_multiset(pt, ps)
        common = gset & pset
        TP += sum(common.values())
        FP += sum((pset - gset).values())
        FN += sum((gset - pset).values())
    _, _, f1 = _prf1(TP, FP, FN)
    return f1


def _slot_value_f1_macro(
    gold_texts: List[str], gold_slots: List[List[Dict]],
    pred_texts: List[str], pred_slots: List[List[Dict]]
) -> float:
    types = set()
    for lst in (gold_slots, pred_slots):
        for ex in lst:
            for s in (ex or []):
                if s.get("name"):
                    types.add(str(s["name"]))
    if not types:
        return 0.0

    f1_list = []
    for t in sorted(types):
        TP = FP = FN = 0
        for gt, gs, pt, ps in zip(gold_texts, gold_slots, pred_texts, pred_slots):
            gset = _type_value_multiset(gt, gs)
            pset = _type_value_multiset(pt, ps)
            g_t = Counter({(k, v): c for (k, v), c in gset.items() if k == t})
            p_t = Counter({(k, v): c for (k, v), c in pset.items() if k == t})
            common = g_t & p_t
            TP += sum(common.values())
            FP += sum((p_t - g_t).values())
            FN += sum((g_t - p_t).values())
        _, _, f1 = _prf1(TP, FP, FN)
        if (TP + FP + FN) > 0:
            f1_list.append(f1)
    return sum(f1_list) / len(f1_list) if f1_list else 0.0


# -----------------------------
# Main compute (template-aligned)
# -----------------------------
def compute_slot_f1(pred_path: str, use_post_process: bool = True):
    """Compute slot value-level F1 (micro & macro) by extracting slots from
    'answer' (gold) and 'after' (pred) using the same HF model. Report:
      - All
      - group_slot=0  (before_score == 0)
      - group_slot>0  (before_score > 0)

    Args:
        pred_path: path to predictions JSON
        use_post_process: whether to apply post_processing() before scoring
    """
    with open(pred_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if use_post_process:
        data = post_processing(data)

    items = [it for it in data if ("answer" in it and "after" in it)]
    if not items:
        print("=== Slot F1 (value-level, micro) ===")
        print("All: 0.00")
        print("group_slot=0: 0.00")
        print("group_slot>0: 0.00")
        print("=== Slot F1 (value-level, macro) ===")
        print("All: 0.00")
        print("group_slot=0: 0.00")
        print("group_slot>0: 0.00")
        return {
            "slot_micro": {"all": 0.0, "group_eq_0": 0.0, "group_gt_0": 0.0},
            "slot_macro": {"all": 0.0, "group_eq_0": 0.0, "group_gt_0": 0.0},
            "counts": {"all": 0, "group_eq_0": 0, "group_gt_0": 0},
        }

    gold_texts = [it["answer"] if isinstance(it["answer"], str) else "" for it in items]
    pred_texts = [it["after"]  if isinstance(it["after"],  str) else "" for it in items]
    groups = [0 if abs(float(it.get("before_score", 0.0))) < 1e-12 else 1 for it in items]  # 0: eq 0, 1: > 0

    slot_pipe = _load_slot_pipe()
    gold_slots = _extract_slots(gold_texts, slot_pipe, batch_size=64)
    pred_slots = _extract_slots(pred_texts, slot_pipe, batch_size=64)

    # overall
    micro_all = _slot_value_f1_micro(gold_texts, gold_slots, pred_texts, pred_slots)
    macro_all = _slot_value_f1_macro(gold_texts, gold_slots, pred_texts, pred_slots)

    # subgroup
    idx_b0 = [i for i, g in enumerate(groups) if g == 0]
    idx_bp = [i for i, g in enumerate(groups) if g == 1]

    def _take(L, idxs): return [L[i] for i in idxs]
    def _micro(idxs):
        return _slot_value_f1_micro(_take(gold_texts, idxs), _take(gold_slots, idxs),
                                    _take(pred_texts, idxs), _take(pred_slots, idxs))
    def _macro(idxs):
        return _slot_value_f1_macro(_take(gold_texts, idxs), _take(gold_slots, idxs),
                                    _take(pred_texts, idxs), _take(pred_slots, idxs))

    micro_b0 = _micro(idx_b0) if idx_b0 else 0.0
    micro_bp = _micro(idx_bp) if idx_bp else 0.0
    macro_b0 = _macro(idx_b0) if idx_b0 else 0.0
    macro_bp = _macro(idx_bp) if idx_bp else 0.0

    # Print in template style (two sections)
    print("=== Slot F1 (value-level, micro) ===")
    print(f"All: {micro_all*100:.2f}")
    print(f"group_slot=0: {micro_b0*100:.2f}")
    print(f"group_slot>0: {micro_bp*100:.2f}")

    print("=== Slot F1 (value-level, macro) ===")
    print(f"All: {macro_all*100:.2f}")
    print(f"group_slot=0: {macro_b0*100:.2f}")
    print(f"group_slot>0: {macro_bp*100:.2f}")

    return {
        "slot_micro": {"all": micro_all, "group_eq_0": micro_b0, "group_gt_0": micro_bp},
        "slot_macro": {"all": macro_all, "group_eq_0": macro_b0, "group_gt_0": macro_bp},
        "counts": {"all": len(items), "group_eq_0": len(idx_b0), "group_gt_0": len(idx_bp)},
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute slot value-level F1 (micro & macro) by extracting slots from 'answer' (gold) and 'after' (pred)."
    )
    parser.add_argument("--pred_path", required=True, help="Path to predictions JSON")
    parser.add_argument("--post_process", action="store_true", help="Enable post_processing() step")
    args = parser.parse_args()

    compute_slot_f1(args.pred_path, use_post_process=args.post_process)


if __name__ == "__main__":
    main()
