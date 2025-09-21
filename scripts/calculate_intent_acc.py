#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
from collections import defaultdict

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from utils.postprocessing import post_processing


def _norm_label(s: str) -> str:
    """Looser match: collapse spaces + casefold."""
    if not isinstance(s, str):
        return ""
    return " ".join(s.split()).casefold()


def _infer_intents(texts, pipe, batch_size: int = 64):
    """Batch infer intent labels with a HF pipeline."""
    out = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        preds = pipe(batch)
        for cand_list in preds:  # list of {label, score}
            best = max(cand_list, key=lambda x: float(x["score"]))
            out.append(str(best["label"]))
    return out


def compute_intent_acc(pred_path: str, use_post_process: bool = True):
    """Compute intent accuracy by inferring intents from 'answer' (gold) and 'after' (pred),
    reporting All, group_intent=0, group_intent>0. Grouping is by `before_score`.

    Args:
        pred_path: path to predictions JSON
        use_post_process: whether to apply post_processing() before scoring
    """
    with open(pred_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if use_post_process:
        data = post_processing(data)

    # Keep items that contain both answer/after
    items = [it for it in data if ("answer" in it and "after" in it)]
    if not items:
        print("=== Intent Accuracy ===")
        print("All: 0.00")
        print("group_intent=0: 0.00")
        print("group_intent>0: 0.00")
        return {
            "intent_acc_all": 0.0,
            "intent_acc_group_eq_0": 0.0,
            "intent_acc_group_gt_0": 0.0,
            "totals": {
                "all": {"correct": 0, "total": 0},
                "group_eq_0": {"correct": 0, "total": 0},
                "group_gt_0": {"correct": 0, "total": 0},
            },
        }

    gold_texts = [it["answer"] if isinstance(it["answer"], str) else "" for it in items]
    pred_texts = [it["after"]  if isinstance(it["after"],  str) else "" for it in items]
    groups = [0 if abs(float(it.get("before_score", 0.0))) < 1e-12 else 1 for it in items]  # 0: eq 0, 1: > 0

    # Single intent model for both gold/pred inference
    model_id = "cartesinus/xlm-r-base-amazon-massive-intent"
    device = 0 if torch.cuda.is_available() else -1
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_id)
    pipe = pipeline(
        "text-classification",
        model=mdl,
        tokenizer=tok,
        device=device,
        return_all_scores=True,
        truncation=True,
    )

    gold_labels = _infer_intents(gold_texts, pipe, batch_size=64)
    pred_labels = _infer_intents(pred_texts, pipe, batch_size=64)

    totals_all = {"correct": 0, "total": 0}
    totals_b0  = {"correct": 0, "total": 0}  # before_score == 0
    totals_bp  = {"correct": 0, "total": 0}  # before_score > 0

    for g, p, grp in zip(gold_labels, pred_labels, groups):
        g_norm = _norm_label(g)
        p_norm = _norm_label(p)
        ok = int(g_norm != "" and g_norm == p_norm)

        totals_all["correct"] += ok; totals_all["total"] += 1
        if grp == 0:
            totals_b0["correct"] += ok; totals_b0["total"] += 1
        else:
            totals_bp["correct"] += ok; totals_bp["total"] += 1

    def acc(t):  # accuracy helper
        n = t.get("total", 0)
        return 0.0 if n == 0 else t.get("correct", 0) / n

    all_acc = acc(totals_all)
    b0_acc  = acc(totals_b0)
    bp_acc  = acc(totals_bp)

    print("=== Intent Accuracy ===")
    print(f"All: {all_acc*100:.2f}")
    print(f"group_intent=0: {b0_acc*100:.2f}")
    print(f"group_intent>0: {bp_acc*100:.2f}")

    return {
        "intent_acc_all": all_acc,
        "intent_acc_group_eq_0": b0_acc,
        "intent_acc_group_gt_0": bp_acc,
        "totals": {
            "all": dict(totals_all),
            "group_eq_0": dict(totals_b0),
            "group_gt_0": dict(totals_bp),
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute Intent Accuracy by inferring intents on 'answer' and 'after' (same model)."
    )
    parser.add_argument("--pred_path", required=True, help="Path to predictions JSON")
    parser.add_argument("--post_process", action="store_true", help="Enable post_processing() step")
    args = parser.parse_args()

    compute_intent_acc(args.pred_path, use_post_process=args.post_process)


if __name__ == "__main__":
    main()
