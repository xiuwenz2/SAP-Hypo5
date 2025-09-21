#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils.postprocessing import post_processing

# Min–max thresholds from your earlier scaling
NLI_MIN = 0.0028752246871590614
NLI_MAX = 0.9661698341369629


def _minmax_norm(xs, lo, hi):
    rng = float(hi) - float(lo)
    return [max(0.0, min(1.0, (float(x) - float(lo)) / rng)) for x in xs]


def compute_nli_score(pred_path: str, use_post_process: bool = True):
    """Compute mean normalized NLI (avg of ref→hyp & hyp→ref), reporting All, group_nli=0, group_nli>0.
    Grouping is by `before_score` in each item.

    Args:
        pred_path: path to predictions JSON
        use_post_process: whether to apply post_processing() before scoring
    """
    with open(pred_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if use_post_process:
        data = post_processing(data)

    pairs = []
    flags = []  # 1 => before_score==0, 0 => >0
    for it in data:
        if "answer" in it and "after" in it:
            pairs.append((it["answer"], it["after"]))
            flags.append(1 if abs(float(it.get("before_score", 0.0))) < 1e-12 else 0)

    if not pairs:
        print("=== NLI (entailment, avg, normalized) ===")
        print("All: nan")
        print("group_nli=0: nan")
        print("group_nli>0: nan")
        return {
            "nli_all": float("nan"),
            "nli_group_eq_0": float("nan"),
            "nli_group_gt_0": float("nan"),
            "counts": {"all": 0, "group_eq_0": 0, "group_gt_0": 0},
        }

    model_name = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name, cache_dir=".cache")
    mdl = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=3, cache_dir=".cache"
    ).to(device)
    mdl.eval()

    @torch.no_grad()
    def _probs(left, right, bs=64):
        outs = []
        for i in range(0, len(left), bs):
            enc = tok(
                left[i:i+bs], right[i:i+bs],
                padding=True, truncation=True, max_length=tok.model_max_length,
                return_tensors="pt",
            ).to(device)
            logits = mdl(**enc).logits
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
            outs.append(probs[:, 0])  # entailment @ index 0 (matching your original)
        return np.concatenate(outs, axis=0)

    refs = [r for r, _ in pairs]
    hyps = [h for _, h in pairs]
    rh = _probs(refs, hyps)
    hr = _probs(hyps, refs)
    raw = ((rh + hr) / 2.0).tolist()

    # Normalize to [0,1] using your thresholds
    scores = _minmax_norm(raw, NLI_MIN, NLI_MAX)

    totals_all = defaultdict(float); cnt_all = 0
    totals_b0  = defaultdict(float); cnt_b0  = 0
    totals_bp  = defaultdict(float); cnt_bp  = 0

    for s, flg in zip(scores, flags):
        totals_all["sum"] += s; cnt_all += 1
        if flg == 1:
            totals_b0["sum"] += s; cnt_b0 += 1
        else:
            totals_bp["sum"] += s; cnt_bp += 1

    def mean(sumv, cnt): return (sumv / cnt) if cnt > 0 else float("nan")
    all_m = mean(totals_all["sum"], cnt_all)
    b0_m  = mean(totals_b0["sum"],  cnt_b0)
    bp_m  = mean(totals_bp["sum"],  cnt_bp)

    print("=== NLI (entailment, avg, normalized) ===")
    print(f"All: {all_m:.6f}")
    print(f"group_nli=0: {b0_m:.6f}")
    print(f"group_nli>0: {bp_m:.6f}")

    return {
        "nli_all": all_m,
        "nli_group_eq_0": b0_m,
        "nli_group_gt_0": bp_m,
        "counts": {"all": cnt_all, "group_eq_0": cnt_b0, "group_gt_0": cnt_bp},
    }


def main():
    parser = argparse.ArgumentParser(description="Compute NLI entailment (avg, normalized)")
    parser.add_argument(
        "--pred_path",
        required=True,
        help="Path to predictions JSON",
    )
    parser.add_argument(
        "--post_process",
        action="store_true",
        help="Enable post_processing() step",
    )
    args = parser.parse_args()

    compute_nli_score(args.pred_path, use_post_process=args.post_process)


if __name__ == "__main__":
    main()
