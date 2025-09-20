#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
from collections import defaultdict

import torch
from bert_score import score as bert_score
from utils.postprocessing import post_processing

# Min–max thresholds from your earlier scaling
BERT_MIN = -0.1180
BERT_MAX = 1.0


def _minmax_norm(xs, lo, hi):
    rng = float(hi) - float(lo)
    return [max(0.0, min(1.0, (float(x) - float(lo)) / rng)) for x in xs]


def compute_bertscore(pred_path: str):
    """Compute normalized BERTScore F1 (rescaled baseline -> then min–max) for:
       All, group_bert=0, group_bert>0. Grouping is by `before_score`.
    """
    with open(pred_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    data = post_processing(data)

    refs = []
    hyps = []
    flags = []  # 1 => before_score==0, 0 => >0
    for it in data:
        if "answer" in it and "after" in it:
            refs.append(it["answer"])
            hyps.append(it["after"])
            flags.append(1 if abs(float(it.get("before_score", 0.0))) < 1e-12 else 0)

    if not refs:
        print("=== BERTScore (F1, rescaled + normalized) ===")
        print("All: nan")
        print("group_bert=0: nan")
        print("group_bert>0: nan")
        return {
            "bert_all": float("nan"),
            "bert_group_eq_0": float("nan"),
            "bert_group_gt_0": float("nan"),
            "counts": {"all": 0, "group_eq_0": 0, "group_gt_0": 0},
        }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, F1 = bert_score(
        cands=hyps,
        refs=refs,
        lang="en",
        rescale_with_baseline=True,  # keep as before
        device=device,
        batch_size=64,
        model_type=None,
    )
    raw = F1.tolist()

    # Normalize to [0,1] using your thresholds
    scores = _minmax_norm(raw, BERT_MIN, BERT_MAX)

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

    print("=== BERTScore (F1, rescaled + normalized) ===")
    print(f"All: {all_m:.6f}")
    print(f"group_bert=0: {b0_m:.6f}")
    print(f"group_bert>0: {bp_m:.6f}")

    return {
        "bert_all": all_m,
        "bert_group_eq_0": b0_m,
        "bert_group_gt_0": bp_m,
        "counts": {"all": cnt_all, "group_eq_0": cnt_b0, "group_gt_0": cnt_bp},
    }


def main():
    parser = argparse.ArgumentParser(description="Compute BERTScore F1 (rescaled + normalized)")
    parser.add_argument("--pred_path", default="./res/sap/Qwen2-7B-Instruct/beam4_test.json",
                        help="Path to predictions JSON")
    args = parser.parse_args()
    compute_bertscore(args.pred_path)


if __name__ == "__main__":
    main()
