import json
import argparse
from collections import defaultdict

import jiwer
from utils.postprocessing import post_processing

# word-level normalization for WER
TRANSFORM = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(),
])


def compute_csid_wer(pred_path: str):
    """Compute WER from CSID only, reporting All, group_wer=0, group_wer>0.
    Grouping is by `before_score` in each item.
    """
    with open(pred_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if data:  # keep original behavior
        data = data[:-1]

    data = post_processing(data)

    totals_all = defaultdict(int)
    totals_b0  = defaultdict(int)  # before_score == 0
    totals_bp  = defaultdict(int)  # before_score > 0

    for it in data:
        ref  = it["answer"]
        hyp  = it["after"]

        out = jiwer.process_words(
            ref, hyp,
            reference_transform=TRANSFORM,
            hypothesis_transform=TRANSFORM,
        )
        C, S, D, I = int(out.hits), int(out.substitutions), int(out.deletions), int(out.insertions)
        N = C + S + D

        for dst in (totals_all, totals_b0 if abs(float(it.get("before_score", 0.0))) < 1e-12 else totals_bp):
            dst["C"] += C; dst["S"] += S; dst["D"] += D; dst["I"] += I; dst["Nref"] += N

    def wer(t):
        N = t.get("Nref", 0)
        return 0.0 if N == 0 else (t.get("S", 0) + t.get("D", 0) + t.get("I", 0)) / N

    all_wer = wer(totals_all)
    b0_wer  = wer(totals_b0)
    bp_wer  = wer(totals_bp)

    print("=== WER (CSID only) ===")
    print(f"All: {all_wer*100:.2f}")
    print(f"group_wer=0: {b0_wer*100:.2f}")
    print(f"group_wer>0: {bp_wer*100:.2f}")

    return {
        "wer_all": all_wer,
        "wer_group_eq_0": b0_wer,
        "wer_group_gt_0": bp_wer,
        "totals": {"all": dict(totals_all), "group_eq_0": dict(totals_b0), "group_gt_0": dict(totals_bp)},
    }


def main():
    parser = argparse.ArgumentParser(description="Compute WER from CSID only (jiwer v3)")
    parser.add_argument("--pred_path", default="./res/sap/Qwen2-7B-Instruct/beam4_test.json", help="Path to predictions JSON")
    args = parser.parse_args()

    compute_csid_wer(args.pred_path)


if __name__ == "__main__":
    main()
