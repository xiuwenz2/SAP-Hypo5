#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
from typing import List, Dict, Any, Optional

import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from utils.postprocessing import post_processing


def _load_items(pred_path: str) -> List[Dict[str, Any]]:
    """Load JSON or JSONL; keep items that have at least 'after' and 'answer'."""
    items: List[Dict[str, Any]] = []
    with open(pred_path, "r", encoding="utf-8") as f:
        head = f.read(2048)
        f.seek(0)
        if head.lstrip().startswith("["):
            items = json.load(f)
        else:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))

    records = [r for r in items if all(k in r for k in ("after", "answer"))]

    # Keep original behavior: if non-empty, drop the last item
    if records:
        records = records[:-1]
    return records


def _group_is_noerr(rec: Dict[str, Any]) -> bool:
    """
    Grouping rule:
      - If 'before_score' exists: noerr if before_score == 0
      - Otherwise: noerr if before == answer (string equality)
    """
    if "before_score" in rec:
        try:
            return float(rec.get("before_score", 0.0)) == 0.0
        except Exception:
            return False
    return rec.get("before", "") == rec.get("answer", "")


def _cosine_pairwise(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cosine_similarity(u, v, dim=1)


def compute_mean_cosine_grouped(
    pred_path: str,
    model: str = "Qwen/Qwen3-Embedding-8B",
    cache_dir: Optional[str] = None,
    batch_size: int = 32,
    device: Optional[str] = None,
):
    """
    Compute cosine similarity between 'after' and 'answer' embeddings,
    grouped into All / group_eq_0 / group_gt_0 (consistent with reference style).
    """
    records = _load_items(pred_path)
    records = post_processing(records)

    if not records:
        print("=== Cosine similarity (after vs answer) ===")
        print("All: nan")
        print("group=0: nan")
        print("group>0: nan")
        return {
            "cos_all": float("nan"),
            "cos_group_eq_0": float("nan"),
            "cos_group_gt_0": float("nan"),
            "counts": {"all": 0, "group_eq_0": 0, "group_gt_0": 0},
        }

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    st_model = SentenceTransformer(model, cache_folder=cache_dir)
    st_model.to(device)

    afters  = [r["after"] for r in records]
    answers = [r["answer"] for r in records]
    is_noerr = [_group_is_noerr(r) for r in records]  # True => group_eq_0, False => group_gt_0

    sum_all = 0.0; cnt_all = 0
    sum_b0  = 0.0; cnt_b0  = 0   # group_eq_0
    sum_bp  = 0.0; cnt_bp  = 0   # group_gt_0

    with torch.inference_mode():
        for i in tqdm(range(0, len(afters), batch_size), desc="Embedding (grouped)"):
            a_batch = afters[i: i + batch_size]
            r_batch = answers[i: i + batch_size]
            f_batch = is_noerr[i: i + batch_size]

            em_a = st_model.encode(
                a_batch, convert_to_tensor=True, normalize_embeddings=True,
                device=device, batch_size=batch_size, show_progress_bar=False
            )
            em_r = st_model.encode(
                r_batch, convert_to_tensor=True, normalize_embeddings=True,
                device=device, batch_size=batch_size, show_progress_bar=False
            )

            sims = _cosine_pairwise(em_a, em_r)  # (B,)
            B = sims.numel()

            # All
            sum_all += float(sims.sum().item())
            cnt_all += B

            # Grouped accumulation
            if any(f_batch):
                mask = torch.tensor(f_batch, device=sims.device, dtype=torch.bool)
                if mask.any():
                    sum_b0 += float(sims[mask].sum().item())
                    cnt_b0 += int(mask.sum().item())
                if (~mask).any():
                    sum_bp += float(sims[~mask].sum().item())
                    cnt_bp += int((~mask).sum().item())
            else:
                # All go to group_gt_0
                sum_bp += float(sims.sum().item())
                cnt_bp += B

            # Cleanup
            del em_a, em_r, sims
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    cos_all = (sum_all / max(cnt_all, 1)) if cnt_all > 0 else float("nan")
    cos_b0  = (sum_b0  / max(cnt_b0, 1)) if cnt_b0  > 0 else float("nan")
    cos_bp  = (sum_bp  / max(cnt_bp, 1)) if cnt_bp  > 0 else float("nan")

    print("=== Cosine similarity (after vs answer) ===")
    print(f"All: {cos_all:.6f}")
    print(f"group=0: {cos_b0:.6f}")
    print(f"group>0: {cos_bp:.6f}")

    return {
        "cos_all": cos_all,
        "cos_group_eq_0": cos_b0,
        "cos_group_gt_0": cos_bp,
        "counts": {"all": cnt_all, "group_eq_0": cnt_b0, "group_gt_0": cnt_bp},
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute mean cosine similarity between 'after' and 'answer' grouped by before_score (framework-unified)."
    )
    parser.add_argument(
        "--pred_path",
        default="./res/sap/Qwen2-7B-Instruct/beam4_test.json",
        help="Path to predictions JSON/JSONL",
    )
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-8B", help="SentenceTransformer embedding model")
    parser.add_argument("--cache_dir", default="/projects/bedl/.cache", help="Cache dir for SentenceTransformer")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default=None, help="Override device, e.g., cuda / cpu")

    args = parser.parse_args()

    compute_mean_cosine_grouped(
        pred_path=args.pred_path,
        model=args.model,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
