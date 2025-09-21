# SAP-Hypo5

### Evaluation
Evaluate LLM-corrected ASR transcripts from a single JSON/JSONL file.

#### Usage
```bash
bash scripts/evaluate.sh <path/to/json>
```

- Stage 1. WER – CSID-based WER with default post-processing.
- Stage 2. Semantic Scores – Q-Embedding cosine, BERTScore (F1), MENLI/NLI.
- Stage 3. Downstream Metrics – Intent accuracy and slot filling F1.

#### Input Format
```
[{before: ASR hypothesis (raw),  after: corrected hypothesis (LLM output),  answer: ground-truth reference,
  before_score: WER(before, answer),  after_score: WER(after, answer)}]
```
