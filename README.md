# SAP-Hypo5

This repository provides a comprehensive pipeline for fine-tuning and multi-dimensional evaluation of large language models in ASR hypothesis correction for dysarthric speech recognition. Please find the SAP-Hypo5 dataset [here](https://huggingface.co/datasets/xiuwenz2/SAP-Hypo5).

### Fine-tuning
We release our fine-tuned meta-llama/Llama-3.1-8B on the SAP-Hypo5 dataset ([model](https://huggingface.co/xiuwenz2/Llama-3.1-8B-ft-SAP-Hypo5)) following [Hypo2Trans](https://github.com/Hypotheses-Paradise/Hypo2Trans). Set ```train_on_inputs=False``` for faster convergence and better performance.

### Inference
```
python inference.py \
      --test_data_path <path/to/test.json> \
      --out_dir <dir/to/output>
```
Note: Prompter and prompt templates are directly copied from the original repository.

### Evaluation
Evaluate LLM-corrected ASR transcripts from a single JSON/JSONL file.

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
