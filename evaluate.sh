#!/bin/bash
# Usage:
#   ./evaluate.sh PRED_PATH
#
# Defaults:
#   stage=1, stop_stage=3
#
# Stages:
#   1) WER
#   2) Semantic metrics (Q-Embedding, BERTScore, MENLI)
#   3) Intent & Slot filling

set -e

PRED_PATH=$1

STAGE=1
STOP_STAGE=3

if [ -z "$PRED_PATH" ]; then
  echo "Usage: $0 PRED_PATH"
  exit 1
fi

PYTHON=python3   # change if needed

echo
echo ">>> Config: stage=$STAGE, stop_stage=$STOP_STAGE"
echo ">>> PRED_PATH=$PRED_PATH"

########################################
# Stage 1: WER
########################################
if [ $STAGE -le 1 ] && [ $STOP_STAGE -ge 1 ]; then
  echo
  echo "=== Stage 1: WER ==="
  echo ">>> Running calculate_wer.py"
  $PYTHON -m scripts.calculate_wer --pred_path "$PRED_PATH" --post_process
fi

########################################
# Stage 2: Semantic metrics
########################################
if [ $STAGE -le 2 ] && [ $STOP_STAGE -ge 2 ]; then
  echo
  echo "=== Stage 2: Semantic metrics ==="

  echo ">>> Running calculate_qemb.py"
  $PYTHON -m scripts.calculate_qemb --pred_path "$PRED_PATH"

  echo
  echo ">>> Running calculate_bertscore.py"
  $PYTHON -m scripts.calculate_bertscore --pred_path "$PRED_PATH"

  echo
  echo ">>> Running calculate_menli.py"
  $PYTHON -m scripts.calculate_menli --pred_path "$PRED_PATH"
fi

########################################
# Stage 3: Intent & Slot filling
########################################
if [ $STAGE -le 3 ] && [ $STOP_STAGE -ge 3 ]; then
  echo
  echo "=== Stage 3: Intent & Slot filling ==="

  # Intent Accuracy
  echo
  echo ">>> Running calculate_intent_acc.py"
  $PYTHON -m scripts.calculate_intent_acc --pred_path "$PRED_PATH"

  # # Slot Filling
  echo
  echo ">>> Running calculate_slot_f1.py"
  $PYTHON -m scripts.calculate_slot_f1 --pred_path "$PRED_PATH"
fi
