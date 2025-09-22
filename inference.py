import os
import sys
import json
import re
import gc
from argparse import ArgumentParser
from pathlib import Path

import torch
import transformers
import numpy as np
from tqdm import tqdm

from utils.prompter import Prompter
import editdistance

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel, PeftConfig

device = "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
load_8bit = True  # Set to False if bitsandbytes is unavailable

def build_prompts(prompter, inputs_ls):
    """
    Build a plain prompt string from input list.
    """
    input1 = inputs_ls[0]
    input2 = ". ".join(inputs_ls[1:]) + "." if len(inputs_ls) > 1 else None
    return prompter.generate_prompt(input=input1, input2=input2)

def clean_text(text):
    """
    Lightweight text cleaning used only when --clean_text is enabled:
    - Remove non-word chars (keep underscore)
    - Remove isolated apostrophes
    - Remove digits
    """
    text = re.sub(r"[^\w\s']", '', text)
    text = re.sub(r"(?<!\w)'|'(?!\w)", '', text)
    text = re.sub(r"[\d]", '', text)
    return text

def calculate_wer(pre, ref):
    wer_score = editdistance.eval(pre, ref) / len(ref)
    return wer_score

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--num_test_pairs", type=int, default=1000000)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--clean_text", action="store_false", help="Enable optional text cleaning for pred/gt/best_hypo")

    parser.add_argument("--ckpt_path", type=str, default="xiuwenz2/Llama-3.1-8B-ft-SAP-Hypo5", help="Path to LoRA adapter checkpoint directory.")
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to test JSON file.")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for results.")

    args = parser.parse_args()
    prompter = Prompter("H2T-LoRA")

    os.makedirs(args.out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    # Llama 3.1 tokenizer usually has pad_token; if not, fallback to eos
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Left padding is safer for batched causal generation
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        load_in_8bit=load_8bit,
        device_map="auto",
        torch_dtype=torch.float16 if not load_8bit else None,
        # trust_remote_code=True,
    )

    # Load LoRA adapter (keep consistent with how you saved it)
    peft_config = PeftConfig.from_pretrained(args.ckpt_path)
    model = PeftModel.from_pretrained(
        model,
        args.ckpt_path,
        torch_dtype=torch.float16 if not load_8bit else None,
        device_map="auto",
    )

    model.eval()
    if (not load_8bit) and torch.__version__ >= "2" and sys.platform != "win32":
        try:
            model = torch.compile(model)
        except Exception:
            pass

    with open(args.test_data_path, 'r') as f:
        test_data = json.load(f)
    test_data = test_data[:min(args.num_test_pairs, len(test_data))]
    print(f"Running inference on {len(test_data)} samples...")

    before_2, after_2 = 0, 0
    results = []

    # Greedy decoding only
    gen_cfg = GenerationConfig(
        num_beams=1,
        do_sample=False,
        max_new_tokens=64,
        repetition_penalty=args.repetition_penalty,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    for i in tqdm(range(len(test_data))):

        item = test_data[i]
        inputs = item['input']
        gt = item.get('output_2', item.get('output', ""))

        prompt = build_prompts(prompter, inputs)

        inputs_pt = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs_pt,
                generation_config=gen_cfg,
                return_dict_in_generate=True
            )
        pred = tokenizer.decode(out.sequences[0], skip_special_tokens=True)
        pred = prompter.get_response(pred)

        if args.clean_text:
            pred = re.sub(r'</s>', '', pred).lower()
            pred = clean_text(pred)

        best_hypo = clean_text(inputs[0])
        gt = clean_text(gt)
        pred = re.sub(r'\n+.+', '', pred)

        wer_prediction = calculate_wer(pred.split(), gt.split())
        wer_best_hypo = calculate_wer(best_hypo.split(), gt.split())

        before_2 += wer_best_hypo
        after_2 += wer_prediction

        results.append({
            'before': best_hypo,
            'after': pred,
            'answer': gt,
            'before_score': wer_best_hypo,
            'after_score': wer_prediction,
        })

        # You can comment out the following prints for speed
        print('before:::', best_hypo)
        print('after :::', pred)
        print('answer:::', gt)
        print('before score', wer_best_hypo)
        print('after score', wer_prediction)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    n = len(test_data)
    results.append({
        'num of test pairs': len(test_data),
        'before_2': before_2 / n,
        'after_2': after_2 / n,
    })

    with open(f"{args.out_dir}/{args.split}.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

