#!/usr/bin/env python3
"""
Actor-only math evaluation on a local/json HF dataset.

This script is designed for fair, apples-to-apples comparison between
G1/G2/G3 actor checkpoints:

- same actor-only evaluation path
- same dataset
- same prompt formatting
- same generation settings
- one primary metric: exact accuracy via math_verify
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openrlhf.utils.logging_utils import init_logger
from openrlhf.utils.math_verifier import verify_llm_answer

logger = init_logger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Actor-only exact-accuracy evaluation")
    parser.add_argument("--model_checkpoint", type=str, required=True, help="HF-loadable actor checkpoint")
    parser.add_argument("--eval_dataset", type=str, required=True, help="Dataset name or local json/jsonl path")
    parser.add_argument("--eval_split", type=str, default="test", help="Dataset split")
    parser.add_argument("--input_key", type=str, default="question", help="Prompt field")
    parser.add_argument("--answer_key", type=str, default="answer", help="Gold final-answer field")
    parser.add_argument("--eval_max_samples", type=int, default=None, help="Optional sample cap")
    parser.add_argument("--batch_size", type=int, default=8, help="Generation batch size")
    parser.add_argument("--prompt_max_len", type=int, default=1024, help="Prompt truncation length")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Generation budget")
    parser.add_argument("--temperature", type=float, default=0.0, help="0 means greedy decoding")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p if sampling enabled")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="Generation repetition penalty")
    parser.add_argument(
        "--prompt_suffix",
        type=str,
        default="",
        help="Optional fixed suffix appended to each question for all models",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Model dtype",
    )
    parser.add_argument("--seed", type=int, default=43, help="Random seed")
    parser.add_argument("--output_file", type=str, required=True, help="Path to summary json output")
    parser.add_argument(
        "--predictions_file",
        type=str,
        default=None,
        help="Optional jsonl file to save per-example generations",
    )
    parser.add_argument("--mismatch_limit", type=int, default=20, help="How many mismatches to store in summary")
    return parser.parse_args()


def get_torch_dtype(dtype_name: str):
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "float16":
        return torch.float16
    return torch.bfloat16


def load_eval_dataset(eval_dataset: str, eval_split: str):
    eval_dataset_path = Path(eval_dataset).expanduser()

    if eval_dataset_path.is_file():
        if eval_dataset_path.suffix.lower() not in {".json", ".jsonl"}:
            raise ValueError(f"Unsupported local dataset file: {eval_dataset_path}")
        logger.info(f"Loading local dataset file: {eval_dataset_path}")
        return load_dataset("json", data_files={eval_split: str(eval_dataset_path)})[eval_split]

    if eval_dataset_path.is_dir():
        candidates = sorted(
            p for p in eval_dataset_path.iterdir()
            if p.is_file() and p.suffix.lower() in {".json", ".jsonl"}
        )
        if not candidates:
            raise ValueError(f"No .json/.jsonl files found under {eval_dataset_path}")
        if len(candidates) > 1:
            raise ValueError(
                f"Multiple dataset files found under {eval_dataset_path}: {[p.name for p in candidates]}. "
                "Pass the exact file path instead."
            )
        logger.info(f"Loading local dataset directory via file: {candidates[0]}")
        return load_dataset("json", data_files={eval_split: str(candidates[0])})[eval_split]

    logger.info(f"Loading hub dataset: {eval_dataset} [{eval_split}]")
    if eval_dataset == "openai/gsm8k":
        return load_dataset(eval_dataset, name="main")[eval_split]
    return load_dataset(eval_dataset)[eval_split]


def build_prompts_and_answers(dataset, input_key: str, answer_key: str, prompt_suffix: str):
    prompts: List[str] = []
    answers: List[str] = []
    for row in dataset:
        prompt = row[input_key] if input_key in row else row.get("question", row.get("prompt", ""))
        answer = row[answer_key] if answer_key in row else row.get("answer", "")
        if prompt_suffix:
            prompt = f"{prompt.rstrip()}\n{prompt_suffix}"
        prompts.append(prompt)
        answers.append(str(answer))
    return prompts, answers


def main():
    args = parse_args()
    set_seed(args.seed)

    torch_dtype = get_torch_dtype(args.dtype)

    logger.info("Loading actor-only evaluation model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_checkpoint,
        torch_dtype=torch_dtype,
        device_map=args.device,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    model.eval()

    dataset = load_eval_dataset(args.eval_dataset, args.eval_split)
    if args.eval_max_samples is not None and args.eval_max_samples > 0:
        dataset = dataset.select(range(min(args.eval_max_samples, len(dataset))))

    prompts, answers = build_prompts_and_answers(dataset, args.input_key, args.answer_key, args.prompt_suffix)

    logger.info(f"Evaluating {len(prompts)} samples")
    logger.info(f"answer_key={args.answer_key}, temperature={args.temperature}, batch_size={args.batch_size}")

    predictions = []
    mismatches = []
    correct = 0
    total = 0

    do_sample = args.temperature > 0

    for start in tqdm(range(0, len(prompts), args.batch_size), desc="Evaluating"):
        end = min(start + args.batch_size, len(prompts))
        batch_prompts = prompts[start:end]
        batch_answers = answers[start:end]

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.prompt_max_len,
        )
        input_ids = inputs["input_ids"].to(args.device)
        attention_mask = inputs["attention_mask"].to(args.device)
        prompt_lengths = attention_mask.sum(dim=1).tolist()

        generate_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": do_sample,
            "repetition_penalty": args.repetition_penalty,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            generate_kwargs["temperature"] = args.temperature
            generate_kwargs["top_p"] = args.top_p

        with torch.no_grad():
            outputs = model.generate(**generate_kwargs)

        outputs = outputs.detach().cpu()

        for local_idx, (gold_answer, prompt, prompt_len) in enumerate(zip(batch_answers, batch_prompts, prompt_lengths)):
            seq = outputs[local_idx]
            generated_ids = seq[prompt_len:]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            try:
                is_correct = bool(verify_llm_answer(response, gold_answer))
            except Exception:
                is_correct = False

            total += 1
            correct += 1 if is_correct else 0

            record = {
                "idx": start + local_idx,
                "question": prompt,
                "gold_answer": gold_answer,
                "response": response,
                "is_correct": is_correct,
            }
            predictions.append(record)
            if not is_correct and len(mismatches) < args.mismatch_limit:
                mismatches.append(record)

    accuracy = correct / total if total else 0.0
    results = {
        "metadata": {
            "model_checkpoint": args.model_checkpoint,
            "eval_dataset": args.eval_dataset,
            "eval_split": args.eval_split,
            "input_key": args.input_key,
            "answer_key": args.answer_key,
            "eval_max_samples": args.eval_max_samples,
            "batch_size": args.batch_size,
            "prompt_max_len": args.prompt_max_len,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
            "prompt_suffix": args.prompt_suffix,
            "seed": args.seed,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "metrics": {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        },
        "mismatches": mismatches,
    }

    output_path = Path(args.output_file).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    if args.predictions_file:
        predictions_path = Path(args.predictions_file).expanduser().resolve()
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        with predictions_path.open("w") as f:
            for row in predictions:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    logger.info("=" * 80)
    logger.info("ACTOR-ONLY ACCURACY RESULTS")
    logger.info("=" * 80)
    logger.info(f"accuracy: {accuracy:.4%} ({correct}/{total})")
    logger.info(f"results saved to: {output_path}")
    if args.predictions_file:
        logger.info(f"predictions saved to: {args.predictions_file}")


if __name__ == "__main__":
    main()
