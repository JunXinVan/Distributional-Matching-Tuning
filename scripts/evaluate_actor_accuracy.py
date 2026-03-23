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

Important evaluation note:
- many math checkpoints are trained on long-form solutions but are evaluated on
  short final answers
- verifier failures should be tracked explicitly instead of being silently
  merged into ordinary mistakes
- answer extraction should be configurable so we can compare "free-form long
  reasoning" against "final-answer only" protocols without changing code
"""

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

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

DEFAULT_MATH_FINAL_ANSWER_SUFFIX = (
    "Solve the problem carefully, but in your final response output only the final answer. "
    "Put the final answer in \\boxed{} and do not include any explanation."
)


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
    parser.add_argument(
        "--prompt_preset",
        type=str,
        default="raw_question",
        choices=["raw_question", "math_final_answer_only"],
        help="Prompt preset. 'math_final_answer_only' appends an instruction to emit only the boxed final answer.",
    )
    parser.add_argument(
        "--response_extraction",
        type=str,
        default="boxed_or_full",
        choices=["full_response", "boxed_only", "boxed_or_full", "last_line", "last_line_or_full"],
        help="How to turn a raw model response into the text passed to the verifier.",
    )
    parser.add_argument(
        "--verifier_error_policy",
        type=str,
        default="record_as_incorrect",
        choices=["record_as_incorrect", "raise"],
        help="Whether verifier exceptions should be recorded as incorrect examples or crash the run.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument(
        "--backend",
        type=str,
        default="transformers",
        choices=["transformers", "vllm"],
        help="Inference backend. Use vllm for high-throughput multi-GPU evaluation.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Model dtype",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Tensor parallel size for vLLM backend.",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.85,
        help="vLLM gpu_memory_utilization.",
    )
    parser.add_argument(
        "--max_num_seqs",
        type=int,
        default=256,
        help="vLLM max_num_seqs.",
    )
    parser.add_argument(
        "--enable_prefix_caching",
        action="store_true",
        default=False,
        help="Enable prefix caching for vLLM backend.",
    )
    parser.add_argument("--seed", type=int, default=43, help="Random seed")
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of dataset shards")
    parser.add_argument("--shard_idx", type=int, default=0, help="Current shard index in [0, num_shards)")
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


def resolve_prompt_suffix(prompt_suffix: str, prompt_preset: str) -> str:
    if prompt_suffix:
        return prompt_suffix
    if prompt_preset == "math_final_answer_only":
        return DEFAULT_MATH_FINAL_ANSWER_SUFFIX
    return ""


def build_prompts_and_answers(dataset, input_key: str, answer_key: str, prompt_suffix: str):
    prompts: List[str] = []
    answers: List[str] = []
    example_ids: List[str] = []
    for row_idx, row in enumerate(dataset):
        prompt = row[input_key] if input_key in row else row.get("question", row.get("prompt", ""))
        answer = row[answer_key] if answer_key in row else row.get("answer", "")
        if prompt_suffix:
            prompt = f"{prompt.rstrip()}\n{prompt_suffix}"
        prompts.append(prompt)
        answers.append(str(answer))
        example_ids.append(str(row.get("idx", row_idx)))
    return prompts, answers, example_ids


def extract_boxed_content(text: str) -> str | None:
    matches = re.findall(r"\\boxed\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text, flags=re.DOTALL)
    if matches:
        return matches[-1].strip()
    return None


def extract_last_nonempty_line(text: str) -> str | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    return lines[-1]


def prepare_response_for_verifier(response: str, mode: str) -> Tuple[str, str]:
    response = response.strip()

    if mode == "full_response":
        return response, "full_response"

    if mode == "boxed_only":
        boxed = extract_boxed_content(response)
        return (boxed or "", "boxed_only")

    if mode == "boxed_or_full":
        boxed = extract_boxed_content(response)
        if boxed:
            return boxed, "boxed"
        return response, "full_response_fallback"

    if mode == "last_line":
        last_line = extract_last_nonempty_line(response)
        return (last_line or "", "last_line")

    if mode == "last_line_or_full":
        last_line = extract_last_nonempty_line(response)
        if last_line:
            return last_line, "last_line"
        return response, "full_response_fallback"

    raise ValueError(f"Unknown response extraction mode: {mode}")


def classify_verifier_exception(exc: Exception) -> str:
    message = str(exc).lower()
    if "timeout" in message or isinstance(exc, TimeoutError):
        return "timeout"
    if "parse" in message:
        return "parse_error"
    return exc.__class__.__name__


def verify_response(
    response: str,
    gold_answer: str,
    extraction_mode: str,
    verifier_error_policy: str,
) -> Tuple[bool, Dict[str, str | bool]]:
    candidate_response, extraction_used = prepare_response_for_verifier(response, extraction_mode)
    details: Dict[str, str | bool] = {
        "candidate_response": candidate_response,
        "extraction_used": extraction_used,
        "verifier_error": False,
        "verifier_error_type": "",
        "verifier_error_message": "",
    }

    if not candidate_response.strip():
        details["verifier_error"] = True
        details["verifier_error_type"] = "empty_candidate"
        details["verifier_error_message"] = "No candidate answer after response extraction."
        if verifier_error_policy == "raise":
            raise ValueError(details["verifier_error_message"])
        return False, details

    try:
        is_correct = bool(verify_llm_answer(candidate_response, gold_answer, raise_on_error=True))
        return is_correct, details
    except Exception as exc:
        details["verifier_error"] = True
        details["verifier_error_type"] = classify_verifier_exception(exc)
        details["verifier_error_message"] = repr(exc)
        if verifier_error_policy == "raise":
            raise
        return False, details


def generate_with_transformers(args, prompts: List[str]):
    torch_dtype = get_torch_dtype(args.dtype)

    logger.info("Loading actor-only evaluation model with transformers backend...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_checkpoint,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    model = model.to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    model.eval()

    do_sample = args.temperature > 0

    for start in tqdm(range(0, len(prompts), args.batch_size), desc="Evaluating"):
        end = min(start + args.batch_size, len(prompts))
        batch_prompts = prompts[start:end]

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
        responses = []
        for local_idx, prompt_len in enumerate(prompt_lengths):
            seq = outputs[local_idx]
            generated_ids = seq[prompt_len:]
            responses.append(tokenizer.decode(generated_ids, skip_special_tokens=True).strip())
        yield start, batch_prompts, responses


def generate_with_vllm(args, prompts: List[str]):
    try:
        from vllm import LLM, SamplingParams
    except ImportError as e:
        raise ImportError(
            "vLLM backend requested but vllm is not installed in the current environment."
        ) from e

    logger.info("Loading actor-only evaluation model with vLLM backend...")
    logger.info(
        "vLLM config: tp=%s, gpu_memory_utilization=%.2f, max_num_seqs=%s, prefix_caching=%s",
        args.tensor_parallel_size,
        args.gpu_memory_utilization,
        args.max_num_seqs,
        args.enable_prefix_caching,
    )

    llm = LLM(
        model=args.model_checkpoint,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        trust_remote_code=True,
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        enable_prefix_caching=args.enable_prefix_caching,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        max_tokens=args.max_new_tokens,
        skip_special_tokens=True,
        truncate_prompt_tokens=args.prompt_max_len,
    )

    for start in tqdm(range(0, len(prompts), args.batch_size), desc="Evaluating"):
        end = min(start + args.batch_size, len(prompts))
        batch_prompts = prompts[start:end]
        outputs = llm.generate(batch_prompts, sampling_params)
        responses = [output.outputs[0].text.strip() for output in outputs]
        yield start, batch_prompts, responses


def main():
    args = parse_args()
    set_seed(args.seed)
    args.prompt_suffix = resolve_prompt_suffix(args.prompt_suffix, args.prompt_preset)

    dataset = load_eval_dataset(args.eval_dataset, args.eval_split)
    if args.eval_max_samples is not None and args.eval_max_samples > 0:
        dataset = dataset.select(range(min(args.eval_max_samples, len(dataset))))

    if args.num_shards < 1:
        raise ValueError(f"--num_shards must be >= 1, got {args.num_shards}")
    if not 0 <= args.shard_idx < args.num_shards:
        raise ValueError(
            f"--shard_idx must be in [0, {args.num_shards}), got {args.shard_idx}"
        )
    if args.num_shards > 1:
        shard_indices = list(range(args.shard_idx, len(dataset), args.num_shards))
        dataset = dataset.select(shard_indices)
        logger.info(
            f"Using dataset shard {args.shard_idx}/{args.num_shards} with {len(dataset)} samples"
        )

    prompts, answers, example_ids = build_prompts_and_answers(
        dataset, args.input_key, args.answer_key, args.prompt_suffix
    )

    logger.info(f"Evaluating {len(prompts)} samples")
    logger.info(
        "answer_key=%s, backend=%s, temperature=%s, batch_size=%s",
        args.answer_key,
        args.backend,
        args.temperature,
        args.batch_size,
    )

    predictions = []
    mismatches = []
    correct = 0
    total = 0
    verifier_error_count = 0
    verifier_error_breakdown: Dict[str, int] = {}
    extraction_counts: Dict[str, int] = {}

    if args.backend == "vllm":
        batch_iterator = generate_with_vllm(args, prompts)
    else:
        batch_iterator = generate_with_transformers(args, prompts)

    for start, batch_prompts, responses in batch_iterator:
        end = min(start + len(batch_prompts), len(prompts))
        batch_answers = answers[start:end]

        for local_idx, (gold_answer, prompt, response) in enumerate(zip(batch_answers, batch_prompts, responses)):
            is_correct, verification_details = verify_response(
                response=response,
                gold_answer=gold_answer,
                extraction_mode=args.response_extraction,
                verifier_error_policy=args.verifier_error_policy,
            )

            total += 1
            correct += 1 if is_correct else 0
            extraction_used = str(verification_details["extraction_used"])
            extraction_counts[extraction_used] = extraction_counts.get(extraction_used, 0) + 1
            if verification_details["verifier_error"]:
                verifier_error_count += 1
                error_type = str(verification_details["verifier_error_type"])
                verifier_error_breakdown[error_type] = verifier_error_breakdown.get(error_type, 0) + 1

            record = {
                "idx": start + local_idx,
                "example_id": example_ids[start + local_idx],
                "question": prompt,
                "gold_answer": gold_answer,
                "response": response,
                "candidate_response": verification_details["candidate_response"],
                "extraction_used": extraction_used,
                "is_correct": is_correct,
                "verifier_error": verification_details["verifier_error"],
                "verifier_error_type": verification_details["verifier_error_type"],
                "verifier_error_message": verification_details["verifier_error_message"],
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
            "backend": args.backend,
            "eval_max_samples": args.eval_max_samples,
            "batch_size": args.batch_size,
            "prompt_max_len": args.prompt_max_len,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
            "tensor_parallel_size": args.tensor_parallel_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_num_seqs": args.max_num_seqs,
            "enable_prefix_caching": args.enable_prefix_caching,
            "prompt_suffix": args.prompt_suffix,
            "prompt_preset": args.prompt_preset,
            "response_extraction": args.response_extraction,
            "verifier_error_policy": args.verifier_error_policy,
            "seed": args.seed,
            "num_shards": args.num_shards,
            "shard_idx": args.shard_idx,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "metrics": {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "verifier_error_count": verifier_error_count,
        },
        "verifier_error_breakdown": verifier_error_breakdown,
        "extraction_counts": extraction_counts,
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
    logger.info(f"verifier_error_count: {verifier_error_count}")
    if verifier_error_breakdown:
        logger.info(f"verifier_error_breakdown: {verifier_error_breakdown}")
    logger.info(f"extraction_counts: {extraction_counts}")
    logger.info(f"results saved to: {output_path}")
    if args.predictions_file:
        logger.info(f"predictions saved to: {args.predictions_file}")


if __name__ == "__main__":
    main()
