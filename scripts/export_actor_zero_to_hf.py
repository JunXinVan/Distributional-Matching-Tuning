#!/usr/bin/env python3
"""
Export a DeepSpeed ZeRO actor checkpoint to a Hugging Face model directory.

This is intended for post-training evaluation where the training-time actor
checkpoint exists only as ZeRO sharded files such as:

    .../_actor/global_step500/

The script reconstructs the actor weights on CPU, loads them into the base
causal LM, and writes a standard Hugging Face directory that can be consumed
by `from_pretrained(...)`.
"""

import argparse
from pathlib import Path

import torch
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Export a ZeRO actor checkpoint to HF format")
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="Base model path/name used to instantiate the architecture",
    )
    parser.add_argument(
        "--zero_checkpoint_dir",
        type=str,
        default=None,
        help="Directory that contains global_step subdirectories and optional latest file",
    )
    parser.add_argument(
        "--zero_tag_dir",
        type=str,
        default=None,
        help="Exact ZeRO tag directory, e.g. .../_actor/global_step500",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Checkpoint tag such as global_step500. Optional if --zero_tag_dir is provided.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for Hugging Face model files",
    )
    parser.add_argument(
        "--save_dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "bfloat16", "float16"],
        help="Dtype to save the exported model in",
    )
    parser.add_argument(
        "--max_shard_size",
        type=str,
        default="5GB",
        help="HF save_pretrained max_shard_size",
    )
    parser.add_argument(
        "--safe_serialization",
        action="store_true",
        help="Save weights as safetensors",
    )
    return parser.parse_args()


def resolve_zero_root_and_tag(args):
    if args.zero_tag_dir:
        tag_dir = Path(args.zero_tag_dir).expanduser().resolve()
        if not tag_dir.is_dir():
            raise FileNotFoundError(f"ZeRO tag directory does not exist: {tag_dir}")
        return tag_dir.parent, tag_dir.name

    if not args.zero_checkpoint_dir:
        raise ValueError("Either --zero_checkpoint_dir or --zero_tag_dir must be provided.")

    checkpoint_root = Path(args.zero_checkpoint_dir).expanduser().resolve()
    if not checkpoint_root.is_dir():
        raise FileNotFoundError(f"ZeRO checkpoint root does not exist: {checkpoint_root}")

    if args.tag:
        return checkpoint_root, args.tag

    latest_file = checkpoint_root / "latest"
    if latest_file.is_file():
        return checkpoint_root, latest_file.read_text().strip()

    raise ValueError(
        f"Could not infer checkpoint tag under {checkpoint_root}. "
        "Pass --tag or use --zero_tag_dir."
    )


def get_save_dtype(name: str):
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    return torch.bfloat16


def main():
    args = parse_args()
    checkpoint_root, tag = resolve_zero_root_and_tag(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[export] base_model={args.base_model}")
    print(f"[export] checkpoint_root={checkpoint_root}")
    print(f"[export] tag={tag}")
    print(f"[export] output_dir={output_dir}")

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float32,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    # This reconstructs fp32 weights on CPU from ZeRO shards.
    model = load_state_dict_from_zero_checkpoint(model, str(checkpoint_root), tag=tag)

    save_dtype = get_save_dtype(args.save_dtype)
    model = model.to(save_dtype)
    model.save_pretrained(
        output_dir,
        safe_serialization=args.safe_serialization,
        max_shard_size=args.max_shard_size,
    )
    tokenizer.save_pretrained(output_dir)
    print("[export] done")


if __name__ == "__main__":
    main()
