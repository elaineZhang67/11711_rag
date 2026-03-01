#!/usr/bin/env python3
from __future__ import annotations

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() :
    p = argparse.ArgumentParser(description="Merge a LoRA adapter into base model weights for inference.")
    p.add_argument("--base-model", required=True)
    p.add_argument("--adapter-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    return p.parse_args()


def main() :
    args = parse_args()
    if args.dtype == "bf16":
        dtype = torch.bfloat16
    elif args.dtype == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model = model.merge_and_unload()
    model.save_pretrained(args.out_dir, safe_serialization=True)

    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tok.save_pretrained(args.out_dir)
    print(f"Merged model saved to {args.out_dir}")


if __name__ == "__main__":
    main()
