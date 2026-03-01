#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _read_questions(path) :
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        t = line.strip()
        if t:
            rows.append(t)
    return rows


def _read_refs(path) :
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    return {str(k): str(v).strip() for k, v in obj.items()}


def _build_examples(questions, refs) :
    examples = []
    for i, q in enumerate(questions, start=1):
        a = refs.get(str(i), "").strip()
        if not a:
            continue
        examples.append({"question": q, "answer": a})
    return examples


def _prompt_for_question(q) :
    return (
        "You are answering factual questions about Pittsburgh and Carnegie Mellon University.\n"
        "Answer directly and concisely.\n\n"
        f"Question: {q}\n"
        "Answer:"
    )


def _tokenize_example(tokenizer, ex, max_length) :
    prompt = _prompt_for_question(ex["question"])
    target = ex["answer"]

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    all_ids = tokenizer(
        prompt + " " + target,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
    )["input_ids"]

    labels = list(all_ids)
    prompt_len = min(len(prompt_ids), len(labels))
    for i in range(prompt_len):
        labels[i] = -100

    return {
        "input_ids": all_ids,
        "attention_mask": [1] * len(all_ids),
        "labels": labels,
    }


class ListDataset:
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


class CausalDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        pad_id = self.tokenizer.pad_token_id
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids = []
        attention_mask = []
        labels = []
        for f in features:
            l = len(f["input_ids"])
            pad_n = max_len - l
            input_ids.append(f["input_ids"] + [pad_id] * pad_n)
            attention_mask.append(f["attention_mask"] + [0] * pad_n)
            labels.append(f["labels"] + [-100] * pad_n)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def parse_args() :
    p = argparse.ArgumentParser(description="Fine-tune a causal LM with LoRA on QA pairs.")
    p.add_argument("--base-model", default="Qwen/Qwen2.5-14B-Instruct")
    p.add_argument("--questions", default="data/train/questions.txt")
    p.add_argument("--references", default="data/train/reference_answers.json")
    p.add_argument("--output-dir", default="data/models/qwen25_14b_lora")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--eval-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument(
        "--target-modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    p.add_argument("--no-4bit", action="store_true", help="Disable 4-bit loading (requires much more GPU memory).")
    return p.parse_args()


def main() :
    args = parse_args()
    random.seed(args.seed)

    questions = _read_questions(args.questions)
    refs = _read_refs(args.references)
    examples = _build_examples(questions, refs)
    if not examples:
        raise RuntimeError("No training examples found.")

    random.shuffle(examples)
    n_eval = int(len(examples) * max(0.0, min(0.5, args.eval_ratio)))
    eval_examples = examples[:n_eval]
    train_examples = examples[n_eval:]
    if not train_examples:
        raise RuntimeError("No train examples left after split.")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if not args.no_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model_kwargs = {"trust_remote_code": True}
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model_kwargs["device_map"] = "auto" if torch.cuda.is_available() else None

    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    if quantization_config is not None:
        model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=args.target_modules,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    train_items = [_tokenize_example(tokenizer, ex, args.max_length) for ex in train_examples]
    eval_items = [_tokenize_example(tokenizer, ex, args.max_length) for ex in eval_examples] if eval_examples else []

    train_ds = ListDataset(train_items)
    eval_ds = ListDataset(eval_items) if eval_items else None

    fp16 = torch.cuda.is_available() and not torch.cuda.is_bf16_supported()
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    targs = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=args.save_steps if eval_ds is not None else None,
        save_strategy="steps",
        load_best_model_at_end=False,
        report_to=[],
        fp16=fp16,
        bf16=bf16,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        max_grad_norm=0.3,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=CausalDataCollator(tokenizer),
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    print(f"Saved LoRA adapter to {out_dir}")
    print("Next: merge adapter for inference with scripts/merge_lora.py")


if __name__ == "__main__":
    main()
