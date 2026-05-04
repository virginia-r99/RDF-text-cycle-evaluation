#!/usr/bin/env python3

# Example:
# python train_lora_webnlg_co_ie_multilingual.py \
#   --data_root ../../WebNLG_CO \
#   --output_root ./runs_ie_qwen \
#   --model Qwen/Qwen3-4B-Instruct-2507 \
#   --gradient_checkpointing --trust_remote_code

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    TrainerCallback,
)
from trl import SFTConfig, SFTTrainer

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


LLM_MODELS = [
    "Qwen/Qwen3-4B-Instruct-2507",
    "CohereLabs/tiny-aya-global",
    "HuggingFaceTB/SmolLM3-3B",
    "BSC-LT/salamandra-2b-instruct",
]

TARGET_LANGS = ["en", "es", "ca", "gl", "eu"]

LANGUAGE_TRIPLESETS = {
    "en": {
        "tripleset_tags": ["modifiedtripleset"],
        "triple_tags": ["mtriple"],
        "lex_lang": "en",
    },
    "es": {
        "tripleset_tags": ["spanishtripleset"],
        "triple_tags": ["striple"],
        "lex_lang": "es",
    },
    "ca": {
        "tripleset_tags": ["catalantripleset"],
        "triple_tags": ["ctriple"],
        "lex_lang": "ca",
    },
    "gl": {
        "tripleset_tags": ["galiciantripleset"],
        "triple_tags": ["gtriple"],
        "lex_lang": "gl",
    },
    "eu": {
        "tripleset_tags": ["basquetripleset"],
        "triple_tags": ["btriple"],
        "lex_lang": "eu",
    },
}

LEXICALISATION_TAG_CANDIDATES = ["lex", "text"]

LANG_NAME = {
    "en": "English",
    "es": "Spanish",
    "ca": "Catalan",
    "gl": "Galician",
    "eu": "Basque",
}

EMPTY_TOKEN = {
    "en": "[NONE]",
    "es": "[NINGUNA]",
    "ca": "[CAP]",
    "gl": "[NINGUNHA]",
    "eu": "[BATERE EZ]",
}


def make_prompt(lang: str, text: str) -> str:
    return (
        f"Extract all RDF triples expressed in the following {LANG_NAME[lang]} text. "
        f"Preserve the original facts and do not infer or add information.\n\n"
        f"Text:\n{normalize_space(text)}\n\n"
        f"Return triples in the format:\n"
        f"[subject | predicate | object]\n"
        f"One triple per line. If none, return {EMPTY_TOKEN[lang]}."
    )


def expected_answer(triples: Sequence[str], lang: str) -> str:
    if not triples:
        return EMPTY_TOKEN[lang]
    return "\n".join(f"[{t}]" for t in triples)


def build_messages(lang: str, source_text: str) -> List[Dict[str, str]]:
    return [
        {
            "role": "user",
            "content": make_prompt(lang, source_text),
        }
    ]


@dataclass
class Example:
    split: str
    bucket: str
    xml_file: str
    category: str
    eid: str
    lang: str
    triples: List[str]
    source_text: str
    target: str
    prompt: List[Dict[str, str]]
    align_key: str
    num_lexicalisations: int


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def safe_text(node: Optional[ET.Element]) -> str:
    if node is None:
        return ""
    return normalize_space("".join(node.itertext()))


def entry_attr(entry: ET.Element, *names: str) -> str:
    for name in names:
        if name in entry.attrib:
            return entry.attrib[name]
    lower_map = {k.lower(): v for k, v in entry.attrib.items()}
    for name in names:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return ""


def extract_triples(entry: ET.Element, lang: str) -> List[str]:
    spec = LANGUAGE_TRIPLESETS[lang]
    triple_tags = {t.lower() for t in spec["triple_tags"]}

    for tripleset_tag in spec["tripleset_tags"]:
        triples: List[str] = []
        for node in entry:
            if node.tag.lower() == tripleset_tag.lower():
                for child in node:
                    if child.tag.lower() in triple_tags:
                        txt = safe_text(child)
                        if txt:
                            triples.append(txt)
                if triples:
                    return triples
    return []


def extract_lexicalisations(entry: ET.Element, lang: str) -> List[Tuple[str, str]]:
    target_lex_lang = LANGUAGE_TRIPLESETS[lang]["lex_lang"]
    valid_tags = {t.lower() for t in LEXICALISATION_TAG_CANDIDATES}
    rows: List[Tuple[str, str]] = []

    for node in entry.iter():
        if node.tag.lower() not in valid_tags:
            continue
        node_lang = (node.attrib.get("lang", "") or "").lower()
        if node_lang != target_lex_lang:
            continue
        text = safe_text(node)
        if not text:
            continue
        lid = node.attrib.get("lid", "") or ""
        rows.append((lid, text))
    return rows


def infer_bucket_from_path(xml_path: Path, split: str) -> str:
    split_lower = split.lower()
    parts = [p.lower() for p in xml_path.parts]
    if split_lower in parts:
        idx = parts.index(split_lower)
        rel_parts = xml_path.parts[idx + 1:]
        if len(rel_parts) >= 2:
            return rel_parts[0]
    return ""


def parse_split(root_dir: Path, split: str, lang: str) -> List[Example]:
    split_dir = root_dir / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Missing split directory: {split_dir}")

    if split.lower() in {"train", "dev"}:
        xml_files = sorted(split_dir.rglob("*.xml"))
    else:
        xml_files = sorted(split_dir.glob("*.xml"))

    examples: List[Example] = []
    for xml_path in xml_files:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        bucket = infer_bucket_from_path(xml_path, split)

        entry_nodes = [n for n in root.iter() if n.tag.lower() == "entry"]
        for entry_idx, entry in enumerate(entry_nodes):
            category = entry_attr(entry, "category") or xml_path.stem
            eid = entry_attr(entry, "eid", "id") or f"entry_{entry_idx}"
            triples = extract_triples(entry, lang)
            if not triples:
                continue

            lex_rows = extract_lexicalisations(entry, lang)
            if not lex_rows:
                continue

            first_text = lex_rows[0][1]
            prompt = build_messages(lang, first_text)
            target = expected_answer(triples, lang)
            align_key = f"{split}|{bucket}|{xml_path.name}|{eid}|{lang}"

            examples.append(
                Example(
                    split=split,
                    bucket=bucket,
                    xml_file=xml_path.name,
                    category=category,
                    eid=eid,
                    lang=lang,
                    triples=triples,
                    source_text=first_text,
                    target=target,
                    prompt=prompt,
                    align_key=align_key,
                    num_lexicalisations=len(lex_rows),
                )
            )
    return examples


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_torch_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def find_lora_target_modules(model: nn.Module) -> List[str]:
    names = set()
    linear_classes = (nn.Linear,)
    try:
        import bitsandbytes as bnb
        linear_classes = linear_classes + (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt)
    except Exception:
        pass

    for name, module in model.named_modules():
        if isinstance(module, linear_classes):
            leaf = name.split(".")[-1]
            if leaf != "lm_head":
                names.add(leaf)

    preferred = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "up_proj", "down_proj", "gate_proj",
        "Wqkv", "out_proj", "fc1", "fc2",
    ]
    picked = [n for n in preferred if n in names]
    return picked if picked else sorted(names)


class SaveAdapterConfigCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        return control


def sanitize_run_name(model_name: str, suffix: str) -> str:
    model_part = model_name.replace("/", "__")
    suffix = re.sub(r"[^A-Za-z0-9._-]+", "_", suffix)
    return f"{model_part}__{suffix}"


def build_hf_dataset(examples: List[Example]) -> Dataset:
    rows = []
    for ex in examples:
        rows.append(
            {
                "prompt": ex.prompt,
                "completion": [{"role": "assistant", "content": ex.target}],
                "align_key": ex.align_key,
                "lang": ex.lang,
                "bucket": ex.bucket,
                "category": ex.category,
                "eid": ex.eid,
                "xml_file": ex.xml_file,
                "num_lexicalisations": ex.num_lexicalisations,
            }
        )
    return Dataset.from_list(rows)


def estimate_warmup_steps(
    n_examples: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    num_train_epochs: float,
    warmup_ratio: float,
) -> int:
    effective_batch = max(1, per_device_train_batch_size * gradient_accumulation_steps)
    steps_per_epoch = max(1, math.ceil(n_examples / effective_batch))
    total_steps = max(1, int(math.ceil(steps_per_epoch * num_train_epochs)))
    if warmup_ratio <= 0:
        return 0
    return max(1, int(round(total_steps * warmup_ratio)))


def find_latest_checkpoint(output_dir: Path) -> Optional[Path]:
    checkpoints = []
    for p in output_dir.glob("checkpoint-*"):
        if not p.is_dir():
            continue
        m = re.match(r"checkpoint-(\d+)$", p.name)
        if m:
            checkpoints.append((int(m.group(1)), p))
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints[-1][1]


def is_experiment_already_finished(output_dir: Path) -> bool:
    final_adapter_dir = output_dir / "final_adapter"
    final_metrics_path = output_dir / "final_metrics.json"
    run_config_path = output_dir / "run_config.json"
    return final_adapter_dir.exists() and final_metrics_path.exists() and run_config_path.exists()


def load_existing_metrics(output_dir: Path) -> Dict[str, float]:
    metrics_path = output_dir / "final_metrics.json"
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def cleanup_training_objects(trainer=None, model=None, tokenizer=None):
    try:
        if trainer is not None:
            if hasattr(trainer, "model") and trainer.model is not None:
                try:
                    trainer.model.cpu()
                except Exception:
                    pass
            if hasattr(trainer, "accelerator"):
                try:
                    trainer.accelerator.free_memory()
                except Exception:
                    pass
    except Exception:
        pass

    try:
        if model is not None:
            model.cpu()
    except Exception:
        pass

    del trainer, model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def run_multilingual_experiment(
    model_name: str,
    parsed_data: Dict[str, Dict[str, List[Example]]],
    output_root: Path,
    args: argparse.Namespace,
) -> Dict[str, object]:
    run_name = sanitize_run_name(model_name, "ie_multilingual_all_langs")
    output_dir = output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    train_examples: List[Example] = []
    dev_examples: List[Example] = []
    per_lang_counts = {}
    for lang in TARGET_LANGS:
        train_examples.extend(parsed_data[lang]["train"])
        dev_examples.extend(parsed_data[lang]["dev"])
        per_lang_counts[lang] = {
            "train": len(parsed_data[lang]["train"]),
            "dev": len(parsed_data[lang]["dev"]),
        }

    metadata = {
        "experiment_name": "IE-multilingual-all-languages",
        "model_name": model_name,
        "task": "ie",
        "training_mode": "multilingual",
        "train_splits": ["train"],
        "eval_splits": ["dev"],
        "target_langs": TARGET_LANGS,
        "epochs": args.num_train_epochs,
        "n_train_total": len(train_examples),
        "n_dev_total": len(dev_examples),
        "per_language_counts": per_lang_counts,
        "entry_level_training": True,
        "one_example_per_entry": True,
        "fewshot_in_training": False,
        "zero_shot_instruction_training": True,
    }

    if len(train_examples) == 0:
        raise ValueError("No training examples found.")
    if len(dev_examples) == 0:
        raise ValueError("No dev examples found.")

    final_adapter_dir = output_dir / "final_adapter"
    if is_experiment_already_finished(output_dir):
        print(f"Skipping multilingual IE training because final adapter already exists: {final_adapter_dir}")
        existing_metrics = load_existing_metrics(output_dir)
        return {
            "status": "skipped_already_trained",
            "experiment_name": metadata["experiment_name"],
            "output_dir": str(output_dir),
            "final_adapter_dir": str(final_adapter_dir),
            "metrics": existing_metrics,
            "metadata": metadata,
        }

    latest_checkpoint = find_latest_checkpoint(output_dir)
    if latest_checkpoint is not None:
        print(f"Resuming multilingual IE training from checkpoint: {latest_checkpoint}")
    else:
        print("Starting fresh multilingual IE training")

    dtype = get_torch_dtype()
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        torch_dtype=dtype,
        device_map="auto",
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    target_modules = find_lora_target_modules(model)
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    train_dataset = build_hf_dataset(train_examples)
    eval_dataset = build_hf_dataset(dev_examples)

    warmup_steps = estimate_warmup_steps(
        n_examples=len(train_examples),
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
    )

    sft_args = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        warmup_steps=warmup_steps,
        logging_steps=args.logging_steps,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps if args.eval_strategy == "steps" else None,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps if args.save_strategy == "steps" else None,
        save_total_limit=args.save_total_limit,
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        report_to="none",
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=args.dataloader_num_workers,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        seed=args.seed,
        max_length=args.max_length,
        completion_only_loss=True,
        assistant_only_loss=False,
        dataset_num_proc=1,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2), SaveAdapterConfigCallback()],
    )

    with (output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                **metadata,
                "target_modules": target_modules,
                "training_args": {
                    "max_length": args.max_length,
                    "per_device_train_batch_size": args.per_device_train_batch_size,
                    "per_device_eval_batch_size": args.per_device_eval_batch_size,
                    "gradient_accumulation_steps": args.gradient_accumulation_steps,
                    "num_train_epochs": args.num_train_epochs,
                    "learning_rate": args.learning_rate,
                    "weight_decay": args.weight_decay,
                    "warmup_ratio_requested": args.warmup_ratio,
                    "warmup_steps_effective": warmup_steps,
                    "lora_r": args.lora_r,
                    "lora_alpha": args.lora_alpha,
                    "lora_dropout": args.lora_dropout,
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    train_result = trainer.train(
        resume_from_checkpoint=str(latest_checkpoint) if latest_checkpoint is not None else None
    )

    trainer.save_model(str(final_adapter_dir))
    tokenizer.save_pretrained(str(final_adapter_dir))

    metrics = trainer.evaluate()
    if train_result is not None and hasattr(train_result, "metrics"):
        for k, v in train_result.metrics.items():
            metrics[f"train_{k}" if not k.startswith("train_") else k] = v

    with (output_dir / "final_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    result = {
        "status": "ok_resumed" if latest_checkpoint is not None else "ok_trained",
        "experiment_name": metadata["experiment_name"],
        "output_dir": str(output_dir),
        "final_adapter_dir": str(final_adapter_dir),
        "metrics": metrics,
        "metadata": metadata,
    }

    del trainer, model, tokenizer, train_dataset, eval_dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="Path to WebNLG_CO root")
    parser.add_argument("--output_root", type=str, required=True, help="Directory where the run will be saved")
    parser.add_argument("--model", type=str, required=True, choices=LLM_MODELS)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--num_train_epochs", type=float, default=2.0)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--eval_strategy", type=str, default="epoch", choices=["epoch", "steps"])
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_strategy", type=str, default="epoch", choices=["epoch", "steps"])
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    args = parser.parse_args()

    seed_everything(args.seed)

    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    print("Parsing datasets once for multilingual IE entry-level training...")
    parsed_data = {
        lang: {
            "train": parse_split(data_root, "train", lang),
            "dev": parse_split(data_root, "dev", lang),
        }
        for lang in TARGET_LANGS
    }

    summary_counts = []
    for lang in TARGET_LANGS:
        summary_counts.append(
            {
                "lang": lang,
                "train_entries": len(parsed_data[lang]["train"]),
                "dev_entries": len(parsed_data[lang]["dev"]),
            }
        )
    print(json.dumps(summary_counts, indent=2, ensure_ascii=False))

    result = run_multilingual_experiment(
        model_name=args.model,
        parsed_data=parsed_data,
        output_root=output_root,
        args=args,
    )

    summary = {
        "experiment_name": result["experiment_name"],
        "status": result["status"],
        "output_dir": result["output_dir"],
        "final_adapter_dir": result["final_adapter_dir"],
        "eval_loss": result["metrics"].get("eval_loss") if result.get("metrics") else None,
        "n_train_total": result["metadata"].get("n_train_total"),
        "n_dev_total": result["metadata"].get("n_dev_total"),
        "target_langs": result["metadata"].get("target_langs"),
    }

    with (output_root / f"summary__{args.model.replace('/', '__')}__ie_multilingual.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nTraining finished.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
