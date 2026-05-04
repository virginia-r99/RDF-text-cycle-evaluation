#!/usr/bin/env python3
# coding: utf-8

from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse
import gc
import json
import re
import time
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore")
pd.set_option("display.max_colwidth", 180)

TOKEN = "..."

# =========================
# Defaults
# =========================
DEFAULT_DATA_ROOT = Path("../../WebNLG_CO")
DEFAULT_OUTPUT_DIR = Path("./outputs_zero_shot_ie_trained")
TARGET_LANGS = ["en", "es", "ca", "gl", "eu"]
EVAL_SPLIT = "test"

MAX_NEW_TOKENS = 512
TEMPERATURE = 0.0
DO_SAMPLE = False
TOP_P = 1.0
REPETITION_PENALTY = 1.0

TRUST_REMOTE_CODE = True
SAVE_EVERY = 50
LIMIT_PER_SPLIT = None
OVERWRITE_EXISTING = False

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

# Same prompt family as the training code
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zero-shot IE evaluation for trained multilingual LoRA model")
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base_model", type=str, required=True,
                        help="Base model used in training, e.g. Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--adapter_path", type=str, required=True,
                        help="Path to final_adapter from training")
    parser.add_argument("--limit_per_split", type=int, default=LIMIT_PER_SPLIT)
    parser.add_argument("--overwrite_existing", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--save_every", type=int, default=SAVE_EVERY)
    return parser.parse_args()


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def safe_text(node: Optional[ET.Element]) -> str:
    if node is None:
        return ""
    return normalize_ws("".join(node.itertext()))


def split_triple_text(triple_text: str) -> Tuple[str, str, str]:
    txt = normalize_ws(triple_text)
    parts = [p.strip() for p in re.split(r"\s*\|\s*", txt)]
    if len(parts) >= 3:
        s = parts[0]
        p = parts[1]
        o = " | ".join(parts[2:])
        return s, p, o
    return txt, "", ""


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


def extract_lexicalisations(entry: ET.Element, lang: str) -> List[Dict[str, str]]:
    target_lex_lang = LANGUAGE_TRIPLESETS[lang]["lex_lang"]
    valid_tags = {t.lower() for t in LEXICALISATION_TAG_CANDIDATES}
    lex_rows = []

    for node in entry.iter():
        if node.tag.lower() not in valid_tags:
            continue
        node_lang = (node.attrib.get("lang", "") or "").lower()
        if node_lang != target_lex_lang:
            continue
        text = safe_text(node)
        if not text:
            continue
        lex_rows.append(
            {
                "lex": text,
                "comment": node.attrib.get("comment", ""),
                "lid": node.attrib.get("lid", ""),
                "lex_lang": node_lang,
            }
        )
    return lex_rows


def lexicalisations_to_json(lexicalisations: List[Dict[str, str]]) -> str:
    payload = []
    for i, lex in enumerate(lexicalisations, start=1):
        lid = (lex.get("lid") or f"lex_{i}").strip()
        payload.append({lid: lex.get("lex", "")})
    return json.dumps(payload, ensure_ascii=False)


def build_align_key(split: str, category: str, eid: str, triple_count: int) -> str:
    return f"{split}|||{category}|||{eid}|||{triple_count}"


def infer_triple_bucket_from_path(xml_path: Path, split: str) -> str:
    split_lower = split.lower()
    path_parts = [p.lower() for p in xml_path.parts]
    if split_lower in path_parts:
        split_idx = path_parts.index(split_lower)
        rel_parts = xml_path.parts[split_idx + 1:]
        if len(rel_parts) >= 2:
            return rel_parts[0]
    return ""


def parse_xml_file(xml_path: Path, split: str, lang: str) -> List[Dict]:
    fallback_category = xml_path.stem
    tree = ET.parse(xml_path)
    root = tree.getroot()

    rows = []
    entry_nodes = [n for n in root.iter() if n.tag.lower() == "entry"]
    triple_bucket = infer_triple_bucket_from_path(xml_path, split)

    for entry_idx, entry in enumerate(entry_nodes):
        eid = entry_attr(entry, "eid", "id") or f"entry_{entry_idx}"
        size_attr = entry_attr(entry, "size")
        category_attr = entry_attr(entry, "category") or fallback_category

        triples = extract_triples(entry, lang=lang)
        if not triples:
            continue

        triple_count = int(size_attr) if str(size_attr).isdigit() else len(triples)
        triples_struct = [
            {"subject": s, "predicate": p, "object": o, "raw": raw}
            for raw in triples
            for (s, p, o) in [split_triple_text(raw)]
        ]

        lexicalisations = extract_lexicalisations(entry, lang=lang)

        # Only one text per entry, prefer id1
        if lexicalisations:
            id1 = [x for x in lexicalisations if (x.get("lid") or "").strip().lower() == "id1"]
            lexicalisations = [id1[0] if id1 else lexicalisations[0]]
        else:
            lexicalisations = [
                {
                    "lex": "",
                    "comment": "",
                    "lid": "",
                    "lex_lang": LANGUAGE_TRIPLESETS[lang]["lex_lang"],
                }
            ]

        first_reference = lexicalisations[0].get("lex", "")
        align_key = build_align_key(split, category_attr, eid, triple_count)

        rows.append(
            {
                "align_key": align_key,
                "split": split,
                "lang": lang,
                "category": category_attr,
                "xml_file": xml_path.name,
                "xml_path": str(xml_path),
                "triple_bucket": triple_bucket,
                "eid": eid,
                "size": triple_count,
                "num_triples": triple_count,
                "entry_idx": entry_idx,
                "num_lexicalisations": len(lexicalisations),
                "reference": first_reference,
                "lexicalisations": lexicalisations_to_json(lexicalisations),
                "triples": triples,
                "triples_struct": triples_struct,
                "triple_text": "\n".join(f"- [{t}]" for t in triples),
            }
        )
    return rows


def parse_language_split(data_root: Path, split: str, lang: str) -> pd.DataFrame:
    split_dir = data_root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Missing split directory: {split_dir}")

    xml_files = sorted(split_dir.rglob("*.xml")) if split.lower() in {"train", "dev"} else sorted(split_dir.glob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"No XML files found in: {split_dir}")

    rows = []
    for xml_file in tqdm(xml_files, desc=f"Parsing {lang}/{split}"):
        rows.extend(parse_xml_file(xml_file, split=split, lang=lang))

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"Parsed 0 rows from {split_dir} for lang={lang}")
    return df


# =========================
# Same prompt as training
# =========================
def make_prompt(lang: str, text: str) -> str:
    return (
        f"Extract all RDF triples expressed in the following {LANG_NAME[lang]} text. "
        f"Preserve the original facts and do not infer or add information.\n\n"
        f"Text:\n{normalize_ws(text)}\n\n"
        f"Return triples in the format:\n"
        f"[subject | predicate | object]\n"
        f"One triple per line. If none, return {EMPTY_TOKEN[lang]}."
    )


def build_messages(lang: str, source_text: str) -> List[Dict[str, str]]:
    return [
        {
            "role": "user",
            "content": make_prompt(lang, source_text),
        }
    ]


def extract_predicted_triples(text: str, lang: str) -> List[str]:
    if not isinstance(text, str) or not text.strip():
        return []

    txt = text.strip()
    if normalize_ws(txt).upper() == EMPTY_TOKEN[lang].upper():
        return []

    matches = re.findall(r"\[(.*?)\]", txt, flags=re.DOTALL)
    triples = []
    for m in matches:
        candidate = normalize_ws(m)
        parts = [p.strip() for p in re.split(r"\s*\|\s*", candidate)]
        if len(parts) >= 3:
            s = parts[0]
            p = parts[1]
            o = " | ".join(parts[2:])
            triples.append(f"{s} | {p} | {o}")

    deduped = []
    seen = set()
    for t in triples:
        key = t.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(t)
    return deduped


def predicted_triples_to_text(triples: List[str], lang: str) -> str:
    if not triples:
        return EMPTY_TOKEN[lang]
    return "\n".join([f"[{t}]" for t in triples])


def expected_answer(triples: List[str], lang: str) -> str:
    if not triples:
        return EMPTY_TOKEN[lang]
    return "\n".join([f"[{t}]" for t in triples])


def get_model_family(model_name: str) -> str:
    name = model_name.lower()
    if "salamandra" in name:
        return "salamandra"
    if "qwen" in name:
        return "qwen"
    if "smollm" in name:
        return "smollm"
    if "aya" in name or "cohere" in name:
        return "aya"
    return "generic"


def _hf_kwargs() -> Dict[str, str]:
    return {"token": TOKEN} if TOKEN else {}


def load_model_and_tokenizer(base_model: str, adapter_path: str):
    family = get_model_family(base_model)

    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path if Path(adapter_path).exists() else base_model,
        trust_remote_code=TRUST_REMOTE_CODE,
        **_hf_kwargs()
    )

    if family == "salamandra":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=TRUST_REMOTE_CODE,
            **_hf_kwargs(),
        )
    elif family == "qwen":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=TRUST_REMOTE_CODE,
            **_hf_kwargs(),
        )
    elif family in {"aya", "smollm"}:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            trust_remote_code=TRUST_REMOTE_CODE,
            **_hf_kwargs(),
        )
        if torch.cuda.is_available():
            model = model.to("cuda")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="auto",
            trust_remote_code=TRUST_REMOTE_CODE,
            **_hf_kwargs(),
        )

    model = PeftModel.from_pretrained(model, adapter_path)

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model.eval()
    return tokenizer, model


def apply_chat_template_or_fallback(tokenizer, messages: List[Dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass

    return "\n\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages]) + "\n\nASSISTANT:"


def build_model_inputs(tokenizer, prompt_text: str, model, model_name: str) -> Dict[str, torch.Tensor]:
    family = get_model_family(model_name)

    if family == "salamandra":
        input_ids = tokenizer.encode(
            prompt_text,
            add_special_tokens=False,
            return_tensors="pt"
        )
        return {"input_ids": input_ids.to(model.device)}

    if family in {"qwen", "smollm"}:
        return tokenizer([prompt_text], return_tensors="pt").to(model.device)

    if family == "aya":
        input_ids = tokenizer(
            prompt_text,
            add_special_tokens=False,
            return_tensors="pt"
        )["input_ids"]
        return {"input_ids": input_ids.to(model.device)}

    return tokenizer(prompt_text, return_tensors="pt").to(model.device)


@torch.inference_mode()
def generate_one(model, tokenizer, prompt_text: str, model_name: str, max_new_tokens: int) -> Dict[str, str]:
    model_inputs = build_model_inputs(tokenizer, prompt_text, model, model_name)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "repetition_penalty": REPETITION_PENALTY,
    }

    if DO_SAMPLE:
        gen_kwargs.update({
            "do_sample": True,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
        })
    else:
        gen_kwargs.update({"do_sample": False})

    generated = model.generate(**model_inputs, **gen_kwargs)
    input_len = model_inputs["input_ids"].shape[1]
    output_ids = generated[0][input_len:]
    text = tokenizer.decode(output_ids, skip_special_tokens=True)
    return {"raw_generation": text}


def cleanup_model(model=None, tokenizer=None):
    if model is not None:
        try:
            model.cpu()
        except Exception:
            pass
        del model

    if tokenizer is not None:
        del tokenizer

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def output_csv_path(output_dir: Path, adapter_path: str) -> Path:
    model_name = Path(adapter_path).name
    return output_dir / f"generations__{model_name}.csv"


def compute_set_metrics(gold: List[str], pred: List[str]) -> Dict[str, float]:
    gold_set = set(gold)
    pred_set = set(pred)

    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    exact_match = float(gold_set == pred_set)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": exact_match,
    }


def prepare_dataset(data_root: Path, limit_per_split: Optional[int]) -> pd.DataFrame:
    all_dfs = []
    for lang in TARGET_LANGS:
        df_part = parse_language_split(data_root, EVAL_SPLIT, lang)
        if limit_per_split is not None:
            df_part = df_part.head(limit_per_split).copy()
        all_dfs.append(df_part)

    df_all = pd.concat(all_dfs, ignore_index=True)

    run_rows = []
    for _, row in df_all.iterrows():
        messages = build_messages(row["lang"], row["reference"])
        prompt_text = make_prompt(row["lang"], row["reference"])

        run_rows.append(
            {
                "lang": row["lang"],
                "split": row["split"],
                "category": row["category"],
                "eid": row["eid"],
                "size": row["size"],
                "num_triples": row["num_triples"],
                "triple_bucket": row["triple_bucket"],
                "xml_file": row["xml_file"],
                "xml_path": row["xml_path"],
                "align_key": row["align_key"],
                "num_lexicalisations": row["num_lexicalisations"],
                "lexicalisations": row["lexicalisations"],
                "reference": row["reference"],
                "triples": json.dumps(row["triples"], ensure_ascii=False),
                "triples_struct": json.dumps(row["triples_struct"], ensure_ascii=False),
                "prompt": prompt_text,
                "messages": json.dumps(messages, ensure_ascii=False),
            }
        )

    return pd.DataFrame(run_rows)


def run_generation(
    base_model: str,
    adapter_path: str,
    run_df: pd.DataFrame,
    output_dir: Path,
    args: argparse.Namespace
) -> pd.DataFrame:
    out_path = output_csv_path(output_dir, adapter_path)
    key_cols = ["lang", "split", "category", "eid", "align_key"]

    existing_df = None
    completed_keys = set()

    if out_path.exists() and not args.overwrite_existing:
        print(f"Found existing results at {out_path}")
        existing_df = pd.read_csv(out_path)
        if not existing_df.empty:
            required_cols = set(key_cols + ["generation_status"])
            if required_cols.issubset(existing_df.columns):
                completed_df = existing_df[existing_df["generation_status"] == "ok"].copy()
                completed_keys = set(
                    tuple(x) for x in completed_df[key_cols].drop_duplicates().itertuples(index=False, name=None)
                )

    run_df = run_df.copy()
    run_df["_run_key"] = list(run_df[key_cols].itertuples(index=False, name=None))
    missing_df = run_df[~run_df["_run_key"].isin(completed_keys)].copy()
    run_df.drop(columns=["_run_key"], inplace=True)
    missing_df.drop(columns=["_run_key"], inplace=True)

    if missing_df.empty and existing_df is not None and not args.overwrite_existing:
        print(f"All entries already present in {out_path}")
        return existing_df

    tokenizer, model = load_model_and_tokenizer(base_model, adapter_path)
    new_results = []

    prompt_text_cache: Dict[str, str] = {}
    for messages_json in missing_df["messages"].unique().tolist():
        messages = json.loads(messages_json)
        prompt_text_cache[messages_json] = apply_chat_template_or_fallback(tokenizer, messages)

    sorted_df = missing_df.copy()
    sorted_df["prompt_len_chars"] = sorted_df["prompt"].str.len()
    sorted_df = sorted_df.sort_values(
        ["lang", "split", "num_triples", "prompt_len_chars", "category", "eid"]
    )

    print(f"Running zero-shot evaluation on {len(sorted_df)} missing entries")

    try:
        for idx, (_, row) in enumerate(
            tqdm(sorted_df.iterrows(), total=len(sorted_df), desc="Generating"),
            start=1
        ):
            prompt_text = prompt_text_cache[row["messages"]]
            t0 = time.time()

            try:
                gen = generate_one(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_text=prompt_text,
                    model_name=base_model,
                    max_new_tokens=args.max_new_tokens,
                )
                raw_generation = gen["raw_generation"]
                predicted_triples = extract_predicted_triples(raw_generation, row["lang"])
                extracted = predicted_triples_to_text(predicted_triples, row["lang"])
                gold_triples = json.loads(row["triples"])
                metrics = compute_set_metrics(gold_triples, predicted_triples)
                status = "ok"
                error = ""
            except Exception as e:
                raw_generation = ""
                predicted_triples = []
                extracted = ""
                metrics = {
                    "tp": 0, "fp": 0, "fn": 0,
                    "precision": 0.0, "recall": 0.0, "f1": 0.0, "exact_match": 0.0,
                }
                status = "error"
                error = repr(e)

            new_results.append(
                {
                    **row.drop(labels=["prompt_len_chars"]).to_dict(),
                    "base_model": base_model,
                    "adapter_path": adapter_path,
                    "raw_generation": raw_generation,
                    "predicted_triples": json.dumps(predicted_triples, ensure_ascii=False),
                    "extracted_triples_text": extracted,
                    "generation_status": status,
                    "generation_error": error,
                    "latency_sec": round(time.time() - t0, 4),
                    "timestamp_utc": pd.Timestamp.utcnow().isoformat(),
                    **metrics,
                }
            )

            if idx % args.save_every == 0:
                checkpoint_df = pd.DataFrame(new_results)
                if existing_df is not None and not existing_df.empty:
                    merged_df = pd.concat([existing_df, checkpoint_df], ignore_index=True)
                else:
                    merged_df = checkpoint_df

                merged_df = (
                    merged_df.sort_values("timestamp_utc")
                    .drop_duplicates(subset=key_cols, keep="last")
                )
                merged_df.to_csv(out_path, index=False)
                print(f"Saved checkpoint: {out_path} ({idx}/{len(sorted_df)})")

        new_result_df = pd.DataFrame(new_results)
        if existing_df is not None and not existing_df.empty:
            result_df = pd.concat([existing_df, new_result_df], ignore_index=True)
        else:
            result_df = new_result_df

        result_df = (
            result_df.sort_values("timestamp_utc")
            .drop_duplicates(subset=key_cols, keep="last")
        )
        result_df.to_csv(out_path, index=False)
        return result_df

    finally:
        cleanup_model(model, tokenizer)


def write_summary(results_df: pd.DataFrame, output_dir: Path):
    ok_df = results_df[results_df["generation_status"] == "ok"].copy()

    by_lang_rows = []
    for lang, g in ok_df.groupby("lang"):
        tp = g["tp"].sum()
        fp = g["fp"].sum()
        fn = g["fn"].sum()

        micro_p = tp / (tp + fp) if (tp + fp) else 0.0
        micro_r = tp / (tp + fn) if (tp + fn) else 0.0
        micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0

        by_lang_rows.append(
            {
                "lang": lang,
                "rows": len(g),
                "micro_precision": micro_p,
                "micro_recall": micro_r,
                "micro_f1": micro_f1,
                "macro_precision": g["precision"].mean(),
                "macro_recall": g["recall"].mean(),
                "macro_f1": g["f1"].mean(),
                "exact_match": g["exact_match"].mean(),
                "avg_latency_sec": g["latency_sec"].mean(),
            }
        )

    by_lang_df = pd.DataFrame(by_lang_rows).sort_values("lang")

    tp = ok_df["tp"].sum()
    fp = ok_df["fp"].sum()
    fn = ok_df["fn"].sum()
    overall = {
        "rows": len(ok_df),
        "micro_precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "micro_recall": tp / (tp + fn) if (tp + fn) else 0.0,
        "micro_f1": (2 * (tp / (tp + fp) if (tp + fp) else 0.0) * (tp / (tp + fn) if (tp + fn) else 0.0) /
                     ((tp / (tp + fp) if (tp + fp) else 0.0) + (tp / (tp + fn) if (tp + fn) else 0.0)))
                    if ((tp + fp) and (tp + fn) and ((tp / (tp + fp)) + (tp / (tp + fn)) > 0)) else 0.0,
        "macro_precision": ok_df["precision"].mean() if len(ok_df) else 0.0,
        "macro_recall": ok_df["recall"].mean() if len(ok_df) else 0.0,
        "macro_f1": ok_df["f1"].mean() if len(ok_df) else 0.0,
        "exact_match": ok_df["exact_match"].mean() if len(ok_df) else 0.0,
    }

    by_lang_path = output_dir / "summary_by_lang.csv"
    overall_path = output_dir / "summary_overall.json"
    xlsx_path = output_dir / "zero_shot_eval_summary.xlsx"

    by_lang_df.to_csv(by_lang_path, index=False)
    with open(overall_path, "w", encoding="utf-8") as f:
        json.dump(overall, f, ensure_ascii=False, indent=2)

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        results_df.to_excel(writer, sheet_name="entry_level", index=False)
        by_lang_df.to_excel(writer, sheet_name="summary_by_lang", index=False)
        pd.DataFrame([overall]).to_excel(writer, sheet_name="overall", index=False)

    print("\nSummary by language:")
    print(by_lang_df.to_string(index=False))
    print("\nOverall:")
    print(json.dumps(overall, indent=2, ensure_ascii=False))


def main() -> None:
    global LIMIT_PER_SPLIT, OVERWRITE_EXISTING
    args = parse_args()
    LIMIT_PER_SPLIT = args.limit_per_split
    OVERWRITE_EXISTING = args.overwrite_existing

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Visible GPU:", torch.cuda.get_device_name(0))

    run_df = prepare_dataset(args.data_root, args.limit_per_split)

    print(f"Prepared rows: {len(run_df)}")
    print("\nSummary by language:")
    summary = (
        run_df.groupby(["lang", "split"])
        .agg(
            rows=("align_key", "size"),
            unique_instances=("align_key", "nunique"),
            categories=("category", "nunique")
        )
        .reset_index()
    )
    print(summary.to_string(index=False))

    results_df = run_generation(
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        run_df=run_df,
        output_dir=args.output_dir,
        args=args,
    )

    combined_path = args.output_dir / "generations_zero_shot_trained_model.csv"
    results_df.to_csv(combined_path, index=False)
    print(f"\nCombined results: {combined_path}")

    write_summary(results_df, args.output_dir)


if __name__ == "__main__":
    main()