#!/usr/bin/env python3
# coding: utf-8

from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import gc
import json
import random
import re
import time
import warnings
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore")
pd.set_option("display.max_colwidth", 180)

# Optional Hugging Face token. You can also leave it empty and rely on `huggingface-cli login`.
TOKEN = "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"

# =========================
# Defaults
# =========================
DEFAULT_DATA_ROOT = Path("/home/vramon/notebooks/Rdf_text_EKAW/WebNLG_CO")
PARSE_SPLITS = ["train"]  # Only need train split for the few-shot anchors
TARGET_LANGS = ["en", "es", "ca", "gl", "eu"]

LLM_MODELS = [
    "Qwen/Qwen3-4B-Instruct-2507",
    "CohereLabs/tiny-aya-global"
]

FEWSHOT_TRIPLE_SIZES = [1, 3, 7]
FEWSHOT_RANDOM_SEED = 13

MAX_NEW_TOKENS = 512
TEMPERATURE = 0.0
DO_SAMPLE = False
TOP_P = 1.0
REPETITION_PENALTY = 1.0

USE_4BIT = False
TRUST_REMOTE_CODE = True
TORCH_DTYPE = "auto"
DEVICE_MAP = "auto"

SAVE_EVERY = 50
LIMIT_PER_SPLIT = None
OVERWRITE_EXISTING = False

PROMPTS = {
    "en": {
        "system": "You are an assistant that verbalizes RDF triples faithfully and naturally.",
        "instruction": """In English, structured data is commonly represented as triples, with the format [subject, predicate, object]. Based on these triples, generate a single-paragraph text composed of complete, grammatically correct, and natural sentences.

INSTRUCTIONS:
- Generate the text solely from the input triples.
- Return the final verbalization with this format: The final verbalization is: [verbalization output]
- Insert the verbalization of the input triples within square brackets [], without adding anything else.""",
        "answer_prefix": "The final verbalization is:",
        "example_label": "Example",
        "input_label": "Input triples",
        "output_label": "Output",
        "target_label": "Now verbalize the following input triples.",
    },
    "es": {
        "system": "Eres un asistente que verbaliza tripletas RDF de forma fiel y natural.",
        "instruction": """En español, los datos estructurados se representan comúnmente como tríos, con el formato [sujeto, predicado, objeto]. Basándose en estos tríos, genere un texto de un solo párrafo compuesto por oraciones completas, gramaticalmente correctas y naturales.

INSTRUCCIONES:
- Genere el texto únicamente a partir de las tripletas de entrada.
- Devuelva la verbalización final con este formato: La verbalización final es: [salida de verbalización]
- Entre corchetes [] inserte la verbalización de las tripletas de entrada, sin añadir nada más.""",
        "answer_prefix": "La verbalización final es:",
        "example_label": "Ejemplo",
        "input_label": "Tripletas de entrada",
        "output_label": "Salida",
        "target_label": "Ahora verbalice las siguientes tripletas de entrada.",
    },
    "ca": {
        "system": "Ets un assistent que verbalitza tripletes RDF de manera fidel i natural.",
        "instruction": """En català, les dades estructurades es representen habitualment com a tríos, amb el format [subjecte, predicat, objecte]. Basant-se en aquests tríos, generi un text d’un sol paràgraf compost per oracions completes, gramaticalment correctes i naturals.

INSTRUCCIONS:
- Generi el text únicament a partir de les tripletes d’entrada.
- Retorni la verbalització final amb aquest format: La verbalització final és: [sortida de verbalització]
- Entre claudàtors [] insereixi la verbalització de les tripletes d’entrada, sense afegir res més.""",
        "answer_prefix": "La verbalització final és:",
        "example_label": "Exemple",
        "input_label": "Tripletes d'entrada",
        "output_label": "Sortida",
        "target_label": "Ara verbalitzi les següents tripletes d’entrada.",
    },
    "gl": {
        "system": "Es un asistente que verbaliza tripletas RDF de maneira fiel e natural.",
        "instruction": """En galego, os datos estruturados represéntanse habitualmente como tripletas, co formato [suxeito, predicado, obxecto]. A partir destas tripletas, xere un texto dun único parágrafo composto por oracións completas, gramaticalmente correctas e naturais.

INSTRUCIÓNS:
- Xere o texto unicamente a partir das tripletas de entrada.
- Devolva a verbalización final con este formato: A verbalización final é: [saída da verbalización]
- Entre corchetes [] insira a verbalización das tripletas de entrada, sen engadir nada máis.""",
        "answer_prefix": "A verbalización final é:",
        "example_label": "Exemplo",
        "input_label": "Tripletas de entrada",
        "output_label": "Saída",
        "target_label": "Agora verbalice as seguintes tripletas de entrada.",
    },
    "eu": {
        "system": "RDF hirukoteak modu fidel eta naturalean hitzez ematen dituen laguntzailea zara.",
        "instruction": """Euskaraz, datu egituratuak maiz hirukote gisa adierazten dira, [subjektua, predikatua, objektua] formatuan. Hirukote horietan oinarrituta, sortu paragrafo bakarreko testu bat, esaldi oso, zuzen eta naturalekin.

JARRAIBIDEAK:
- Sortu testua sarrerako hirukoteetatik soilik abiatuta.
- Itzuli azken hitzezko adierazpena formatu honekin: Azken verbalizazioa hau da: [verbalizazioaren irteera]
- Kortxeteen [] artean sartu sarrerako hirukoteen verbalizazioa, besterik gehitu gabe.""",
        "answer_prefix": "Azken verbalizazioa hau da:",
        "example_label": "Adibidea",
        "input_label": "Sarrerako hirukoteak",
        "output_label": "Irteera",
        "target_label": "Orain verbalizatu honako sarrerako hirukote hauek.",
    },
}

LANGUAGE_TRIPLESETS = {
    "en": {"tripleset_tags": ["modifiedtripleset"], "triple_tags": ["mtriple"], "lex_lang": "en"},
    "es": {"tripleset_tags": ["spanishtripleset"], "triple_tags": ["striple"], "lex_lang": "es"},
    "ca": {"tripleset_tags": ["catalantripleset"], "triple_tags": ["ctriple"], "lex_lang": "ca"},
    "gl": {"tripleset_tags": ["galiciantripleset"], "triple_tags": ["gtriple"], "lex_lang": "gl"},
    "eu": {"tripleset_tags": ["basquetripleset"], "triple_tags": ["btriple"], "lex_lang": "eu"},
}

LEXICALISATION_TAG_CANDIDATES = ["lex", "text"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cyclic Few-shot WebNLG verbalisation benchmark")
    parser.add_argument("--input_csv", type=Path, required=True, help="Path to the IE results CSV")
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output_dir", type=Path, default=Path("./outputs_cycle_verbalisation"))
    parser.add_argument("--limit_per_split", type=int, default=LIMIT_PER_SPLIT)
    parser.add_argument("--overwrite_existing", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--save_every", type=int, default=SAVE_EVERY)
    parser.add_argument("--seed", type=int, default=FEWSHOT_RANDOM_SEED)
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
        rel_parts = xml_path.parts[split_idx + 1 :]
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
        if not lexicalisations:
            lexicalisations = [
                {
                    "lex": "",
                    "comment": "",
                    "lid": "",
                    "lex_lang": LANGUAGE_TRIPLESETS[lang]["lex_lang"],
                }
            ]

        align_key = build_align_key(split, category_attr, eid, triple_count)
        first_reference = lexicalisations[0].get("lex", "") if lexicalisations else ""

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
    for xml_file in tqdm(xml_files, desc=f"Parsing {lang}/{split} for Anchors"):
        rows.extend(parse_xml_file(xml_file, split=split, lang=lang))

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"Parsed 0 rows from {split_dir} for lang={lang}")
    return df


def get_instance_row(df: pd.DataFrame, align_key: str, lang: str) -> pd.Series:
    rows = df[(df["align_key"] == align_key) & (df["lang"] == lang)].copy()
    if rows.empty:
        raise KeyError(f"Missing align_key={align_key} for lang={lang}")
    return rows.iloc[0]


def render_triples(triples: List[str]) -> str:
    return "\n".join([f"[{t}]" for t in triples])


def expected_answer(lang: str, verbalization: str) -> str:
    return f"{PROMPTS[lang]['answer_prefix']} [{normalize_ws(verbalization)}]"


def build_user_prompt(lang: str, fewshot_rows: List[pd.Series], target_triples: List[str]) -> str:
    cfg = PROMPTS[lang]
    shots = []
    for i, row in enumerate(fewshot_rows, start=1):
        shots.append(
            f"{cfg['example_label']} {i}:\n"
            f"{cfg['input_label']}:\n{render_triples(row['triples'])}\n\n"
            f"{cfg['output_label']}:\n{expected_answer(lang, row['reference'])}"
        )

    target_block = (
        f"{cfg['target_label']}\n"
        f"{cfg['input_label']}:\n{render_triples(target_triples)}"
    )
    return "\n\n".join([cfg["instruction"]] + shots + [target_block])


def build_messages(lang: str, user_prompt: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": PROMPTS[lang]["system"]},
        {"role": "user", "content": user_prompt},
    ]


def extract_bracketed_verbalization(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    matches = re.findall(r"\[(.*?)\]", text, flags=re.DOTALL)
    if matches:
        return normalize_ws(matches[-1])
    return normalize_ws(text)


def extract_prefixed_answer(text: str, lang: str) -> str:
    prefixes = [
        PROMPTS[lang]["answer_prefix"],
        "The final verbalization is:",
        "La verbalización final es:",
        "La verbalització final és:",
        "A verbalización final é:",
        "Azken verbalizazioa hau da:",
    ]
    txt = normalize_ws(text)
    for prefix in prefixes:
        if txt.lower().startswith(prefix.lower()):
            txt = txt[len(prefix):].strip()
            break
    return extract_bracketed_verbalization(txt)


def parse_bracketed_triples(text: str) -> List[str]:
    """Helper to convert the CSV IE output back into a list of strings for the prompt."""
    if pd.isna(text) or not isinstance(text, str) or not text.strip():
        return []
    empty_markers = {"[NONE]", "[NINGUNA]", "[CAP]", "[NINGUNHA]", "[BATERE EZ]"}
    if normalize_ws(text).upper() in {x.upper() for x in empty_markers}:
        return []
    matches = re.findall(r"\[(.*?)\]", text, flags=re.DOTALL)
    return [normalize_ws(m) for m in matches]


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


def resolve_torch_dtype(dtype_name: str):
    if dtype_name == "auto":
        return "auto"
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported TORCH_DTYPE: {dtype_name}")


def model_slug(model_name: str) -> str:
    return model_name.replace("/", "__")


def _hf_kwargs() -> Dict[str, str]:
    return {"token": TOKEN} if TOKEN else {}


def load_model_and_tokenizer(model_name: str):
    family = get_model_family(model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=TRUST_REMOTE_CODE,
        **_hf_kwargs()
    )

    if family == "salamandra":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=TRUST_REMOTE_CODE,
            **_hf_kwargs(),
        )

    elif family == "qwen":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=TRUST_REMOTE_CODE,
            **_hf_kwargs(),
        )

    elif family == "aya":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=TRUST_REMOTE_CODE,
            **_hf_kwargs(),
        )
        if torch.cuda.is_available():
            model = model.to("cuda")

    elif family == "smollm":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=TRUST_REMOTE_CODE,
            **_hf_kwargs(),
        )
        if torch.cuda.is_available():
            model = model.to("cuda")

    else:
        kwargs = {"trust_remote_code": TRUST_REMOTE_CODE}
        dtype = resolve_torch_dtype(TORCH_DTYPE)
        kwargs["torch_dtype"] = dtype
        if DEVICE_MAP is not None:
            kwargs["device_map"] = DEVICE_MAP
        if USE_4BIT:
            kwargs["load_in_4bit"] = True
        kwargs.update(_hf_kwargs())
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model.eval()
    return tokenizer, model


def apply_chat_template_or_fallback(tokenizer, messages: List[Dict[str, str]], model_name: str) -> str:
    family = get_model_family(model_name)
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            if family == "salamandra":
                date_string = datetime.today().strftime("%Y-%m-%d")
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    date_string=date_string,
                )
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
    family = get_model_family(model_name)
    model_inputs = build_model_inputs(tokenizer, prompt_text, model, model_name)

    if family == "aya":
        generated = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.1,
            top_p=0.95,
            use_cache=True,
        )
    else:
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
    import gc
    import torch

    if model is not None:
        try:
            model.cpu()          # move weights off GPU first
        except Exception:
            pass
        del model

    if tokenizer is not None:
        del tokenizer

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def output_csv_path(output_dir: Path, model_name: str) -> Path:
    return output_dir / f"cycle_generations__{model_slug(model_name)}__Qwen__Qwen3-4B-Instruct-2507.csv"


def prepare_dataset(data_root: Path, input_csv: Path, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, List[pd.Series]], pd.DataFrame]:
    # 1. Parse XML to extract Few-Shot Anchors from the 'train' split
    all_dfs = []
    for lang in TARGET_LANGS:
        for split in PARSE_SPLITS:
            df_part = parse_language_split(data_root, split, lang)
            all_dfs.append(df_part)
    df_all = pd.concat(all_dfs, ignore_index=True)

    instance_view = df_all[[
        "lang", "align_key", "split", "category", "eid", "size", "num_triples", "triples",
        "triple_text", "reference", "lexicalisations", "num_lexicalisations", "triple_bucket", "xml_file", "xml_path"
    ]].copy()

    langs_per_instance = instance_view.groupby("align_key")["lang"].nunique().reset_index(name="n_langs")
    valid_align_keys = set(langs_per_instance.loc[langs_per_instance["n_langs"] == len(TARGET_LANGS), "align_key"])

    train_en = instance_view[
        (instance_view["lang"] == "en")
        & (instance_view["split"] == "train")
        & (instance_view["align_key"].isin(valid_align_keys))
    ].copy()

    selected_rows = []
    used_categories = set()
    for target_size in FEWSHOT_TRIPLE_SIZES:
        candidates = train_en[train_en["size"] == target_size].copy()
        candidates = candidates[~candidates["category"].isin(used_categories)].copy()
        if candidates.empty:
            raise ValueError(
                f"Could not find a shared train example with size={target_size} from a new category across all languages."
            )
        candidates = candidates.sample(frac=1.0, random_state=seed + target_size)
        chosen = candidates.iloc[0]
        selected_rows.append(chosen)
        used_categories.add(chosen["category"])

    fewshot_anchor_df = pd.DataFrame(selected_rows).reset_index(drop=True)
    fewshot_examples = {lang: [] for lang in TARGET_LANGS}
    for _, anchor in fewshot_anchor_df.iterrows():
        for lang in TARGET_LANGS:
            fewshot_examples[lang].append(get_instance_row(df_all, anchor["align_key"], lang))

    # 2. Parse INPUT CSV to get the extracted triples for the cyclic verbalisation
    print(f"\nLoading previously generated IE triples from {input_csv}...")
    run_df_raw = pd.read_csv(input_csv)
    if LIMIT_PER_SPLIT is not None:
        run_df_raw = run_df_raw.head(LIMIT_PER_SPLIT).copy()

    run_rows = []
    for _, row in run_df_raw.iterrows():
        row_dict = row.to_dict()
        lang = row_dict.get("lang")
        
        # Get target text directly from the IE extraction step
        extracted_text = row_dict.get("extracted_triples_text", "")
        target_triples = parse_bracketed_triples(extracted_text)
        
        prompt = build_user_prompt(lang, fewshot_examples[lang], target_triples)
        messages = build_messages(lang, prompt)
        
        row_dict["cycle_verb_prompt"] = prompt
        row_dict["cycle_verb_messages"] = json.dumps(messages, ensure_ascii=False)
        run_rows.append(row_dict)

    run_df = pd.DataFrame(run_rows)
    return df_all, fewshot_anchor_df, fewshot_examples, run_df


def write_manifests(output_dir: Path, run_df: pd.DataFrame, fewshot_examples: Dict[str, List[pd.Series]], args: argparse.Namespace):
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "cycle_fewshot_manifest.csv"
    run_df.to_csv(manifest_path, index=False)

    fewshot_export_rows = []
    for lang in TARGET_LANGS:
        for shot_idx, row in enumerate(fewshot_examples[lang], start=1):
            fewshot_export_rows.append(
                {
                    "lang": lang,
                    "shot_idx": shot_idx,
                    "align_key": row["align_key"],
                    "split": row["split"],
                    "category": row["category"],
                    "triple_bucket": row.get("triple_bucket", ""),
                    "eid": row["eid"],
                    "size": row["size"],
                    "num_triples": row["num_triples"],
                    "triples": json.dumps(row["triples"], ensure_ascii=False),
                    "reference": row["reference"],
                    "num_lexicalisations": row["num_lexicalisations"],
                    "lexicalisations": row["lexicalisations"],
                }
            )
    pd.DataFrame(fewshot_export_rows).to_csv(output_dir / "cycle_fewshot_examples.csv", index=False)

    config_snapshot = {
        "llm_models": LLM_MODELS,
        "parse_splits": PARSE_SPLITS,
        "target_langs": TARGET_LANGS,
        "fewshot_triple_sizes": FEWSHOT_TRIPLE_SIZES,
        "fewshot_random_seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "temperature": TEMPERATURE,
        "do_sample": DO_SAMPLE,
        "top_p": TOP_P,
        "repetition_penalty": REPETITION_PENALTY,
        "use_4bit": USE_4BIT,
        "trust_remote_code": TRUST_REMOTE_CODE,
        "torch_dtype": TORCH_DTYPE,
        "device_map": DEVICE_MAP,
        "limit_per_split": args.limit_per_split,
        "overwrite_existing": args.overwrite_existing,
        "data_root": str(args.data_root),
        "input_csv": str(args.input_csv),
        "output_dir": str(output_dir),
    }
    with open(output_dir / "cycle_run_config.json", "w", encoding="utf-8") as f:
        json.dump(config_snapshot, f, ensure_ascii=False, indent=2)


def run_generation_for_model(
    model_name: str,
    run_df: pd.DataFrame,
    output_dir: Path,
    args: argparse.Namespace
) -> pd.DataFrame:
    out_path = output_csv_path(output_dir, model_name)

    # Key to uniquely identify the row in the CSV (including the previous model_name which ran the IE)
    key_cols = ["lang", "split", "category", "eid", "align_key", "model_name"]

    existing_df = None
    completed_keys = set()

    if out_path.exists() and not args.overwrite_existing:
        print(f"Found existing results at {out_path}")
        existing_df = pd.read_csv(out_path)

        if not existing_df.empty:
            required_cols = set(key_cols + ["cycle_verb_status"])
            if required_cols.issubset(existing_df.columns):
                completed_df = existing_df[existing_df["cycle_verb_status"] == "ok"].copy()
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

    tokenizer, model = load_model_and_tokenizer(model_name)
    new_results = []
    family = get_model_family(model_name)

    prompt_text_cache: Dict[str, str] = {}
    for messages_json in missing_df["cycle_verb_messages"].unique().tolist():
        messages = json.loads(messages_json)
        prompt_text_cache[messages_json] = apply_chat_template_or_fallback(
            tokenizer, messages, model_name
        )

    sorted_df = missing_df.copy()
    sorted_df["prompt_len_chars"] = sorted_df["cycle_verb_prompt"].str.len()
    sorted_df = sorted_df.sort_values(
        ["lang", "split", "num_triples", "prompt_len_chars", "category", "eid"]
    )

    print(
        f"Running {model_name} ({family}) on {len(sorted_df)} missing entries"
        + (f" / {len(run_df)} total" if len(sorted_df) != len(run_df) else "")
    )

    try:
        for idx, (_, row) in enumerate(
            tqdm(
                sorted_df.iterrows(),
                total=len(sorted_df),
                desc=f"Generating with {model_name}"
            ),
            start=1
        ):
            prompt_text = prompt_text_cache[row["cycle_verb_messages"]]
            t0 = time.time()

            try:
                gen = generate_one(
                    model, tokenizer, prompt_text, model_name, args.max_new_tokens
                )
                raw_generation = gen["raw_generation"]
                extracted = extract_prefixed_answer(raw_generation, row["lang"])
                status = "ok"
                error = ""
            except Exception as e:
                raw_generation = ""
                extracted = ""
                status = "error"
                error = repr(e)

            new_results.append(
                {
                    **row.drop(labels=["prompt_len_chars"]).to_dict(),
                    "cycle_verb_model_name": model_name,
                    "cycle_verb_raw_generation": raw_generation,
                    "cycle_verb_extracted_verbalization": extracted,
                    "cycle_verb_status": status,
                    "cycle_verb_error": error,
                    "cycle_verb_latency_sec": round(time.time() - t0, 4),
                    "cycle_verb_timestamp_utc": pd.Timestamp.utcnow().isoformat(),
                }
            )

            if idx % args.save_every == 0:
                checkpoint_df = pd.DataFrame(new_results)

                if existing_df is not None and not existing_df.empty:
                    merged_df = pd.concat([existing_df, checkpoint_df], ignore_index=True)
                else:
                    merged_df = checkpoint_df

                merged_df = (
                    merged_df.sort_values("cycle_verb_timestamp_utc")
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
            result_df.sort_values("cycle_verb_timestamp_utc")
            .drop_duplicates(subset=key_cols, keep="last")
        )

        result_df.to_csv(out_path, index=False)
        return result_df

    finally:
        cleanup_model(model, tokenizer)


def main() -> None:
    global LIMIT_PER_SPLIT, OVERWRITE_EXISTING
    args = parse_args()
    LIMIT_PER_SPLIT = args.limit_per_split
    OVERWRITE_EXISTING = args.overwrite_existing

    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Visible GPU:", torch.cuda.get_device_name(0))

    df_all, fewshot_anchor_df, fewshot_examples, run_df = prepare_dataset(args.data_root, args.input_csv, args.seed)

    print(f"Parsed 'train' reference rows for anchors: {len(df_all)}")
    print("\nSummary by language/split (Train only):")
    summary = (
        df_all.groupby(["lang", "split"])
        .agg(rows=("align_key", "size"), unique_instances=("align_key", "nunique"), categories=("category", "nunique"))
        .reset_index()
    )
    print(summary.to_string(index=False))

    print("\nSelected few-shot anchors:")
    print(fewshot_anchor_df[["align_key", "category", "eid", "size", "reference", "num_lexicalisations"]].to_string(index=False))

    for lang in TARGET_LANGS:
        print(f"\n=== {lang.upper()} few-shot examples ===")
        fewshot_df = pd.DataFrame(
            [
                {
                    "category": row["category"],
                    "eid": row["eid"],
                    "size": row["size"],
                    "num_triples": row["num_triples"],
                    "num_lexicalisations": row["num_lexicalisations"],
                    "reference": row["reference"][:120],
                }
                for row in fewshot_examples[lang]
            ]
        )
        print(fewshot_df.to_string(index=False))

    write_manifests(args.output_dir, run_df, fewshot_examples, args)
    print(f"\nManifest rows loaded from CSV: {len(run_df)}")
    
    print_cols = ["lang", "split", "category", "eid", "size", "num_triples"]
    if "model_name" in run_df.columns:
        print_cols.insert(0, "model_name")
        
    print(run_df[print_cols].head(10).to_string(index=False))

    all_results = []
    for model_name in LLM_MODELS:
        model_df = run_generation_for_model(model_name, run_df, args.output_dir, args)
        all_results.append(model_df)

    results_df = pd.concat(all_results, ignore_index=True)
    combined_path = args.output_dir / "cycle_generations_all_models.csv"
    results_df.to_csv(combined_path, index=False)
    print(f"\nCombined results: {combined_path}")

    group_cols = ["cycle_verb_model_name", "lang", "split"]
    if "model_name" in results_df.columns:
        group_cols.insert(1, "model_name") # The original IE model name

    qc = (
        results_df.groupby(group_cols)
        .agg(
            rows=("align_key", "size"),
            ok=("cycle_verb_status", lambda s: (s == "ok").sum()),
            nonempty=("cycle_verb_extracted_verbalization", lambda s: s.fillna("").str.len().gt(0).sum()),
            avg_latency_sec=("cycle_verb_latency_sec", "mean"),
        )
        .reset_index()
    )
    print("\nQuick sanity check:")
    print(qc.to_string(index=False))


if __name__ == "__main__":
    main()