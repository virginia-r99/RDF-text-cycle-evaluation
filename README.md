# Generating vs Reconstructing Knowledge: Code and Data for WebNLG ES‑CO

This repository contains the code, datasets, and experiment scripts accompanying the paper *“Generating vs Reconstructing Knowledge: A Multilingual Evaluation of RDF–Text Asymmetry in Low-Resource Languages”*. It implements the full RDF → text → RDF evaluation pipeline over English, Spanish, and Spain’s co‑official languages (Catalan, Galician, Basque) using the new WebNLG ES‑CO corpus and a graph‑centric cycle consistency framework. 

We provide:

- The WebNLG_ES_CO dataset for English, Spanish, Catalan, Galician, and Basque, aligned with WebNLG/WebNLG ES structure. 
- Scripts and notebooks to construct the multilingual triples and verbalisations from translation outputs and Wikidata‑based label harvesting. 
- Few‑shot prompting and LoRA‑based supervised adaptation experiments for both RDF‑to‑text verbalisation and text‑to‑RDF information extraction. 
- Evaluation scripts and artefacts for local triple‑level and global graph‑level metrics, including cycle consistency statistics. 

***

## Repository structure

```text
.
│   README.md
│   requirements.txt
│   .gitattributes
│   LICENSE
│
├── dataset_adaptation/
│   │   rebuild_triples_xml.ipynb
│   │   rebuild_verbalisations_xml.ipynb
│   │   revoting_triples.ipynb
│   │   revoting_verbalisations.ipynb
│   │   translate_WebNLG_triples.py
│   │   translate_WebNLG_verbalisations.py
│   │
│   └── results/
│       ├── triples/
│       │   entity_translations_ca_revoted_with_output_token.csv
│       │   entity_translations_eu_revoted_with_output_token.csv
│       │   entity_translations_gl_revoted_with_output_token.csv
│       │   relation_translations_ca_revoted_with_output_token.csv
│       │   relation_translations_eu_revoted_with_output_token.csv
│       │   relation_translations_gl_revoted_with_output_token.csv
│       │
│       └── verbalisations/
│           registry_webnlg_en_ca.revoted.csv
│           registry_webnlg_en_eu.revoted.csv
│           registry_webnlg_en_gl.revoted.csv
│
├── IE/
│   ├── Few-shot/
│   │   │   webnlg_fewshot_ie_benchmark.py
│   │   │
│   │   ├── evaluation/
│   │   │   instance_level_ie_metrics.xlsx
│   │   │   summary_ie_by_model_lang.csv
│   │   │   summary_ie_by_model_lang.xlsx
│   │   │   summary_ie_by_model_lang_category.csv
│   │   │   summary_ie_by_model_lang_category.xlsx
│   │   │   summary_ie_by_model_lang_split.csv
│   │   │   summary_ie_by_model_lang_split.xlsx
│   │   │   summary_ie_micro_by_model_lang.csv
│   │   │   summary_ie_micro_by_model_lang.xlsx
│   │   │   webnlg_ie_evaluation_metrics.ipynb
│   │   │
│   │   └── outputs/
│   │       fewshot_examples.csv
│   │       fewshot_manifest.csv
│   │       generations__BSC-LT__salamandra-2b-instruct.csv
│   │       generations__CohereLabs__tiny-aya-global.csv
│   │       generations__HuggingFaceTB__SmolLM3-3B.csv
│   │       generations__Qwen__Qwen3-4B-Instruct-2507.csv
│   │       run_config.json
│   │
│   └── LoRA/
│       │   eval_zero_shot_ie_trained.py
│       │   train_lora_webnlg_co_ie_multilingual.py
│       │
│       └── runs_ie_qwen/
│           │   summary__Qwen__Qwen3-4B-Instruct-2507__ie_multilingual.json
│           │
│           └── zero_shot_eval/
│               generations_zero_shot_trained_model.csv
│               generations__final_adapter.csv
│               summary_by_lang.csv
│               summary_overall.json
│               zero_shot_eval_summary.xlsx
│
├── verbalisation/
│   │   webnlg_fewshot_verbalisation_benchmark.py
│   │
│   ├── evaluation/
│   │   summary_by_model_lang.csv
│   │   summary_by_model_lang_category.csv
│   │   summary_by_model_lang_EKAW_verb.xlsx
│   │   summary_by_model_lang_split.csv
│   │   webnlg_generation_evaluation_bertscore_prf_bleu.ipynb
│   │
│   └── outputs/
│       fewshot_examples.csv
│       fewshot_manifest.csv
│       generations__BSC-LT__salamandra-2b-instruct.csv
│       generations__CohereLabs__tiny-aya-global.csv
│       generations__HuggingFaceTB__SmolLM3-3B.csv
│       generations__Qwen__Qwen3-4B-Instruct-2507.csv
│       run_config.json
│
└── WebNLG_ES_CO/
    ├── dev/
    │   ├── 1triples/ ... 7triples/
    │   └── test/
    └── train/
        ├── 1triples/ ... 7triples/
```

At a high level:

- `dataset_adaptation/`: construction of WebNLG ES‑CO triples and verbalisations from Spanish WebNLG plus MT/LLM outputs. 
- `IE/`: information extraction (text‑to‑RDF) experiments, both few‑shot and LoRA‑fine‑tuned, including metrics and model generations. 
- `verbalisation/`: RDF‑to‑text few‑shot benchmarks and evaluation notebooks. 
- `WebNLG_ES_CO/`: final multilingual XML corpora aligned with WebNLG splits and triple‑count folders. 

***

## WebNLG ES‑CO dataset

The `WebNLG_ES_CO` folder hosts the multilingual WebNLG variants used to evaluate RDF ↔ text transformations in English (en), Spanish (es), Catalan (ca), Galician (gl), and Basque (eu). 

- `train/`, `dev/`, `test/` reproduce the original WebNLG splits, with subfolders `1triples`–`7triples` grouping XML files by triple count. 
- English and Spanish come from WebNLG and WebNLG ES, while Catalan, Galician, and Basque verbalisations are automatically translated from Spanish using an ensemble of multilingual MT/LLM systems with agreement‑based selection and semantic voting. 
- RDF triples are translated separately: entities via Wikidata labels/aliases in the target language when available, otherwise via MT plus revoting; relations via the same MT ensemble, since WebNLG predicates do not have direct KB label alignments. 

This “silver” corpus is designed to stress‑test multilingual RDF↔text consistency, prioritising structural and factual alignment rather than stylistic perfection. 

***

## Dataset adaptation pipeline

The `dataset_adaptation` directory contains the scripts and notebooks to reconstruct WebNLG ES‑CO from raw translations, Wikidata labels, and revoting decisions. 

### Key scripts and notebooks

- `translate_WebNLG_triples.py`: translates RDF triples (entities and predicates) from Spanish into Catalan, Galician, and Basque using multiple MT/LLM backends and a revoting strategy. 
- `translate_WebNLG_verbalisations.py`: translates Spanish verbalisations into the target languages, generating multiple candidates and selecting a final output with language ID, string similarity, and sentence‑embedding similarity to the source. 
- `revoting_triples.ipynb`: consolidates entity and relation translations, filters out invalid language outputs, and resolves ties via semantic similarity. 
- `revoting_verbalisations.ipynb`: applies the same revoting logic at sentence level to obtain robust silver verbalisations. 
- `rebuild_triples_xml.ipynb`: rebuilds WebNLG‑style XML triple files from CSV registries and per‑language translation tables. 
- `rebuild_verbalisations_xml.ipynb`: rebuilds per‑language XML verbalisations aligned with the translated triples. 

### Results files

- `results/triples/*.csv`: store per‑language entity and relation translations, including the selected output token for each MT/LLM candidate. 
- `results/verbalisations/registry_webnlg_en_{ca,eu,gl}.revoted.csv`: track links between English references and their translated verbalisations, plus the model choices used in revoting. 

**Typical use:**

1. Run `translate_WebNLG_triples.py` and `translate_WebNLG_verbalisations.py` to generate multilingual triple and verbalisation candidates. 
2. Inspect and, if needed, modify thresholds in `revoting_triples.ipynb` and `revoting_verbalisations.ipynb` to adjust language and similarity filters. 
3. Execute `rebuild_triples_xml.ipynb` and `rebuild_verbalisations_xml.ipynb` to regenerate the XML corpora under `WebNLG_ES_CO/`. 

***

## RDF‑to‑text verbalisation experiments

The `verbalisation` directory implements the RDF‑to‑text (RDF → text) experiments described in the paper for all five languages and multiple compact multilingual LLMs. 

### Few‑shot benchmark

- `webnlg_fewshot_verbalisation_benchmark.py`:
  - Loads WebNLG ES‑CO triples from `WebNLG_ES_CO/{train,dev,test}`. 
  - Builds parallel instruction prompts in English, Spanish, Catalan, Galician, and Basque asking the model to generate a single‑paragraph verbalisation from triples, with a fixed bracketed output format. 
  - Queries several models (e.g., Salamandra‑2B‑Instruct, tiny‑aya‑global, SmolLM3‑3B, Qwen3‑4B‑Instruct‑2507) in few‑shot mode and logs generations to CSV. 

- `outputs/`:
  - `fewshot_manifest.csv`, `fewshot_examples.csv`: define the sampling strategy and concrete few‑shot examples per language. 
  - `generations__*csv`: raw verbalisation outputs for each model and language combination. 
  - `run_config.json`: configuration used for the few‑shot runs (models, prompts, splits). 

### Evaluation

- `evaluation/webnlg_generation_evaluation_bertscore_prf_bleu.ipynb`:
  - Computes surface metrics: BLEU, ROUGE‑L, METEOR, chrF++. 
  - Computes semantic metrics: BERTScore and embedding cosine similarity using multilingual E5 representations. 
  - Derives an expansion ratio (generated length / reference length) to analyse omissions and hallucinations. 
- `evaluation/summary_by_model_lang*.csv`:
  - Aggregate metrics by model, language, category, and split, corresponding to Table 1 in the paper. 

These artefacts support RQ1 and RQ2 by quantifying how well models preserve the semantics of triples in verbalisation across languages. 

***

## Text‑to‑RDF information extraction experiments

The `IE` directory contains the text‑to‑RDF information extraction (IE) experiments, spanning few‑shot prompting and LoRA‑based supervised adaptation on top of multilingual LLMs. 

### Few‑shot IE

Under `IE/Few-shot/`:

- `webnlg_fewshot_ie_benchmark.py`:
  - Loads textual references from WebNLG ES‑CO and constructs instruction‑style prompts asking the model to extract all explicitly stated RDF triples in a strict `[subject | predicate | object]` format, with one triple per line and no explanations. 
  - Instantiates parallel prompts in English, Spanish, Catalan, Galician, and Basque by direct translation of the instructions while preserving the same structural constraints. 
  - Queries the same family of compact LLMs (Qwen3‑4B‑Instruct‑2507, SmolLM3‑3B, tiny‑aya‑global, Salamandra‑2B‑Instruct) and dumps extracted triples to CSV under `outputs/`. 

- `outputs/`:
  - `fewshot_manifest.csv`, `fewshot_examples.csv`: describe which instances and languages are used in the few‑shot probes. 
  - `generations__*csv`: raw triple predictions for each model and language. 
  - `run_config.json`: configuration for few‑shot IE runs. 

- `evaluation/webnlg_ie_evaluation_metrics.ipynb`:
  - Computes exact triple F1 (strict subject–predicate–object matches). 
  - Computes relaxed metrics: swap‑aware F1 (subject/object swapped but same relation), and soft F1 based on lexical/semantic similarity to capture near‑matches. 
  - Derives graph‑level metrics: graph similarity, information retention (fraction of gold triples recovered) and hallucination rate (fraction of predicted triples not present in gold). 
  - Aggregates results by model and language into `summary_ie_*` CSV/XLSX files, corresponding to Table 2 in the paper. 

### LoRA‑based IE fine‑tuning

Under `IE/LoRA/`:

- `train_lora_webnlg_co_ie_multilingual.py`:
  - Fine‑tunes an instruction‑tuned base model (Qwen3‑4B‑Instruct‑2507) on IE over WebNLG ES‑CO using LoRA adapters. 
  - Uses simplified prompts that fix the desired output format and minimise variation during training. 
  - Trains jointly across languages to encourage cross‑lingual generalisation while preserving a shared triple schema. 

- `eval_zero_shot_ie_trained.py`:
  - Evaluates the fine‑tuned adapter in a zero‑shot IE setting on held‑out splits and all languages. 
  - Produces triple‑level and graph‑level metrics analogous to the few‑shot evaluation. 

- `runs_ie_qwen/`:
  - `summary__Qwen__Qwen3-4B-Instruct-2507__ie_multilingual.json`: summarises pre‑ and post‑fine‑tuning performance across languages and metrics (Exact F1, soft F1, hallucination, omission, graph similarity). 
  - `zero_shot_eval/`:
    - `generations_zero_shot_trained_model.csv`, `generations__final_adapter.csv`: predictions of the fine‑tuned model. 
    - `summary_by_lang.csv`, `summary_overall.json`, `zero_shot_eval_summary.xlsx`: aggregated IE scores after adaptation, matching the “Fine Tune” rows in Table 2. 

These components support RQ1, RQ2, and RQ4 by showing the large gap between verbalisation and extraction, and the gains from lightweight supervised adaptation, especially for low‑resource languages. 

***

## Metrics and analysis

The evaluation notebooks in `verbalisation/evaluation/` and `IE/Few-shot/evaluation/` implement the graph‑centric cycle consistency framework used in the paper. 

- **Verbalisation metrics (RDF→text)**:
  - Surface: BLEU, ROUGE‑L, METEOR, chrF++. 
  - Semantic: BERTScore and E5‑based cosine similarity. 
  - Structural: expansion ratio and its deviation from 1 to capture over‑ and under‑generation. 

- **Information extraction metrics (text→RDF)**:
  - Exact triple F1, soft triple F1, and swap‑aware F1. 
  - Predicate‑only F1 to isolate relation detection. 
  - Graph similarity, information retention, and hallucination rates to quantify global graph preservation and error types. 

The resulting summary files reproduce the tables and analyses for multilingual degradation, verbalisation vs extraction asymmetry, and the impact of LoRA fine‑tuning across languages. 

***

## Installation

Create a virtual environment and install dependencies:

```bash
git clone https://github.com/virginia-r99/RDF-text-cycle-evaluation.git
cd your-repo-webnlg-es-co

python -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

All experiments were run with Python 3.10+ and standard scientific/NLP libraries (Hugging Face, PyTorch, etc.), as detailed in the environment configuration of the original project. 

***

## Example workflows

Below are example end‑to‑end workflows that mirror the paper’s experiments. 

### 1. Rebuild the WebNLG ES‑CO corpus

1. Generate multilingual triple and sentence candidates:
   - Run `dataset_adaptation/translate_WebNLG_triples.py`.
   - Run `dataset_adaptation/translate_WebNLG_verbalisations.py`.
2. Apply revoting and quality filters:
   - Open `dataset_adaptation/revoting_triples.ipynb` and `dataset_adaptation/revoting_verbalisations.ipynb` and run the cells to select final candidates.
3. Rebuild XML files:
   - Execute `dataset_adaptation/rebuild_triples_xml.ipynb`.
   - Execute `dataset_adaptation/rebuild_verbalisations_xml.ipynb`.
4. Verify that the resulting XMLs in `WebNLG_ES_CO/{train,dev,test}` follow the original WebNLG structure and splits.

### 2. Run verbalisation (RDF→text) few‑shot evaluation

1. Configure model names and prompts in `verbalisation/outputs/run_config.json` if needed.
2. Run:
   ```bash
   python verbalisation/webnlg_fewshot_verbalisation_benchmark.py
   ```
   This will generate per‑model, per‑language CSVs under `verbalisation/outputs/`. 
3. Open `verbalisation/evaluation/webnlg_generation_evaluation_bertscore_prf_bleu.ipynb` and run the notebook to compute metrics and write summaries to the `evaluation/` folder. 

### 3. Run few‑shot IE (text→RDF) benchmark

1. Adjust model configuration in `IE/Few-shot/run_config.json` or in the script arguments.
2. Run:
   ```bash
   python IE/Few-shot/webnlg_fewshot_ie_benchmark.py
   ```
   which will write triple generations to `IE/Few-shot/outputs/`. 
3. Open `IE/Few-shot/evaluation/webnlg_ie_evaluation_metrics.ipynb` to compute IE metrics, hallucination/omission rates, and graph similarity, and export the aggregated CSV/XLSX files. 

### 4. Train and evaluate LoRA IE adapters

1. Launch LoRA fine‑tuning:
   ```bash
   python IE/LoRA/train_lora_webnlg_co_ie_multilingual.py
   ```
   This will train Qwen3‑4B‑Instruct‑2507 IE adapters on WebNLG ES‑CO and save them to a specified output directory. 
2. Evaluate the fine‑tuned model in zero‑shot IE:
   ```bash
   python IE/LoRA/eval_zero_shot_ie_trained.py
   ```
   Producing CSV/JSON/XLSX summaries in `IE/LoRA/runs_ie_qwen/zero_shot_eval/` that correspond to the “Fine Tune” rows in the paper’s IE table. 

***
## Contact

For questions, feedback, or issues related to this repository, please open a GitHub issue or contact the authors of the accompanying paper.

- **Virginia Ramón-Ferrer** — virginia.ramon@upm.es

**Affiliation**: Ontology Engineering Group, Universidad Politécnica de Madrid, Madrid, Spain.
