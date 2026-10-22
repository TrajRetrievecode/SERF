# SERF code release

SERF routes a budget of client calls to the sub-questions of a query that still need evidence, tests whether necessary evidence sits on an offline client, and selects the cited pages across the called clients. This package runs the evaluation set, the two public datasets and the control analyses of the paper and produces its tables and result figures.

## Layout

```text
analysis/            Result analyses, table generators, and the figure generator.
configs/             Protocols and fixed query memberships.
scripts/             Preparation, indexing, experiment, and aggregation scripts.
src/                 Routing, retrieval, generation, and scoring modules.
MANIFEST.md          One entry per file with the paper result it serves.
requirements.txt     Python dependencies.
```

## Environment

Use Linux, Python 3.11 and CUDA for embeddings and generation. The results of the paper were produced with Python 3.11.15, the package versions pinned in `requirements.txt`, the CUDA 12.8 build of PyTorch and NVIDIA RTX A6000 GPUs. Create the virtual environment outside the package directory, because the confirmation scripts of step 8 pin every module loaded from inside it. Run every command from the package directory.

```bash
python -m venv ../serf-venv
source ../serf-venv/bin/activate
python -m pip install -r requirements.txt
export SERF_ROOT="$PWD"
export SERF_DATA="$SERF_ROOT/data"
export SERF_MODELS="$SERF_ROOT/models"
export SERF_FINRAG="$SERF_DATA/datasets/FinRAGBench-V"
export SERF_VISDOM="$SERF_DATA/datasets/VisDoM"
export SERF_FINRAG_MANIFEST="$SERF_ROOT/artifacts/stage2/pdf_corpus/page_index_manifest.json"
export SERF_CLIP="$SERF_MODELS/openai/clip-vit-base-patch32"
export SERF_BGE="$SERF_MODELS/BAAI/bge-m3"
export SERF_GENERATOR="$SERF_MODELS/Qwen/Qwen2.5-7B-Instruct"
export SERF_QWEN_EMBED="$SERF_MODELS/Qwen/Qwen3-VL-Embedding-2B"
export SERF_GOLD_TASKS="$SERF_DATA/annotation/gold_v1/tasks_gold_v1.json"
export SERF_GOLD_MEDIA="$SERF_DATA/annotation/gold_v1"
export SERF_GOLD_CACHE="$SERF_ROOT/artifacts/gold_v3/cache"
export PYTHONPATH="$SERF_ROOT/src:$SERF_ROOT/src/public_e2e_v1:$SERF_ROOT/scripts:$SERF_ROOT/scripts/public_e2e_v1:$SERF_ROOT/scripts/serf_v3:$SERF_ROOT/scripts/serf_v3_devconfirm:$SERF_ROOT/analysis:$SERF_ROOT/scripts/public_paired_v1:$SERF_ROOT/scripts/enumeration:$SERF_ROOT/analysis/controls/attribution:$SERF_ROOT/analysis/controls/reference:$SERF_ROOT/analysis/controls/composite:$SERF_ROOT/analysis/controls/global_selection"
export PYTHONDONTWRITEBYTECODE=1
python -m unittest discover -s scripts -p test_formal_gold_v1.py
python -m unittest discover -s analysis/controls/global_selection/tests -p test_global_selection.py
```

`SERF_ROOT` holds every output under `artifacts/` and the derived data under `data/`. Link `data/` to external storage if needed. The dataset, checkpoint, manifest, task, and image variables point to the inputs. The evaluation runner reads the task file at the path above.

Set `CUDA_VISIBLE_DEVICES` to the devices in use. The generation of the one-page variant and the external methods uses four devices with tensor parallel size 4. Every other generation command uses one device with tensor parallel size 1. The evaluation embedding builder expects one idle GPU. Adjust the memory fraction of each command to the devices.

## Inputs

Prepare [FinRAGBench-V](https://github.com/zhaosuifeng/FinRAGBench-V) under `SERF_FINRAG` with `queries/queries_ch.json`, `queries/queries_en.json`, `qrels/qrels_ch.tsv`, `qrels/qrels_en.tsv`, and the QA PDF archives `pdfs_for_QA/pdf_ch.tar.gz` and `pdfs_for_QA/pdf_en.tar.gz`.

Prepare [VisDoM](https://github.com/MananSuri27/VisDoM) under `SERF_VISDOM` with `feta_tab`, `paper_tab`, `scigraphvqa`, `slidevqa`, and `spiqa`, each with its task CSV and its `docs/` directory of PDFs. The SciGraph CSV is `scigraphqa.csv`.

| Variable | Checkpoint |
|---|---|
| `SERF_CLIP` | `openai/clip-vit-base-patch32` |
| `SERF_BGE` | `BAAI/bge-m3` |
| `SERF_GENERATOR` | `Qwen/Qwen2.5-7B-Instruct` |
| `SERF_QWEN_EMBED` | `Qwen/Qwen3-VL-Embedding-2B` |

The Qwen embedding checkpoint includes `scripts/qwen3_vl_embedding.py`. Its visual utilities come from the installed `qwen-vl-utils` package.

The FinRAG manifest at `SERF_FINRAG_MANIFEST` holds a `documents` array. Each entry has `source_path`, `output_path`, and `lang`. Each `output_path` names a gzip JSONL file with one row per page: `page_number_1`, `text`, `text_characters`, and `lang`. Paths resolve from the package directory or are absolute. The PDF stem is the document key. Step 1 builds this manifest from the PDFs with PyMuPDF.

The package includes `data/annotation/gold_v1/tasks_gold_v1.json`, an array of 45 evaluation tasks with `canonical_id`, `question`, `candidate_pages`, and `participation_scenarios`. A candidate page has `candidate_id`, `client_id`, and `media_path`. A scenario has `scenario_id` and `online_clients`.

The package includes `data/annotation/gold_v1/fedfinrag_resolve_gold_v1.json`, an object with `dataset_version`, `task_count` and 45 `annotations`. An annotation has `canonical_id`, `atomic_claims`, `reference_answer`, `minimum_sufficient_client_sets`, `participation_assessments`, `candidate_assessments`, and annotation and adjudication provenance. `annotation_provenance`, present for the items whose first pass was recorded, names the `source_model` of that pass and marks `human_annotation` false because the pass was model-assisted. `gold_provenance`, present for every item, records the adjudication and the confirmation by the human curator in `human_curator_confirmed`. A candidate assessment has `candidate_id`, `stance` and `usable_for_claim_ids`, and the runner treats `support` as relevant evidence. A participation assessment has `scenario_id` and `answerability`. The hashes of both files are in `configs/formal_gold_v1_protocol.json`.

The package includes the 80 candidate page images in `data/annotation/gold_v1/media/`. Each task's `media_path` resolves under `SERF_GOLD_MEDIA`, whose default is `data/annotation/gold_v1/`. [LICENSE.md](data/annotation/gold_v1/LICENSE.md) covers the page renderings under Apache License 2.0 and the questions, candidate selections, client assignments, scenarios and labels under CC BY 4.0; [NOTICE](data/annotation/gold_v1/NOTICE) records the image source.

Step 7 builds the embedding cache at `SERF_GOLD_CACHE`. `images.json` lists candidate IDs, image paths, media paths, and hashes. `queries.json` lists `query_key`, `canonical_id`, `kind`, `slot_id`, and `text`, with query keys `<canonical_id>::full` or `<canonical_id>::<slot_id>`. `image_embeddings.npy` and `query_embeddings.npy` hold normalized vectors in those row orders. `manifest.json` records their hashes and the 1,048,576-pixel cap. The builder reads the image and query instructions from the evaluation protocol.

## Run order

Start with an empty `artifacts/` directory. The plan and generation writers append to existing files, so a repeated step needs a fresh `SERF_ROOT` or the removal of its earlier outputs.

### 1. Prepare page text and build the public indices

```bash
python scripts/preparation/prepare_finrag_pdf_corpus.py --workers 8
python scripts/public_e2e_v1/build_visdom_page_text.py --workers 8
for D in finrag visdom; do
  python scripts/public_e2e_v1/build_clip_page_index.py --dataset "$D" --clip "$SERF_CLIP"
  python scripts/public_e2e_v1/build_dense_text_index.py --dataset "$D" --model "$SERF_BGE"
  python scripts/public_e2e_v1/build_slot_queries.py --dataset "$D" --clip "$SERF_CLIP" --dense "$SERF_BGE"
done
```

The CLIP builder also builds the TF-IDF channel. The dense builder uses bge-m3. The sub-question representations use all three channels. The four settings cross 8 or 16 clients with the `seen_query` and `unseen_family` tracks.

### 2. Select the retriever weights and the prompt

Both scripts select records that lie in the training portion of at least one setting and in no test portion; some of these records lie in the development portion of another setting. The reported runs use the weights recorded in `configs/public_e2e_v1/protocol_v3.json` and prompt version `v1` of `src/public_e2e_v1/e2e_generation.py`.

```bash
for D in finrag visdom; do
  python scripts/public_e2e_v1/select_retriever_delivered.py --dataset "$D"
  python scripts/public_e2e_v1/select_prompt.py --dataset "$D"     --tensor-parallel-size 4 --gpu-memory-utilization 0.11
done
```

### 3. Run the one-page variant and the external methods

Use four visible devices.

```bash
for D in finrag visdom; do
  python scripts/public_e2e_v1/run_slot_plan.py --dataset "$D" \
    --protocol configs/public_e2e_v1/protocol_v3.json --out-suffix _slot
  python scripts/public_e2e_v1/run_e2e.py --dataset "$D" --stages generate score \
    --protocol configs/public_e2e_v1/protocol_v3.json --out-suffix _slot \
    --tensor-parallel-size 4 --gpu-memory-utilization 0.11
done
python scripts/public_e2e_v1/aggregate.py --tag _slot
```

### 4. Run the paired availability scenarios

Use one visible device. The answerable scenario of a pair takes as many clients offline as the unanswerable one, chosen among the clients without a relevant document.

```bash
for D in finrag visdom; do
  python scripts/public_paired_v1/run_paired_plan.py --dataset "$D" \
    --protocol configs/public_paired_v1/protocol_paired_v1.json
  cp "artifacts/public_e2e_v1/${D}_slot/generation_cache.jsonl" \
     "artifacts/public_paired_v1/${D}/generation_cache.jsonl"
  python scripts/public_paired_v1/run_paired_e2e.py --dataset "$D" --stages generate score \
    --protocol configs/public_paired_v1/protocol_paired_v1.json \
    --tensor-parallel-size 1 --gpu-memory-utilization 0.36
done
python scripts/public_paired_v1/aggregate_paired.py
```

### 5. Run SERF on the public datasets

```bash
for D in finrag visdom; do
  python scripts/serf_v3/run_v3_plan.py --dataset "$D" --split test \
    --protocol configs/serf_v3/protocol_serf_v3_public.json
  python scripts/serf_v3/merge_caches.py "artifacts/serf_v3/${D}_test/generation_cache.jsonl" \
    "artifacts/public_e2e_v1/${D}_slot/generation_cache.jsonl" \
    "artifacts/public_paired_v1/${D}/generation_cache.jsonl"
  python scripts/serf_v3/run_v3_e2e.py --dataset "$D" --stages generate score \
    --protocol configs/serf_v3/protocol_serf_v3_public.json --out-suffix _test \
    --tensor-parallel-size 1 --gpu-memory-utilization 0.36
done
python scripts/serf_v3/aggregate_v3.py
```

### 6. Run the component ablation

```bash
for D in finrag visdom; do
  python scripts/serf_v3_ablation/run_v3_abl_plan.py --dataset "$D" --split test \
    --protocol configs/serf_v3_ablation/protocol_v3_ablation.json
done
```

The plans give evidence recall, client calls, and routing abstention. The tables print for Centralized the mean number of clients that hold its cited pages, computed by the `reference` control analysis of step 9, and a dash before that analysis has run. No generation is needed.

### 7. Run the evaluation set and the single-client-offline enumeration

Use the packaged tasks, annotations and candidate images. The Qwen3-VL-Embedding-2B checkpoint at `SERF_QWEN_EMBED` is the only external input for this step.

```bash
python scripts/preparation/build_gold_visual_embeddings.py --root "$SERF_ROOT" --model "$SERF_QWEN_EMBED"
python scripts/run_formal_gold_v1.py --root "$SERF_ROOT"
python scripts/enumeration/enumerate_single_offline.py
python scripts/enumeration/margins_readonly.py
```

### 8. Run the development confirmation

The confirmation population is fixed in `configs/serf_v3_devconfirm/confirm_uids_finrag.json` and `confirm_uids_visdom.json`: 288 pairs of a setting and a query over 234 FinRAG queries and 603 pairs over 487 VisDoM queries. The protocol holds the family-component bootstrap rule and the label and page-text digests. `bind_plans.py` writes a copy of the protocol whose plan hashes are those of the development plans of this run and records the environment that the later scripts check.

```bash
for D in finrag visdom; do
  python scripts/serf_v3/run_v3_plan.py --dataset "$D" --split dev \
    --protocol configs/serf_v3/protocol_serf_v3_public.json
done
python scripts/serf_v3_devconfirm/bind_plans.py
P=artifacts/serf_v3_devconfirm/runtime_protocol.json
python scripts/serf_v3_devconfirm/make_dev_genplan.py "$P"
for D in finrag visdom; do
  python scripts/serf_v3_devconfirm/run_devconfirm_e2e.py --dataset "$D" --stages generate \
    --protocol configs/serf_v3/protocol_serf_v3_public.json --out-suffix _dev \
    --tensor-parallel-size 1 --gpu-memory-utilization 0.36
done
python scripts/serf_v3_devconfirm/dev_evidence.py "$P"
for D in finrag visdom; do
  python scripts/serf_v3_devconfirm/run_devconfirm_e2e.py --dataset "$D" --stages score \
    --protocol configs/serf_v3/protocol_serf_v3_public.json --out-suffix _dev
done
python scripts/serf_v3_devconfirm/aggregate_devconfirm.py "$P"
```

The C1 and C2 tests and the descriptive comparisons of Appendix D are written to `artifacts/serf_v3_devconfirm/evidence_dev.json`. The answer-level aggregates of the confirmation population and of the full development portion are written as separate files beside it.

### 9. Run the control analyses

Run the following commands, also available together as `bash analysis/run_controls.sh`, after steps 1 through 8. Outputs are created under `$SERF_ROOT/artifacts/controls/<job>/`; use a fresh output directory for each run.

`attribution` produces component panels and paired comparisons for each router on the public datasets and evaluation set, including the aggregates for Table 3. `reference` produces owner-contact, per-method-plus-test, online-only-threshold and content-abstention comparisons for Section 4.3 and Figure 2. `composite` produces the two-client question sets, new query embeddings, plans and complete-evidence aggregates for Table 4. `global_selection` produces global page ranking with pooled evidence selection and the availability test, owner contacts, and paired comparisons on the public test portions, composite questions and evaluation set for Tables 3 and 4. `audits` produces the selection-population overlap counts and confirmation analyses on untouched queries.

The composite encoder uses the first GPU exposed by `CUDA_VISIBLE_DEVICES`, or the first device with at least 20480 MiB free if that variable is unset; its checkpoints come from `SERF_CLIP` and `SERF_BGE`.

```bash
# Attribution: cached public scores and evaluation embeddings; no generation.
J=analysis/controls/attribution
O="$SERF_ROOT/artifacts/controls/attribution"
P="$J/protocol_fairness_v1_A_attribution.json"
for D in finrag visdom; do
  python "$J/run_public.py" --dataset "$D" --design "$P" --out "$O"
done
python "$J/check_reproduction.py" --design "$P" --out "$O"
python "$J/run_evaluation.py" --design "$P" --out "$O"
python "$J/aggregate.py" --design "$P" --input "$O" --out "$O/results"

# Reference: existing plans, training score extraction, and system-level CSVs.
J=analysis/controls/reference
O="$SERF_ROOT/artifacts/controls/reference"
python "$J/public_rows.py" --root "$SERF_ROOT" --out "$O"
python "$J/evaluation_rows.py" --root "$SERF_ROOT" --out "$O"
python "$J/aggregate.py" --input "$O" --inputs "$SERF_ROOT/artifacts" --out "$O/results"

# Composite: construct, encode new queries, route, and aggregate.
J=analysis/controls/composite
O="$SERF_ROOT/artifacts/controls/composite"
P="$J/protocol_fairness_v1_C_composite.json"
python "$J/prepare_composites.py" --protocol "$P" --out "$O"
python "$J/encode_composites.py" --protocol "$P" --out "$O"
python "$J/run_composites.py" --protocol "$P" --out "$O"
python "$J/aggregate.py" --protocol "$P" --input "$O" --out "$O"

# Global selection: public queries, composite queries, evaluation, then paired statistics.
J=analysis/controls/global_selection
O="$SERF_ROOT/artifacts/controls/global_selection"
P="$J/protocol_fairness_v1_D_global_selection.json"
for D in finrag visdom; do
  python "$J/run_public.py" --root "$SERF_ROOT" --dataset "$D" --design "$P" --out "$O"
done
python "$J/run_composites.py" --root "$SERF_ROOT" --design "$P" --out "$O"
python "$J/run_evaluation.py" --root "$SERF_ROOT" --design "$P" --out "$O"
python "$J/aggregate.py" --design "$P" --input "$O" --out "$O"

# Audits read inputs and print JSON; shell redirection stores the audit outputs.
mkdir -p artifacts/controls/audits
python analysis/controls/audits/audit_selection_overlap.py > artifacts/controls/audits/selection_overlap.json
python analysis/controls/audits/audit_confirmation_untouched.py > artifacts/controls/audits/confirmation_untouched.json
```

## Tables and figures

Run these commands in order after the experiment steps. The JSON analyses feed the table and figure generators, and the tables use the LaTeX macros defined in `analysis/paper_macros.tex`.

```bash
mkdir -p artifacts/analysis artifacts/paper/tables artifacts/paper/figures
python analysis/gold_v3.py
python analysis/server_reanalysis.py > artifacts/analysis/reanalysis_results.json
python analysis/v3_ablation_readonly.py > artifacts/analysis/v3_ablation.json
python analysis/v3_ablation_by_subquestions.py > artifacts/analysis/v3_ablation_by_subquestions.json
python analysis/collect_facts.py
python analysis/gen_table_gold.py
python analysis/gen_table_public_main.py
python analysis/gen_table_attribution.py
python analysis/gen_table_composite.py
python analysis/gen_table_paired.py
python analysis/gen_table_ablation.py
python analysis/gen_table_subquestions.py
python analysis/gen_table_calls.py
python analysis/format_tables.py
python analysis/make_figures.py
```

| Command under `analysis/` | Output under `artifacts/` | Paper result |
|---|---|---|
| `gold_v3.py` | `analysis/gold_v3_results.json`, `analysis/gold_v3_items.csv` | Table 1 |
| `server_reanalysis.py` | `analysis/reanalysis_results.json` | Figure 2 |
| `v3_ablation_readonly.py` | `analysis/v3_ablation.json` | Table 5 |
| `v3_ablation_by_subquestions.py` | `analysis/v3_ablation_by_subquestions.json` | results by number of sub-questions |
| `collect_facts.py` | `analysis/collected_facts.json` | Figures 2, 3, 4 |
| `gen_table_gold.py` | `paper/tables/tab_gold_main_budget.tex` | Table 1 |
| `gen_table_public_main.py` | `paper/tables/tab_public_main.tex` | Table 2 |
| `gen_table_attribution.py` | `paper/tables/tab_attribution.tex` | Table 3; controls `attribution`, `composite` and `global_selection` |
| `gen_table_composite.py` | `paper/tables/tab_composite.tex` | Table 4; controls `attribution`, `composite` and `global_selection` |
| `gen_table_paired.py` | `paper/tables/tab_public_paired.tex` | paired differences by setting |
| `gen_table_ablation.py` | `paper/tables/tab_ablation.tex` | Table 5, input of the stacked form |
| `gen_table_subquestions.py` | `paper/tables/tab_by_subquestions.tex` | results by number of sub-questions |
| `gen_table_calls.py` | `paper/tables/tab_public_calls.tex` | client calls by setting |
| `format_tables.py` | the tables above, formatted, and `paper/tables/tab_ablation_stacked.tex` | Tables 1 to 5 |
| `make_figures.py` | `paper/figures/fig_test.pdf`, `fig_margin.pdf`, `fig_budget.pdf` | Figures 2, 3, 4 |

`bash analysis/run_all.sh` runs the same sequence. Figure 2 uses the missingness-proxy metrics of the reference run with every method from step 3, the values of each method with the availability test from `server_reanalysis.py`, SERF from step 5, and the evaluation enumeration from step 7.

## Names in the outputs

`FedResolve-v3` is SERF. `FedResolve` and `SERF-v1` are the one-page variant. `FedResolve-noResidual`, `FedResolve-noEarlyStop`, `FedResolve-noAvailability`, and `FedResolve-noSlots` are the ablations without the residual update, the early stop, the availability test, and the sub-questions. The external methods appear as `RAGRoute`, `UniversalRAG`, `CoRe-MMRAG`, `S2G-RAG` and `S2G-RAG+avail`. Control rows retain the method name and their variant suffix: `+both` for the components of SERF in attribution and `pooled_share` in the composite job. Attribution's `FedResolve` with `+sel` is SERF with pooled evidence selection.

`Centralized` in `global_selection` has variants `base` (global page ranking), `+sel` (pooled evidence selection), and `+sel+test` (selection with the availability test). `owner_contacts` counts the distinct owners of its cited pages; the Tables 3 and 4 generator uses the control's `contacts` summary field.
