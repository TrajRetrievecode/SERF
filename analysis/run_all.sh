#!/usr/bin/env bash
set -euo pipefail
cd "${SERF_ROOT:?Set SERF_ROOT to the package directory}"
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
