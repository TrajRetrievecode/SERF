#!/usr/bin/env bash
set -euo pipefail
cd "${SERF_ROOT:?Set SERF_ROOT to the package directory}"
export PYTHONDONTWRITEBYTECODE=1

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
