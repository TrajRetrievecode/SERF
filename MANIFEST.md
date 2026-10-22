`MANIFEST.md` — One line per package file.
`README.md` — Setup, inputs, nine experiment and control steps, and result mapping.
`analysis/collect_facts.py` — Figures 2, 3 and 4.
`analysis/controls/attribution/aggregate.py` — Component attribution; Table 3.
`analysis/controls/attribution/attribution.py` — Component attribution; Table 3.
`analysis/controls/attribution/check_reproduction.py` — Component attribution; Table 3.
`analysis/controls/attribution/protocol_fairness_v1_A_attribution.json` — Component attribution; Table 3.
`analysis/controls/attribution/run_evaluation.py` — Component attribution; Table 3.
`analysis/controls/attribution/run_public.py` — Component attribution; Table 3.
`analysis/controls/audits/audit_confirmation_untouched.py` — Read-only selection and confirmation audits.
`analysis/controls/audits/audit_selection_overlap.py` — Read-only selection and confirmation audits.
`analysis/controls/composite/aggregate.py` — Composite questions; Table 4.
`analysis/controls/composite/composite_common.py` — Composite questions; Table 4.
`analysis/controls/composite/encode_composites.py` — Composite questions; Table 4.
`analysis/controls/composite/prepare_composites.py` — Composite questions; Table 4.
`analysis/controls/composite/protocol_fairness_v1_C_composite.json` — Composite questions; Table 4.
`analysis/controls/composite/run_composites.py` — Composite questions; Table 4.
`analysis/controls/reference/aggregate.py` — Reference controls; Section 4.3 and Figure 2.
`analysis/controls/reference/b_common.py` — Reference controls; Section 4.3 and Figure 2.
`analysis/controls/reference/evaluation_rows.py` — Reference controls; Section 4.3 and Figure 2.
`analysis/controls/reference/protocol_fairness_v1_B_reference.json` — Reference controls; Section 4.3 and Figure 2.
`analysis/controls/reference/public_rows.py` — Reference controls; Section 4.3 and Figure 2.
`analysis/controls/reference/registered_enumeration_checks.json` — Reference controls; Section 4.3 and Figure 2.
`analysis/format_tables.py` — Export and format Tables 1 to 5, supporting tables and Figures 2 to 4.
`analysis/gen_table_ablation.py` — Table 5.
`analysis/gen_table_attribution.py` — Component attribution; Table 3.
`analysis/gen_table_calls.py` — Calls by setting.
`analysis/gen_table_composite.py` — Composite questions; Table 4.
`analysis/gen_table_gold.py` — Table 1.
`analysis/gen_table_paired.py` — Paired differences by setting.
`analysis/gen_table_public_main.py` — Table 2.
`analysis/gen_table_subquestions.py` — Results by sub-question count.
`analysis/gold_v3.py` — Experiment, preparation or result analysis.
`analysis/make_figures.py` — Figures 2, 3 and 4.
`analysis/paper_macros.tex` — LaTeX macros used by the exported tables.
`analysis/run_all.sh` — Export and format Tables 1 to 5, supporting tables and Figures 2 to 4.
`analysis/run_controls.sh` — Run attribution, reference, composite, global selection and audit controls.
`analysis/server_reanalysis.py` — Experiment, preparation or result analysis.
`analysis/v3_ablation_by_subquestions.py` — Experiment, preparation or result analysis.
`analysis/v3_ablation_readonly.py` — Experiment, preparation or result analysis.
`configs/enumeration/protocol_enumeration_v1.json` — Experiment protocol or confirmation membership.
`configs/formal_gold_v1_protocol.json` — Experiment protocol or confirmation membership.
`configs/public_e2e_v1/protocol_v3.json` — Experiment protocol or confirmation membership.
`configs/public_paired_v1/protocol_paired_v1.json` — Experiment protocol or confirmation membership.
`configs/serf_v3/protocol_serf_v3.json` — Experiment protocol or confirmation membership.
`configs/serf_v3/protocol_serf_v3_public.json` — Experiment protocol or confirmation membership.
`configs/serf_v3_ablation/protocol_v3_ablation.json` — Experiment protocol or confirmation membership.
`configs/serf_v3_devconfirm/confirm_uids_finrag.json` — Experiment protocol or confirmation membership.
`configs/serf_v3_devconfirm/confirm_uids_visdom.json` — Experiment protocol or confirmation membership.
`configs/serf_v3_devconfirm/gold_digest_finrag.json` — Experiment protocol or confirmation membership.
`configs/serf_v3_devconfirm/gold_digest_visdom.json` — Experiment protocol or confirmation membership.
`configs/serf_v3_devconfirm/pagetext_digest_finrag.json` — Experiment protocol or confirmation membership.
`configs/serf_v3_devconfirm/pagetext_digest_visdom.json` — Experiment protocol or confirmation membership.
`configs/serf_v3_devconfirm/protocol_serf_v3_devconfirm.json` — Experiment protocol or confirmation membership.
`requirements.txt` — Python dependencies.
`scripts/enumeration/enumerate_single_offline.py` — Experiment, preparation or result analysis.
`scripts/enumeration/margins_readonly.py` — Experiment, preparation or result analysis.
`scripts/preparation/build_gold_visual_embeddings.py` — Experiment, preparation or result analysis.
`scripts/preparation/prepare_finrag_pdf_corpus.py` — Experiment, preparation or result analysis.
`scripts/public_e2e_v1/aggregate.py` — Experiment, preparation or result analysis.
`scripts/public_e2e_v1/build_clip_page_index.py` — Experiment, preparation or result analysis.
`scripts/public_e2e_v1/build_dense_text_index.py` — Experiment, preparation or result analysis.
`scripts/public_e2e_v1/build_slot_queries.py` — Experiment, preparation or result analysis.
`scripts/public_e2e_v1/build_visdom_page_text.py` — Experiment, preparation or result analysis.
`scripts/public_e2e_v1/run_e2e.py` — Experiment, preparation or result analysis.
`scripts/public_e2e_v1/run_slot_plan.py` — Experiment, preparation or result analysis.
`scripts/public_e2e_v1/select_prompt.py` — Experiment, preparation or result analysis.
`scripts/public_e2e_v1/select_retriever_delivered.py` — Experiment, preparation or result analysis.
`scripts/public_e2e_v1/show_tables.py` — Experiment, preparation or result analysis.
`scripts/public_paired_v1/aggregate_paired.py` — Experiment, preparation or result analysis.
`scripts/public_paired_v1/run_paired_e2e.py` — Experiment, preparation or result analysis.
`scripts/public_paired_v1/run_paired_plan.py` — Experiment, preparation or result analysis.
`scripts/run_formal_gold_v1.py` — Experiment, preparation or result analysis.
`scripts/serf_v3/aggregate_v3.py` — Experiment, preparation or result analysis.
`scripts/serf_v3/e2e_v3_adapter.py` — Experiment, preparation or result analysis.
`scripts/serf_v3/merge_caches.py` — Experiment, preparation or result analysis.
`scripts/serf_v3/run_v3_e2e.py` — Experiment, preparation or result analysis.
`scripts/serf_v3/run_v3_plan.py` — Experiment, preparation or result analysis.
`scripts/serf_v3_ablation/run_v3_abl_plan.py` — Experiment, preparation or result analysis.
`scripts/serf_v3_devconfirm/aggregate_devconfirm.py` — Experiment, preparation or result analysis.
`scripts/serf_v3_devconfirm/bind_plans.py` — Experiment, preparation or result analysis.
`scripts/serf_v3_devconfirm/dev_evidence.py` — Experiment, preparation or result analysis.
`scripts/serf_v3_devconfirm/digest_util.py` — Experiment, preparation or result analysis.
`scripts/serf_v3_devconfirm/make_dev_genplan.py` — Experiment, preparation or result analysis.
`scripts/serf_v3_devconfirm/pincheck.py` — Experiment, preparation or result analysis.
`scripts/serf_v3_devconfirm/run_devconfirm_e2e.py` — Experiment, preparation or result analysis.
`scripts/test_formal_gold_v1.py` — Experiment, preparation or result analysis.
`src/fedresolve_core.py` — Shared experiment implementation.
`src/public_e2e_v1/e2e_common.py` — Shared experiment implementation.
`src/public_e2e_v1/e2e_corpus.py` — Shared experiment implementation.
`src/public_e2e_v1/e2e_generation.py` — Shared experiment implementation.
`src/public_e2e_v1/e2e_pagefed.py` — Shared experiment implementation.
`src/public_e2e_v1/e2e_records.py` — Shared experiment implementation.
`src/public_e2e_v1/e2e_retrieval.py` — Shared experiment implementation.
`src/public_e2e_v1/e2e_routing.py` — Shared experiment implementation.
`src/public_e2e_v1/e2e_slot_adapters.py` — Shared experiment implementation.
`src/public_e2e_v1/e2e_slots.py` — Shared experiment implementation.
`src/public_e2e_v1/e2e_stats.py` — Shared experiment implementation.
`data/annotation/gold_v1/tasks_gold_v1.json` — 45 evaluation tasks, candidate selections, client assignments and scenarios; step 7.
`data/annotation/gold_v1/fedfinrag_resolve_gold_v1.json` — 45 evaluation annotations with annotation and adjudication provenance; step 7.
`data/annotation/gold_v1/NOTICE` — Apache-2.0 image source attribution.
`data/annotation/gold_v1/media/` — 80 candidate page PNG images; step 7.
`data/annotation/gold_v1/LICENSE.md` — Apache-2.0 page renderings and CC BY 4.0 evaluation material.
`analysis/controls/global_selection/aggregate.py` — Global page ranking with pooled evidence selection; Tables 3 and 4.
`analysis/controls/global_selection/d_common.py` — Global page ranking with pooled evidence selection; Tables 3 and 4.
`analysis/controls/global_selection/protocol_fairness_v1_D_global_selection.json` — Global page ranking with pooled evidence selection; Tables 3 and 4.
`analysis/controls/global_selection/registered/gold_v3_results.json` — SERF result copy for aggregation comparisons; Tables 3 and 4.
`analysis/controls/global_selection/registered/v3_cells.csv` — SERF result copy for aggregation comparisons; Tables 3 and 4.
`analysis/controls/global_selection/run_composites.py` — Global page ranking with pooled evidence selection; Tables 3 and 4.
`analysis/controls/global_selection/run_evaluation.py` — Global page ranking with pooled evidence selection; Tables 3 and 4.
`analysis/controls/global_selection/run_public.py` — Global page ranking with pooled evidence selection; Tables 3 and 4.
`analysis/controls/global_selection/tests/test_global_selection.py` — Synthetic global selection unit tests.
