"""Score each public test query once; reuse the fixed global pool for every variant."""
from __future__ import annotations
import time
import numpy as np
import d_common as d


def run_dataset(args, dataset, protocol, gold, design, sources):
    import run_v3_plan
    import e2e_slot_adapters
    import e2e_v3_adapter
    import fedresolve_core
    import e2e_records
    import e2e_retrieval
    from e2e_corpus import page_text_map
    from public_rows import query_views, require_consolidated_corpus, public_metrics
    from b_common import public_gate, owners_of
    a_public = d.earlier(args, 'attribution', 'run_public.py')
    d.pin(design, run_v3_plan=run_v3_plan, e2e_slot_adapters=e2e_slot_adapters,
          e2e_v3_adapter=e2e_v3_adapter, fedresolve_core=fedresolve_core,
          e2e_records=e2e_records, e2e_retrieval=e2e_retrieval)
    require_consolidated_corpus(args.root, dataset)
    records, _, inventory = e2e_records.load_records(dataset)
    positions = {r.uid: i for i, r in enumerate(records)}
    blend = protocol['retriever']['blend_weight'][dataset]
    dense = protocol['retriever']['dense_weight'][dataset]
    index = e2e_retrieval.PageIndex.load(dataset, blend)
    if dense > 0:
        index.attach_dense(dataset, dense)
    slots = run_v3_plan.SlotQueries(dataset, index, dense)
    texts = page_text_map(dataset)
    configurations = {}
    for clients in protocol['client_scales']:
        for track in protocol['tracks']:
            seed = protocol['seed'] + clients + (0 if dataset == 'finrag' else 50000)
            fed = fedresolve_core.build_federation(sorted(set(index.doc_key_list) | set(inventory)),
                                                  num_clients=clients, seed=seed)
            splits = fedresolve_core.split_records(records, track=track, seed=seed + 101)
            selected = splits['test'] if args.limit is None else splits['test'][:args.limit]
            view = e2e_retrieval.ClientPageView(index, fed.doc_to_client, clients)
            configurations[f'c{clients}_{track}'] = (fed, selected, view)
    allowed = {cfg: {r.uid for r in rows} for cfg, (_, rows, _) in configurations.items()}

    def load(path, predicate):
        return [r for r in d.stream(sources.use(path)) if r['config'] in allowed
                and r['uid'] in allowed[r['config']] and predicate(r)]

    art = args.root / 'artifacts'
    arows = d.index_rows(load(args.a_out / (dataset + '.jsonl'),
                              lambda r: r['method'] == 'FedResolve' and r['variant'] == '+sel'))
    original = load(art / 'serf_v3' / (dataset + '_test') / 'plan.jsonl',
                    lambda r: r['method'] in ('FedResolve-v3', 'Centralized'))
    v3 = d.index_rows(r for r in original if r['method'] == 'FedResolve-v3')
    central = d.index_rows(r for r in original if r['method'] == 'Centralized')
    paired = load(art / 'public_paired_v1' / dataset / 'plan.jsonl', lambda r: True)
    by_scenario = {}
    for r in paired:
        by_scenario.setdefault(d.key(r), []).append(r)
    bowners = d.index_rows(load(args.b_out / 'public_owners.jsonl', lambda r: r['dataset'] == dataset))
    bpaired_rows = load(args.b_out / 'public_paired.jsonl', lambda r: r['dataset'] == dataset)
    bserf = d.index_rows(r for r in bpaired_rows if r['method'] == 'FedResolve-v3')
    bcentral = d.index_rows(r for r in bpaired_rows if r['method'] == 'Centralized')
    rows, refs = [], []
    for cfg, (fed, records_test, view) in configurations.items():
        for record in records_test:
            t0 = time.perf_counter()
            views = query_views(record, positions[record.uid], index, slots, view, gold, blend)
            elapsed = 1000 * (time.perf_counter() - t0)
            for scenario in ('full', 'unneeded_offline', 'qrel_missing'):
                online = run_v3_plan.online_mask(record, fed, scenario, view.num_clients, protocol['seed'])
                scenario_key = cfg, record.uid, scenario, 2
                # The original plans store scenario IDs rather than masks. Use their
                # unchanged mask builder and verify explicit masks when present.
                for r in by_scenario[scenario_key]:
                    if 'offline_clients' in r and r['offline_clients'] != np.flatnonzero(~online).tolist():
                        raise ValueError('Recorded public offline set differs')
                gate = public_gate(by_scenario[scenario_key])
                d.verify_gate(views, online, gold, gate)
                budgets = (1, 2, 3) if scenario == 'full' else (2,)
                for budget in budgets:
                    k = cfg, record.uid, scenario, budget
                    serf = v3[k]
                    computed = {**serf, **a_public.metric_fields(record, serf['evidence_page_uids'])}
                    if scenario == 'full':
                        recorded = arows[k]
                        d.same(computed, recorded, ('selected_clients', 'evidence_page_uids', 'abstain_routing', 'cost',
                                                    'evidence_recall', 'evidence_precision', 'complete_evidence'))
                        ref = d.reference(recorded, 'public')
                    else:
                        recorded = bserf[k]
                        d.same(computed, recorded, ('selected_clients', 'evidence_page_uids', 'abstain_routing',
                                                   'cost', 'evidence_recall', 'evidence_precision'))
                        d.same({'gate': gate}, recorded, ('gate',))
                        ref = d.reference({**recorded, **a_public.metric_fields(record, recorded['evidence_page_uids'])},
                                          'public_paired')
                        ref['answerable'] = scenario == 'unneeded_offline'
                    refs.append(ref)
                    for variant, result in d.public_selections(views, online, budget, gold, view.page_client, gate):
                        row = d.public_row(a_public, dataset, cfg, record, view, index, texts,
                                           protocol, variant, budget, scenario, result, gate, elapsed,
                                           views.evidence_score)
                        row.update(panel='public' if scenario == 'full' else 'public_paired',
                                   offline_clients=np.flatnonzero(~online).tolist())
                        if scenario != 'full':
                            row['answerable'] = scenario == 'unneeded_offline'
                        if variant == 'base':
                            d.same(row, central[k], ('evidence_page_uids', 'selected_clients', 'cost', 'abstain_routing'))
                            if scenario == 'full':
                                d.same(row, bowners[k], ('owner_contacts', 'evidence_recall'))
                            else:
                                b = bcentral[k]
                                d.same(row, b, ('evidence_page_uids', 'evidence_recall', 'abstain_routing'))
                                # Job B's added decision is applied to its recorded
                                # Centralized rows; no selection is reconstructed.
                                abstain = bool(b['abstain_routing']) or gate
                                pages = [] if abstain else b['evidence_page_uids']
                                owners = owners_of(pages, index, view)
                                refs.append(d.reference({**b, **a_public.metric_fields(record, pages),
                                    'selected_page_uids': pages, 'evidence_page_uids': pages,
                                    'abstain_routing': abstain, 'owner_contacts': len(owners),
                                    'answerable': scenario == 'unneeded_offline'}, 'public_paired',
                                    'Centralized + test'))
                        if scenario == 'full' or variant != 'base':
                            rows.append(row)
            print(dataset, cfg, record.uid, flush=True)
    return rows, refs, {cfg: len(v[1]) for cfg, v in configurations.items()}


def main():
    p = d.parser(__doc__)
    p.add_argument('--dataset', choices=('finrag', 'visdom'), required=True)
    args = p.parse_args()
    design = d.setup(args)
    sources = d.Sources()
    public, gold = d.checked_protocol(args, design, sources)
    started = time.time()
    rows, refs, cohorts = run_dataset(args, args.dataset, public, gold, design, sources)
    d.finish(args, args.dataset, rows, refs, sources, started, cohorts=cohorts)


if __name__ == '__main__':
    main()
