"""Read public references and scores; plan only training paired selections."""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True
import time
from collections import defaultdict
from unittest.mock import patch

import numpy as np

from b_common import (SCENARIOS, Writer, check_constants, owners_of, parser, public_gate,
                      read_json, row_key, score_statistics, setup, sha, stream,
                      unique_index, verify_centralized, write_json)


def public_metrics(record, pages):
    """Use exactly the public runner's unit mapping and evidence metric."""
    from run_e2e import gold_units, to_units
    from e2e_common import prf
    scores = prf(to_units(pages, record.gold_unit), gold_units(record, record.gold_unit))
    return {'gold_unit': record.gold_unit, 'evidence_recall': scores['recall'],
            'evidence_precision': scores['precision'], 'evidence_f1': scores['f1'],
            'success': scores['recall'] > 0}


def query_views(record, position, index, slots, view, gold_protocol, blend):
    from e2e_slot_adapters import QueryViews
    start, stop = slots.spans[position]
    return QueryViews(
        full_page=index.scores_for_rows(record.uid, view.all_rows),
        slot_page=slots.page_scores(slice(start, stop), blend),
        rows_by_client=view.rows_by_client, num_clients=view.num_clients,
        claim_query_weight=gold_protocol['fedresolve']['claim_query_weight'],
        full_query_weight=gold_protocol['fedresolve']['full_query_weight'],
        temperature=gold_protocol['fedresolve']['global_temperature'])


def recover_selection(views, online, budget, protocol, called):
    """Reuse v3 selection on recorded calls; no test routing is executed."""
    import e2e_v3_adapter as adapter
    base = {'clients': list(called), 'trace': {'critical_missing_slots': []}}
    with patch.object(adapter, 'fedresolve_adapter', return_value=base):
        return adapter.fedresolve_v3_adapter(views, online, budget, protocol)


def selection_cost(record, pages, rows, called, index, view, texts, protocol):
    from e2e_common import package_bytes
    from e2e_retrieval import CERTIFICATE_BYTES_PER_CALL
    limit = protocol['evidence']['page_text_char_limit']
    calls = len(called)
    cert = CERTIFICATE_BYTES_PER_CALL * calls
    evidence_bytes = sum(len(texts.get(p, '')[:limit].encode('utf-8')) for p in pages)
    return {'client_calls': calls, 'selected_pages': len(pages),
            'serialized_request_bytes': package_bytes({'query': record.query, 'page_budget': 2}) * calls,
            'serialized_response_bytes': sum(package_bytes({
                'client': int(view.page_client[r]), 'page_uid': index.page_uids[r],
                'certificate': {'score': 0.0, 'available': True},
                'text': texts.get(index.page_uids[r], '')[:limit]}) for r in rows),
            'certificate_bytes': cert, 'protocol_bytes': cert + evidence_bytes}


def filtered_plan(path, configs, allowed_uids, methods=None, scenarios=None, budgets=(2,)):
    return unique_index(r for r in stream(path)
                        if r['config'] in configs and r['uid'] in allowed_uids[r['config']]
                        and int(r['budget']) in budgets
                        and (methods is None or r['method'] in methods)
                        and (scenarios is None or r['scenario'] in scenarios))


def require_consolidated_corpus(root, dataset):
    """Stop before corpus helpers can fall back to creating per-document caches."""
    cache = root / 'data/cache/public_e2e_v1'
    for name in (f'{dataset}_pages.json', f'{dataset}_page_text.jsonl.gz'):
        if not (cache / name).is_file():
            raise FileNotFoundError(f'Required read-only consolidated cache is missing: {name}')


def run_dataset(args, dataset, protocol, gold_protocol, outputs):
    from e2e_records import load_records
    from e2e_retrieval import ClientPageView, PageIndex
    from run_paired_plan import SlotQueries, online_mask
    from fedresolve_core import build_federation, split_records
    from e2e_v3_adapter import fedresolve_v3_adapter
    from e2e_corpus import page_text_map
    from e2e_slot_adapters import _critical_missing

    require_consolidated_corpus(args.root, dataset)
    owner_writer, paired_writer, scores_writer = outputs
    records, _, inventory = load_records(dataset)
    positions = {r.uid: i for i, r in enumerate(records)}
    by_uid = {r.uid: r for r in records}
    blend = protocol['retriever']['blend_weight'][dataset]
    dense_weight = protocol['retriever']['dense_weight'][dataset]
    index = PageIndex.load(dataset, blend)
    if dense_weight > 0:
        index.attach_dense(dataset, dense_weight)
    slots = SlotQueries(dataset, index, dense_weight)
    texts = page_text_map(dataset)
    federation_inventory = sorted(set(index.doc_key_list) | set(inventory))
    settings, test_uids = {}, {}
    counts = defaultdict(int)
    for clients in protocol['client_scales']:
        for track in protocol['tracks']:
            config = f'c{clients}_{track}'
            seed = protocol['seed'] + clients + (0 if dataset == 'finrag' else 50000)
            federation = build_federation(federation_inventory, num_clients=clients, seed=seed)
            splits = split_records(records, track=track, seed=seed + 101)
            for name in ('train', 'test'):
                splits[name] = splits[name][:args.limit] if args.limit else splits[name]
            view = ClientPageView(index, federation.doc_to_client, clients)
            settings[config] = (federation, splits, view)
            test_uids[config] = {r.uid for r in splits['test']}

    art = args.root / 'artifacts'
    paths = {
        'public_reference': art / 'public_e2e_v1' / f'{dataset}_slot' / 'plan.jsonl',
        'v3_reference': art / 'serf_v3' / f'{dataset}_test' / 'plan.jsonl',
        'paired': art / 'public_paired_v1' / dataset / 'plan.jsonl'}
    ref1 = filtered_plan(paths['public_reference'], settings, test_uids,
                         ('Centralized',), ('full',), (1, 2, 3))
    ref2 = filtered_plan(paths['v3_reference'], settings, test_uids,
                         ('Centralized',), ('full',), (1, 2, 3))
    expected_refs = {(cfg, uid, 'full', b, 'Centralized')
                     for cfg, uids in test_uids.items() for uid in uids for b in (1, 2, 3)}
    if set(ref1) != expected_refs or set(ref2) != expected_refs:
        raise ValueError('Missing Centralized reference rows or different query sets')
    paired = filtered_plan(paths['paired'], settings, test_uids, scenarios=SCENARIOS)
    serf = filtered_plan(paths['v3_reference'], settings, test_uids,
                         ('FedResolve-v3',), SCENARIOS)
    expected_serf = {(cfg, uid, sc, 2, 'FedResolve-v3') for cfg, uids in test_uids.items()
                     for uid in uids for sc in SCENARIOS}
    if set(serf) != expected_serf:
        raise ValueError('SERF-v3 paired scenarios are required to recover its test selection')
    paired_methods = set(protocol['methods']) - {'Oracle-evidence'}
    expected_paired = {(cfg, uid, sc, 2, method) for cfg, uids in test_uids.items()
                       for uid in uids for sc in SCENARIOS for method in paired_methods}
    if set(paired) != expected_paired:
        raise ValueError('Paired plan is missing a protocol method or scenario')

    for key, row in sorted(ref1.items()):
        view = settings[row['config']][2]
        owners = verify_centralized(row, ref2[key], index, view)
        owner_writer.write({**row, 'variant': 'owner_contacts', 'selected_page_uids': row['evidence_page_uids'], 'owners': owners,
                            'owner_contacts': len(owners), **public_metrics(by_uid[row['uid']], row['evidence_page_uids'])})
        counts['owner_rows'] += 1

    for config, (federation, splits, view) in settings.items():
        for split in ('train', 'test'):
            for number, record in enumerate(splits[split]):
                views = query_views(record, positions[record.uid], index, slots, view, gold_protocol, blend)
                for scenario in SCENARIOS:
                    online = online_mask(record, federation, scenario, view.num_clients, protocol['seed'])
                    stat = score_statistics(views.combined, views.probabilities, online)
                    if split == 'test':
                        scenario_rows = [paired[(config, record.uid, scenario, 2, m)] for m in sorted(paired_methods)]
                        gate = public_gate(scenario_rows)
                        if bool(_critical_missing(views, online, gold_protocol)) != gate:
                            raise ValueError('Recomputed cached-score gate differs from recorded S2G-RAG+avail')
                        v3row = serf[(config, record.uid, scenario, 2, 'FedResolve-v3')]
                        v1row = paired[(config, record.uid, scenario, 2, 'FedResolve')]
                        if v3row['selected_clients'] != v1row['selected_clients'] or v3row['abstain_routing'] != v1row['abstain_routing']:
                            raise ValueError('Recorded v3 routing differs from v1')
                        for row in scenario_rows + [v3row]:
                            paired_writer.write({**row, 'variant': 'recorded', 'selected_page_uids': row['evidence_page_uids'], 'gate': gate,
                                'answerable': scenario == 'unneeded_offline',
                                **public_metrics(record, row['evidence_page_uids'])})
                            counts['paired_rows'] += 1
                        selected = recover_selection(views, online, 2, gold_protocol, v3row['selected_clients'])
                        recovered = [index.page_uids[r] for r in selected['rows']]
                        if not v3row['abstain_routing'] and recovered != v3row['evidence_page_uids']:
                            raise ValueError('Recovered v3 evidence differs when the recorded gate did not abstain')
                    else:
                        selected = fedresolve_v3_adapter(views, online, 2, gold_protocol)
                        gate = bool(selected['trace']['critical_missing_slots'])
                        recovered = [index.page_uids[r] for r in selected['rows']]
                    scores_writer.write({'dataset': dataset, 'config': config, 'method': 'FedResolve-v3',
                        'variant': 'online_rule_inputs', 'budget': 2, 'scenario': scenario, 'uid': record.uid,
                        'split': split, 'selected_clients': [int(c) for c in selected['clients']],
                        'evidence_page_uids': recovered, 'selected_page_uids': recovered,
                        'empty_evidence': not recovered, 'gate': gate,
                        'answerable': scenario == 'unneeded_offline', **stat,
                        'cost': selection_cost(record, recovered, selected['rows'], selected['clients'],
                                               index, view, texts, protocol),
                        **public_metrics(record, recovered)})
                    counts[f'{split}_score_rows'] += 1
                if (number + 1) % 100 == 0:
                    print(f'{dataset} {config} {split}: {number + 1} queries', flush=True)
    return {'counts': dict(counts), 'sources': {k: {'path': str(p), 'sha256': sha(p)} for k, p in paths.items()},
            'settings': {k: {name: len(splits[name]) for name in ('train', 'test')}
                         for k, (_, splits, _) in settings.items()}}


def main():
    p = parser(__doc__)
    p.add_argument('--dataset', choices=('finrag', 'visdom', 'both'), default='both')
    args = p.parse_args()
    setup(args)
    protocol = read_json(args.root / 'configs/public_e2e_v1/protocol_v3.json')
    gold_protocol = read_json(args.root / 'configs/formal_gold_v1_protocol.json')
    check_constants(protocol, gold_protocol)
    started = time.time()
    outputs = [Writer(args.out / name) for name in ('public_owners.jsonl', 'public_paired.jsonl', 'public_scores.jsonl')]
    summary = {'smoke': args.limit is not None, 'limit': args.limit, 'datasets': {}}
    try:
        for dataset in (('finrag', 'visdom') if args.dataset == 'both' else (args.dataset,)):
            summary['datasets'][dataset] = run_dataset(args, dataset, protocol, gold_protocol, outputs)
    finally:
        for writer in outputs:
            writer.close()
    summary['elapsed_seconds'] = time.time() - started
    summary['design_sha256'] = sha(__import__('pathlib').Path(__file__).with_name('protocol_fairness_v1_B_reference.json'))
    summary['protocol_sha256'] = sha(args.root / 'configs/public_e2e_v1/protocol_v3.json')
    summary['frozen_constants_sha256'] = sha(args.root / 'configs/formal_gold_v1_protocol.json')
    summary['code_sha256'] = {str(__file__): sha(__file__),
                              'b_common.py': sha(__import__('b_common').__file__)}
    write_json(args.out / 'public_manifest.json', summary)
    print('Public read-only extraction and training paired planning complete.', flush=True)


if __name__ == '__main__':
    main()
