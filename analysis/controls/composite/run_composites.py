"""Fit training-only thresholds, then plan and score test composites on the CPU."""
from __future__ import annotations

import json
import time

import numpy as np

from composite_common import (CompositeQueries, arguments, evidence_metrics, fit_threshold,
                              fresh, gates, make_views, online_mask, online_statistic,
                              parser, protocol_digest, read_json, read_jsonl, route,
                              selection_variants, settings, write_json, write_jsonl)


def cost_fields(query, budget, called, proposed_rows, evidence_uids, index, client_view, texts, protocol):
    from e2e_common import package_bytes
    from e2e_retrieval import CERTIFICATE_BYTES_PER_CALL
    calls = len(called)
    limit = protocol['cost']['page_text_char_limit']
    evidence_bytes = sum(len(texts.get(u, '')[:limit].encode('utf-8')) for u in evidence_uids)
    response_bytes = sum(package_bytes({
        'client': int(client_view.page_client[r]), 'page_uid': index.page_uids[r],
        'certificate': {'score': 0.0, 'available': True},
        'text': texts.get(index.page_uids[r], '')[:limit]}) for r in proposed_rows)
    return {'client_calls': calls, 'selected_pages': len(evidence_uids),
            'serialized_request_bytes': package_bytes({'query': query, 'page_budget': budget}) * calls,
            'serialized_response_bytes': response_bytes if evidence_uids else CERTIFICATE_BYTES_PER_CALL * calls,
            'certificate_bytes': CERTIFICATE_BYTES_PER_CALL * calls,
            'protocol_bytes': CERTIFICATE_BYTES_PER_CALL * calls + evidence_bytes}


def build_rows(c, views, online, scenario, budget, method, variant, result,
               theta, index, client_view, texts, protocol, elapsed):
    decisions, critical = gates(result, views, online, protocol, theta)
    proposed = [index.page_uids[r] for r in result['rows']]
    ranked = np.argsort(-views.evidence_score)[:protocol['cost']['ranked_candidate_depth']]
    statistic = online_statistic(views, online)
    for policy, abstain in decisions.items():
        evidence = [] if abstain else proposed
        called = [int(c) for c in result['clients']]
        cost = cost_fields(c['query'], budget, called, result['rows'], evidence,
                           index, client_view, texts, protocol)
        metrics = evidence_metrics(c, evidence)
        yield {'row_key': '|'.join((c['dataset'], c['config'], method, variant, policy,
                                   f'b{budget}', scenario, c['uid'])),
               'dataset': c['dataset'], 'config': c['config'], 'portion': 'test',
               'method': method, 'variant': variant, 'policy': policy, 'budget': budget,
               'scenario': scenario, 'uid': c['uid'], 'lang': c['lang'],
               'clients': client_view.num_clients, 'track': c['config'].split('_', 1)[1],
               'selected_clients': called, 'selected_page_uids': list(evidence),
               'evidence_page_uids': list(evidence), 'proposed_page_uids': proposed,
               'ranked_page_uids': [index.page_uids[int(r)] for r in ranked],
               'abstain_routing': bool(abstain), 'availability_decision': bool(critical),
               'online_only_score': statistic if np.isfinite(statistic) else None,
               'theta': theta, 'routing_reason': json.dumps(result['trace'], ensure_ascii=False),
               'critical_missing_slots': critical, 'offline_clients': np.flatnonzero(~online).tolist(),
               'candidate_pages': sum(client_view.rows_by_client[c].size for c in called),
               'gold_unit': c['gold_unit'], 'cross_client_gold': True,
               'generation_key': None, 'evidence_labels': [f'P{i+1}' for i in range(len(evidence))],
               'cost': cost, 'calls': cost['client_calls'], 'client_calls': cost['client_calls'],
               'selected_pages': cost['selected_pages'], 'retrieval_ms': elapsed, **metrics}


def evaluate_dataset(dataset, args, protocol, composites):
    from e2e_corpus import page_text_map
    from e2e_records import load_records
    from e2e_retrieval import ClientPageView, PageIndex
    records, _, inventory = load_records(dataset)
    blend = protocol['retriever']['blend_weight'][dataset]
    dense_weight = protocol['retriever']['dense_weight'][dataset]
    index = PageIndex.load(dataset, blend)
    if dense_weight > 0:
        index.attach_dense(dataset, dense_weight)
    queries = CompositeQueries(args.out / 'queries' / dataset, index, dense_weight)
    if queries.manifest['protocol_sha256'] != protocol_digest(args.protocol) or queries.manifest['limit'] != args.limit:
        raise ValueError('Encoded query protocol or smoke limit differs')
    # Rebuild the same federation from the original records and page inventory.
    configurations = {}
    for config, seed, federation, splits in settings(dataset, protocol, records, index, inventory):
        configurations[config] = (federation, ClientPageView(index, federation.doc_to_client,
                                                          federation.num_clients), splits)
    def view_for(c):
        federation, client_view, splits = configurations[c['config']]
        allowed = {r.uid for r in splits[c['portion']]}
        if not set(c['uids']).issubset(allowed):
            raise ValueError('Composite crosses a protocol portion')
        for part in c['parts']:
            if {federation.doc_to_client[d] for d in part['documents']} != {part['client']}:
                raise ValueError('Rebuilt federation disagrees with composite definition')
        full, slots = queries.scores(c['uid'], blend)
        return make_views(full, slots, client_view, protocol), client_view
    training = []
    for c in composites:
        if c['portion'] != 'train':
            continue
        views, client_view = view_for(c)
        answer = online_mask(c, 'other_offline', client_view.num_clients, protocol['seed'])
        missing = online_mask(c, 'part_offline', client_view.num_clients, protocol['seed'])
        result = route('FedResolve-v3', 'native', views, answer, 2, protocol,
                       client_view, index, c)
        complete = evidence_metrics(c, [index.page_uids[r] for r in result['rows']])['complete_evidence']
        a, b = online_statistic(views, answer), online_statistic(views, missing)
        if not np.isfinite(a) or not np.isfinite(b):
            raise ValueError('Training scenario has no eligible online score')
        training.append({'dataset': dataset, 'config': c['config'], 'uid': c['uid'],
                         'answerable_score': a, 'unanswerable_score': b,
                         'answerable_complete': complete})
    fitted = fit_threshold(training, protocol)
    fitted['dataset'] = dataset
    fitted['counts_by_setting'] = {config: sum(r['config'] == config for r in training)
                                  for config in configurations}
    fitted['protocol_sha256'] = protocol_digest(args.protocol)
    write_json(args.out / 'thresholds' / f'{dataset}.json', fitted)
    write_jsonl(args.out / 'thresholds' / f'{dataset}_training.jsonl', training)
    texts = page_text_map(dataset)
    written = 0
    with fresh(args.out / 'plans' / dataset / 'plan.jsonl').open('x', encoding='utf-8') as f:
        for c in composites:
            if c['portion'] != 'test':
                continue
            views, client_view = view_for(c)
            for budget in protocol['budgets']:
                scenarios = ['full', 'part_offline', 'other_offline'] if budget == 2 else ['full']
                for scenario in scenarios:
                    online = online_mask(c, scenario, client_view.num_clients, protocol['seed'])
                    for method, variant in selection_variants(protocol):
                        start = time.perf_counter()
                        result = route(method, variant, views, online, budget, protocol,
                                       client_view, index, c)
                        elapsed = 1000 * (time.perf_counter() - start)
                        if any(not online[c] for c in result['clients']):
                            raise ValueError('Adapter called an offline client')
                        if len(result['rows']) > budget or len(set(result['rows'])) != len(result['rows']):
                            raise ValueError('Adapter violated unique page budget')
                        for row in build_rows(c, views, online, scenario, budget, method, variant,
                                              result, fitted['theta'], index, client_view,
                                              texts, protocol, elapsed):
                            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                            written += 1
            if written % 1320 == 0:
                print({'stage': 'plan', 'dataset': dataset, 'rows': written}, flush=True)
    return {'dataset': dataset, 'plan_rows': written, 'theta': fitted['theta']}


def main():
    args = arguments(parser(__doc__))
    protocol = read_json(args.protocol)
    construction = read_json(args.out / 'construction.json')
    if construction['protocol_sha256'] != protocol_digest(args.protocol) or construction['limit'] != args.limit:
        raise ValueError('Construction protocol or limit differs')
    fresh(args.out / 'run_summary.json')
    for dataset in protocol['datasets']:
        for path in (args.out / 'thresholds' / f'{dataset}.json',
                     args.out / 'thresholds' / f'{dataset}_training.jsonl',
                     args.out / 'plans' / dataset / 'plan.jsonl'):
            fresh(path)
    started = time.time()
    composites = read_jsonl(args.out / 'composites.jsonl')
    summaries = []
    for dataset in protocol['datasets']:
        summaries.append(evaluate_dataset(dataset, args, protocol,
                                         [c for c in composites if c['dataset'] == dataset]))
    write_json(args.out / 'run_summary.json', {'datasets': summaries,
               'limit': args.limit, 'elapsed_seconds': time.time() - started,
               'protocol_sha256': protocol_digest(args.protocol)})


if __name__ == '__main__':
    main()
