"""Run the evaluation comparisons from cached embeddings without generation."""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True
import importlib.util
import math
import os
import time
from pathlib import Path

import numpy as np

from b_common import (Writer, balanced, parser, read_json, score_statistics, setup,
                      sha, write_json)

RUNNER_SHA = '7cf78be0360ab12ab4e4f48a5cf8137f64ce3a9adae051f9d3b3ab09ebe4e1fe'
V3_SHA = 'f30f5f65db1d37f7cdc90a38d3ed9f2c385703624c22f675f5f467494c4f8eec'
PROTOCOL_SHA = 'dd22af39e288cf9e686e4c186e5828ed3e18b82ce737acb7da776146382e29f6'


def load_module(name, path, digest):
    if sha(path) != digest:
        raise ValueError(f'Evaluation module hash mismatch: {name}')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def find_module(root, explicit, relative_paths):
    if explicit is not None:
        return explicit
    for rel in relative_paths:
        if (root / rel).is_file():
            return root / rel
    raise FileNotFoundError('Existing fixed module not found; supply the documented module path option')


def build_inputs(tasks, images, queries, ie, qe, root):
    """Candidate metadata and cached embeddings only; never annotations."""
    ie = np.asarray(ie, dtype=np.float64)
    qe = np.asarray(qe, dtype=np.float64)
    ie = ie / np.maximum(np.linalg.norm(ie, axis=1, keepdims=True), 1e-12)
    qe = qe / np.maximum(np.linalg.norm(qe, axis=1, keepdims=True), 1e-12)
    ii = {r['candidate_id']: i for i, r in enumerate(images)}
    qi = {r['query_key']: i for i, r in enumerate(queries)}
    if len(ii) != len(images) or len(qi) != len(queries):
        raise ValueError('Duplicate embedding cache identifiers')
    sizes = {}
    for image in images:
        path = Path(image['path'])
        if not path.is_absolute():
            path = root / path
        sizes[image['candidate_id']] = path.stat().st_size
    out = {}
    for task in tasks:
        cid = task['canonical_id']
        candidates = [r['candidate_id'] for r in task['candidate_pages']]
        m = ie[[ii[c] for c in candidates]]
        slots = sorted((i for k, i in qi.items() if k.startswith(f'{cid}::') and not k.endswith('::full')),
                       key=lambda i: queries[i]['slot_id'])
        if not slots:
            raise ValueError('Missing cached sub-question embeddings')
        out[cid] = {'canonical_id': cid, 'candidate_ids': candidates,
                    'candidate_clients': [int(r['client_id']) for r in task['candidate_pages']],
                    'candidate_bytes': [sizes[c] for c in candidates],
                    'full_scores': m @ qe[qi[f'{cid}::full']],
                    'slot_scores': np.vstack([m @ qe[i] for i in slots]), 'image_similarity': m @ m.T}
    return out


def metric_boundary(rf, output, inputs, gold, task, views, online, protocol, scenario_id=None):
    """All label access lives here and in the unchanged metric/oracle functions."""
    cert = protocol['federation']['certificate_bytes_per_client_call']
    metrics = rf.item_metrics(output, gold, inputs, cert)
    if scenario_id is None:
        # The oracle is used only as a metric labeling function, never as a router input.
        label_output = rf.oracle_client_adapter(inputs, views, online, 2, gold)
        answerable = not bool(label_output['abstain'])
    else:
        assessments = gold['participation_assessments']
        scenarios = task['participation_scenarios']
        if len(assessments) != len(scenarios):
            raise ValueError('Evaluation participation scenario count mismatch')
        labels = {}
        for sc, label in zip(scenarios, assessments):
            if sc['scenario_id'] != label['scenario_id']:
                raise ValueError('Evaluation participation scenario order mismatch')
            labels[sc['scenario_id']] = label['answerability'] == 'answerable'
        answerable = labels[scenario_id]
    return {'answerable': answerable, 'success': bool(metrics['complete_client_set']), **metrics}


def oracle_at_metric_boundary(rf, method, inputs, views, online, budget, gold):
    if method == 'Oracle client set':
        return rf.oracle_client_adapter(inputs, views, online, budget, gold)
    return rf.oracle_evidence_adapter(inputs, online, gold)


def owners(inputs, evidence):
    mapping = dict(zip(inputs['candidate_ids'], inputs['candidate_clients']))
    return sorted({int(mapping[c]) for c in evidence})


def row_for(output, inputs, metrics, protocol, method, scenario, budget, gate=False, **extra):
    cert = protocol['federation']['certificate_bytes_per_client_call']
    calls = int(output['client_calls'])
    payload = sum(inputs['candidate_bytes'][inputs['candidate_ids'].index(c)] for c in output['evidence_ids'])
    return {'dataset': 'evaluation_set', 'config': 'c8_evaluation', 'method': method,
            'variant': 'plus_availability_test' if method.endswith('+avail') else 'recorded',
            'budget': budget, 'scenario': scenario, 'uid': inputs['canonical_id'],
            'selected_clients': [int(c) for c in output['queried_clients']],
            'evidence_page_uids': list(output['evidence_ids']), 'selected_page_uids': list(output['evidence_ids']),
            'owners': owners(inputs, output['evidence_ids']), 'owner_contacts': len(owners(inputs, output['evidence_ids'])),
            'abstain_routing': bool(output['abstain']), 'abstain': bool(output['abstain']), 'gate': gate,
            'cost': {'client_calls': calls, 'selected_pages': len(output['evidence_ids']),
                     'certificate_bytes': calls * cert, 'protocol_bytes': calls * cert + payload,
                     'serialized_request_bytes': None, 'serialized_response_bytes': None},
            'cost_note': 'Evaluation runner measures certificate plus image payload bytes; serialized byte fields are unavailable.',
            **metrics, **extra}


def v3_output(rf, v3_select, inputs, views, online, protocol):
    base = rf.ADAPTERS['FedResolve'](inputs, views, online, 2, protocol)
    evidence = v3_select(inputs, base['queried_clients'], 2, protocol, views['evidence_score'])
    gate = bool(base['trace']['critical_missing_slots'])
    return {**base, 'evidence_ids': evidence, 'abstain': gate or not evidence}, gate


def close(a, b):
    if isinstance(a, dict):
        return isinstance(b, dict) and set(a) == set(b) and all(close(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(close(x, y) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return (math.isnan(a) and math.isnan(b)) or abs(a - b) <= 1e-12 * max(1, abs(b))
    return a == b


def reproduce(rf, tasks, inputs_by_id, gold_by_id, protocol, recorded, enumeration, limited):
    """Check paired decisions before new results; check full summaries in a full run."""
    checked = 0
    for task in tasks:
        cid = task['canonical_id']
        inputs, gold = inputs_by_id[cid], gold_by_id[cid]
        views = rf.score_views(inputs, protocol)
        for scenario in task['participation_scenarios']:
            online = np.zeros(8, dtype=bool)
            online[scenario['online_clients']] = True
            for method in tuple(rf.ADAPTERS) + ('Oracle client set',):
                output = (rf.ADAPTERS[method](inputs, views, online, 2, protocol) if method in rf.ADAPTERS
                          else oracle_at_metric_boundary(rf, method, inputs, views, online, 2, gold))
                metrics = metric_boundary(rf, output, inputs, gold, task, views, online, protocol, scenario['scenario_id'])
                want = next(d for d in recorded['participation_main_budget'][method]['decisions']
                            if d['scenario_id'] == scenario['scenario_id'] and d['canonical_id'] == cid)
                if bool(output['abstain']) != bool(want['abstain']) or metrics['success'] != bool(want['complete_client_set']):
                    raise ValueError(f'Evaluation paired reproduction failed for {method} {cid}')
                checked += 1
    if limited:
        return {'paired_decisions_checked': checked, 'full_summary_check': 'omitted for smoke'}
    for budget in (1, 2, 3):
        for method in rf.METHODS:
            metrics = []
            for task in tasks:
                cid = task['canonical_id']
                inputs, gold = inputs_by_id[cid], gold_by_id[cid]
                views = rf.score_views(inputs, protocol)
                online = np.ones(8, dtype=bool)
                output = (rf.ADAPTERS[method](inputs, views, online, budget, protocol) if method in rf.ADAPTERS
                          else oracle_at_metric_boundary(rf, method, inputs, views, online, budget, gold))
                metrics.append(rf.item_metrics(output, gold, inputs, protocol['federation']['certificate_bytes_per_client_call']))
            summary = rf.aggregate(metrics, protocol['seed'] + budget * 100 + rf.METHODS.index(method))
            if not close(summary, recorded['summary_by_budget'][str(budget)][method]):
                raise ValueError(f'Evaluation full summary reproduction failed for {method} budget {budget}')
    rows = []
    for task in tasks:
        cid = task['canonical_id']
        inputs, gold = inputs_by_id[cid], gold_by_id[cid]
        views = rf.score_views(inputs, protocol)
        for client in sorted(set(inputs['candidate_clients'])):
            online = np.ones(8, dtype=bool)
            online[client] = False
            output = rf.ADAPTERS['FedResolve'](inputs, views, online, 2, protocol)
            metrics = metric_boundary(rf, output, inputs, gold, task, views, online, protocol)
            rows.append({**metrics, 'abstain': bool(output['abstain'])})
    summary = balanced(rows)
    want = enumeration['all']['FedResolve']
    if len(rows) != 176 or any(summary[k] != want[k] for k in ('answerable', 'answerable_success', 'unanswerable', 'unanswerable_abstain')):
        raise ValueError('Evaluation enumeration reproduction failed')
    return {'paired_decisions_checked': checked, 'full_summaries_checked': 30, 'enumeration': summary}


def main():
    p = parser(__doc__)
    p.add_argument('--runner', type=Path)
    p.add_argument('--gold-v3', type=Path)
    p.add_argument('--cache', type=Path)
    p.add_argument('--recorded', type=Path)
    p.add_argument('--enumeration', type=Path)
    args = p.parse_args()
    setup(args)
    started = time.time()
    runner_path = find_module(args.root, args.runner, ('scripts/run_formal_gold_v1.py',))
    rf = load_module('run_formal_gold_v1', runner_path, RUNNER_SHA)
    v3_path = find_module(args.root, args.gold_v3, ('analysis/gold_v3.py',))
    v3 = load_module('gold_v3', v3_path, V3_SHA)
    protocol_path = args.root / 'configs/formal_gold_v1_protocol.json'
    protocol = read_json(protocol_path)
    gold_path = args.root / 'data/annotation/gold_v1/fedfinrag_resolve_gold_v1.json'
    tasks_path = Path(os.environ.get('SERF_GOLD_TASKS', str(args.root / 'data/annotation/gold_v1/tasks_gold_v1.json')))
    cache = args.cache or Path(os.environ.get('SERF_GOLD_CACHE', str(args.root / 'artifacts/gold_v3/cache')))
    recorded_path = args.recorded or args.root / 'artifacts/formal_gold_v1/formal_results.json'
    enum_path = args.enumeration or args.root / 'artifacts/enumeration/enumeration_summary.json'
    recorded, enumeration = read_json(recorded_path), read_json(enum_path)
    cache_sha = recorded['cache_manifest_sha256']
    for path, digest in ((runner_path, RUNNER_SHA), (protocol_path, PROTOCOL_SHA),
                         (gold_path, protocol['dataset']['gold_json_sha256']),
                         (tasks_path, protocol['dataset']['tasks_json_sha256']), (cache / 'manifest.json', cache_sha)):
        if sha(path) != digest:
            raise ValueError(f'Evaluation input hash mismatch: {path.name}')
    if recorded['protocol_sha256'] != PROTOCOL_SHA or recorded['cache_manifest_sha256'] != cache_sha:
        raise ValueError('Recorded run is not the fixed run')
    manifest = read_json(cache / 'manifest.json')
    for name, digest in manifest['artifacts'].items():
        if not (cache / name).resolve().is_relative_to(cache.resolve()) or sha(cache / name) != digest:
            raise ValueError('Evaluation cache artifact mismatch')
    all_tasks = read_json(tasks_path)
    if len(all_tasks) != 45:
        raise ValueError('Expected 45 fixed tasks')
    tasks = all_tasks[:args.limit] if args.limit else all_tasks
    gold_by_id = {r['canonical_id']: r for r in read_json(gold_path)['annotations']}
    inputs_by_id = build_inputs(tasks, read_json(cache / 'images.json'), read_json(cache / 'queries.json'),
                                np.load(cache / 'image_embeddings.npy'), np.load(cache / 'query_embeddings.npy'), args.root)
    reproduction = reproduce(rf, tasks, inputs_by_id, gold_by_id, protocol, recorded, enumeration, bool(args.limit))
    owner_writer, pair_writer, score_writer = (Writer(args.out / name) for name in
                                              ('evaluation_owners.jsonl', 'evaluation_paired.jsonl', 'evaluation_scores.jsonl'))
    scenario_counts = {'enumerated_176': 0, 'paired_80': 0}
    try:
        for task in tasks:
            cid = task['canonical_id']
            inputs, gold = inputs_by_id[cid], gold_by_id[cid]
            views = rf.score_views(inputs, protocol)
            for budget in (1, 2, 3):
                online = np.ones(8, dtype=bool)
                output = rf.ADAPTERS['Centralized'](inputs, views, online, budget, protocol)
                metrics = metric_boundary(rf, output, inputs, gold, task, views, online, protocol)
                owner_writer.write(row_for(output, inputs, metrics, protocol, 'Centralized', 'full', budget))
            recorded_offline = {}
            scenarios = []
            for sc in task['participation_scenarios']:
                offline = sorted(set(range(8)) - set(sc['online_clients']))
                if len(offline) != 1:
                    raise ValueError('Evaluation paired scenario is not single-client-offline')
                recorded_offline[offline[0]] = sc['scenario_id']
                online = np.zeros(8, dtype=bool)
                online[sc['online_clients']] = True
                scenarios.append(('paired_80', sc['scenario_id'], online, offline[0], sc['scenario_id']))
            for client in sorted(set(inputs['candidate_clients'])):
                online = np.ones(8, dtype=bool)
                online[client] = False
                scenarios.append(('enumerated_176', f'offline_{client}', online, client, recorded_offline.get(client, '')))
            for family, sid, online, client, recorded_sid in scenarios:
                scenario_counts[family] += 1
                central = rf.ADAPTERS['Centralized'](inputs, views, online, 2, protocol)
                serf_v1 = rf.ADAPTERS['FedResolve'](inputs, views, online, 2, protocol)
                serf_v3, gate = v3_output(rf, v3.v3_select, inputs, views, online, protocol)
                stat = score_statistics(views['combined'], views['probabilities'], online)
                extra = {'scenario_family': family, 'scenario_id': sid, 'offline_client': client,
                         'recorded_scenario': recorded_sid}
                gated = {**central, 'abstain': bool(central['abstain']) or gate}
                for method, output in (('Centralized', central), ('Centralized+avail', gated),
                                       ('FedResolve', serf_v1), ('FedResolve-v3', serf_v3)):
                    metrics = metric_boundary(rf, output, inputs, gold, task, views, online, protocol,
                                              sid if family == 'paired_80' else None)
                    pair_writer.write(row_for(output, inputs, metrics, protocol, method, sid, 2, gate,
                                               offline_is_necessary=not metrics['answerable'], **extra))
                raw_output = {**serf_v3, 'abstain': not serf_v3['evidence_ids']}
                metrics = metric_boundary(rf, raw_output, inputs, gold, task, views, online, protocol,
                                          sid if family == 'paired_80' else None)
                score_writer.write(row_for(raw_output, inputs, metrics, protocol, 'FedResolve-v3', sid, 2, gate,
                                           split='evaluation', empty_evidence=not raw_output['evidence_ids'], **stat, **extra))
    finally:
        owner_writer.close()
        pair_writer.close()
        score_writer.close()
    if not args.limit and scenario_counts != {'enumerated_176': 176, 'paired_80': 80}:
        raise ValueError('Scenario counts changed')
    write_json(args.out / 'evaluation_manifest.json', {'smoke': args.limit is not None, 'limit': args.limit,
        'design_sha256': sha(Path(__file__).with_name('protocol_fairness_v1_B_reference.json')),
        'reproduction': reproduction, 'scenario_counts': scenario_counts, 'elapsed_seconds': time.time() - started,
        'sources': {str(path): sha(path) for path in (runner_path, v3_path, protocol_path, tasks_path, gold_path,
                                                    cache / 'manifest.json', recorded_path, enum_path)}})
    print('Evaluation evaluation reproduction and new reference rows complete.', flush=True)


if __name__ == '__main__':
    main()
