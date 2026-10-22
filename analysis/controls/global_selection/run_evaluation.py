"""Drive the fixed metric runner on cached inputs; copy recorded SERF rows."""
from __future__ import annotations
import time
import numpy as np
import d_common as d


def selections(rf, selector, inputs, views, online, budget, protocol, gate, paired=False):
    base = rf.ADAPTERS['Centralized'](inputs, views, online, budget, protocol)
    called = sorted({c for c in inputs['candidate_clients'] if online[c]})
    evidence = selector(inputs, called, budget, protocol, views['evidence_score'])
    for variant in (('+sel', '+sel+test') if paired else ('base', '+sel')):
        out = dict(base)
        if variant != 'base':
            out.update(evidence_ids=evidence, abstain=not evidence)
        if variant == '+sel+test':
            out['abstain'] = bool(out['abstain']) or gate
        out['trace'] = {**out['trace'], 'variant': variant, 'availability_decision': gate}
        yield variant, out


def metric_rows(rf, b_eval, decisions, inputs, annotations, tasks, protocol):
    """Annotations cross only the earlier job's fixed metric boundary."""
    for row in decisions:
        task = tasks[row['uid']]
        online = np.ones(len(protocol['federation']['client_ids']), dtype=bool)
        online[row['offline_clients']] = False
        inp = inputs[row['uid']]
        views = rf.score_views(inp, protocol)
        metrics = b_eval.metric_boundary(rf, row['selection'], inp, annotations[row['uid']], task,
                                         views, online, protocol,
                                         row['scenario'] if row['panel'] == 'evaluation_paired' else None)
        result = b_eval.row_for(row['selection'], inp, metrics, protocol, 'Centralized',
                                row['scenario'], row['budget'], row['availability_decision'])
        result.update({k: v for k, v in row.items() if k != 'selection'})
        result['selected_pages'] = len(result['selected_page_uids'])
        result['cost']['selected_pages'] = result['selected_pages']
        result.update(result['cost'])
        yield result


def run(args, design, sources):
    from evaluation_rows import RUNNER_SHA, V3_SHA, find_module, build_inputs
    # This module is named run_evaluation too. Load Job A explicitly to avoid
    # resolving our entrypoint through the ordinary import name.
    a_eval = d.earlier(args, 'attribution', 'run_evaluation.py')
    b_eval = d.earlier(args, 'reference', 'evaluation_rows.py')
    rf_path = find_module(args.root, args.runner, ('scripts/run_formal_gold_v1.py',))
    rf = d.module('run_formal_gold_v1', sources.use(rf_path), RUNNER_SHA)
    v3_path = a_eval.find_selector(args.root, args.selector, V3_SHA)
    selector = d.module('d_gold_v3', sources.use(v3_path), V3_SHA).v3_select
    protocol_path = sources.use(args.root / 'configs/formal_gold_v1_protocol.json')
    protocol = d.read(protocol_path)
    if d.digest(protocol_path) != design['input_protocol_sha256']['formal_gold_v1_protocol.json']:
        raise ValueError('Fixed evaluation protocol differs')
    cache = args.cache or d.Path(d.os.environ.get('SERF_GOLD_CACHE', str(args.root / 'artifacts/gold_v3/cache')))
    manifest_path = sources.use(cache / 'manifest.json')
    recorded = d.read(sources.use(args.root / 'artifacts/formal_gold_v1/formal_results.json'))
    if d.digest(manifest_path) != recorded['cache_manifest_sha256']:
        raise ValueError('Evaluation cache differs from the recorded run')
    if recorded['protocol_sha256'] != d.digest(protocol_path):
        raise ValueError('Recorded evaluation protocol differs')
    for name, expected in d.read(manifest_path)['artifacts'].items():
        path = cache / name
        if cache.resolve() not in path.resolve().parents or d.digest(sources.use(path)) != expected:
            raise ValueError('Fixed evaluation cache artifact differs')
    task_path = args.tasks or d.Path(d.os.environ.get('SERF_GOLD_TASKS', str(args.root / 'data/annotation/gold_v1/tasks_gold_v1.json')))
    gold_path = args.gold or args.root / 'data/annotation/gold_v1/fedfinrag_resolve_gold_v1.json'
    for path, field in ((task_path, 'tasks_json_sha256'), (gold_path, 'gold_json_sha256')):
        if d.digest(sources.use(path)) != protocol['dataset'][field]:
            raise ValueError('Fixed evaluation input differs: ' + field)
    tasks = d.read(task_path)
    if len(tasks) != 45:
        raise ValueError('Expected 45 fixed questions')
    tasks = tasks if args.limit is None else tasks[:args.limit]
    inputs = build_inputs(tasks, d.read(cache / 'images.json'), d.read(cache / 'queries.json'),
                           np.load(cache / 'image_embeddings.npy'), np.load(cache / 'query_embeddings.npy'), args.root)
    arows = d.index_rows({**r, 'config': 'c8_evaluation', 'dataset': 'evaluation_set'}
                        for r in d.stream(sources.use(args.a_out / 'evaluation.jsonl'))
                        if r['method'] == 'FedResolve' and r['variant'] == '+sel' and r['uid'] in inputs)
    bpaired = list(d.stream(sources.use(args.b_out / 'evaluation_paired.jsonl')))
    paired = {(r['uid'], r['scenario'], r['method']): r for r in bpaired
              if r['scenario_family'] == 'paired_80' and r['uid'] in inputs}
    if len(paired) != sum(r['scenario_family'] == 'paired_80' and r['uid'] in inputs for r in bpaired):
        raise ValueError('Duplicate recorded evaluation scenario')
    bowners = d.index_rows(r for r in d.stream(sources.use(args.b_out / 'evaluation_owners.jsonl'))
                           if r['uid'] in inputs)
    decisions, refs = [], []
    for task in tasks:
        uid = task['canonical_id']
        inp = inputs[uid]
        views = rf.score_views(inp, protocol)
        scenarios = [('full', np.ones(8, dtype=bool), (2, 3), False)]
        for sc in task['participation_scenarios']:
            online = np.zeros(8, dtype=bool)
            online[sc['online_clients']] = True
            scenarios.append((sc['scenario_id'], online, (2,), True))
        for scenario, online, budgets, is_pair in scenarios:
            gate = False if not is_pair else paired[uid, scenario, 'FedResolve-v3']['gate']
            gate = d.verify_gate(views, online, protocol, gate)
            if is_pair:
                for method in ('FedResolve-v3', 'Centralized+avail', 'Centralized'):
                    rec = paired[uid, scenario, method]
                    if rec['offline_client'] not in np.flatnonzero(~online):
                        raise ValueError('Evaluation recorded offline set differs')
                    d.same({'gate': gate}, rec, ('gate',))
            for budget in budgets:
                ref = paired[uid, scenario, 'FedResolve-v3'] if is_pair else arows['c8_evaluation', uid, 'full', budget]
                if bool(ref['abstain_routing']) != (gate or not ref['selected_page_uids']):
                    raise ValueError('Recorded evaluation SERF gate differs')
                if int(ref['client_calls']) != len(ref['selected_clients']):
                    raise ValueError('Recorded evaluation SERF calls differ')
                refs.append(d.reference(ref, 'evaluation_paired' if is_pair else 'evaluation'))
                if is_pair:
                    refs.append(d.reference(paired[uid, scenario, 'Centralized+avail'],
                                            'evaluation_paired', 'Centralized + test'))
                for variant, output in selections(rf, selector, inp, views, online, budget, protocol, gate, is_pair):
                    decisions.append({'panel': 'evaluation_paired' if is_pair else 'evaluation',
                        'dataset': 'evaluation_set', 'config': 'c8_evaluation', 'method': 'Centralized',
                        'variant': variant, 'budget': budget, 'scenario': scenario, 'uid': uid,
                        'selection': output, 'offline_clients': np.flatnonzero(~online).tolist(),
                        'availability_decision': gate})
    # Save the complete score-only decisions before annotation content is loaded.
    with d.fresh(args.out / 'evaluation_decisions.jsonl') as f:
        for row in decisions:
            d.emit(f, row)
    annotations = {r['canonical_id']: r for r in d.read(gold_path)['annotations']}
    task_by_id = {t['canonical_id']: t for t in tasks}
    rows = list(metric_rows(rf, b_eval, decisions, inputs, annotations, task_by_id, protocol))
    # Labels are passed directly into the same fixed metric boundary solely
    # to check the saved SERF metrics. There is no routing recomputation.
    for rec in refs:
        inp = inputs[rec['uid']]
        selection = {'queried_clients': rec['selected_clients'], 'evidence_ids': rec['selected_page_uids'],
                     'client_calls': rec['client_calls'], 'abstain': rec['abstain_routing'], 'confidence': 0.0}
        metrics = rf.item_metrics(selection, annotations[rec['uid']], inp,
                                  protocol['federation']['certificate_bytes_per_client_call'])
        d.same(metrics, rec, ('strict_success', 'evidence_recall', 'evidence_precision', 'citation_f1', 'bytes'))
    for row in rows:
        if row['variant'] == 'base':
            d.same(row, bowners[d.key(row)], ('selected_page_uids', 'owner_contacts', 'strict_success',
                                             'evidence_recall', 'evidence_precision', 'citation_f1', 'bytes'))
    if args.limit is None and sum(r['panel'] == 'evaluation_paired' and r['variant'] == '+sel' for r in rows) != 80:
        raise ValueError('Expected 80 fixed paired scenarios')
    return rows, refs, len(tasks)


def main():
    p = d.parser(__doc__)
    for option in ('runner', 'selector', 'cache', 'tasks', 'gold'):
        p.add_argument('--' + option, type=d.Path)
    args = p.parse_args()
    design = d.setup(args)
    sources = d.Sources()
    started = time.time()
    rows, refs, queries = run(args, design, sources)
    d.finish(args, 'evaluation', rows, refs, sources, started, queries=queries,
             enumeration='Omitted: the original enumerator has no added-method interface')


if __name__ == '__main__':
    main()
