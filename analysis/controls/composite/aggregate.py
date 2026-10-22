"""Aggregate composite plans into JSON and CSV result tables."""
from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
from e2e_common import ROOT

import numpy as np
from e2e_stats import mcnemar, paired_bootstrap
from composite_common import (arguments, fresh, parser, protocol_digest, read_json,
                              read_jsonl, selection_variants, write_json)

METRICS = ('part_recall', 'complete_evidence', 'evidence_recall', 'calls',
           'selected_pages', 'abstain_routing')


def average(values):
    return float(np.mean(values)) if values else None


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def exact_mcnemar(a, b):
    """Reuse the protocol test, with a stable tail for very large discordance."""
    discordant = sum(bool(x) != bool(y) for x, y in zip(a, b))
    if discordant < 1024:
        return mcnemar(a, b)
    # The upstream math.comb * 0.5**n implementation can overflow/underflow.
    from scipy.special import bdtr
    treatment_only = sum(bool(x) and not bool(y) for x, y in zip(a, b))
    control_only = discordant - treatment_only
    return {'treatment_only_correct': treatment_only, 'control_only_correct': control_only,
            'discordant': discordant,
            'p_value': min(1.0, 2 * float(bdtr(min(treatment_only, control_only), discordant, 0.5))),
            'test': 'exact_binomial', 'numerics': 'Stable binomial tail for large discordance'}


def slices(protocol):
    for method, variant in selection_variants(protocol):
        for policy in protocol['policies']:
            for budget in protocol['budgets']:
                for scenario in (('full', 'part_offline', 'other_offline') if budget == 2 else ('full',)):
                    yield method, variant, policy, budget, scenario


def validate_rows(rows, definitions, protocol):
    test = {(c['dataset'], c['config'], c['uid']): c for c in definitions if c['portion'] == 'test'}
    if len(test) != sum(c['portion'] == 'test' for c in definitions):
        raise ValueError('Duplicate test composite definition')
    expected = set(slices(protocol))
    seen = defaultdict(set)
    for r in rows:
        key = r['dataset'], r['config'], r['uid']
        cell = r['method'], r['variant'], r['policy'], r['budget'], r['scenario']
        if key not in test or cell not in expected:
            raise ValueError('Unexpected plan composite or cell')
        if cell in seen[key]:
            raise ValueError('Duplicate plan row')
        seen[key].add(cell)
        if r['calls'] != len(r['selected_clients']) or r['calls'] != r['cost']['client_calls']:
            raise ValueError('Call accounting mismatch')
        if r['selected_pages'] != len(r['evidence_page_uids']):
            raise ValueError('Selected page accounting mismatch')
        if r['abstain_routing'] and r['evidence_page_uids']:
            raise ValueError('An abstained row cites evidence')
    if any(seen[key] != expected for key in test):
        raise ValueError('Incomplete plans; every composite must have every protocol cell')


def aggregate(rows, protocol):
    configs = [f'c{c}_{t}' for c in protocol['client_scales'] for t in protocol['tracks']]
    grouped = defaultdict(dict)
    for r in rows:
        key = (r['dataset'], r['config'], r['method'], r['variant'],
               r['policy'], r['budget'], r['scenario'])
        if r['uid'] in grouped[key]:
            raise ValueError('Duplicate row in aggregation')
        grouped[key][r['uid']] = r
    cells, means, balanced, balanced_means = [], [], [], []
    statistics, pooled, counts = [], [], []
    for dataset in protocol['datasets']:
        for method, variant, policy, budget, scenario in slices(protocol):
            identity = {'dataset': dataset, 'method': method, 'variant': variant,
                        'policy': policy, 'budget': budget, 'scenario': scenario}
            setting_cells = []
            for config in configs:
                observations = list(grouped[(dataset, config, method, variant, policy, budget, scenario)].values())
                cell = {**identity, 'config': config, 'records': len(observations),
                        **{m: average([float(r[m]) for r in observations]) for m in METRICS}}
                cells.append(cell)
                setting_cells.append(cell)
            complete = all(c['records'] > 0 for c in setting_cells)
            means.append({**identity, 'settings_available': sum(c['records'] > 0 for c in setting_cells),
                          'settings_required': len(configs),
                          'records': sum(c['records'] for c in setting_cells),
                          **{m: average([c[m] for c in setting_cells]) if complete else None for m in METRICS}})
            if method == 'FedResolve-v3' and variant == 'native':
                continue
            a_all, b_all = [], []
            by_metric = defaultdict(list)
            for config in configs:
                treatment = grouped[(dataset, config, 'FedResolve-v3', 'native', policy, budget, scenario)]
                control = grouped[(dataset, config, method, variant, policy, budget, scenario)]
                if set(treatment) != set(control):
                    raise ValueError('Paired comparison uid mismatch')
                uids = sorted(treatment)
                for metric in ('part_recall', 'complete_evidence', 'calls'):
                    result = paired_bootstrap([treatment[u][metric] for u in uids],
                                              [control[u][metric] for u in uids],
                                              iterations=protocol['statistics']['bootstrap_iterations'],
                                              seed=protocol['statistics']['bootstrap_seed'])
                    row = {**identity, 'config': config, 'metric': metric,
                           'reference': 'FedResolve-v3', **clean(result)}
                    statistics.append(row)
                    by_metric[metric].append(row)
                a_all.extend(treatment[u]['complete_evidence'] for u in uids)
                b_all.extend(control[u]['complete_evidence'] for u in uids)
            pooled.append({**identity, 'reference': 'FedResolve-v3', 'pairs': len(a_all),
                           **exact_mcnemar(a_all, b_all),
                           'note': 'Pooled setting-query pairs can share source queries across settings.'})
            for metric, setting_rows in by_metric.items():
                counts.append({**identity, 'metric': metric, 'reference': 'FedResolve-v3',
                               'settings_excluding_zero': sum(r['significant_at_95'] for r in setting_rows),
                               'settings_available': sum(r['n'] > 0 for r in setting_rows),
                               'settings_required': len(configs)})
        for method, variant in selection_variants(protocol):
            for policy in protocol['policies']:
                setting_balanced = []
                identity = {'dataset': dataset, 'method': method, 'variant': variant,
                            'policy': policy, 'budget': 2}
                for config in configs:
                    answer = grouped[(dataset, config, method, variant, policy, 2, 'other_offline')]
                    missing = grouped[(dataset, config, method, variant, policy, 2, 'part_offline')]
                    if set(answer) != set(missing):
                        raise ValueError('Availability scenario uid mismatch')
                    uids = sorted(answer)
                    success = [bool(answer[u]['complete_evidence']) and not answer[u]['abstain_routing'] for u in uids]
                    abstain = [missing[u]['abstain_routing'] for u in uids]
                    row = {**identity, 'config': config, 'paired_records': len(uids),
                           'answerable_complete_success': average(success),
                           'unanswerable_abstention': average(abstain),
                           'balanced_accuracy': average([0.5 * (a + b) for a, b in zip(success, abstain)])}
                    balanced.append(row)
                    setting_balanced.append(row)
                complete = all(r['paired_records'] > 0 for r in setting_balanced)
                balanced_means.append({**identity, 'settings_required': len(configs),
                                       'settings_available': sum(r['paired_records'] > 0 for r in setting_balanced),
                                       **{m: average([r[m] for r in setting_balanced]) if complete else None
                                          for m in ('answerable_complete_success', 'unanswerable_abstention', 'balanced_accuracy')}})
    return {'per_setting': cells, 'four_setting_means': means, 'paired_balanced_accuracy': balanced,
            'balanced_four_setting_means': balanced_means, 'paired_bootstrap': statistics,
            'interval_setting_counts': counts, 'pooled_mcnemar': pooled}


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with fresh(path).open('x', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = parser(__doc__)
    p.add_argument('--input', type=Path, default=ROOT / 'artifacts/controls/composite', help='Composite job artifact directory')
    args = arguments(p)
    protocol = read_json(args.protocol)
    construction = read_json(args.input / 'construction.json')
    if construction['protocol_sha256'] != protocol_digest(args.protocol) or construction['limit'] != args.limit:
        raise ValueError('Input protocol or smoke limit differs')
    definitions = read_jsonl(args.input / 'composites.jsonl')
    rows = [r for d in protocol['datasets'] for r in read_jsonl(args.input / 'plans' / d / 'plan.jsonl')]
    validate_rows(rows, definitions, protocol)
    if args.limit is not None:
        keep = set()
        for dataset in protocol['datasets']:
            for clients in protocol['client_scales']:
                for track in protocol['tracks']:
                    config = f'c{clients}_{track}'
                    selected = [c for c in definitions if c['dataset'] == dataset and c['config'] == config and c['portion'] == 'test'][:args.limit]
                    keep.update((c['dataset'], c['config'], c['uid']) for c in selected)
        rows = [r for r in rows if (r['dataset'], r['config'], r['uid']) in keep]
    output = aggregate(rows, protocol)
    output['construction'] = construction
    output['thresholds'] = {d: read_json(args.input / 'thresholds' / f'{d}.json') for d in protocol['datasets']}
    output['protocol_sha256'] = protocol_digest(args.protocol)
    output['limit'] = args.limit
    result_dir = args.out / 'results'
    # Preflight all targets, so reruns do not partially replace a table.
    for name in ('composite.json', 'composite.csv', 'construction.csv', 'threshold_sweep.csv',
                 'four_setting_means.csv', 'paired_balanced_accuracy.csv', 'balanced_four_setting_means.csv',
                 'paired_bootstrap.csv', 'interval_setting_counts.csv', 'pooled_mcnemar.csv'):
        fresh(result_dir / name)
    write_json(result_dir / 'composite.json', clean(output))
    write_csv(result_dir / 'composite.csv', output['per_setting'])
    for name in ('four_setting_means', 'paired_balanced_accuracy', 'balanced_four_setting_means',
                 'paired_bootstrap', 'interval_setting_counts', 'pooled_mcnemar'):
        write_csv(result_dir / f'{name}.csv', output[name])
    write_csv(result_dir / 'construction.csv', construction['settings'])
    write_csv(result_dir / 'threshold_sweep.csv', [
        {'dataset': d, 'chosen_theta': t['theta'], 'training_composites': t['training_composites'], **r}
        for d, t in output['thresholds'].items() for r in t['sweep']])
    print(json.dumps({'results': str(result_dir), 'rows': len(rows)}))


if __name__ == '__main__':
    main()
