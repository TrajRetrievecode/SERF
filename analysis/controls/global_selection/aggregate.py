"""Validate control rows and emit all specified comparisons as JSON and CSV."""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import os
ROOT = Path(os.environ.get('SERF_ROOT', str(HERE.parents[2])))
sys.path.append(str(ROOT / 'src/public_e2e_v1'))
import d_common as d
from e2e_stats import paired_bootstrap, mcnemar

METRICS = ('evidence_recall', 'evidence_precision', 'citation_f1', 'complete_evidence',
           'part_recall', 'strict_success', 'contacts', 'selected_pages', 'abstain_routing',
           'client_calls', 'protocol_bytes', 'bytes')
CELL = ('panel', 'dataset', 'config', 'method', 'variant', 'budget', 'scenario')


def groups(rows, fields):
    result = defaultdict(list)
    for row in rows:
        result[tuple(row[k] for k in fields)].append(row)
    return result


def contact(row):
    return row['client_calls'] if row['method'] == 'FedResolve-v3' else row['owner_contacts']


def validate(rows, manifests):
    unique = set()
    for row in rows:
        k = tuple(row[f] for f in CELL) + (row['uid'],)
        if k in unique:
            raise ValueError('Duplicate output row: ' + str(k))
        unique.add(k)
        pages = row['selected_page_uids']
        if len(set(pages)) != len(pages) or len(pages) > int(row['budget']):
            raise ValueError('Page budget or uniqueness violation')
        if row['selected_pages'] != len(pages):
            raise ValueError('Selected-page count differs')
        if row['method'] == 'Centralized':
            if row['owner_contacts'] != len(set(row['owners'])):
                raise ValueError('Owner contacts differ')
            if row['panel'] not in ('evaluation', 'evaluation_paired') and row['abstain_routing'] and pages:
                raise ValueError('Public abstention cites evidence')
        for field in METRICS:
            if field in row and not np.isfinite(float(row[field])):
                raise ValueError('Nonfinite metric: ' + field)
    for stage, manifest in manifests.items():
        selected = [r for r in rows if r.get('_stage') == stage]
        new = [r for r in selected if r['method'] == 'Centralized' and r['variant'] != 'recorded']
        refs = [r for r in selected if r['method'] != 'Centralized']
        if len(new) != manifest['rows'] or len(refs) != manifest['references']:
            raise ValueError('Downloaded output count differs from manifest')
        if stage in ('finrag', 'visdom'):
            for cfg, count in manifest['cohorts'].items():
                cohort = [r for r in new if r['config'] == cfg]
                cells = [(b, 'full', v) for b in (1, 2, 3) for v in ('base', '+sel', '+sel+test')]
                cells += [(2, s, v) for s in ('unneeded_offline', 'qrel_missing') for v in ('+sel', '+sel+test')]
                check_cells(cohort, cells, count)
        elif stage == 'composite':
            for ds, counts in manifest['cohorts'].items():
                for cfg, count in counts.items():
                    cohort = [r for r in new if r['dataset'] == ds and r['config'] == cfg]
                    check_cells(cohort, [(b, s, v) for b in (2, 3)
                        for s in ('full', 'part_offline', 'other_offline')
                        for v in ('base', '+sel', '+sel+test')], count)
        elif stage == 'evaluation':
            full = [r for r in new if r['panel'] == 'evaluation']
            check_cells(full, [(b, 'full', v) for b in (2, 3) for v in ('base', '+sel')], manifest['queries'])
            paired = [r for r in new if r['panel'] == 'evaluation_paired']
            a = {(r['uid'], r['scenario']) for r in paired if r['variant'] == '+sel'}
            b = {(r['uid'], r['scenario']) for r in paired if r['variant'] == '+sel+test'}
            if a != b or (manifest['limit'] is None and len(a) != 80):
                raise ValueError('Evaluation paired universe differs')


def check_cells(rows, cells, count):
    actual = groups(rows, ('budget', 'scenario', 'variant'))
    if count == 0:
        if actual:
            raise ValueError('Unexpected nonempty setting')
        return
    if set(actual) != set(cells):
        raise ValueError('Incomplete variant, budget or scenario cells')
    uids = None
    for key in cells:
        ids = {r['uid'] for r in actual[key]}
        if len(ids) != count or len(actual[key]) != count or (uids is not None and ids != uids):
            raise ValueError('Query cohorts differ between cells')
        uids = ids


def aligned(a, b):
    aa = {(r['config'], r['uid'], r['scenario']): r for r in a}
    bb = {(r['config'], r['uid'], r['scenario']): r for r in b}
    if len(aa) != len(a) or len(bb) != len(b) or set(aa) != set(bb):
        raise ValueError('Paired query universe differs')
    keys = sorted(aa)
    return [aa[k] for k in keys], [bb[k] for k in keys]


def exact(a, b):
    # Use the imported test normally. Large discordance can overflow or
    # underflow its integer-to-float tail; the stable tail is the same test.
    discordant = sum(bool(x) != bool(y) for x, y in zip(a, b))
    if discordant >= 1024:
        from scipy.stats import binom
        x = sum(bool(v) and not bool(w) for v, w in zip(a, b))
        y = sum(bool(w) and not bool(v) for v, w in zip(a, b))
        n = x + y
        return {'treatment_only_correct': x, 'control_only_correct': y, 'discordant': n,
                'p_value': min(1.0, 2 * float(binom.cdf(min(x, y), n, 0.5))), 'test': 'exact_binomial'}
    return mcnemar(a, b)


def reverse_stat(row, binary=False):
    out = {**row, 'direction': 'control minus SERF' if row['direction'] == 'SERF minus control'
           else 'control minus treatment'}
    if binary:
        out['treatment_only_correct'], out['control_only_correct'] = row['control_only_correct'], row['treatment_only_correct']
    else:
        out.update(treatment_mean=row['control_mean'], control_mean=row['treatment_mean'],
                   difference=-row['difference'], ci_low=-row['ci_high'], ci_high=-row['ci_low'])
    return out


def summarize(rows):
    table = []
    for k, rs in sorted(groups(rows, CELL).items()):
        item = {**dict(zip(CELL, k)), 'n': len(rs)}
        for metric in METRICS:
            if all(metric in r for r in rs):
                item[metric] = float(np.mean([r[metric] for r in rs]))
        table.append(item)
    return table


def macro(table, fields, metrics):
    out = []
    for k, rs in sorted(groups([r for r in table if r['dataset'] != 'evaluation_set'], fields).items()):
        if len({r['config'] for r in rs}) != len(rs):
            raise ValueError('Duplicate setting summary')
        valid = {r['config'] for r in rs} == set(d.SETTINGS) and all(r['n'] > 0 for r in rs)
        item = {**dict(zip(fields, k)), 'config': 'mean_four_settings',
                'settings': len(rs), 'n': sum(r['n'] for r in rs), 'complete_four_settings': valid}
        for metric in metrics:
            if any(metric in r for r in rs):
                item[metric] = float(np.mean([r[metric] for r in rs])) if valid and all(r.get(metric) is not None for r in rs) else None
        out.append(item)
    return out


def primary_statistics(rows, design):
    cells = groups(rows, ('panel', 'dataset', 'config', 'budget', 'scenario'))
    boots, mc, unavailable = [], [], []
    pooled = defaultdict(lambda: ([], []))
    for k, rs in sorted(cells.items()):
        panel, ds, cfg, budget, scenario = k
        serf = [r for r in rs if r['method'] == 'FedResolve-v3']
        if panel in ('public_paired', 'evaluation_paired'):
            continue
        for variant in sorted({r['variant'] for r in rs if r['method'] == 'Centralized'}):
            controls = [r for r in rs if r['method'] == 'Centralized' and r['variant'] == variant]
            header = dict(zip(('panel', 'dataset', 'config', 'budget', 'scenario'), k))
            header.update(control='Centralized', variant=variant, direction='SERF minus control')
            if not serf:
                if panel != 'composite' or budget != 3 or scenario == 'full':
                    raise ValueError('Required recorded SERF rows are missing')
                unavailable.append({**header, 'reason': 'Job C has no recorded budget-3 offline SERF rows'})
                unavailable.append({**header, 'direction': 'control minus SERF',
                                    'reason': 'Job C has no recorded budget-3 offline SERF rows'})
                continue
            a, b = aligned(serf, controls)
            metrics = ('part_recall', 'complete_evidence', 'contacts') if panel == 'composite' else (
                      ('evidence_recall',) if panel == 'evaluation' else ('evidence_recall', 'contacts'))
            seed = design['statistics']['bootstrap_seeds'][panel]
            for metric in metrics:
                result = paired_bootstrap([r[metric] for r in a], [r[metric] for r in b],
                                           iterations=10000, seed=seed)
                row = {**header, 'metric': metric, **result}
                boots.extend([row, reverse_stat(row)])
            if panel == 'evaluation':
                row = {**header, 'metric': 'strict_success',
                       **exact([r['strict_success'] for r in a], [r['strict_success'] for r in b])}
                mc.extend([row, reverse_stat(row, True)])
            if panel == 'composite':
                pk = ds, budget, scenario, variant
                pooled[pk][0].extend(a)
                pooled[pk][1].extend(b)
    for (ds, budget, scenario, variant), (a, b) in sorted(pooled.items()):
        a, b = aligned(a, b)
        row = {'panel': 'composite', 'dataset': ds, 'config': 'pooled_settings', 'budget': budget,
               'scenario': scenario, 'variant': variant, 'direction': 'SERF minus control',
               'control': 'Centralized', 'metric': 'complete_evidence', 'n': len(a),
               **exact([r['complete_evidence'] for r in a], [r['complete_evidence'] for r in b])}
        mc.extend([row, reverse_stat(row, True)])
    counts = []
    fields = ('panel', 'dataset', 'budget', 'scenario', 'variant', 'direction', 'metric')
    for k, rs in sorted(groups([r for r in boots if r['dataset'] != 'evaluation_set'], fields).items()):
        counts.append({**dict(zip(fields, k)), 'settings_available': len(rs), 'settings_expected': 4,
            'interval_excludes_zero': sum(r['significant_at_95'] for r in rs),
            'positive': sum(r['ci_low'] > 0 for r in rs), 'negative': sum(r['ci_high'] < 0 for r in rs)})
    return boots, counts, mc, unavailable


def side_success(row):
    if not row['answerable']:
        return bool(row['abstain_routing'])
    success = (row['complete_evidence'] if row['panel'] == 'composite' else
               row['complete_client_set'] if row['panel'] == 'evaluation_paired' else row['evidence_recall'] > 0)
    return bool(success) and not row['abstain_routing']


def paired_tables(rows):
    selected = [r for r in rows if 'answerable' in r and r['scenario'] != 'full']
    fields = ('panel', 'dataset', 'config', 'method', 'variant', 'budget')
    tables, tests = [], []
    for k, rs in sorted(groups(selected, fields).items()):
        ans = [r for r in rs if r['answerable']]
        una = [r for r in rs if not r['answerable']]
        if k[0] != 'evaluation_paired':
            if {r['uid'] for r in ans} != {r['uid'] for r in una}:
                raise ValueError('Scenario sides have different query cohorts')
        ao, uo = sum(side_success(r) for r in ans), sum(side_success(r) for r in una)
        ar, ur = ao / len(ans) if ans else None, uo / len(una) if una else None
        tables.append({**dict(zip(fields, k)), 'n': len(rs), 'answerable': len(ans), 'unanswerable': len(una),
                       'answerable_success': ao, 'unanswerable_abstain': uo,
                       'answerable_rate': ar, 'unanswerable_rate': ur,
                       'false_abstentions': sum(r['abstain_routing'] for r in ans),
                       'balanced_accuracy': (ar + ur) / 2 if ar is not None and ur is not None else None})
    for k, rs in sorted(groups(selected, ('panel', 'dataset', 'config', 'budget')).items()):
        combos = groups(rs, ('method', 'variant'))
        comparisons = []
        for ctrl in combos:
            if ctrl[0] == 'Centralized':
                comparisons.append((('FedResolve-v3', 'recorded'), ctrl, 'SERF minus control'))
        comparisons += [(('Centralized', '+sel+test'), ('Centralized', '+sel'), 'treatment minus control'),
                        (('Centralized', '+sel+test'), ('Centralized + test', 'recorded'), 'treatment minus control')]
        for t, ctrl, direction in comparisons:
            if t not in combos or ctrl not in combos:
                # Only the documented budget-3 composite SERF comparison is unavailable.
                if t[0] == 'FedResolve-v3' and k[0] == 'composite' and k[3] == 3:
                    continue
                if ctrl[0] == 'Centralized + test' and k[0] == 'composite':
                    continue
                raise ValueError('Missing paired comparison method')
            a, b = aligned(combos[t], combos[ctrl])
            for side in (True, False):
                pairs = [(x, y) for x, y in zip(a, b) if x['answerable'] == side]
                if any(x['answerable'] != y['answerable'] for x, y in zip(a, b)):
                    raise ValueError('Recorded scenario labels differ')
                row = {**dict(zip(('panel', 'dataset', 'config', 'budget'), k)),
                       'treatment': ' '.join(t), 'control': ' '.join(ctrl), 'direction': direction,
                       'side': 'answerable' if side else 'unanswerable', 'n': len(pairs),
                       **exact([side_success(x) for x, y in pairs], [side_success(y) for x, y in pairs])}
                tests.extend([row, reverse_stat(row, True)])
    return tables, tests


def reproduction(rows, registered, smoke):
    checks = []
    for rec in registered['public']:
        if rec['method'] not in ('FedResolve-v3', 'Centralized') or rec['scenario'] != 'full':
            continue
        selected = [r for r in rows if r['panel'] == 'public' and r['dataset'] == rec['dataset']
                    and r['config'] == rec['config'] and r['budget'] == int(rec['budget'])
                    and r['method'] == rec['method'] and (r['method'] != 'Centralized' or r['variant'] == 'base')]
        if not selected:
            raise ValueError('Missing registered public comparison cell')
        for metric in ('evidence_recall', 'selected_pages', 'client_calls', 'protocol_bytes', 'abstain_routing'):
            actual = float(np.mean([r[metric] for r in selected]))
            match = len(selected) == int(rec['n']) and np.isclose(actual, float(rec[metric]), atol=1e-12, rtol=1e-12)
            checks.append({'panel': 'public', 'dataset': rec['dataset'], 'config': rec['config'],
                           'method': rec['method'], 'budget': int(rec['budget']), 'metric': metric,
                           'actual': actual, 'recorded': float(rec[metric]), 'match': None if smoke else bool(match)})
    for budget in (2, 3):
        selected = [r for r in rows if r['panel'] == 'evaluation' and r['method'] == 'FedResolve-v3' and r['budget'] == budget]
        if not selected:
            raise ValueError('Missing evaluation SERF questions')
        want = registered['evaluation']['online'][str(budget)]['SERF-v3']
        for metric in ('strict_success', 'evidence_recall', 'evidence_precision', 'citation_f1', 'bytes', 'client_calls'):
            actual = float(np.mean([r[metric] for r in selected]))
            checks.append({'panel': 'evaluation', 'budget': budget, 'method': 'FedResolve-v3', 'metric': metric,
                           'actual': actual, 'recorded': want[metric], 'match': None if smoke else
                           len(selected) == 45 and bool(np.isclose(actual, want[metric], atol=1e-12, rtol=1e-12))})
    if not smoke and any(not r['match'] for r in checks):
        raise ValueError('Recorded aggregate reproduction failed')
    return {'status': 'subset_not_comparable' if smoke else 'pass', 'checks': checks}


def csv_file(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with d.fresh(path) as f:
        writer = csv.DictWriter(f, fields, lineterminator='\n')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (list, dict)) else v for k, v in d.clean(row).items()})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--limit', type=int)
    p.add_argument('--design', type=Path, default=HERE / 'protocol_fairness_v1_D_global_selection.json')
    args = p.parse_args()
    if args.limit is not None and args.limit < 1:
        p.error('--limit must be positive')
    destination = args.out.resolve()
    allowed = (ROOT / 'artifacts/controls/global_selection').resolve()
    smoke = (ROOT / 'artifacts/controls/global_selection_smoke').resolve()
    if destination not in (allowed, smoke):
        raise ValueError('Outputs must stay inside the control artifact directory')
    design = d.read(args.design)
    import e2e_stats
    d.pin(design, e2e_stats=e2e_stats)
    for name, expected in design['registered_result_sha256'].items():
        if d.digest(HERE / 'registered' / name) != expected:
            raise ValueError('Registered result copy differs: ' + name)
    rows, manifests = [], {}
    for stage in ('finrag', 'visdom', 'composite', 'evaluation'):
        manifest = d.read(args.input / (stage + '_manifest.json'))
        manifests[stage] = manifest
        if manifest['design_sha256'] != d.digest(args.design):
            raise ValueError('Downloaded design differs')
        if manifest['limit'] is not None and args.limit != manifest['limit']:
            raise ValueError('Smoke outputs require the same explicit limit')
        for suffix in ('.jsonl', '_references.jsonl'):
            for row in d.stream(args.input / (stage + suffix)):
                rows.append({**row, '_stage': stage, 'contacts': contact(row)})
    validate(rows, manifests)
    # Apply a local subset only after checking complete downloaded files.
    if args.limit is not None:
        cohorts = {}
        chosen = defaultdict(list)
        for row in rows:
            k = row['panel'], row['dataset'], row['config']
            if row['uid'] not in chosen[k] and len(chosen[k]) < args.limit:
                chosen[k].append(row['uid'])
        rows = [r for r in rows if r['uid'] in chosen[r['panel'], r['dataset'], r['config']]]
    with (HERE / 'registered/v3_cells.csv').open(encoding='utf-8-sig') as f:
        registered_public = list(csv.DictReader(f))
    registered = {'public': registered_public, 'evaluation': d.read(HERE / 'registered/gold_v3_results.json')}
    rep = reproduction(rows, registered, args.limit is not None)
    table = summarize(rows)
    # Explicit empty settings are visible, and four-setting means stay null.
    for stage in ('finrag', 'visdom', 'composite'):
        cohorts = manifests[stage]['cohorts']
        if stage != 'composite':
            cohorts = {stage: cohorts}
        for ds, cfgs in cohorts.items():
            for cfg, count in cfgs.items():
                if count == 0:
                    panel = 'composite' if stage == 'composite' else 'public'
                    cells = [(b, s, v) for b in ((2, 3) if stage == 'composite' else (1, 2, 3))
                             for s in (('full', 'part_offline', 'other_offline') if stage == 'composite' else ('full',))
                             for v in ('base', '+sel', '+sel+test')]
                    for b, s, v in cells:
                        table.append({'panel': panel, 'dataset': ds, 'config': cfg, 'n': 0,
                                      'method': 'Centralized', 'variant': v, 'budget': b, 'scenario': s,
                                      **{m: None for m in METRICS}})
    means = macro(table, tuple(f for f in CELL if f != 'config'), METRICS)
    boot, counts, mc, unavailable = primary_statistics(rows, design)
    balanced, side_tests = paired_tables(rows)
    for ds, cfgs in manifests['composite']['cohorts'].items():
        for cfg, count in cfgs.items():
            if count == 0:
                for b in (2, 3):
                    for v in ('base', '+sel', '+sel+test'):
                        balanced.append({'panel': 'composite', 'dataset': ds, 'config': cfg,
                            'method': 'Centralized', 'variant': v, 'budget': b, 'n': 0,
                            'answerable': 0, 'unanswerable': 0, 'answerable_success': 0,
                            'unanswerable_abstain': 0, 'false_abstentions': 0,
                            'answerable_rate': None, 'unanswerable_rate': None, 'balanced_accuracy': None})
    bal_means = macro(balanced, ('panel', 'dataset', 'method', 'variant', 'budget'),
                      ('answerable_rate', 'unanswerable_rate', 'balanced_accuracy'))
    tables = {'per_setting': table, 'mean_four_settings': means, 'paired_bootstrap': boot,
              'interval_setting_counts': counts, 'mcnemar': mc, 'paired_missingness': balanced,
              'paired_missingness_mean_four_settings': bal_means, 'paired_side_mcnemar': side_tests,
              'unavailable_comparisons': unavailable, 'reproduction': rep['checks']}
    out = args.out / 'results'
    out.mkdir(parents=True, exist_ok=True)
    d.save(out / 'global_selection.json', {'tables': tables, 'reproduction_status': rep['status'],
           'limit': args.limit, 'manifests': manifests, 'enumeration': 'Omitted: original script requires changes'})
    flattened = []
    for name, values in tables.items():
        csv_file(out / (name + '.csv'), values)
        flattened.extend({'table': name, **r} for r in values)
    csv_file(out / 'global_selection.csv', flattened)
    print(json.dumps({'output': str(out), 'reproduction': rep['status'], 'rows': len(rows)}))


if __name__ == '__main__':
    main()
