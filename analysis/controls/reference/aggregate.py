"""Aggregate reference rows and package result tables into four comparator tables."""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
from e2e_common import ROOT
from b_common import (balanced, clean, read_json, rule_abstains, select_training_theta,
                      sha, stream, sweep, write_json)
from e2e_stats import mcnemar, paired_bootstrap
from enumerate_single_offline import clopper_pearson

BOOTSTRAP_SEED = 20260805
RESAMPLES = 10000
SETTINGS = {'c8_seen_query', 'c8_unseen_family', 'c16_seen_query', 'c16_unseen_family'}


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('x', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (__import__('json').dumps(clean(v), sort_keys=True) if isinstance(v, (dict, list, tuple)) else v)
                             for k, v in clean(row).items()})


def emit(out, name, tables, **details):
    write_json(out / f'{name}.json', {'table': tables, **details})
    write_csv(out / f'{name}.csv', tables)


def summarize_groups(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    return [{**dict(zip(keys, key)), **balanced(group)} for key, group in sorted(groups.items())]


def macro_means(table, keys, rates, counts, require_four=True):
    groups = defaultdict(list)
    for row in table:
        groups[tuple(row.get(k) for k in keys)].append(row)
    output = []
    for key, rows in sorted(groups.items()):
        if rows[0]['dataset'] == 'evaluation_set':
            continue
        configs = {r['config'] for r in rows}
        if require_four and configs != SETTINGS:
            raise ValueError(f'Expected four public settings for {key}, found {configs}')
        if len(configs) != len(rows):
            raise ValueError('Duplicate setting summary')
        result = {**dict(zip(keys, key)), 'config': 'mean_over_settings', 'settings': len(rows)}
        for field in counts:
            result[field] = sum(r[field] for r in rows)
        for field in rates:
            values = [r[field] for r in rows]
            result[field] = float(np.mean(values)) if all(v is not None for v in values) else None
        output.append(result)
    return output


def limit_rows(rows, limit):
    if not limit:
        return rows
    uids = defaultdict(list)
    out = []
    for row in rows:
        key = row['dataset'], row['config'], row.get('split', 'test')
        if row['uid'] not in uids[key] and len(uids[key]) < limit:
            uids[key].append(row['uid'])
        if row['uid'] in uids[key]:
            out.append(row)
    return out


def validate_downloads(data, manifests):
    """Check complete exports before applying any local smoke limit."""
    for ds, manifest in manifests['public']['datasets'].items():
        counts = manifest['counts']
        for name, count_key in (('public_owners', 'owner_rows'), ('public_paired', 'paired_rows')):
            if sum(r['dataset'] == ds for r in data[name]) != counts[count_key]:
                raise ValueError(f'Incomplete input {name} for {ds}')
        scores = [r for r in data['public_scores'] if r['dataset'] == ds]
        for split in ('train', 'test'):
            if sum(r['split'] == split for r in scores) != counts[f'{split}_score_rows']:
                raise ValueError(f'Incomplete input {split} scores for {ds}')
        for cfg, split_counts in manifest['settings'].items():
            cohorts = {}
            for split in ('train', 'test'):
                selected = [r for r in scores if r['config'] == cfg and r['split'] == split]
                key_set = {(r['uid'], r['scenario']) for r in selected}
                if len(key_set) != len(selected) or len(selected) != split_counts[split] * 2:
                    raise ValueError('Duplicate or incomplete public score scenarios')
                pairs = paired_map(selected)
                cohorts[split] = set(pairs)
            if cohorts['train'] & cohorts['test']:
                raise ValueError('Public training and test cohorts overlap')
            expected_owner_keys = {(u, b) for u in cohorts['test'] for b in (1, 2, 3)}
            actual_owner_keys = {(r['uid'], r['budget']) for r in data['public_owners']
                                 if r['dataset'] == ds and r['config'] == cfg}
            if actual_owner_keys != expected_owner_keys:
                raise ValueError('Public owners and scores have different test cohorts')
    if {r['dataset'] for r in data['public_scores']} != set(manifests['public']['datasets']):
        raise ValueError('Unexpected dataset in input public scores')
    total = sum(manifests['evaluation']['scenario_counts'].values())
    if len(data['evaluation_scores']) != total or len(data['evaluation_paired']) != 4 * total:
        raise ValueError('Incomplete input evaluation scenarios')
    tasks = min(manifests['evaluation'].get('limit') or 45, 45)
    if len(data['evaluation_owners']) != tasks * 3:
        raise ValueError('Incomplete input evaluation owners')
    for family, count in manifests['evaluation']['scenario_counts'].items():
        rows = [r for r in data['evaluation_scores'] if r['scenario_family'] == family]
        keys = {(r['uid'], r['scenario_id']) for r in rows}
        if len(keys) != count or len(rows) != count:
            raise ValueError('Duplicate or missing evaluation score scenario')
        for method in ('Centralized', 'Centralized+avail', 'FedResolve', 'FedResolve-v3'):
            selected = [r for r in data['evaluation_paired'] if r['scenario_family'] == family and r['method'] == method]
            if len(selected) != count or {(r['uid'], r['scenario_id']) for r in selected} != keys:
                raise ValueError('Evaluation method scenario universe mismatch')


def limit_downloads(data, limit):
    if limit is None:
        return data
    allowed = defaultdict(list)
    for name in ('public_scores', 'evaluation_scores'):
        for r in data[name]:
            key = r['dataset'], r['config'], r['split']
            if r['uid'] not in allowed[key] and len(allowed[key]) < limit:
                allowed[key].append(r['uid'])
    return {name: [r for r in rows if r['uid'] in allowed[(r['dataset'], r['config'],
                        r.get('split', 'evaluation' if r['dataset'] == 'evaluation_set' else 'test'))]]
            for name, rows in data.items()}


def owner_table(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['dataset'], row['config'], row['budget'])].append(row)
    table = []
    for (ds, cfg, budget), group in sorted(groups.items()):
        if len({r['uid'] for r in group}) != len(group):
            raise ValueError('Duplicate owner contact query')
        counts = [sum(r['owner_contacts'] == n for r in group) for n in range(4)]
        if sum(counts) != len(group):
            raise ValueError('Owner count exceeds budget')
        table.append({'dataset': ds, 'config': cfg, 'method': 'Centralized', 'budget': budget,
            'scenario': 'full', 'queries': len(group), 'owner_contact_sum': sum(r['owner_contacts'] for r in group),
            'mean_owner_contacts': float(np.mean([r['owner_contacts'] for r in group])),
            **{f'owners_{n}_count': counts[n] for n in range(4)},
            **{f'owners_{n}_share': counts[n] / len(group) for n in range(4)}})
    table += macro_means(table, ('dataset', 'method', 'budget', 'scenario'),
                         ('mean_owner_contacts',) + tuple(f'owners_{n}_share' for n in range(4)),
                         ('queries', 'owner_contact_sum') + tuple(f'owners_{n}_count' for n in range(4)))
    return table


def paired_map(rows):
    out = defaultdict(dict)
    for row in rows:
        uid = row['uid']
        side = bool(row['answerable'])
        if side in out[uid]:
            raise ValueError('Duplicate side in public query pair')
        out[uid][side] = row
    if any(set(v) != {True, False} for v in out.values()):
        raise ValueError('Public scenario sides have different query universes')
    return out


def binary_outcomes(row):
    return bool(row['success']) and not bool(row['abstain']) if row['answerable'] else bool(row['abstain'])


def pair_statistics(treatment, control):
    A, B = paired_map(treatment), paired_map(control)
    if set(A) != set(B):
        raise ValueError('Comparison methods have different public query universes')
    uids = sorted(A)
    ta = [int(binary_outcomes(A[u][True])) for u in uids]
    ca = [int(binary_outcomes(B[u][True])) for u in uids]
    tu = [int(binary_outcomes(A[u][False])) for u in uids]
    cu = [int(binary_outcomes(B[u][False])) for u in uids]
    return {'balanced': paired_bootstrap(np.mean([ta, tu], axis=0), np.mean([ca, cu], axis=0),
                                        iterations=RESAMPLES, seed=BOOTSTRAP_SEED),
            'answerable_mcnemar': mcnemar(ta, ca), 'unanswerable_mcnemar': mcnemar(tu, cu)}


def macro_statistics(comparisons):
    """One global uid cluster draw preserves overlaps between the four settings."""
    if set(comparisons) != SETTINGS:
        raise ValueError('Macro statistics require the four protocol settings')
    all_uids = sorted({r['uid'] for a, _ in comparisons.values() for r in a})
    positions = {u: i for i, u in enumerate(all_uids)}
    membership, differences = [], []
    for cfg in sorted(comparisons):
        a, b = (paired_map(rows) for rows in comparisons[cfg])
        if set(a) != set(b):
            raise ValueError('Macro comparison query universe mismatch')
        mask = np.zeros(len(all_uids))
        diff = np.zeros(len(all_uids))
        for uid in a:
            mask[positions[uid]] = 1
            diff[positions[uid]] = np.mean([int(binary_outcomes(a[uid][side])) - int(binary_outcomes(b[uid][side]))
                                           for side in (True, False)])
        membership.append(mask)
        differences.append(diff)
    membership, differences = np.array(membership), np.array(differences)
    point = float(np.mean(differences.sum(axis=1) / membership.sum(axis=1)))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = []
    for _ in range(RESAMPLES):
        weights = rng.multinomial(len(all_uids), np.full(len(all_uids), 1 / len(all_uids)))
        denominators = membership @ weights
        if np.all(denominators > 0):
            draws.append(float(np.mean((differences @ weights) / denominators)))
    if not draws:
        raise ValueError('No valid question-cluster bootstrap draws')
    low, high = np.quantile(draws, [0.025, 0.975])
    return {'difference': point, 'ci_low': float(low), 'ci_high': float(high),
            'iterations': RESAMPLES, 'valid_draws': len(draws), 'seed': BOOTSTRAP_SEED,
            'unit': 'uid clusters, shared across settings', 'unique_queries': len(all_uids)}


def evaluation_statistics(treatment, control):
    key = lambda r: (r['uid'], r['scenario_id'])
    a, b = ({key(r): r for r in rows} for rows in (treatment, control))
    if set(a) != set(b):
        raise ValueError('Evaluation comparison scenario universe mismatch')
    clusters = sorted({k[0] for k in a})
    if not clusters:
        return {'difference': None, 'ci_low': None, 'ci_high': None, 'iterations': RESAMPLES,
                'valid_draws': 0, 'unit': 'question clusters', 'questions': 0}
    cells = []
    for uid in clusters:
        ans, una, da, du = 0, 0, 0, 0
        for k in a:
            if k[0] != uid:
                continue
            if a[k]['answerable'] != b[k]['answerable']:
                raise ValueError('Evaluation answerability mismatch')
            delta = int(binary_outcomes(a[k])) - int(binary_outcomes(b[k]))
            if a[k]['answerable']:
                ans += 1
                da += delta
            else:
                una += 1
                du += delta
        cells.append([ans, una, da, du])
    cells = np.asarray(cells, dtype=float)
    total = cells.sum(axis=0)
    point = 0.5 * (total[2] / total[0] + total[3] / total[1]) if total[0] and total[1] else None
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sums = cells[rng.integers(0, len(cells), size=(RESAMPLES, len(cells)))].sum(axis=1)
    valid = (sums[:, 0] > 0) & (sums[:, 1] > 0)
    draws = 0.5 * (sums[valid, 2] / sums[valid, 0] + sums[valid, 3] / sums[valid, 1])
    low, high = np.quantile(draws, [0.025, 0.975]) if len(draws) else (None, None)
    return {'difference': point, 'ci_low': low, 'ci_high': high, 'iterations': RESAMPLES,
            'valid_draws': int(valid.sum()), 'unit': 'question clusters', 'questions': len(clusters)}


def reference_table(public, evaluation):
    decisions = []
    groups = defaultdict(list)
    for r in public:
        base = {**r, 'abstain': bool(r['abstain_routing']), 'variant': 'recorded_routing'}
        gated = {**r, 'variant': 'plus_availability_test',
                 'abstain': (bool(r['abstain_routing']) or r['gate']) if r['answerable'] else bool(r['gate']),
                 'base_or_gate_abstain': bool(r['abstain_routing']) or r['gate']}
        for row in (base, gated):
            decisions.append(row)
            groups[(row['dataset'], row['config'], row['method'], row['variant'])].append(row)
    for r in evaluation:
        row = {**r, 'variant': 'plus_availability_test' if r['method'] == 'Centralized+avail' else 'recorded_routing'}
        decisions.append(row)
    keys = ('dataset', 'config', 'method', 'variant')
    table = summarize_groups([r for r in decisions if r['dataset'] != 'evaluation_set'], keys)
    table += summarize_groups([r for r in decisions if r['dataset'] == 'evaluation_set'], keys + ('scenario_family',))
    table += macro_means([r for r in table if r['dataset'] != 'evaluation_set'],
                         ('dataset', 'method', 'variant'), ('answerable_rate', 'unanswerable_rate', 'balanced'),
                         ('answerable', 'answerable_success', 'unanswerable', 'unanswerable_abstain', 'false_abstentions'))
    stats, macro = [], defaultdict(dict)
    for (ds, cfg, method, variant), rows in sorted(groups.items()):
        if variant != 'plus_availability_test':
            continue
        ctrl = groups[(ds, cfg, method, 'recorded_routing')]
        stats.append({'dataset': ds, 'config': cfg, 'method': method, 'comparison': 'plus test minus recorded routing',
                      **pair_statistics(rows, ctrl)})
        macro[(ds, method)][cfg] = (rows, ctrl)
    for (ds, method), comparisons in sorted(macro.items()):
        stats.append({'dataset': ds, 'config': 'mean_over_settings', 'method': method,
                      'comparison': 'plus test minus recorded routing', 'balanced': macro_statistics(comparisons)})
    for family in ('enumerated_176', 'paired_80'):
        a = [r for r in decisions if r.get('scenario_family') == family and r['method'] == 'Centralized+avail']
        b = [r for r in decisions if r.get('scenario_family') == family and r['method'] == 'Centralized']
        stats.append({'dataset': 'evaluation_set', 'scenario_family': family, 'comparison': 'Centralized+avail minus Centralized',
                      'balanced': evaluation_statistics(a, b)})
    # Preserve the enumeration summary's block names and count field names.
    blocks = {}
    evaluation_decisions = [r for r in decisions if r['dataset'] == 'evaluation_set' and r['scenario_family'] == 'enumerated_176']
    filters = {'all': lambda r: True, 'recorded_80': lambda r: bool(r['recorded_scenario']),
               'new_only': lambda r: not r['recorded_scenario'],
               'unanswerable_recorded_offline_client': lambda r: not r['answerable'] and bool(r['recorded_scenario']),
               'unanswerable_other_necessary_client': lambda r: not r['answerable'] and not r['recorded_scenario']}
    for name, fn in filters.items():
        blocks[name] = {}
        for method in sorted({r['method'] for r in evaluation_decisions}):
            summary = balanced([r for r in evaluation_decisions if r['method'] == method and fn(r)])
            for side, successes in (('answerable', 'answerable_success'), ('unanswerable', 'unanswerable_abstain')):
                n, k = summary[side], summary[successes]
                summary[f'{side}_ci'] = clopper_pearson(k, n) if n else None
            blocks[name][method] = summary
    return table, stats, blocks


def online_table(public_scores, evaluation_scores):
    training, tables, stats, decisions, macro = {}, [], [], [], defaultdict(dict)
    for ds in sorted({r['dataset'] for r in public_scores}):
        train = [r for r in public_scores if r['dataset'] == ds and r['split'] == 'train']
        test = [r for r in public_scores if r['dataset'] == ds and r['split'] == 'test']
        for form in ('raw', 'share'):
            theta, curve = select_training_theta(train, form)
            training[f'{ds}:{form}'] = {'theta': theta, 'sweep': curve,
                                       'training_config_counts': {cfg: sum(r['config'] == cfg for r in train) for cfg in SETTINGS}}
            for cfg in sorted({r['config'] for r in test}):
                rows = [r for r in test if r['config'] == cfg]
                treat = [{**r, 'abstain': rule_abstains(r, form, theta)} for r in rows]
                ctrl = [{**r, 'abstain': bool(r['gate']) or r['empty_evidence']} for r in rows]
                paired_map(rows)
                for variant, chosen in ((form, treat), ('availability', ctrl)):
                    # Availability is the same for both forms; emit it once.
                    if variant == 'availability' and form == 'share':
                        continue
                    tables.append({'dataset': ds, 'config': cfg, 'method': 'FedResolve-v3', 'variant': variant,
                                   'theta': theta if variant != 'availability' else None, **balanced(chosen)})
                stats.append({'dataset': ds, 'config': cfg, 'variant': form, 'theta': theta,
                              'comparison': 'online rule minus availability', **pair_statistics(treat, ctrl)})
                macro[(ds, form)][cfg] = (treat, ctrl)
                decisions += [{**r, 'variant': form, 'theta': theta} for r in treat]
    public_tables = list(tables)
    tables += macro_means(public_tables, ('dataset', 'method', 'variant', 'theta'),
                          ('answerable_rate', 'unanswerable_rate', 'balanced'),
                          ('answerable', 'answerable_success', 'unanswerable', 'unanswerable_abstain', 'false_abstentions'))
    for (ds, form), comparisons in sorted(macro.items()):
        stats.append({'dataset': ds, 'config': 'mean_over_settings', 'variant': form,
                      'comparison': 'online rule minus availability', 'balanced': macro_statistics(comparisons)})
    maxima = {}
    for family in ('enumerated_176', 'paired_80'):
        rows = [r for r in evaluation_scores if r['scenario_family'] == family]
        availability = [{**r, 'abstain': bool(r['gate']) or r['empty_evidence']} for r in rows]
        tables.append({'dataset': 'evaluation_set', 'config': 'c8_evaluation', 'method': 'FedResolve-v3',
                       'variant': 'availability', 'scenario_family': family, 'theta': None, **balanced(availability)})
        for form in ('raw', 'share'):
            values = sweep(rows, form)
            best = max((v['balanced'] for v in values if v['balanced'] is not None), default=None)
            maxima[f'{family}:{form}'] = {'maximum_balanced': best,
                'maximizing_grid_points': [v['theta'] for v in values if best is not None and v['balanced'] == best],
                'interpretation': 'Descriptive maximum on the fixed evaluation set; no threshold selected for use.'}
            tables += [{'dataset': 'evaluation_set', 'config': 'c8_evaluation', 'method': 'FedResolve-v3',
                        'variant': form, 'scenario_family': family, **v} for v in values]
            tables.append({'dataset': 'evaluation_set', 'config': 'c8_evaluation', 'method': 'FedResolve-v3',
                           'variant': form + '_descriptive_maximum', 'scenario_family': family,
                           'theta': None, 'balanced': best,
                           'maximizing_grid_points': maxima[f'{family}:{form}']['maximizing_grid_points']})
            # Uncertainty for every evaluation grid point is also descriptive.
            for value in values:
                treat = [{**r, 'abstain': rule_abstains(r, form, value['theta'])} for r in rows]
                stats.append({'dataset': 'evaluation_set', 'scenario_family': family, 'variant': form,
                              'theta': value['theta'], 'comparison': 'online rule minus availability',
                              'balanced': evaluation_statistics(treat, availability)})
    return tables, training, maxima, stats


def content_table(inputs):
    table = []
    for name in ('paired_missingness.csv', 'v3_missingness.csv'):
        with (inputs / ('public_paired_v1' if name.startswith('paired') else 'serf_v3') / name).open(encoding='utf-8-sig', newline='') as handle:
            for row in csv.DictReader(handle):
                if name.startswith('v3') and row['kind'] != 'paired':
                    continue
                n = int(row.get('paired_records', row.get('n')))
                cfg = row.get('config') or f"c{row['clients']}_{row['track']}"
                method = row['method']
                mode = 'availability_plus_refusal' if method in ('S2G-RAG+avail', 'FedResolve', 'FedResolve-v3') else 'generator_refusal'
                result = {'dataset': row['dataset'], 'config': cfg, 'method': method, 'variant': mode,
                          'source': name, 'budget': 2, 'answerable': n, 'unanswerable': n}
                for source_field, rate_field, count_field in (
                    ('answerable_success_evidence', 'answerable_rate', 'answerable_success'),
                    ('answerable_success_answer', 'answerable_answer_rate', 'answerable_answer_success'),
                    ('unanswerable_abstain_system', 'unanswerable_rate', 'unanswerable_abstain'),
                    ('unanswerable_abstain_routing', 'unanswerable_routing_rate', 'unanswerable_routing_abstain')):
                    rate = float(row[source_field])
                    count = round(rate * n)
                    if abs(rate * n - count) > 1e-7:
                        raise ValueError('Recorded rates cannot be converted to integer counts')
                    result[rate_field], result[count_field] = rate, count
                result['balanced'] = 0.5 * (result['answerable_rate'] + result['unanswerable_rate'])
                table.append(result)
    table += macro_means(table, ('dataset', 'method', 'variant', 'source', 'budget'),
                         ('answerable_rate', 'answerable_answer_rate', 'unanswerable_rate', 'unanswerable_routing_rate', 'balanced'),
                         ('answerable', 'answerable_success', 'answerable_answer_success', 'unanswerable',
                          'unanswerable_abstain', 'unanswerable_routing_abstain'))
    return table


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=ROOT / 'artifacts/controls/reference')
    p.add_argument('--inputs', type=Path, default=ROOT / 'artifacts')
    p.add_argument('--out', type=Path, default=ROOT / 'artifacts/controls/reference/results')
    p.add_argument('--limit', type=int)
    p.add_argument('--tables-only', action='store_true', help='Write the available B4 results only; no job inputs needed.')
    p.add_argument('--allow-smoke', action='store_true')
    args = p.parse_args()
    if args.limit is not None and args.limit <= 0:
        p.error('--limit must be positive')
    if not args.out.resolve().is_relative_to((ROOT / 'artifacts/controls/reference').resolve()):
        p.error('--out must stay inside artifacts/controls/reference')
    args.out.mkdir(parents=True, exist_ok=True)
    sources = {name: sha(args.inputs / ('public_paired_v1' if name.startswith('paired') else 'serf_v3') / name) for name in ('paired_missingness.csv', 'v3_missingness.csv')}
    content = content_table(args.inputs)
    content_path = args.out / 'content_abstention.json'
    if content_path.exists():
        existing = read_json(content_path)
        if existing['table'] != content or existing['source_sha256'] != sources or not (args.out / 'content_abstention.csv').exists():
            raise ValueError('Existing B4 output does not match the protocol input tables')
    else:
        emit(args.out, 'content_abstention', content, source_sha256=sources,
         definition='Recorded system outcomes on paired scenarios; evidence success and answer success are reported separately.',
         limitation='Generator-refusal group includes routing abstention from an empty page/client selection. The CSVs identify this group, but cannot isolate conditional generator refusal. Routing abstention counts are included.',
         statistics='Summary CSVs cannot reconstruct paired query decisions; no bootstrap or McNemar is inferred from marginal counts.')
    if args.tables_only:
        print('B4 recorded content abstention tables written. B1 through B3 await job outputs.')
        return
    manifests = {name: read_json(args.input / f'{name}_manifest.json') for name in ('public', 'evaluation')}
    smoke = any(m['smoke'] for m in manifests.values()) or args.limit is not None
    if smoke and not args.allow_smoke:
        raise ValueError('Smoke inputs cannot produce final results; use --allow-smoke in a separate output directory')
    design_path = HERE / 'protocol_fairness_v1_B_reference.json'
    if not design_path.is_file():
        raise ValueError('Input job design does not match the package protocol')
    if any(m['design_sha256'] != sha(design_path) for m in manifests.values()):
        raise ValueError('Job manifest design hash mismatch')
    names = ('public_owners', 'public_paired', 'public_scores',
             'evaluation_owners', 'evaluation_paired', 'evaluation_scores')
    data = {name: list(stream(args.input / f'{name}.jsonl')) for name in names}
    validate_downloads(data, manifests)
    data = limit_downloads(data, args.limit)
    def load(name):
        return data[name]
    owners = owner_table(load('public_owners') + load('evaluation_owners'))
    emit(args.out, 'owner_contacts', owners, smoke=smoke, manifests=manifests,
         definition='Distinct owners of cited pages; original centralized central call accounting is preserved in the input rows.')
    reference, reference_stats, enumeration = reference_table(load('public_paired'), load('evaluation_paired'))
    reference += [{**r, 'variant': 'recorded_system', 'metric_level': 'system'} for r in content]
    registered_enum = read_json(args.inputs / 'enumeration/enumeration_summary.json')
    for block, family in (('all', 'enumerated_176'), ('recorded_80', 'paired_80')):
        reference += [{'dataset': 'evaluation_set', 'config': 'c8_evaluation', 'method': method,
                       'variant': 'enumeration_reference', 'source': 'enumeration_summary.json',
                       'scenario_family': family, **summary} for method, summary in registered_enum[block].items()]
    emit(args.out, 'reference_with_test', reference, smoke=smoke, statistics=reference_stats,
         enumeration_summary=enumeration, recorded_system_tables=content,
         recorded_evaluation=read_json(args.inputs / 'enumeration/enumeration_summary.json'),
         definition='Public unanswerable +test success uses the router-independent gate alone. Answerable success requires relevant evidence and no routing or gate abstention. Evaluation retains fixed complete-client-set semantics.')
    online, training, maxima, online_stats = online_table(load('public_scores'), load('evaluation_scores'))
    emit(args.out, 'online_only_rule', online, smoke=smoke, training_selection=training,
         evaluation_maxima=maxima, statistics=online_stats,
         recorded_margins=read_json(args.inputs / 'enumeration/margins_summary.json'),
         recorded_enumeration=read_json(args.inputs / 'enumeration/enumeration_summary.json'),
         definition='Replace the availability gate of fixed SERF-v3 with a strict online raw-score or all-eligible softmax-share threshold. Public thresholds use training rows only; evaluation maxima are descriptive.')
    write_csv(args.out / 'reference_with_test_statistics.csv', reference_stats)
    write_csv(args.out / 'online_only_rule_statistics.csv', online_stats)
    write_json(args.out / 'aggregation_manifest.json', {'smoke': smoke, 'server_manifests': manifests,
                                                       'recorded_table_sha256': sources,
                                                       'job_protocol_sha256': sha(design_path)})
    print('All four JSON and CSV tables written.')


if __name__ == '__main__':
    main()
