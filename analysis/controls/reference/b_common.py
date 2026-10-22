"""Shared score statistics, row validation and exclusive output helpers."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np

from e2e_common import ROOT
SCENARIOS = ('unneeded_offline', 'qrel_missing')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(clean(value), handle, indent=2, allow_nan=False)
        handle.write('\n')


def stream(path):
    with Path(path).open(encoding='utf-8-sig') as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


class Writer:
    def __init__(self, path):
        self.handle = Path(path).open('x', encoding='utf-8', newline='\n')

    def write(self, row):
        self.handle.write(json.dumps(clean(row), allow_nan=False) + '\n')

    def close(self):
        self.handle.close()


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--limit', type=int, default=None)
    return p


def setup(args):
    if args.limit is not None and args.limit <= 0:
        raise ValueError('--limit must be positive')
    args.root = args.root.resolve()
    args.out = args.out.resolve()
    allowed = args.root / 'artifacts' / 'controls'
    if args.out.name not in ('reference', 'reference_smoke') or args.out.parent != allowed:
        raise ValueError('Output must be the reference or reference_smoke job artifact directory')
    args.out.mkdir(parents=True, exist_ok=True)
    for rel in ('scripts/public_paired_v1', 'scripts/serf_v3', 'scripts/public_e2e_v1',
                'src/public_e2e_v1', 'src', 'scripts'):
        sys.path.insert(0, str(args.root / rel))


def check_constants(public, gold):
    expected = dict(global_temperature=0.07, claim_query_weight=0.75, full_query_weight=0.25,
                    minimum_marginal_gain=0.05, missingness_raw_margin=0.02,
                    missingness_probability_ratio=1.15)
    if gold['fedresolve'] != expected:
        raise ValueError('Evaluation routing constants changed')
    if public['seed'] != 20260728 or public['client_scales'] != [8, 16] or public['tracks'] != ['seen_query', 'unseen_family']:
        raise ValueError('Protocol public seeds or settings changed')
    if public['retriever']['blend_weight'] != {'finrag': 0.25, 'visdom': 0.5}:
        raise ValueError('Protocol public blend changed')
    if public['retriever']['dense_weight'] != {'finrag': 0.5, 'visdom': 0.0}:
        raise ValueError('Protocol dense weights changed')


def row_key(row):
    return row['config'], row['uid'], row['scenario'], int(row['budget']), row['method']


def unique_index(rows):
    out = {}
    for row in rows:
        key = row_key(row)
        if key in out:
            raise ValueError(f'Duplicate plan key: {key}')
        out[key] = row
    return out


def score_statistics(combined, probabilities, online):
    """Share retains the denominator over offline as well as online clients."""
    z = np.asarray(combined)
    p = np.asarray(probabilities)
    online = np.asarray(online, dtype=bool)
    if z.ndim != 2 or z.shape != p.shape or z.shape[1] != online.size or not len(z):
        raise ValueError('Incompatible score views')
    raw, share = [], []
    for j in range(len(z)):
        order = np.argsort(-z[j])
        best = next((int(c) for c in order if online[c] and np.isfinite(z[j, c])), None)
        if best is None:
            return {'raw_min': None, 'share_min': None, 'no_online_eligible': True}
        raw.append(float(z[j, best]))
        share.append(float(p[j, best]))
    return {'raw_min': min(raw), 'share_min': min(share), 'no_online_eligible': False}


def grid(form):
    return [i / 200 for i in range(-200 if form == 'raw' else 0, 201)]


def rule_abstains(row, form, theta):
    value = row[f'{form}_min']
    return bool(row.get('empty_evidence', False)) or value is None or value < theta


def owners_of(pages, index, view):
    owners = sorted({int(view.page_client[index.uid_to_row[p]]) for p in pages})
    if any(c < 0 for c in owners):
        raise ValueError('Cited page has no federation owner')
    return owners


def verify_centralized(first, second, index, view):
    if first['evidence_page_uids'] != second['evidence_page_uids']:
        raise ValueError('Centralized pages differ between existing public sources')
    owners = owners_of(first['evidence_page_uids'], index, view)
    for row in (first, second):
        if sorted(set(row['selected_clients'])) != owners:
            raise ValueError('Centralized selected_clients does not equal cited-page owners')
    return owners


def public_gate(rows):
    """Assert that S2G routing did not add an unrelated abstention."""
    by = {r['method']: r for r in rows}
    if len(by) != len(rows):
        raise ValueError('Duplicate method in paired scenario')
    base, gated, serf = (by[m] for m in ('S2G-RAG', 'S2G-RAG+avail', 'FedResolve'))
    if base['abstain_routing'] or not base['evidence_page_uids']:
        raise ValueError('Cannot isolate availability gate from empty S2G evidence')
    gate = bool(gated['abstain_routing'])
    if serf['selected_clients'] and bool(serf['abstain_routing']) != gate:
        raise ValueError('Availability gate disagrees with FedResolve')
    if not serf['selected_clients'] and not serf['abstain_routing']:
        raise ValueError('Empty SERF selection must abstain')
    return gate


def balanced(rows, abstain_key='abstain', success_key='success'):
    ans = [r for r in rows if r['answerable']]
    una = [r for r in rows if not r['answerable']]
    ao = sum(bool(r[success_key]) and not bool(r[abstain_key]) for r in ans)
    uo = sum(bool(r[abstain_key]) for r in una)
    ar = ao / len(ans) if ans else None
    ur = uo / len(una) if una else None
    return {'answerable': len(ans), 'answerable_success': ao, 'answerable_rate': ar,
            'unanswerable': len(una), 'unanswerable_abstain': uo, 'unanswerable_rate': ur,
            'false_abstentions': sum(bool(r[abstain_key]) for r in ans),
            'balanced': 0.5 * (ar + ur) if ar is not None and ur is not None else None}


def sweep(rows, form):
    out = []
    for theta in grid(form):
        decisions = [{**r, 'abstain': rule_abstains(r, form, theta)} for r in rows]
        out.append({'theta': theta, **balanced(decisions)})
    return out


def select_training_theta(rows, form):
    if not rows or any(r['split'] != 'train' for r in rows):
        raise ValueError('Threshold selection requires training rows only')
    values = sweep(rows, form)
    if any(v['balanced'] is None for v in values):
        raise ValueError('Training selection needs both scenario sides')
    best = max(v['balanced'] for v in values)
    return min(v['theta'] for v in values if v['balanced'] == best), values
