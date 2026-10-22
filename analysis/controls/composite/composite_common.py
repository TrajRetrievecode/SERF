"""Shared construction, imported selection, metrics and immutable output helpers."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from e2e_common import ROOT
from unittest.mock import patch

import numpy as np

from fedresolve_core import build_federation, family_key, split_records, stable_u64
from e2e_slots import decompose
from e2e_slot_adapters import ADAPTERS, QueryViews, _critical_missing, softmax
import e2e_v3_adapter


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--out', type=Path, default=ROOT / 'artifacts/controls/composite')
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--protocol', type=Path, default=Path(__file__).resolve().parent /
                   'protocol_fairness_v1_C_composite.json')
    return p


def arguments(p):
    args = p.parse_args()
    if args.limit is not None and args.limit < 1:
        p.error('--limit must be positive')
    args.out = args.out.resolve()
    if not args.out.is_relative_to((ROOT / 'artifacts/controls/composite').resolve()):
        p.error('--out must stay inside artifacts/controls/composite')
    args.out.mkdir(parents=True, exist_ok=True)
    return args


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def fresh(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f'Existing output must not be modified: {path}')
    return path


def write_json(path, value):
    path = fresh(path)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def write_jsonl(path, rows):
    with fresh(path).open('x', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def protocol_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def settings(dataset, protocol, records, index, inventory):
    docs = sorted(set(index.doc_key_list) | set(inventory))
    for clients in protocol['client_scales']:
        seed = protocol['seed'] + clients + (0 if dataset == 'finrag' else 50000)
        federation = build_federation(docs, num_clients=clients, seed=seed)
        for track in protocol['tracks']:
            yield f'c{clients}_{track}', seed, federation, split_records(
                records, track=track, seed=seed + 101)


def stripped(text):
    return str(text).strip().rstrip('?？').strip()


def construct(records, span_by_uid, federation, seed, dataset, config, portion, limit=None):
    candidates = []
    owners = {}
    families = {}
    for record in records:
        start, stop = span_by_uid[record.uid]
        units = record.gold_pages if dataset == 'finrag' else record.doc_keys
        if stop - start != 1 or not record.doc_keys or not units:
            continue
        clients = federation.gold_clients(record)
        if len(clients) == 1:
            candidates.append(record)
            owners[record.uid] = int(clients[0])
            families[record.uid] = {family_key(d) for d in record.doc_keys}
    candidates.sort(key=lambda r: (stable_u64(r.uid, seed), r.uid))
    used, output = set(), []
    attempted = 0
    for i, left in enumerate(candidates):
        if left.uid in used:
            continue
        right = next((r for r in candidates[i + 1:]
                      if r.uid not in used and r.lang == left.lang
                      and owners[r.uid] != owners[left.uid]
                      and families[r.uid].isdisjoint(families[left.uid])), None)
        if right is None:
            continue
        used.update((left.uid, right.uid))
        attempted += 1
        chinese = left.lang.lower().split('-')[0] in ('ch', 'zh', 'cn')
        query = left.query + ('以及' if chinese else ' and ') + right.query
        slots = decompose(query)
        if len(slots) != 2 or [stripped(s) for s in slots] != [
                stripped(left.query), stripped(right.query)]:
            continue
        parts = []
        for number, r in enumerate((left, right), 1):
            units = r.gold_pages if dataset == 'finrag' else r.doc_keys
            parts.append({'part': number, 'uid': r.uid, 'text': r.query,
                          'gold_units': sorted(set(units)),
                          'documents': sorted(set(r.doc_keys)),
                          'families': sorted(families[r.uid]), 'client': owners[r.uid]})
        output.append({'dataset': dataset, 'config': config, 'portion': portion,
                       'seed': seed, 'uid': f'comp:{left.uid}|{right.uid}',
                       'uids': [left.uid, right.uid], 'texts': [left.query, right.query],
                       'query': query, 'slot_texts': slots, 'lang': left.lang,
                       'parts': parts, 'gold_unit': 'page' if dataset == 'finrag' else 'document',
                       'gold_units': sorted(set(parts[0]['gold_units']) | set(parts[1]['gold_units'])),
                       'gold_clients': sorted([owners[left.uid], owners[right.uid]])})
    count = len(candidates)
    kept = len(output)
    processed = output if limit is None else output[:limit]
    audit = {'dataset': dataset, 'config': config, 'portion': portion,
             'seed': seed, 'source_queries': len(records), 'candidates': count,
             'attempted_pairs': attempted, 'kept_pairs': kept,
             'share_candidates_paired': 2 * attempted / count if count else None,
             'share_candidates_retained': 2 * kept / count if count else None,
             'share_pairs_kept': kept / attempted if attempted else None,
             'processed_composites': len(processed), 'limit': limit}
    return processed, audit


def online_mask(composite, scenario, clients, seed):
    online = np.ones(clients, dtype=bool)
    if scenario == 'part_offline':
        online[composite['parts'][1]['client']] = False
    elif scenario == 'other_offline':
        pool = [c for c in range(clients) if c not in composite['gold_clients']]
        if not pool:
            raise ValueError('No non-gold client for the answerable paired scenario')
        chosen = min(pool, key=lambda c: (
            stable_u64(f"unneeded_offline|{composite['uid']}|{c}", seed), c))
        online[chosen] = False
    elif scenario != 'full':
        raise ValueError(scenario)
    return online


def make_views(full, slots, client_view, protocol):
    p = protocol['fedresolve']
    return QueryViews(full_page=full, slot_page=slots,
                      rows_by_client=client_view.rows_by_client,
                      num_clients=client_view.num_clients,
                      claim_query_weight=p['claim_query_weight'],
                      full_query_weight=p['full_query_weight'],
                      temperature=p['global_temperature'])


def pooled_selection(views, online, budget, protocol, routed):
    """Run the imported pooled selector with an already fixed routing result.

    The upstream function combines routing and selection. Temporarily inject its
    routing dependency, restoring it on exit. No source is copied or modified;
    execution is deliberately single threaded.
    """
    routed = {**routed, 'trace': {**routed.get('trace', {}), 'critical_missing_slots': []}}
    with patch.object(e2e_v3_adapter, 'fedresolve_adapter', return_value=routed):
        return e2e_v3_adapter.fedresolve_v3_adapter(views, online, budget, protocol)


def route(method, variant, views, online, budget, protocol, client_view, index, composite):
    adapter = e2e_v3_adapter.fedresolve_v3_adapter if method == 'FedResolve-v3' else ADAPTERS[method]
    result = adapter(views, online, budget, protocol,
                     page_client=client_view.page_client, visual=index.visual,
                     gold_clients=composite['gold_clients'])
    if variant == 'native':
        return result
    if variant != 'pooled_share' or method not in ('RAGRoute', 'UniversalRAG', 'S2G-RAG'):
        raise ValueError((method, variant))
    accepted, steps = [], []
    for k, client in enumerate(result['clients']):
        if method == 'RAGRoute':
            scores = views.full_client
        elif method == 'UniversalRAG':
            scores = views.slot_client.max(axis=0)
        else:
            slot = result['trace']['gap_steps'][k]['slot']
            scores = views.slot_client[slot]
        eligible = online & np.isfinite(scores)
        share = float(softmax(np.where(eligible, scores, -np.inf),
                              protocol['fedresolve']['global_temperature'])[client])
        steps.append({'client': int(client), 'share': share,
                      'accepted': share >= protocol['fedresolve']['minimum_marginal_gain']})
        if share < protocol['fedresolve']['minimum_marginal_gain']:
            break
        accepted.append(int(client))
    result = {**result, 'clients': accepted,
              'trace': {**result['trace'], 'minimum_share_steps': steps}}
    return pooled_selection(views, online, budget, protocol, result)


def selection_variants(protocol):
    for method in protocol['methods']:
        yield method, 'native'
        if method in ('RAGRoute', 'UniversalRAG', 'S2G-RAG'):
            yield method, 'pooled_share'


def evidence_metrics(composite, page_uids):
    units = set(page_uids) if composite['gold_unit'] == 'page' else {
        u.rsplit('#p', 1)[0] for u in page_uids}
    hits = [bool(units.intersection(part['gold_units'])) for part in composite['parts']]
    gold = set(composite['gold_units'])
    return {'part_recall': sum(hits) / 2, 'complete_evidence': all(hits),
            'part_hits': hits, 'evidence_recall': len(units & gold) / len(gold)}


def online_statistic(views, online):
    values = np.where(online[None, :] & np.isfinite(views.combined), views.combined, -np.inf)
    return float(values.max(axis=1).min())


def gates(result, views, online, protocol, theta):
    empty = not result['rows']
    critical = _critical_missing(views, online, protocol)
    return {'native': bool(result['abstain']),
            'availability': bool(critical) or empty,
            'online_only': online_statistic(views, online) < theta or empty}, critical


def fit_threshold(training, protocol):
    if not training:
        raise ValueError('No training composites; threshold cannot be fitted')
    answer = np.array([r['answerable_score'] for r in training])
    missing = np.array([r['unanswerable_score'] for r in training])
    complete = np.array([r['answerable_complete'] for r in training], dtype=bool)
    sweep = []
    for tick in range(-200, 202):
        theta = tick / 200
        score = float((complete & (answer >= theta)).mean() / 2 + (missing < theta).mean() / 2)
        sweep.append({'theta': theta, 'balanced_accuracy': score})
    best = max(sweep, key=lambda r: (r['balanced_accuracy'], -r['theta']))
    return {'theta': best['theta'], 'training_balanced_accuracy': best['balanced_accuracy'],
            'training_composites': len(training), 'sweep': sweep}


class CompositeQueries:
    """Load new vectors and use the existing SlotQueries.page_scores for both views."""
    def __init__(self, root, index, dense_weight):
        from scipy import sparse
        from run_v3_plan import SlotQueries
        manifest = read_json(Path(root) / 'manifest.json')
        self.manifest = manifest
        self.positions = {uid: i for i, uid in enumerate(manifest['uids'])}
        self.spans = manifest['spans']
        self.slots = SlotQueries.__new__(SlotQueries)
        self.full = SlotQueries.__new__(SlotQueries)
        for holder, prefix in ((self.slots, 'slot'), (self.full, 'full')):
            holder.index = index
            holder.dense_weight = dense_weight
            holder.visual = np.load(Path(root) / f'{prefix}_visual.npy')
            holder.sparse = sparse.load_npz(Path(root) / f'{prefix}_text.npz').tocsr()
            holder.dense = np.load(Path(root) / f'{prefix}_dense.npy')
        n = len(self.positions)
        if n != len(manifest['uids']) or len(self.spans) != n:
            raise ValueError('Composite manifest uid/span mismatch')
        for holder, count in ((self.full, n), (self.slots, 2 * n)):
            if any(a.shape[0] != count for a in (holder.visual, holder.sparse, holder.dense)):
                raise ValueError('Composite feature row mismatch')
            if holder.sparse.shape[1] != index.text.shape[1] or holder.visual.shape[1] != index.visual.shape[1]:
                raise ValueError('Composite feature dimension mismatch')
            if dense_weight > 0 and holder.dense.shape[1] != index.dense.shape[1]:
                raise ValueError('Composite dense dimension mismatch')
        if self.spans != [[2 * i, 2 * i + 2] for i in range(n)]:
            raise ValueError('Composite cache must have exactly two consecutive slots per uid')

    def scores(self, uid, blend):
        i = self.positions[uid]
        start, stop = self.spans[i]
        return (self.full.page_scores(slice(i, i + 1), blend)[0],
                self.slots.page_scores(slice(start, stop), blend))
