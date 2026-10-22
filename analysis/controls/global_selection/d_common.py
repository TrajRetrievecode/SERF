"""Read-only dependencies, global selection, checked references and fresh outputs."""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True
import argparse
import os
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace
import numpy as np

JOB = 'global_selection'
ROOT = Path(os.environ.get('SERF_ROOT', str(Path(__file__).resolve().parents[3])))
HERE = Path(__file__).resolve().parent
SETTINGS = ('c8_seen_query', 'c8_unseen_family', 'c16_seen_query', 'c16_unseen_family')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
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
        raise ValueError('Nonfinite output value')
    return value


def fresh(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open('x', encoding='utf-8', newline='\n')


def save(path, value):
    with fresh(path) as f:
        json.dump(clean(value), f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def stream(path):
    with Path(path).open(encoding='utf-8-sig') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def emit(f, row):
    f.write(json.dumps(clean(row), ensure_ascii=False, allow_nan=False) + '\n')


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--limit', type=int)
    p.add_argument('--design', type=Path, default=HERE / 'protocol_fairness_v1_D_global_selection.json')
    for job, short in [('attribution', 'a'), ('reference', 'b'), ('composite', 'c')]:
        p.add_argument('--' + short + '-out', type=Path)
    return p


def setup(args):
    if args.limit is not None and args.limit < 1:
        raise ValueError('--limit must be positive')
    args.root, args.out = args.root.resolve(), args.out.resolve()
    if args.out not in [args.root / 'artifacts/controls' / n for n in (JOB, JOB + '_smoke')]:
        raise ValueError('Output must be this job or its smoke artifact directory')
    args.out.mkdir(parents=True, exist_ok=True)
    for short, job in [('a', 'attribution'), ('b', 'reference'), ('c', 'composite')]:
        setattr(args, short + '_out', getattr(args, short + '_out') or args.root / 'artifacts/controls' / job)
    # The launchers add these paths too; standalone commands retain the same imports.
    for rel in ('scripts/serf_v3_ablation', 'scripts/serf_v3', 'scripts/public_paired_v1',
                'scripts/public_e2e_v1', 'src/public_e2e_v1', 'src', 'scripts'):
        sys.path.append(str(args.root / rel))
    for job in ('attribution', 'reference', 'composite'):
        base = args.root / 'analysis/controls' / job
        sys.path.append(str(base))
    design = read(args.design)
    required = ('attribution/attribution.py', 'attribution/run_public.py',
                'attribution/run_evaluation.py', 'reference/b_common.py',
                'reference/public_rows.py', 'reference/evaluation_rows.py',
                'composite/composite_common.py')
    for rel in required:
        path = args.root / 'analysis/controls' / rel
        if digest(path) != design['earlier_helpers_sha256'][rel]:
            raise ValueError('Earlier helper differs from protocol: ' + rel)
    import importlib
    pin(design, **{name: importlib.import_module(name) for name in
                  ('run_v3_plan', 'e2e_slot_adapters', 'e2e_v3_adapter', 'fedresolve_core',
                   'e2e_retrieval', 'e2e_records')})
    return design


def module(name, path, expected=None):
    path = Path(path)
    if expected is not None and digest(path) != expected:
        raise ValueError('Source hash mismatch: ' + name)
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def earlier(args, job, filename):
    path = args.root / 'analysis/controls' / job / filename
    if not path.is_file():
        raise FileNotFoundError('Missing earlier job helper: ' + filename)
    return module('d_' + job + '_' + path.stem, path)


def pin(design, **modules):
    for name, mod in modules.items():
        if digest(mod.__file__) != design['source_sha256'][name]:
            raise ValueError('Fixed module differs: ' + name)


class Sources:
    def __init__(self):
        self.paths = {}

    def use(self, path):
        path = Path(path)
        self.paths[str(path)] = digest(path)
        return path


def checked_protocol(args, design, sources):
    from b_common import check_constants
    public_path = sources.use(args.root / 'configs/public_e2e_v1/protocol_v3.json')
    gold_path = sources.use(args.root / 'configs/formal_gold_v1_protocol.json')
    if digest(public_path) != design['input_protocol_sha256']['PROTOCOL_v3.json']:
        raise ValueError('Public protocol differs from protocol')
    if digest(gold_path) != design['input_protocol_sha256']['formal_gold_v1_protocol.json']:
        raise ValueError('Evaluation protocol differs from protocol')
    public, gold = read(public_path), read(gold_path)
    check_constants(public, gold)
    if design['constants'] != gold['fedresolve']:
        raise ValueError('Design constants differ')
    return public, gold


def key(row):
    return row['config'], row['uid'], row['scenario'], int(row['budget'])


def index_rows(rows):
    result = {}
    for row in rows:
        k = key(row)
        if k in result:
            raise ValueError('Duplicate reference: ' + str(k))
        result[k] = row
    return result


def limited(rows, limit):
    chosen = {}
    out = []
    for row in rows:
        k = row['dataset'], row['config']
        uids = chosen.setdefault(k, [])
        if row['uid'] not in uids and (limit is None or len(uids) < limit):
            uids.append(row['uid'])
        if row['uid'] in uids:
            out.append(row)
    return out


def same(got, recorded, fields):
    for field in fields:
        a, b = got[field], recorded[field]
        if isinstance(a, (int, float, np.generic)) and isinstance(b, (int, float, np.generic)):
            equal = math.isclose(float(a), float(b), abs_tol=1e-12, rel_tol=1e-12)
        else:
            equal = a == b
        if not equal:
            raise ValueError('Recorded value differs: ' + field)


def verify_gate(views, online, protocol, recorded):
    from e2e_slot_adapters import _critical_missing
    obj = SimpleNamespace(**views) if isinstance(views, dict) else views
    gate = bool(_critical_missing(obj, online, protocol))
    if gate != bool(recorded):
        raise ValueError('Availability decision differs from earlier source')
    return gate


def public_selections(views, online, budget, protocol, page_client, gate):
    from e2e_slot_adapters import centralized_adapter
    from attribution import public_selection
    base = centralized_adapter(views, online, budget, protocol, page_client=page_client)
    eligible = [c for c in range(views.num_clients)
                if online[c] and views.rows_by_client[c].size]
    pooled = public_selection(views, eligible, budget, protocol)
    for variant, rows, abstain in [('base', base['rows'], base['abstain']),
                                   ('+sel', pooled, not pooled),
                                   ('+sel+test', pooled, bool(gate) or not pooled)]:
        proposed_owners = sorted({int(page_client[r]) for r in rows})
        yield variant, {'rows': list(rows), 'clients': proposed_owners, 'abstain': bool(abstain),
                        'trace': {'access': 'centralized global index', 'variant': variant,
                                  'availability_decision': bool(gate)}, 'eligible_clients': eligible}


def public_row(a_public, dataset, config, record, view, index, texts, protocol,
               variant, budget, scenario, result, gate, elapsed=0.0, evidence_score=None):
    ranked = (np.argsort(-evidence_score)[:protocol['evidence']['ranked_candidate_depth']]
              if evidence_score is not None else [])
    row = a_public.plan_row(dataset, config, view.num_clients, config.split('_', 1)[1],
                            record, view, index, texts, protocol, 'Centralized', variant,
                            budget, result, ranked, elapsed)
    owners = [] if result['abstain'] else list(result['clients'])
    # Ranking happens at one central store. Pool eligibility is not a client call.
    calls = int(bool(result['rows']))
    from e2e_common import package_bytes
    from e2e_retrieval import CERTIFICATE_BYTES_PER_CALL
    cost = {**row['cost'], 'client_calls': calls,
            'serialized_request_bytes': package_bytes({'query': record.query, 'page_budget': budget}) * calls,
            'certificate_bytes': CERTIFICATE_BYTES_PER_CALL * calls,
            'protocol_bytes': row['protocol_bytes'] - row['certificate_bytes'] + CERTIFICATE_BYTES_PER_CALL * calls}
    if result['abstain']:
        cost['serialized_response_bytes'] = CERTIFICATE_BYTES_PER_CALL * calls
    return {**row, 'row_key': '|'.join(map(str, (dataset, config, variant, budget, scenario, record.uid))),
            'scenario': scenario, 'selected_clients': owners, 'owners': owners,
            'owner_contacts': len(owners), 'proposed_clients': list(result['clients']),
            'pool_clients': result['eligible_clients'], 'availability_decision': bool(gate),
            'candidate_pages': row['candidate_pages'] if variant == 'base' else
                               sum(view.rows_by_client[c].size for c in result['eligible_clients']),
            'cost': cost, **cost}


def reference(row, panel, method='FedResolve-v3', variant='recorded'):
    pages = list(row.get('selected_page_uids', row.get('evidence_page_uids', [])))
    calls = row.get('client_calls', row.get('calls', row.get('cost', {}).get('client_calls')))
    if calls is None:
        raise ValueError('Recorded client calls are missing')
    return {**row, 'panel': panel, 'method': method, 'variant': variant,
            'selected_page_uids': pages, 'evidence_page_uids': pages,
            'selected_pages': len(pages), 'client_calls': calls,
            'abstain_routing': bool(row.get('abstain_routing', row.get('abstain', False)))}


def finish(args, stage, rows, refs, sources, started, **extra):
    import time
    with fresh(args.out / (stage + '.jsonl')) as f:
        for row in rows:
            emit(f, row)
    with fresh(args.out / (stage + '_references.jsonl')) as f:
        for row in refs:
            emit(f, row)
    save(args.out / (stage + '_manifest.json'), {'limit': args.limit, 'rows': len(rows),
         'references': len(refs), 'sources': sources.paths, 'design_sha256': digest(args.design),
         'elapsed_seconds': time.time() - started, **extra})
