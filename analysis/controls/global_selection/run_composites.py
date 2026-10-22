"""Read Job C's composites and vectors; apply global selection without encoding."""
from __future__ import annotations
import time
from types import SimpleNamespace
import numpy as np
import d_common as d


def protocol_path(args):
    name = 'protocol_fairness_v1_C_composite.json'
    for path in (args.c_out / name, args.root / 'analysis/controls/composite' / name):
        if path.is_file():
            return path
    raise FileNotFoundError('Job C protocol is missing from its artifact or script directory')


def run_dataset(args, dataset, composites, design, sources, public, gold):
    import composite_common as c
    from e2e_records import load_records
    from e2e_retrieval import PageIndex, ClientPageView
    from public_rows import require_consolidated_corpus
    from e2e_corpus import page_text_map
    a_public = d.earlier(args, 'attribution', 'run_public.py')
    c_protocol_path = protocol_path(args)
    c_protocol = d.read(sources.use(c_protocol_path))
    if c_protocol['fedresolve'] != design['constants'] or any(
        c_protocol['retriever'][k] != public['retriever'][k] for k in ('blend_weight', 'dense_weight')):
        raise ValueError('Composite constants differ')
    plan = list(d.stream(sources.use(args.c_out / 'plans' / dataset / 'plan.jsonl')))
    central = d.index_rows(r for r in plan if r['method'] == 'Centralized'
                           and r['variant'] == 'native' and r['policy'] == 'native')
    serf = d.index_rows(r for r in plan if r['method'] == 'FedResolve-v3'
                      and r['variant'] == 'native' and r['policy'] == 'native')
    gates = {}
    masks = {}
    for row in plan:
        k = row['config'], row['uid'], row['scenario']
        pair = bool(row['availability_decision']), row['offline_clients']
        if k in gates and (gates[k], masks[k]) != pair:
            raise ValueError('Composite gate or offline mask is method dependent')
        gates[k], masks[k] = pair
    require_consolidated_corpus(args.root, dataset)
    records, _, inventory = load_records(dataset)
    index = PageIndex.load(dataset, public['retriever']['blend_weight'][dataset])
    dense = public['retriever']['dense_weight'][dataset]
    if dense > 0:
        index.attach_dense(dataset, dense)
    cache = args.c_out / 'queries' / dataset
    for name in ('manifest.json', 'full_visual.npy', 'full_text.npz', 'full_dense.npy',
                 'slot_visual.npy', 'slot_text.npz', 'slot_dense.npy'):
        sources.use(cache / name)
    queries = c.CompositeQueries(cache, index, dense)
    if queries.manifest['protocol_sha256'] != d.digest(c_protocol_path):
        raise ValueError('Composite vector protocol differs')
    settings = {cfg: (fed, splits, ClientPageView(index, fed.doc_to_client, fed.num_clients))
                for cfg, _, fed, splits in c.settings(dataset, c_protocol, records, index, inventory)}
    texts = page_text_map(dataset)
    rows, refs, missing = [], [], []
    for composite in composites:
        cfg, uid = composite['config'], composite['uid']
        fed, splits, view = settings[cfg]
        if not set(composite['uids']).issubset({r.uid for r in splits['test']}):
            raise ValueError('Composite does not belong to the test portion')
        for part in composite['parts']:
            if {fed.doc_to_client[x] for x in part['documents']} != {part['client']}:
                raise ValueError('Composite federation differs')
        full, slots = queries.scores(uid, public['retriever']['blend_weight'][dataset])
        views = c.make_views(full, slots, view, c_protocol)
        for scenario in ('full', 'part_offline', 'other_offline'):
            sk = cfg, uid, scenario
            online = np.ones(view.num_clients, dtype=bool)
            online[masks[sk]] = False
            if not np.array_equal(online, c.online_mask(composite, scenario, view.num_clients, c_protocol['seed'])):
                raise ValueError('Recorded composite offline mask differs')
            gate = d.verify_gate(views, online, gold, gates[sk])
            for budget in (2, 3):
                k = cfg, uid, scenario, budget
                if k in serf:
                    r = serf[k]
                    d.same(c.evidence_metrics(composite, r['selected_page_uids']), r,
                           ('part_recall', 'complete_evidence', 'evidence_recall', 'part_hits'))
                    if bool(r['availability_decision']) != gate or r['offline_clients'] != masks[sk]:
                        raise ValueError('Recorded composite SERF decision differs')
                    if int(r['cost']['client_calls']) != len(r['selected_clients']):
                        raise ValueError('Recorded composite SERF cost differs')
                    ref = d.reference(r, 'composite')
                    if scenario != 'full':
                        ref['answerable'] = scenario == 'other_offline'
                    refs.append(ref)
                else:
                    if budget != 3 or scenario == 'full':
                        raise ValueError('Missing required Job C SERF row')
                    missing.append({'dataset': dataset, 'config': cfg, 'uid': uid,
                                    'budget': budget, 'scenario': scenario,
                                    'reason': 'Job C records offline scenarios at budget 2 only'})
                record = SimpleNamespace(uid=uid, query=composite['query'],
                                          gold_unit=composite['gold_unit'],
                                          gold_pages=composite['gold_units'], doc_keys=composite['gold_units'])
                for variant, result in d.public_selections(views, online, budget, gold, view.page_client, gate):
                    row = d.public_row(a_public, dataset, cfg, record, view, index, texts, public,
                                       variant, budget, scenario, result, gate,
                                       evidence_score=views.evidence_score)
                    row.update(panel='composite', offline_clients=masks[sk],
                               **c.evidence_metrics(composite, row['selected_page_uids']))
                    if scenario != 'full':
                        row['answerable'] = scenario == 'other_offline'
                    # Job C charges owners; D retains the central-store cost
                    # separately and verifies identical base evidence and contacts.
                    if variant == 'base' and k in central:
                        recorded = central[k]
                        d.same(row, recorded, ('selected_page_uids', 'selected_clients', 'abstain_routing',
                                               'part_recall', 'complete_evidence', 'selected_pages'))
                        if row['owner_contacts'] != recorded['client_calls']:
                            raise ValueError('Job C Centralized contact count differs')
                    rows.append(row)
    return rows, refs, missing


def main():
    args = d.parser(__doc__).parse_args()
    design = d.setup(args)
    sources = d.Sources()
    public, gold = d.checked_protocol(args, design, sources)
    started = time.time()
    composites = d.limited([r for r in d.stream(sources.use(args.c_out / 'composites.jsonl'))
                            if r['portion'] == 'test'], args.limit)
    rows, refs, missing = [], [], []
    for dataset in ('finrag', 'visdom'):
        a, b, m = run_dataset(args, dataset, [r for r in composites if r['dataset'] == dataset],
                               design, sources, public, gold)
        rows.extend(a)
        refs.extend(b)
        missing.extend(m)
    d.finish(args, 'composite', rows, refs, sources, started, unavailable_serf_pairs=missing,
             cohorts={ds: {cfg: sum(r['dataset'] == ds and r['config'] == cfg for r in composites)
                           for cfg in d.SETTINGS} for ds in ('finrag', 'visdom')})


if __name__ == '__main__':
    main()
