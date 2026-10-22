"""Synthetic tests run imported selectors, adapters, metrics and complete CPU stages."""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

JOB = Path(__file__).resolve().parents[1]
WORKSPACE = JOB.parents[2]
for rel in ('src', 'src/public_e2e_v1', 'scripts/serf_v3',
            'scripts/public_e2e_v1', 'scripts/public_paired_v1', 'scripts', 'analysis',
            'analysis/controls/attribution', 'analysis/controls/reference', 'analysis/controls/composite'):
    sys.path.append(str(WORKSPACE / rel))
sys.path.insert(0, str(JOB))
import numpy as np
from scipy import sparse
import d_common as d
import e2e_slot_adapters as adapters
import e2e_v3_adapter
import run_v3_plan
import fedresolve_core
import e2e_retrieval
import e2e_records
import e2e_corpus
import composite_common as c
import public_rows as b_public
import evaluation_rows as b_eval
import run_formal_gold_v1 as rf
import gold_v3

ap = d.module('fixture_a_public', WORKSPACE / 'analysis/controls/attribution/run_public.py')
rp = d.module('fixture_d_public', JOB / 'run_public.py')
rc = d.module('fixture_d_composite', JOB / 'run_composites.py')
re = d.module('fixture_d_evaluation', JOB / 'run_evaluation.py')
agg = d.module('fixture_d_aggregate', JOB / 'aggregate.py')
DESIGN = d.read(JOB / 'protocol_fairness_v1_D_global_selection.json')
PUBLIC = d.read(WORKSPACE / 'configs/public_e2e_v1/protocol_v3.json')
GOLD = d.read(WORKSPACE / 'configs/formal_gold_v1_protocol.json')
CP = d.read(WORKSPACE / 'analysis/controls/composite/protocol_fairness_v1_C_composite.json')


def write_rows(path, rows):
    with d.fresh(path) as f:
        for row in rows:
            d.emit(f, row)


def earlier(args, job, filename):
    return d.module('fixture_' + job + '_' + Path(filename).stem,
                    WORKSPACE / 'analysis/controls' / job / filename)


def tiny():
    index = SimpleNamespace(page_uids=['a#p1', 'a#p2', 'b#p1', 'c#p1'],
                doc_key_list=['a', 'b', 'c'], doc_keys=np.array([0, 0, 1, 2]),
                uid_to_row={'a#p1': 0, 'a#p2': 1, 'b#p1': 2, 'c#p1': 3}, visual=np.eye(4))
    view = e2e_retrieval.ClientPageView(index, {'a': 0, 'b': 1, 'c': 2}, 4)
    views = adapters.QueryViews(np.array([.99, .98, .8, .01]),
                                np.array([[.99, .98, 0., 0.], [0., .1, .8, .01]]),
                                view.rows_by_client, 4, .75, .25, .07)
    record = SimpleNamespace(uid='q', query='first and second', gold_pages=('a#p1', 'b#p1'),
                             doc_keys=('a', 'b'), gold_unit='page')
    return index, view, views, record


class FakeIndex:
    def __init__(self, records):
        self.doc_key_list = sorted({x.doc_keys[0] for x in records})
        self.page_uids = [doc + '#p' + str(p) for doc in self.doc_key_list for p in (1, 2)]
        self.uid_to_row = {uid: i for i, uid in enumerate(self.page_uids)}
        self.doc_keys = np.repeat(np.arange(len(self.doc_key_list)), 2)
        self.visual = np.eye(len(self.page_uids), dtype=np.float32)
        self.text = sparse.eye(len(self.page_uids), format='csr', dtype=np.float32)
        self.dense = None
        self.calls = 0
        self.records = {r.uid: r for r in records}

    def scores_for_rows(self, uid, rows):
        self.calls += 1
        scores = np.linspace(.01, .2, len(self.page_uids))
        record = self.records[uid]
        scores[self.uid_to_row[record.gold_pages[0]]] = .99
        scores[self.uid_to_row[record.gold_pages[0]] + 1] = .95
        return scores[rows]

    def attach_dense(self, dataset, weight):
        self.dense = self.visual.copy()


class FakeSlots:
    def __init__(self, records, index):
        self.records, self.index = records, index
        self.spans = [(i, i + 1) for i in range(len(records))]
        self.decomposition = {'synthetic': True}
        self.calls = 0

    def page_scores(self, span, blend):
        self.calls += 1
        # Avoid counting the separate shared full-query scoring call.
        n = self.index.calls
        scores = self.index.scores_for_rows(self.records[span.start].uid,
                                             np.arange(len(self.index.page_uids)))
        self.index.calls = n
        return scores[None, :]


def records_for(ds):
    return [SimpleNamespace(uid=f'q{doc}_{q}', query=f'What is value {q} for document {doc}?',
                doc_keys=(f'doc{doc}',), gold_pages=(f'doc{doc}#p1',), lang='en', dataset=ds,
                gold_unit='page' if ds == 'finrag' else 'document')
            for doc in range(24) for q in range(5)]


class SelectionTests(unittest.TestCase):
    def test_registered_constants_and_hashes(self):
        self.assertEqual(DESIGN['constants'], GOLD['fedresolve'])
        self.assertEqual(DESIGN['retriever']['blend_weight'], PUBLIC['retriever']['blend_weight'])
        self.assertEqual(DESIGN['statistics']['bootstrap_iterations'], 10000)
        for name, expected in DESIGN['registered_result_sha256'].items():
            self.assertEqual(d.digest(JOB / 'registered' / name), expected)

    def test_coverage_and_multiple_pages_per_owner(self):
        index, view, views, _ = tiny()
        results = dict(d.public_selections(views, np.ones(4, bool), 2, GOLD, view.page_client, False))
        self.assertEqual(results['base']['rows'], [0, 1])
        self.assertEqual(results['+sel']['rows'], [0, 2])
        one_client = np.array([True, False, False, False])
        self.assertEqual(dict(d.public_selections(views, one_client, 3, GOLD, view.page_client, False))['+sel']['rows'], [0, 1])
        self.assertEqual(results['+sel']['eligible_clients'], [0, 1, 2])

    def test_exact_base_and_ties(self):
        _, view, views, _ = tiny()
        views.evidence_score[:] = 1
        native = adapters.centralized_adapter(views, np.ones(4, bool), 3, GOLD, page_client=view.page_client)
        results = dict(d.public_selections(views, np.ones(4, bool), 3, GOLD, view.page_client, False))
        self.assertEqual(results['base']['rows'], native['rows'])
        self.assertIs(e2e_v3_adapter.fedresolve_adapter, adapters.fedresolve_adapter)

    def test_gate_all_clients_and_mismatch(self):
        _, _, views, _ = tiny()
        online = np.array([False, True, True, False])
        self.assertTrue(d.verify_gate(views, online, GOLD, True))
        with self.assertRaises(ValueError):
            d.verify_gate(views, online, GOLD, False)
        self.assertFalse(d.verify_gate(views, np.ones(4, bool), GOLD, False))

    def test_no_online_pool(self):
        _, view, views, _ = tiny()
        for result in dict(d.public_selections(views, np.zeros(4, bool), 2, GOLD, view.page_client, False)).values():
            self.assertTrue(result['abstain'])
            self.assertEqual(result['rows'], [])

    def test_public_page_document_and_cost(self):
        index, view, views, record = tiny()
        results = dict(d.public_selections(views, np.ones(4, bool), 2, GOLD, view.page_client, True))
        rows = {v: d.public_row(ap, 'finrag', 'c8_seen_query', record, view, index, {}, PUBLIC,
                               v, 2, 'full', result, True) for v, result in results.items()}
        self.assertEqual(rows['base']['owner_contacts'], 1)
        self.assertEqual(rows['+sel']['owner_contacts'], 2)
        self.assertEqual(rows['+sel']['client_calls'], 1)
        self.assertEqual(rows['+sel']['evidence_recall'], 1)
        self.assertEqual(rows['+sel+test']['owner_contacts'], 0)
        self.assertEqual(rows['+sel+test']['selected_pages'], 0)
        self.assertEqual(rows['+sel+test']['protocol_bytes'], 512)
        record.gold_unit = 'document'
        self.assertEqual(ap.metric_fields(record, ['a#p1', 'a#p2'])['evidence_recall'], .5)

    def test_gold_selection_and_label_boundary(self):
        inputs = {'canonical_id': 'q', 'candidate_ids': ['a', 'aa', 'b'],
                  'candidate_clients': [0, 0, 1], 'candidate_bytes': [10, 11, 12],
                  'full_scores': np.array([.99, .98, .8]),
                  'slot_scores': np.array([[.99, .98, 0], [0, .1, .8]])}
        views = rf.score_views(inputs, GOLD)
        variants = dict(re.selections(rf, gold_v3.v3_select, inputs, views, np.ones(8, bool), 2, GOLD, False))
        self.assertEqual(variants['base']['evidence_ids'], ['a', 'aa'])
        self.assertEqual(variants['+sel']['evidence_ids'], ['a', 'b'])
        self.assertEqual(variants['+sel']['queried_clients'], [0, 1])
        paired = dict(re.selections(rf, gold_v3.v3_select, inputs, views, np.ones(8, bool), 2, GOLD, True, True))
        self.assertTrue(paired['+sel+test']['abstain'])
        self.assertEqual(paired['+sel+test']['evidence_ids'], ['a', 'b'])
        gold = {'minimum_sufficient_client_sets': [[0, 1]], 'candidate_assessments': [
            {'candidate_id': 'a', 'stance': 'support'}, {'candidate_id': 'b', 'stance': 'support'}]}
        metrics = rf.item_metrics(paired['+sel+test'], gold, inputs, 512)
        self.assertEqual(metrics['evidence_recall'], 1)
        self.assertEqual(metrics['strict_success'], 0)
        self.assertEqual(metrics['bytes'], 534)

    def test_limits_duplicates_and_reference_failure(self):
        rows = [{'dataset': 'finrag', 'config': 'c8_seen_query', 'uid': uid,
                 'scenario': s, 'budget': 2} for uid in ('z', 'a') for s in ('full', 'qrel_missing')]
        self.assertEqual({r['uid'] for r in d.limited(rows, 1)}, {'z'})
        with self.assertRaises(ValueError):
            d.index_rows(rows + rows[:1])
        with self.assertRaises(ValueError):
            d.same({'x': 1}, {'x': 2}, ('x',))
        with self.assertRaises(ValueError):
            agg.aligned(rows[:2], rows[2:])

    def test_bootstrap_signs_and_exact_large_counts(self):
        row = {'panel': 'public', 'dataset': 'finrag', 'config': 'c8_seen_query', 'budget': 2,
               'scenario': 'full', 'uid': 'q', 'evidence_recall': 1, 'contacts': 1,
               'method': 'FedResolve-v3', 'variant': 'recorded'}
        control = {**row, 'method': 'Centralized', 'variant': '+sel', 'evidence_recall': 0, 'contacts': 2}
        boot, counts, _, _ = agg.primary_statistics([row, control], DESIGN)
        self.assertEqual(boot[0]['iterations'], 10000)
        self.assertEqual(boot[1]['difference'], -boot[0]['difference'])
        self.assertEqual(boot[1]['ci_low'], -boot[0]['ci_high'])
        self.assertTrue(any(r['negative'] for r in counts))
        self.assertEqual(agg.exact([1, 0], [0, 1])['p_value'], 1)
        self.assertGreater(agg.exact([1]*600 + [0]*600, [0]*600 + [1]*600)['p_value'], .9)

    def test_macro_missing_setting_and_incomplete_cells(self):
        rows = [{'panel': 'public', 'dataset': 'finrag', 'config': cfg, 'method': 'Centralized',
                 'variant': 'base', 'budget': 2, 'scenario': 'full', 'n': 1, 'contacts': i}
                for i, cfg in enumerate(d.SETTINGS)]
        fields = tuple(f for f in agg.CELL if f != 'config')
        self.assertEqual(agg.macro(rows, fields, ('contacts',))[0]['contacts'], 1.5)
        self.assertIsNone(agg.macro(rows[:3], fields, ('contacts',))[0]['contacts'])
        with self.assertRaises(ValueError):
            agg.check_cells([], [(2, 'full', '+sel')], 1)

    def test_fresh_outputs_and_sandbox(self):
        with tempfile.TemporaryDirectory(dir=JOB / 'tests') as folder:
            path = Path(folder) / 'fresh.json'
            d.save(path, {'ok': 1})
            with self.assertRaises(FileExistsError):
                d.save(path, {'ok': 2})
            args = SimpleNamespace(root=Path(folder), out=Path(folder) / 'elsewhere', limit=1)
            with self.assertRaises(ValueError):
                d.setup(args)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=JOB / 'tests')
        self.base = Path(self.temp.name)
        self.args = SimpleNamespace(root=self.base, a_out=self.base / 'A', b_out=self.base / 'B',
                    c_out=self.base / 'C', out=self.base / 'output', design=JOB / 'protocol_fairness_v1_D_global_selection.json',
                    limit=2, runner=None, selector=None, cache=None, tasks=None, gold=None)

    def tearDown(self):
        self.temp.cleanup()

    def test_composite_protocol_script_layout(self):
        path = self.args.root / 'analysis/controls/composite/protocol_fairness_v1_C_composite.json'
        d.save(path, CP)
        self.assertEqual(rc.protocol_path(self.args), path)
        d.save(self.args.c_out / path.name, CP)
        self.assertEqual(rc.protocol_path(self.args), self.args.c_out / path.name)
        args = SimpleNamespace(root=self.base / 'absent', c_out=self.base / 'also_absent')
        with self.assertRaises(FileNotFoundError):
            rc.protocol_path(args)

    def make_public(self, ds):
        records = records_for(ds)
        index = FakeIndex(records)
        slots = FakeSlots(records, index)
        texts = {p: 'synthetic text' for p in index.page_uids}
        arows, vrows, paired, owners, bpaired = [], [], [], [], []
        for clients in (8, 16):
            for track in PUBLIC['tracks']:
                seed = PUBLIC['seed'] + clients + (50000 if ds == 'visdom' else 0)
                fed = fedresolve_core.build_federation(index.doc_key_list, num_clients=clients, seed=seed)
                split = fedresolve_core.split_records(records, track=track, seed=seed+101)
                view = e2e_retrieval.ClientPageView(index, fed.doc_to_client, clients)
                cfg = f'c{clients}_{track}'
                self.assertGreaterEqual(len(split['test']), 2)
                for record in split['test'][:2]:
                    position = records.index(record)
                    views = b_public.query_views(record, position, index, slots, view, GOLD, PUBLIC['retriever']['blend_weight'][ds])
                    for scenario in ('full', 'unneeded_offline', 'qrel_missing'):
                        online = run_v3_plan.online_mask(record, fed, scenario, clients, PUBLIC['seed'])
                        gate = bool(adapters._critical_missing(views, online, GOLD))
                        for budget in ((1, 2, 3) if scenario == 'full' else (2,)):
                            output = e2e_v3_adapter.fedresolve_v3_adapter(views, online, budget, GOLD)
                            row = ap.plan_row(ds, cfg, clients, track, record, view, index, texts, PUBLIC,
                                              'FedResolve-v3', 'recorded', budget, output, [], 0)
                            row['scenario'] = scenario
                            vrows.append(row)
                            if scenario == 'full':
                                arows.append({**row, 'method': 'FedResolve', 'variant': '+sel'})
                            else:
                                bpaired.append({**row, 'gate': gate, 'answerable': scenario == 'unneeded_offline'})
                            central_result = dict(d.public_selections(views, online, budget, GOLD, view.page_client, gate))['base']
                            central = d.public_row(ap, ds, cfg, record, view, index, texts, PUBLIC,
                                                   'base', budget, scenario, central_result, gate)
                            vrows.append(central)
                            if scenario == 'full':
                                owners.append({**central, 'variant': 'owner_contacts'})
                            else:
                                bpaired.append({**central, 'gate': gate, 'answerable': scenario == 'unneeded_offline'})
                        for method in ('S2G-RAG', 'S2G-RAG+avail', 'FedResolve', 'Centralized'):
                            out = adapters.ADAPTERS[method](views, online, 2, GOLD, page_client=view.page_client)
                            row = ap.plan_row(ds, cfg, clients, track, record, view, index, texts, PUBLIC,
                                              method, 'recorded', 2, out, [], 0)
                            row['scenario'] = scenario
                            row['offline_clients'] = np.flatnonzero(~online).tolist()
                            paired.append(row)
        write_rows(self.args.a_out / (ds + '.jsonl'), arows)
        write_rows(self.args.root / 'artifacts/serf_v3' / (ds + '_test') / 'plan.jsonl', vrows)
        write_rows(self.args.root / 'artifacts/public_paired_v1' / ds / 'plan.jsonl', paired)
        write_rows(self.args.b_out / 'public_owners.jsonl', owners)
        write_rows(self.args.b_out / 'public_paired.jsonl', bpaired)
        index.calls = slots.calls = 0
        with patch.object(d, 'earlier', side_effect=earlier), patch.object(b_public, 'require_consolidated_corpus'), \
             patch.object(e2e_records, 'load_records', return_value=(records, {}, ())), \
             patch.object(e2e_retrieval.PageIndex, 'load', return_value=index), \
             patch.object(run_v3_plan, 'SlotQueries', return_value=slots), \
             patch.object(e2e_corpus, 'page_text_map', return_value=texts), contextlib.redirect_stdout(io.StringIO()):
            rows, refs, cohorts = rp.run_dataset(self.args, ds, PUBLIC, GOLD, DESIGN, d.Sources())
        self.assertEqual(index.calls, 8)
        self.assertEqual(slots.calls, 8)
        self.assertEqual(len(rows), 8 * 13)
        return rows, refs, cohorts, records, index

    def test_public_finrag_pipeline(self):
        rows, refs, cohorts, _, _ = self.make_public('finrag')
        self.assertEqual(sum(cohorts.values()), 8)
        self.assertEqual(len(refs), 8 * 7)
        for row in rows:
            row.update(_stage='finrag', contacts=agg.contact(row))
        for row in refs:
            row.update(_stage='finrag', contacts=agg.contact(row))
        agg.validate(rows + refs, {'finrag': {'cohorts': cohorts, 'rows': len(rows), 'references': len(refs)}})
        balanced, tests = agg.paired_tables(rows + refs)
        self.assertEqual(len(balanced), 16)
        self.assertTrue(tests)

    def test_public_visdom_pipeline(self):
        rows, refs, _, _, _ = self.make_public('visdom')
        self.assertTrue(all(r['evidence_recall'] in (0, 1) for r in rows))

    def test_composite_pipeline_and_missing_references(self):
        records = records_for('finrag')
        index = FakeIndex(records)
        texts = {p: 'synthetic text' for p in index.page_uids}
        d.save(self.args.c_out / 'protocol_fairness_v1_C_composite.json', CP)
        cache = self.args.c_out / 'queries/finrag'
        cache.mkdir(parents=True)
        composites, plan, features = [], [], {}
        for cfg, seed, fed, splits in c.settings('finrag', CP, records, index, ()):
            pairs = [(a, b) for a in splits['test'] for b in splits['test'] if
                     fed.doc_to_client[a.doc_keys[0]] != fed.doc_to_client[b.doc_keys[0]]]
            self.assertTrue(pairs)
            a, b = pairs[0]
            uid = 'comp:' + a.uid + '|' + b.uid
            comp = {'dataset': 'finrag', 'config': cfg, 'uid': uid, 'portion': 'test', 'lang': 'en',
                    'query': a.query + ' and ' + b.query, 'uids': [a.uid, b.uid], 'gold_unit': 'page',
                    'gold_units': [a.gold_pages[0], b.gold_pages[0]],
                    'gold_clients': [fed.doc_to_client[a.doc_keys[0]], fed.doc_to_client[b.doc_keys[0]]],
                    'parts': [{'documents': list(r.doc_keys), 'gold_units': list(r.gold_pages),
                               'client': fed.doc_to_client[r.doc_keys[0]]} for r in (a, b)]}
            composites.append(comp)
            full = np.zeros(len(index.page_uids)) + .01
            slot = np.zeros((2, len(full))) + .01
            for j, record in enumerate((a, b)):
                slot[j, index.uid_to_row[record.gold_pages[0]]] = .99
                full[index.uid_to_row[record.gold_pages[0]]] = .8
            features[uid] = full, slot
            view = e2e_retrieval.ClientPageView(index, fed.doc_to_client, fed.num_clients)
            views = c.make_views(full, slot, view, CP)
            for scenario in ('full', 'part_offline', 'other_offline'):
                online = c.online_mask(comp, scenario, fed.num_clients, CP['seed'])
                for budget in ((2, 3) if scenario == 'full' else (2,)):
                    for method in ('FedResolve-v3', 'Centralized'):
                        output = c.route(method, 'native', views, online, budget, CP, view, index, comp)
                        crun = d.module('fixture_c_run', WORKSPACE / 'analysis/controls/composite/run_composites.py')
                        plan.extend(crun.build_rows(comp, views, online, scenario, budget, method, 'native',
                                                    output, .3, index, view, texts, CP, 0))
        write_rows(self.args.c_out / 'plans/finrag/plan.jsonl', plan)
        for name in ('manifest.json', 'full_visual.npy', 'full_text.npz', 'full_dense.npy',
                     'slot_visual.npy', 'slot_text.npz', 'slot_dense.npy'):
            (cache / name).write_text('{}', encoding='utf-8')
        fake_queries = SimpleNamespace(scores=lambda uid, blend: features[uid],
                    manifest={'protocol_sha256': d.digest(self.args.c_out / 'protocol_fairness_v1_C_composite.json')})
        with patch.object(d, 'earlier', side_effect=earlier), patch.object(b_public, 'require_consolidated_corpus'), \
             patch.object(e2e_records, 'load_records', return_value=(records, {}, ())), \
             patch.object(e2e_retrieval.PageIndex, 'load', return_value=index), \
             patch.object(c, 'CompositeQueries', return_value=fake_queries), \
             patch.object(e2e_corpus, 'page_text_map', return_value=texts):
            rows, refs, missing = rc.run_dataset(self.args, 'finrag', composites, DESIGN, d.Sources(), PUBLIC, GOLD)
        self.assertEqual(len(rows), 4*18)
        self.assertEqual(len(refs), 4*4)
        self.assertEqual(len(missing), 4*2)
        for r in rows + refs:
            r['contacts'] = agg.contact(r)
        boot, counts, mc, unavailable = agg.primary_statistics(rows + refs, DESIGN)
        self.assertEqual(len(unavailable), 4*2*3*2)
        self.assertTrue(mc)
        balanced, tests = agg.paired_tables(rows + refs)
        self.assertTrue(any(r['budget'] == 3 for r in balanced))
        self.assertTrue(tests)

    def test_real_composite_cache_imports_slot_scorer(self):
        records = records_for('finrag')
        index = FakeIndex(records)
        cache = self.base / 'vectors'
        cache.mkdir()
        d.save(cache / 'manifest.json', {'uids': ['comp'], 'spans': [[0, 2]]})
        n = len(index.page_uids)
        for prefix, count in [('full', 1), ('slot', 2)]:
            np.save(cache / (prefix + '_visual.npy'), np.ones((count, n), np.float32))
            np.save(cache / (prefix + '_dense.npy'), np.zeros((count, 0), np.float32))
            sparse.save_npz(cache / (prefix + '_text.npz'), sparse.csr_matrix(np.ones((count, n))))
        queries = c.CompositeQueries(cache, index, 0)
        full, slots = queries.scores('comp', .25)
        self.assertEqual(full.shape, (n,))
        self.assertEqual(slots.shape, (2, n))
        self.assertTrue(np.allclose(full, 1))

    def make_evaluation(self):
        cache = self.base / 'cache'
        cache.mkdir()
        images = []
        for cid in ('a', 'aa', 'b', 'c'):
            path = cache / (cid + '.bin')
            path.write_bytes(b'synthetic image')
            images.append({'candidate_id': cid, 'path': str(path)})
        tasks, annotations, queries, vectors = [], {}, [], []
        for i in range(45):
            uid = 'eval' + str(i)
            scenarios = []
            if i < 40:
                for client in (0, 2):
                    scenarios.append({'scenario_id': uid + '_offline' + str(client),
                                      'online_clients': [c for c in range(8) if c != client]})
            tasks.append({'canonical_id': uid, 'candidate_pages': [
                {'candidate_id': cid, 'client_id': owner} for cid, owner in
                [('a', 0), ('aa', 0), ('b', 1), ('c', 2)]], 'participation_scenarios': scenarios})
            annotations[uid] = {'canonical_id': uid, 'minimum_sufficient_client_sets': [[0, 1]],
                  'candidate_assessments': [{'candidate_id': cid, 'stance': 'support' if cid in ('a', 'b') else 'irrelevant'}
                                           for cid in ('a', 'aa', 'b', 'c')],
                  'participation_assessments': [{'scenario_id': s['scenario_id'],
                        'answerability': 'unanswerable' if 0 not in s['online_clients'] else 'answerable'} for s in scenarios]}
            for slot, vector in [('full', [1., 1.]), ('0', [1., 0.]), ('1', [0., 1.])]:
                queries.append({'query_key': uid + '::' + slot, 'slot_id': -1 if slot == 'full' else int(slot)})
                vectors.append(vector)
        ie = np.array([[1., 0.], [.99, .01], [0., 1.], [.1, .1]])
        qe = np.array(vectors)
        d.save(cache / 'images.json', images)
        d.save(cache / 'queries.json', queries)
        np.save(cache / 'image_embeddings.npy', ie)
        np.save(cache / 'query_embeddings.npy', qe)
        artifacts = {p.name: d.digest(p) for p in cache.iterdir() if p.name.endswith(('.json', '.npy'))}
        d.save(cache / 'manifest.json', {'artifacts': artifacts})
        d.save(self.base / 'artifacts/formal_gold_v1/formal_results.json',
               {'cache_manifest_sha256': d.digest(cache / 'manifest.json'),
                'protocol_sha256': DESIGN['input_protocol_sha256']['formal_gold_v1_protocol.json']})
        tasks_path = self.base / 'tasks.json'
        gold_path = self.base / 'gold.json'
        d.save(tasks_path, tasks)
        d.save(gold_path, {'annotations': list(annotations.values())})
        protocol_path = self.base / 'configs/formal_gold_v1_protocol.json'
        d.save(protocol_path, GOLD)
        inputs = b_eval.build_inputs(tasks, images, queries, ie, qe, self.base)
        arows, bowners, bpaired = [], [], []
        for task in tasks:
            uid = task['canonical_id']
            inp = inputs[uid]
            views = rf.score_views(inp, GOLD)
            for budget in (2, 3):
                online = np.ones(8, bool)
                output = rf.ADAPTERS['FedResolve'](inp, views, online, budget, GOLD)
                output['evidence_ids'] = gold_v3.v3_select(inp, output['queried_clients'], budget, GOLD, views['evidence_score'])
                metrics = rf.item_metrics(output, annotations[uid], inp, 512)
                arows.append({**b_eval.row_for(output, inp, metrics, GOLD, 'FedResolve', 'full', budget),
                              'variant': '+sel'})
                central = rf.ADAPTERS['Centralized'](inp, views, online, budget, GOLD)
                metrics = rf.item_metrics(central, annotations[uid], inp, 512)
                bowners.append(b_eval.row_for(central, inp, metrics, GOLD, 'Centralized', 'full', budget))
            for scenario in task['participation_scenarios']:
                online = np.zeros(8, bool)
                online[scenario['online_clients']] = True
                sid = scenario['scenario_id']
                output, gate = b_eval.v3_output(rf, gold_v3.v3_select, inp, views, online, GOLD)
                central = rf.ADAPTERS['Centralized'](inp, views, online, 2, GOLD)
                gated = {**central, 'abstain': central['abstain'] or gate}
                for method, out in [('FedResolve-v3', output), ('Centralized', central), ('Centralized+avail', gated)]:
                    metrics = b_eval.metric_boundary(rf, out, inp, annotations[uid], task, views, online, GOLD, sid)
                    bpaired.append(b_eval.row_for(out, inp, metrics, GOLD, method, sid, 2, gate,
                            scenario_family='paired_80', offline_client=int(np.flatnonzero(~online)[0])))
        write_rows(self.args.a_out / 'evaluation.jsonl', arows)
        write_rows(self.args.b_out / 'evaluation_owners.jsonl', bowners)
        write_rows(self.args.b_out / 'evaluation_paired.jsonl', bpaired)
        self.args.cache, self.args.tasks, self.args.gold = cache, tasks_path, gold_path
        self.args.runner = WORKSPACE / 'scripts/run_formal_gold_v1.py'
        self.args.selector = WORKSPACE / 'analysis/gold_v3.py'
        real_digest = d.digest

        def fixture_digest(path):
            path = Path(path)
            if path == tasks_path:
                return GOLD['dataset']['tasks_json_sha256']
            if path == gold_path:
                return GOLD['dataset']['gold_json_sha256']
            if path == protocol_path:
                return DESIGN['input_protocol_sha256']['formal_gold_v1_protocol.json']
            return real_digest(path)

        with patch.object(d, 'earlier', side_effect=earlier), patch.object(d, 'digest', side_effect=fixture_digest):
            rows, refs, count = re.run(self.args, DESIGN, d.Sources())
        return rows, refs, count

    def test_evaluation_pipeline_freezes_decisions_and_paired_metrics(self):
        rows, refs, count = self.make_evaluation()
        self.assertEqual(count, 2)
        self.assertEqual(len(rows), 16)
        self.assertEqual(len(refs), 12)
        self.assertTrue((self.args.out / 'evaluation_decisions.jsonl').is_file())
        for r in rows + refs:
            r.update(contacts=agg.contact(r), _stage='evaluation')
        agg.validate(rows + refs, {'evaluation': {'rows': len(rows), 'references': len(refs), 'queries': count, 'limit': 2}})
        balanced, tests = agg.paired_tables(rows + refs)
        self.assertEqual(len(balanced), 4)
        self.assertEqual(len(tests), 16)
        _, _, mc, _ = agg.primary_statistics(rows + refs, DESIGN)
        self.assertEqual(len(mc), 8)

    def test_evaluation_full_45_and_80(self):
        self.args.limit = None
        rows, refs, count = self.make_evaluation()
        self.assertEqual(count, 45)
        self.assertEqual(len(rows), 180 + 160)
        self.assertEqual(sum(r['panel'] == 'evaluation_paired' and r['variant'] == '+sel' for r in rows), 80)
        for r in rows + refs:
            r['contacts'] = agg.contact(r)
        registered = {'public': [], 'evaluation': {'online': {}}}
        for budget in (2, 3):
            serf = [r for r in refs if r['panel'] == 'evaluation' and r['budget'] == budget]
            metrics = ('strict_success', 'evidence_recall', 'evidence_precision', 'citation_f1', 'bytes', 'client_calls')
            registered['evaluation']['online'][str(budget)] = {'SERF-v3': {
                  m: float(np.mean([r[m] for r in serf])) for m in metrics}}
        self.assertEqual(agg.reproduction(rows + refs, registered, False)['status'], 'pass')
        registered['evaluation']['online']['2']['SERF-v3']['evidence_recall'] = -1
        with self.assertRaises(ValueError):
            agg.reproduction(rows + refs, registered, False)

    def test_download_count_and_owner_guards(self):
        index, view, views, record = tiny()
        result = dict(d.public_selections(views, np.ones(4, bool), 2, GOLD, view.page_client, False))['+sel']
        row = d.public_row(ap, 'finrag', 'c8_seen_query', record, view, index, {}, PUBLIC,
                           '+sel', 2, 'full', result, False)
        row.update(panel='public', _stage='finrag', contacts=agg.contact(row))
        with self.assertRaises(ValueError):
            agg.validate([row, row], {})
        with self.assertRaises(ValueError):
            agg.validate([{**row, 'owner_contacts': 99}], {})
        with self.assertRaises(ValueError):
            agg.validate([{**row, 'evidence_recall': float('nan')}], {})
        with self.assertRaises(ValueError):
            agg.validate([row], {'finrag': {'rows': 2, 'references': 0}})

    def test_aggregate_main_json_csv_and_exclusive_creation(self):
        stages = {}
        for ds in ('finrag', 'visdom'):
            # Earlier reference files are read-only in each fixture invocation;
            # keep separate synthetic source trees between the dataset stages.
            self.args.a_out = self.base / (ds + '_A')
            self.args.b_out = self.base / (ds + '_B')
            rows, refs, cohorts, _, _ = self.make_public(ds)
            stages[ds] = rows, refs, {'cohorts': cohorts}
        self.args.a_out, self.args.b_out = self.base / 'eval_A', self.base / 'eval_B'
        rows, refs, count = self.make_evaluation()
        stages['evaluation'] = rows, refs, {'queries': count}
        # Include an explicit empty composite source. This exercises the
        # null macro means rather than silently dropping that experiment.
        stages['composite'] = [], [], {'cohorts': {ds: {cfg: 0 for cfg in d.SETTINGS}
                                                   for ds in ('finrag', 'visdom')}}
        self.args.out = self.base / 'artifacts/controls/global_selection'
        import time
        for stage, (rows, refs, extras) in stages.items():
            d.finish(self.args, stage, rows, refs, d.Sources(), time.time(), **extras)
        with patch.object(agg, 'ROOT', self.base), patch.object(sys, 'argv', ['aggregate.py', '--input', str(self.args.out),
                                       '--out', str(self.args.out), '--limit', '2']), \
             contextlib.redirect_stdout(io.StringIO()):
            agg.main()
        result = d.read(self.args.out / 'results/global_selection.json')
        self.assertEqual(result['reproduction_status'], 'subset_not_comparable')
        self.assertTrue(result['tables']['paired_bootstrap'])
        self.assertTrue((self.args.out / 'results/global_selection.csv').is_file())
        composite_means = [r for r in result['tables']['mean_four_settings'] if r['panel'] == 'composite']
        self.assertTrue(composite_means)
        self.assertTrue(all(r['part_recall'] is None for r in composite_means))
        with patch.object(agg, 'ROOT', self.base), patch.object(sys, 'argv', ['aggregate.py', '--input', str(self.args.out),
                                       '--out', str(self.args.out), '--limit', '2']):
            with self.assertRaises(FileExistsError):
                agg.main()

    def test_production_setup_pins_sources_and_protocol(self):
        import shutil
        required = ('attribution/attribution.py', 'attribution/run_public.py',
                    'attribution/run_evaluation.py', 'reference/b_common.py',
                    'reference/public_rows.py', 'reference/evaluation_rows.py',
                    'composite/composite_common.py')
        for rel in required:
            dst = self.base / 'analysis/controls' / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(WORKSPACE / 'analysis/controls' / rel, dst)
        self.args.out = self.base / 'artifacts/controls/global_selection_smoke'
        design = d.setup(self.args)
        path = self.base / 'configs/public_e2e_v1/protocol_v3.json'
        path.parent.mkdir(parents=True)
        shutil.copyfile(WORKSPACE / 'configs/public_e2e_v1/protocol_v3.json', path)
        shutil.copyfile(WORKSPACE / 'configs/formal_gold_v1_protocol.json', self.base / 'configs/formal_gold_v1_protocol.json')
        public, gold = d.checked_protocol(self.args, design, d.Sources())
        self.assertEqual(public['seed'], PUBLIC['seed'])
        self.assertEqual(gold['fedresolve'], DESIGN['constants'])
        path.write_text('{}', encoding='utf-8')
        with self.assertRaises(ValueError):
            d.checked_protocol(self.args, design, d.Sources())
        helper = self.base / 'analysis/controls/attribution/attribution.py'
        helper.write_text('changed', encoding='utf-8')
        with self.assertRaises(ValueError):
            d.setup(self.args)


if __name__ == '__main__':
    unittest.main(verbosity=2)
