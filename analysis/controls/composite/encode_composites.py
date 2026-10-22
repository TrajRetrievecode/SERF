"""Encode new full and slot queries; this is the only stage that uses a GPU."""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from composite_common import (arguments, fresh, parser, protocol_digest, read_json,
                              read_jsonl, write_json)


def choose_gpu(snapshot):
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible is not None:
        if not visible.strip() or visible.strip() == '-1':
            raise RuntimeError('CUDA_VISIBLE_DEVICES exposes no GPU')
        return {'visible_devices': visible, 'logical_device': 0}
    eligible = [r for r in snapshot if r.get('memory_free_mib', 0) >= 20480]
    if not eligible:
        raise RuntimeError('No GPU has at least 20480 MiB free; retry this stage later')
    return min(eligible, key=lambda r: r['index'])


def unique_composites(rows, dataset, limit=None):
    counts, seen, output = {}, set(), []
    for r in rows:
        if r['dataset'] != dataset:
            continue
        key = r['config'], r['portion']
        counts[key] = counts.get(key, 0) + 1
        if limit is not None and counts[key] > limit:
            continue
        if r['uid'] in seen:
            previous = next(p for p in output if p['uid'] == r['uid'])
            if (r['query'], r['slot_texts']) != (previous['query'], previous['slot_texts']):
                raise ValueError('Repeated uid has different query text')
            continue
        seen.add(r['uid'])
        output.append(r)
    return output


def sparse_vectors(corpus, texts):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize
    from fedresolve_core import mixed_financial_tokens
    vectorizer = TfidfVectorizer(tokenizer=mixed_financial_tokens, lowercase=False,
                                min_df=2, max_df=0.995, sublinear_tf=True, norm='l2',
                                max_features=220000, token_pattern=None)
    vectorizer.fit(corpus)
    return normalize(vectorizer.transform(texts)).astype(np.float32)


def encode_visual(texts, processor, model, torch):
    result = []
    for offset in range(0, len(texts), 256):
        inputs = processor(text=texts[offset:offset + 256], return_tensors='pt',
                           padding=True, truncation=True, max_length=77)
        inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            features = model.get_text_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
        result.append(features.float().cpu().numpy().astype(np.float32))
    return np.concatenate(result)


def encode_dense(texts, tokenizer, model, torch):
    result = []
    for offset in range(0, len(texts), 32):
        inputs = tokenizer(texts[offset:offset + 32], padding=True, truncation=True,
                           max_length=1024, return_tensors='pt')
        inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            hidden = model(**inputs).last_hidden_state[:, 0]
            hidden = torch.nn.functional.normalize(hidden, p=2, dim=-1)
        result.append(hidden.float().cpu().numpy().astype(np.float32))
    return np.concatenate(result)


def save_features(root, full_count, visual, text, dense):
    from scipy import sparse
    # All slot vectors precede the full vectors in this encoding batch.
    for prefix, section in (('slot', slice(0, 2 * full_count)),
                            ('full', slice(2 * full_count, 3 * full_count))):
        np.save(fresh(root / f'{prefix}_visual.npy'), visual[section])
        sparse.save_npz(fresh(root / f'{prefix}_text.npz'), text[section])
        np.save(fresh(root / f'{prefix}_dense.npy'), dense[section])


def main():
    args = arguments(parser(__doc__))
    protocol = read_json(args.protocol)
    construction = read_json(args.out / 'construction.json')
    if construction['protocol_sha256'] != protocol_digest(args.protocol) or construction['limit'] != args.limit:
        raise ValueError('Construction protocol or limit differs from this stage')
    from e2e_common import gpu_snapshot
    from e2e_corpus import pages_for, page_text_map
    from e2e_retrieval import PageIndex
    rows = read_jsonl(args.out / 'composites.jsonl')
    chosen = None
    for dataset in protocol['datasets']:
        started = time.time()
        composites = unique_composites(rows, dataset, args.limit)
        root = args.out / 'queries' / dataset
        fresh(root / 'manifest.json')
        for prefix in ('slot', 'full'):
            for name in ('visual.npy', 'text.npz', 'dense.npy'):
                fresh(root / f'{prefix}_{name}')
        texts = [s for c in composites for s in c['slot_texts']] + [c['query'] for c in composites]
        index = PageIndex.load(dataset, protocol['retriever']['blend_weight'][dataset])
        if texts:
            pages, page_texts = pages_for(dataset), page_text_map(dataset)
            corpus = [page_texts.get(p.page_uid, '') for p in pages]
            text = sparse_vectors(corpus, texts)
            if text.shape[1] != index.text.shape[1]:
                raise ValueError('Refitted sparse vocabulary dimension differs from fixed index')
            if chosen is None:
                chosen = choose_gpu([] if 'CUDA_VISIBLE_DEVICES' in os.environ else gpu_snapshot())
                if 'CUDA_VISIBLE_DEVICES' not in os.environ:
                    os.environ['CUDA_VISIBLE_DEVICES'] = str(chosen['index'])
                import torch
                from transformers import AutoModel, AutoTokenizer, CLIPModel, CLIPProcessor
            # Loading is local only. Models never download or save into existing caches.
            clip_path = os.environ['SERF_CLIP']
            processor = CLIPProcessor.from_pretrained(clip_path, local_files_only=True)
            clip = CLIPModel.from_pretrained(clip_path, dtype=torch.float16,
                                             local_files_only=True).eval().cuda()
            visual = encode_visual(texts, processor, clip, torch)
            del clip
            torch.cuda.empty_cache()
            if dataset == 'finrag':
                dense_path = os.environ['SERF_BGE']
                tokenizer = AutoTokenizer.from_pretrained(dense_path, local_files_only=True)
                model = AutoModel.from_pretrained(dense_path, dtype=torch.float16,
                                                  local_files_only=True).eval().cuda()
                dense = encode_dense(texts, tokenizer, model, torch)
                del model
                torch.cuda.empty_cache()
            else:
                # The VisDoM dense weight is zero; preserve the slot-cache file layout.
                dense = np.zeros((len(texts), 0), dtype=np.float32)
        else:
            from scipy import sparse
            visual = np.zeros((0, index.visual.shape[1]), dtype=np.float32)
            text = sparse.csr_matrix((0, index.text.shape[1]), dtype=np.float32)
            if dataset == 'finrag':
                index.attach_dense(dataset, protocol['retriever']['dense_weight'][dataset])
            dense = np.zeros((0, index.dense.shape[1] if dataset == 'finrag' else 0), dtype=np.float32)
        save_features(root, len(composites), visual, text, dense)
        write_json(root / 'manifest.json', {
            'schema_version': 'public-e2e-slot-query-index-v1', 'dataset': dataset,
            'uids': [c['uid'] for c in composites], 'records': len(composites),
            'spans': [[2 * i, 2 * i + 2] for i in range(len(composites))],
            'slot_texts': 2 * len(composites), 'full_texts': len(composites),
            'decomposition': {'rule': 'Imported decompose, exactly two matching original parts'},
            'protocol_sha256': protocol_digest(args.protocol), 'limit': args.limit,
            'channels': protocol['retriever'], 'gpu': chosen,
            'elapsed_seconds': time.time() - started})
        print({'stage': 'encode', 'dataset': dataset, 'composites': len(composites),
               'seconds': time.time() - started}, flush=True)


if __name__ == '__main__':
    main()
