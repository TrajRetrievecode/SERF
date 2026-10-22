"""Construct training and test composites without encoding or changing any cache."""
from __future__ import annotations

from composite_common import (arguments, construct, parser, protocol_digest, read_json,
                              settings, write_json, write_jsonl)


def main():
    args = arguments(parser(__doc__))
    protocol = read_json(args.protocol)
    from e2e_corpus import CACHE
    from e2e_records import load_records
    from e2e_retrieval import PageIndex
    rows, audits = [], []
    for dataset in protocol['datasets']:
        records, _, inventory = load_records(dataset)
        index = PageIndex.load(dataset, protocol['retriever']['blend_weight'][dataset])
        manifest = read_json(CACHE / f'{dataset}_slot_queries' / 'manifest.json')
        if len(manifest['spans']) != len(records):
            raise ValueError('Original slot manifest record count mismatch')
        if [r.uid for r in records] != [uid for uid, i in sorted(index.query_rows.items(), key=lambda x: x[1])]:
            raise ValueError('Original slot cache record order cannot be verified against page query index')
        spans = {r.uid: manifest['spans'][i] for i, r in enumerate(records)}
        for config, seed, federation, splits in settings(dataset, protocol, records, index, inventory):
            for portion in protocol['construction']['portions']:
                composites, audit = construct(splits[portion], spans, federation, seed,
                                               dataset, config, portion, args.limit)
                rows.extend(composites)
                audits.append(audit)
                print(audit, flush=True)
    write_jsonl(args.out / 'composites.jsonl', rows)
    write_json(args.out / 'construction.json', {'protocol_sha256': protocol_digest(args.protocol),
               'limit': args.limit, 'settings': audits})


if __name__ == '__main__':
    main()
