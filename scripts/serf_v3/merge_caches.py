#!/usr/bin/env python3
"""Merge generation caches into one file, one line per key; exit non-zero on conflicting outputs."""
import json
import sys


def main() -> int:
    out_path, sources = sys.argv[1], sys.argv[2:]
    merged, counts = {}, {}
    for src in sources:
        n = 0
        with open(src, encoding="utf-8") as h:
            for line in h:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                k, v = row["key"], row["output"]
                if k in merged and merged[k] != v:
                    print(json.dumps({"conflict_key": k, "source": src}))
                    return 1
                merged[k] = v
                n += 1
        counts[src] = n
    with open(out_path, "w", encoding="utf-8") as h:
        for k, v in merged.items():
            h.write(json.dumps({"key": k, "output": v}, ensure_ascii=False) + "\n")
    print(json.dumps({"sources": counts, "unique_keys": len(merged), "conflicts": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
