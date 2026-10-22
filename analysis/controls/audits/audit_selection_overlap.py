"""Read-only audit of the prompt and retriever samples, development portions, and confirmation population. Prints JSON."""
import importlib.util
import json
import sys
from pathlib import Path

from e2e_common import ROOT as R
sys.path[:0] = [str(R / p) for p in ("scripts/public_e2e_v1", "src/public_e2e_v1", "src", "scripts")]
from e2e_records import load_records  # noqa: E402
from fedresolve_core import split_records  # noqa: E402


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sp = load_module(R / "scripts/public_e2e_v1/select_prompt.py", "sel_prompt")
sr = load_module(R / "scripts/public_e2e_v1/select_retriever_delivered.py", "sel_retr")
SEED = 20260728
out = {}
for ds in ("finrag", "visdom"):
    records, _, _ = load_records(ds)
    offset = 0 if ds == "finrag" else 50_000
    dev_any, test_any, train_all = set(), set(), None
    for track in ("seen_query", "unseen_family"):
        for clients in (8, 16):
            splits = split_records(records, track=track, seed=SEED + clients + offset + 101)
            dev_any |= {r.uid for r in splits["dev"]}
            test_any |= {r.uid for r in splits["test"]}
            tr = {r.uid for r in splits["train"]}
            train_all = tr if train_all is None else (train_all & tr)
    usable_prompt = set(sp.training_only_uids(records, ds, SEED))
    usable_retr = set(sr.training_only(records, ds, SEED))
    usable_prompt_list = list(sp.training_only_uids(records, ds, SEED))  # sorted(train - test)
    ps = set(usable_prompt_list[:150])  # select_prompt.py: sample = [by_uid[u] for u in usable][: args.sample]
    retr_sample = sr.stratified_sample(records, list(sr.training_only(records, ds, SEED)), 400, SEED)
    rs = {r.uid if hasattr(r, "uid") else r for r in retr_sample}
    conf_path = R / "configs/serf_v3_devconfirm" / f"confirm_uids_{ds}.json"
    conf = json.load(open(conf_path, encoding="utf-8"))
    all_uids = {r.uid for r in records}
    conf_uids = set()
    def walk(o):
        if isinstance(o, dict):
            for v in o.values(): walk(v)
        elif isinstance(o, list):
            for v in o:
                if isinstance(v, str) and v in all_uids: conf_uids.add(v)
                else: walk(v)
    for k, v in conf.items():
        if k in ("excluded_sources", "touched_counts", "touched_union", "disclosed_sources", "completeness_scan"):
            continue
        walk(v)
    out[ds] = {
        "records": len(records),
        "usable_prompt": len(usable_prompt), "usable_retriever": len(usable_retr),
        "usable_is_train_in_every_setting": usable_prompt <= train_all,
        "usable_minus_train_all": len(usable_prompt - train_all),
        "usable_in_dev_of_some_setting": len(usable_prompt & dev_any),
        "usable_in_test_of_some_setting": len(usable_prompt & test_any),
        "prompt_sample": len(ps), "prompt_sample_in_dev_any": len(ps & dev_any), "prompt_sample_in_test_any": len(ps & test_any),
        "retriever_sample": len(rs), "retriever_sample_in_dev_any": len(rs & dev_any), "retriever_sample_in_test_any": len(rs & test_any),
        "confirmation_uids_found": len(conf_uids),
        "prompt_sample_in_confirmation": len(ps & conf_uids), "retriever_sample_in_confirmation": len(rs & conf_uids),
        "confirmation_keys": [k for k, v in conf.items() if isinstance(v, (dict, list))][:20],
    }
print(json.dumps(out, indent=1))
