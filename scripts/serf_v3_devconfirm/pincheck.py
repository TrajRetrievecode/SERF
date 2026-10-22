"""Runtime checks of protocol_serf_v3_devconfirm.json (run_validity).

check(stage): every loaded module file under the project root (outside the third-party package
cache) must be listed in configs/serf_v3_devconfirm/REQUIRED; paths are compared after realpath so a
symlinked root or script directory cannot hide a module.
env_check(stage): the running interpreter and library versions, and the resolved numpy location, must
equal configs/serf_v3_devconfirm/ENVIRONMENT.json written at registration.
environment(): the dict register.sh records (reads package versions without importing vllm or torch).
"""
import importlib.metadata as md
import json
import os
import sys
from pathlib import Path

R = str(Path(os.environ.get("SERF_ROOT", str(Path(__file__).resolve().parents[2])))) + "/"
RR = os.path.realpath(R.rstrip("/")) + "/"
CONF = R + "configs/serf_v3_devconfirm/"


def environment():
    import numpy
    env = {"python": sys.version, "numpy": numpy.__version__, "numpy_file": os.path.realpath(numpy.__file__),
           "project_root_realpath": RR}
    for pkg in ("vllm", "torch", "transformers"):
        try:
            env[pkg] = md.version(pkg)
        except Exception as e:
            env[pkg] = f"unavailable: {type(e).__name__}"
    return env


def env_check(stage):
    registered = json.load(open(CONF + "ENVIRONMENT.json", encoding="utf-8"))
    now = environment()
    diff = sorted(k for k in set(registered) | set(now) if registered.get(k) != now.get(k))
    if diff:
        raise SystemExit(f"{stage}: environment differs from registration: {diff}")


def check(stage):
    required = {os.path.realpath(p) for p in open(CONF + "REQUIRED", encoding="utf-8").read().split("\n") if p}
    # only real files with absolute paths count: some libraries set placeholder __file__ values such as
    # '_classes.py' (torch._classes, torch._ops.ops), which would otherwise resolve against the working directory
    loaded = {os.path.realpath(m.__file__) for m in list(sys.modules.values())
              if isinstance(getattr(m, "__file__", None), str) and os.path.isabs(m.__file__) and os.path.isfile(m.__file__)}
    project = {f for f in loaded if f.startswith(RR) and os.environ.get("SERF_THIRD_PARTY_CACHE", str(Path(R) / "data/cache/python_packages")) + "/" not in f}
    missing = sorted(project - required)
    if missing:
        raise SystemExit(f"{stage}: project modules loaded but not pinned: {missing}")
