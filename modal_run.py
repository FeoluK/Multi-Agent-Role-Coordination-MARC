"""Modal app: MARC general sweep runner — any manifest on cloud GPUs.

Drives any MARC manifest (one container per index) using the same
`marc/run.py` entry point as FarmShare. Config is auto-selected from
the manifest name; override with --config.

Usage (from repo root):

  # Overcooked main sweep (75 runs): vanilla + marc + 3 ablations
  modal run modal_run.py --manifest sweep_manifest.py

  # MPE scaling: vanilla vs marc-arch, N=3/6/9 (18 runs)
  modal run modal_run.py --manifest mpe_scaling_manifest.py

  # Overcooked confirmation battery (30 runs)
  modal run modal_run.py --manifest battery_oc_manifest.py

  # MPE loss-fix battery (28 runs)
  modal run modal_run.py --manifest battery_mpe_manifest.py

  # SMAX scaling (60 runs) — same as modal_smax_scaling.py
  modal run modal_run.py --manifest smax_scaling_manifest.py

  # Smoke test: index 0 only, 50k steps (~5 min, ~$0.10)
  modal run modal_run.py --manifest sweep_manifest.py --smoke

  # Run a subset by index
  modal run modal_run.py --manifest sweep_manifest.py --indices 0,1,2

  # Heavier GPU
  modal run modal_run.py --manifest sweep_manifest.py --gpu A100-80GB

  # Raise parallelism
  modal run modal_run.py --manifest sweep_manifest.py --max-containers 20

Pull results and analyze:

  modal volume get marc-results / ./modal_results
  python marc/rliable_report.py ./modal_results
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import modal

JAXMARL_COMMIT = "2cfd60092438b71b24eb35f5f3d9ddd75d37d0b1"
REPO_ROOT = Path(__file__).parent.resolve()

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "jax[cuda12]==0.4.36",
        "jaxlib==0.4.36",
        "flax==0.10.4",
        "optax==0.2.5",
        "distrax==0.1.5",
        "numpy==1.26.4",
        "scipy==1.12.0",
        "omegaconf>=2.3.0",
        "matplotlib==3.10.9",
        "imageio==2.37.3",
        "pillow==12.2.0",
        f"jaxmarl[algs] @ git+https://github.com/FLAIROx/JaxMARL.git@{JAXMARL_COMMIT}",
    )
    .add_local_dir(str(REPO_ROOT / "marc"), "/root/marc")
)

results_vol = modal.Volume.from_name("marc-results", create_if_missing=True)
RESULTS_DIR = "/results"

app = modal.App("marc-sweep", image=image)


def _parse_spec(line: str) -> dict:
    import shlex
    out = {}
    for tok in shlex.split(line.strip()):
        k, _, v = tok.partition("=")
        out[k] = v
    return out


def _config_for(manifest: str) -> str:
    m = manifest.lower()
    if "smax" in m:
        return "configs/marc_smax.yaml"
    if "mpe" in m:
        return "configs/marc_mpe.yaml"
    if "switch_riddle" in m or "switch riddle" in m:
        return "configs/marc_switch_riddle.yaml"
    if "coin_game" in m or "coin game" in m:
        return "configs/marc_coin_game.yaml"
    return "configs/marc_overcooked.yaml"


@app.function(
    gpu="A10G",
    timeout=6 * 3600,
    volumes={RESULTS_DIR: results_vol},
    max_containers=8,
)
def run_one(idx: int,
            manifest: str = "sweep_manifest.py",
            config: str = "",
            total_timesteps: float | None = None) -> dict:
    import os
    import shlex
    import subprocess
    import sys as _sys
    cwd = "/root/marc"

    raw = subprocess.check_output(
        ["python", manifest, str(idx)],
        cwd=cwd, text=True).strip()
    print(f"[modal idx={idx}] spec: {raw}", flush=True)
    spec = _parse_spec(raw)
    spec_sets = shlex.split(spec.get("SETS", ""))

    cfg = config or _config_for(manifest)

    # Overcooked specs have LAYOUT; MPE/SMAX have ADAPTER
    if "LAYOUT" in spec:
        scenario = spec["LAYOUT"]
        layout_args = ["--layout", scenario]
    else:
        scenario = spec["ADAPTER"]
        layout_args = []

    cmd = [
        _sys.executable, "run.py",
        "--config", cfg,
        "--network-kind", spec["KIND"],
        "--seed", spec["SEED"],
        "--tag", f"{spec['TAG']}_{scenario}",
        "--results-dir", RESULTS_DIR,
        "--gifs-dir", RESULTS_DIR,
        *layout_args,
        *spec_sets,
    ]
    if total_timesteps is not None:
        cmd += ["--set", f"TOTAL_TIMESTEPS={total_timesteps}"]

    print(f"[modal idx={idx}] cmd: {' '.join(cmd)}", flush=True)

    rc = subprocess.run(cmd, cwd=cwd).returncode
    results_vol.commit()
    if rc != 0:
        raise RuntimeError(f"run.py failed: idx={idx} rc={rc} spec={raw}")

    base = f"{spec['TAG']}_{scenario}_{spec['KIND']}_{scenario}_seed{spec['SEED']}"
    p = os.path.join(RESULTS_DIR, base + ".json")
    if os.path.exists(p):
        with open(p) as f:
            return {"idx": idx, **json.load(f)}
    return {"idx": idx, "warning": f"no JSON at {p}", "spec": raw}


@app.local_entrypoint()
def main(
    smoke: bool = False,
    indices: str = "",
    gpu: str = "A10G",
    max_containers: int = 8,
    manifest: str = "sweep_manifest.py",
    config: str = "",
):
    manifest_path = REPO_ROOT / "marc" / manifest
    if not manifest_path.exists():
        raise SystemExit(f"manifest not found: {manifest_path}")

    if indices:
        ids = [int(x) for x in indices.split(",") if x.strip()]
    elif smoke:
        ids = [0]
    else:
        n = int(subprocess.check_output(
            [sys.executable, str(manifest_path), "--count"],
            text=True).strip())
        ids = list(range(n))

    fn = run_one
    if gpu != "A10G" or max_containers != 8:
        fn = fn.with_options(gpu=gpu, max_containers=max_containers)

    cfg = config or _config_for(manifest)
    print(f"[modal] {manifest}: {len(ids)} run(s), config={cfg}, "
          f"gpu={gpu}, max_containers={max_containers}, smoke={smoke}", flush=True)

    if smoke:
        result = fn.remote(0, manifest=manifest, config=config, total_timesteps=50_000)
        print("[smoke result]", json.dumps(result, indent=2, default=str), flush=True)
        return

    n_done = 0
    n_fail = 0
    inputs = [(i, manifest, config) for i in ids]
    for r in fn.starmap(inputs, return_exceptions=True):
        if isinstance(r, Exception):
            n_fail += 1
            print(f"[fail] {type(r).__name__}: {r}", flush=True)
            continue
        n_done += 1
        ev = (r.get("eval") or {}).get("eval_return")
        print(f"[done {n_done}/{len(ids)}] idx={r.get('idx')} "
              f"tag={r.get('tag')} layout={r.get('layout')} "
              f"seed={r.get('seed')} "
              f"final={r.get('final_sparse_return')} eval={ev}",
              flush=True)
    print(f"[modal] sweep complete: {n_done} ok, {n_fail} failed", flush=True)
