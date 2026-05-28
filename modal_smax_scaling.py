"""Modal app: MARC SMAX sweeps on cloud GPUs.

Drives any MARC SMAX manifest (`marc/smax_*_manifest.py`), one container
per manifest index. Each container runs the same `marc/run.py`
invocation a FarmShare `*.sbatch` would — so result `.json` / `.npz`
files are byte-identical in schema to the existing MPE / Overcooked
results and feed straight into `marc/rliable_report.py` with no
analysis-code changes.

Usage (from the repo root, which is where this file lives):

  # default: original 60-run scaling sweep
  modal run modal_smax_scaling.py
  modal run modal_smax_scaling.py --smoke               # only idx 0, 50k steps (~5 min, ~$0.10)
  modal run modal_smax_scaling.py --indices 0,1,30,31   # subset of indices
  modal run modal_smax_scaling.py --gpu A100-80GB       # heavier GPU
  modal run modal_smax_scaling.py --max-containers 16   # raise/lower parallelism

  # follow-up sweeps (budget ladder, 2s3z ablation, etc.) — pass the
  # manifest filename (relative to marc/, must live in marc/):
  modal run modal_smax_scaling.py \
      --manifest smax_followup_manifest.py --gpu A100-80GB

Pull results locally (the volume is keyed by name, so this works from
anywhere with the same Modal account):

  modal volume get marc-smax-results / ./modal_results
  python marc/rliable_report.py ./modal_results

Provenance: image pins `jax==0.4.36` cuda12, JaxMARL @ commit
2cfd60092438b71b24eb35f5f3d9ddd75d37d0b1, and the same flax/optax/distrax
versions as `requirements.txt` so numerics match the FarmShare run.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import modal

JAXMARL_COMMIT = "2cfd60092438b71b24eb35f5f3d9ddd75d37d0b1"
REPO_ROOT = Path(__file__).parent.resolve()

# Image: Linux + Python 3.10 + GPU JAX + pinned MARL deps + the marc/
# source tree (mounted at runtime so iterating on the code doesn't
# trigger an image rebuild).
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

# Persistent volume so result JSONs survive container exit and can be
# pulled with `modal volume get marc-smax-results / <local-path>`.
results_vol = modal.Volume.from_name("marc-smax-results", create_if_missing=True)
RESULTS_DIR = "/results"

app = modal.App("marc-smax-scaling", image=image)


def _parse_spec(line: str) -> dict:
    """smax_scaling_manifest.spec(idx) returns one line of the form
        TAG=foo KIND=marc ADAPTER=smax_3m SEED=30 SETS='--set ... --set ...'
    Parse it into a dict (SETS kept as a single quoted string)."""
    import shlex
    out = {}
    for tok in shlex.split(line.strip()):
        k, _, v = tok.partition("=")
        out[k] = v
    return out


@app.function(
    gpu="A100-80GB",
    timeout=6 * 3600,                 # generous; 5e6 steps usually <2h
    volumes={RESULTS_DIR: results_vol},
    max_containers=3,                 # A100-80GB capacity; smaller pool to reduce preemption
)
def run_one(idx: int,
            manifest: str = "smax_scaling_manifest.py",
            config: str = "configs/marc_smax.yaml",
            total_timesteps: float | None = None) -> dict:
    """One manifest index -> one MARC training run -> JSON+npz on the volume.

    Returns the parsed summary JSON (or a {warning} stub) so the local
    `.map()` driver can stream a one-line per-run report to the user.
    `manifest` is the manifest filename inside /root/marc (any
    smax_*_manifest.py with the same `spec(idx)` / `--count` API works).
    """
    import os
    import shlex
    import subprocess
    import sys as _sys
    cwd = "/root/marc"

    # 1. enumerate the spec from the manifest
    raw = subprocess.check_output(
        ["python", manifest, str(idx)],
        cwd=cwd, text=True).strip()
    print(f"[modal idx={idx} {manifest}] spec: {raw}", flush=True)
    spec = _parse_spec(raw)
    spec_sets = shlex.split(spec.get("SETS", ""))

    # 2. assemble run.py invocation (same shape FarmShare's *.sbatch use)
    cmd = [
        _sys.executable, "run.py",
        "--config", config,
        "--network-kind", spec["KIND"],
        "--seed", spec["SEED"],
        "--tag", f"{spec['TAG']}_{spec['ADAPTER']}",
        "--results-dir", RESULTS_DIR,
        "--gifs-dir", RESULTS_DIR,
        *spec_sets,
    ]
    if total_timesteps is not None:
        cmd += ["--set", f"TOTAL_TIMESTEPS={total_timesteps}"]
    print(f"[modal idx={idx}] cmd: {' '.join(cmd)}", flush=True)

    # 3. run; commit volume so files are durable + visible to client
    rc = subprocess.run(cmd, cwd=cwd).returncode
    results_vol.commit()
    if rc != 0:
        raise RuntimeError(f"run.py failed: idx={idx} rc={rc} spec={raw}")

    # 4. read the summary JSON back as the function's return value
    base = (f"{spec['TAG']}_{spec['ADAPTER']}_{spec['KIND']}_"
            f"{spec['ADAPTER']}_seed{spec['SEED']}")
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
    manifest: str = "smax_scaling_manifest.py",
    config: str = "configs/marc_smax.yaml",
):
    """Drive any SMAX manifest from your laptop. No GPU/JAX needed
    locally — we only call the manifest (pure stdlib) to enumerate."""
    manifest_path = REPO_ROOT / "marc" / manifest
    if not manifest_path.exists():
        raise SystemExit(f"manifest not found: {manifest_path}")

    # Resolve which indices to run.
    if indices:
        ids = [int(x) for x in indices.split(",") if x.strip()]
    elif smoke:
        ids = [0]
    else:
        n = int(subprocess.check_output(
            [sys.executable, str(manifest_path), "--count"],
            text=True).strip())
        ids = list(range(n))

    # Apply runtime overrides only if the user changed defaults — this
    # keeps the base Function's autoscaler pool when defaults are kept.
    fn = run_one
    if gpu != "A10G" or max_containers != 8:
        fn = fn.with_options(gpu=gpu, max_containers=max_containers)

    print(f"[modal] {manifest}: launching {len(ids)} run(s) on {gpu} "
          f"(max_containers={max_containers}, smoke={smoke})", flush=True)

    if smoke:
        # Tiny budget so the smoke is cheap (~5 min, ~$0.10 on A10G).
        result = fn.remote(0, manifest=manifest, config=config,
                           total_timesteps=50_000)
        print("[smoke result]", json.dumps(result, indent=2,
                                           default=str), flush=True)
        return

    # Stream completions as they finish.
    n_done = 0
    n_fail = 0
    # starmap accepts (idx, kwargs); use a generator of (idx, manifest, config)
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
