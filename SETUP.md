# SETUP — from clone to training (Overcooked + MPE)

Everything below is the exact, known-good path. The science runs on
**Stanford FarmShare** (SLURM + GPU); you can also smoke-test locally on
CPU. Replace `<SUNET>` with your Stanford username throughout.

---

## 0. Clone

```bash
git clone <THIS_REPO_URL> marc-checkpoint
cd marc-checkpoint
```

The Python package is `marc/` (run commands from inside it). Configs are
`marc/configs/`. Cluster scripts are `farmshare/`.

---

## 1. Environment (conda + JAX + JaxMARL)

Reproduce the **exact** working env. Python **3.10**, CUDA 12 GPU JAX.

```bash
conda create -y -p ./envs/jaxmarl python=3.10
conda activate ./envs/jaxmarl

# GPU JAX first (must match: jax==0.4.36, cuda12)
pip install "jax[cuda12]==0.4.36"

# JaxMARL pinned to the commit these results were produced with
pip install "git+https://github.com/FLAIROx/JaxMARL.git@2cfd60092438b71b24eb35f5f3d9ddd75d37d0b1"

# the rest, pinned
pip install -r requirements.txt
```

`requirements.txt` is a full `pip freeze` of the working FarmShare env
(includes the cuda wheels and the editable JaxMARL line — if the
editable line errors, the two explicit installs above already cover it;
`pip install -r` the remainder).

Sanity check:

```bash
python -c "import jax, jaxmarl, flax, distrax; print(jax.devices())"
```

CPU-only smoke (no GPU box): prepend `JAX_PLATFORMS=cpu` to any command
and keep `TOTAL_TIMESTEPS` tiny (see §3).

---

## 2. What the code is

- `marc/run.py` — entry point. Loads a YAML config, applies `--set
  KEY=VAL` overrides, trains, evaluates, writes a result `.json` +
  `.npz` + a rollout `.gif`.
- `marc/configs/marc_overcooked.yaml`, `marc_mpe.yaml` — base
  hyperparameters (identical PPO settings; only env + MARC knobs differ).
- `NETWORK_KIND` (set via `--network-kind` or `--set`):
  `vanilla` (IPPO) · `marc_self` · `marc` (full role-latent) ·
  `mappo` (centralized critic) · `cds` (identity-MI diversity).
- `ADAPTER`: `overcooked` (use `--layout`), `mpe_spread_{3,6,9}`,
  `mpe_spread_{3,6,9}_id` (agent-ID baseline variant).
- MARC knobs: `LAMBDA_AUX` (0 = architecture only — the Overcooked-safe
  config), `BETA`, `AUX_NORM`, `AUX_GATE`, `AUX_ANNEAL`,
  `ZERO_TEAMMATE`.
- Manifests (`*_manifest.py`) map a SLURM array index → one run spec;
  the matching `farmshare/*.sbatch` runs them as job arrays.

---

## 3. Run locally (smoke)

MPE, vanilla, tiny budget, CPU:

```bash
cd marc
JAX_PLATFORMS=cpu python run.py \
  --config configs/marc_mpe.yaml --network-kind vanilla --seed 30 \
  --set ADAPTER=mpe_spread_3 --set TOTAL_TIMESTEPS=30000 --set NUM_ENVS=8 \
  --results-dir /tmp/marc_out --gifs-dir /tmp/marc_out
```

MARC architecture-only on an Overcooked layout:

```bash
JAX_PLATFORMS=cpu python run.py \
  --config configs/marc_overcooked.yaml --network-kind marc \
  --layout forced_coord --seed 30 --set LAMBDA_AUX=0 \
  --set TOTAL_TIMESTEPS=50000 \
  --results-dir /tmp/marc_out --gifs-dir /tmp/marc_out
```

Each run writes `<tag>_<kind>_<scenario>_seed<seed>.json` (with an
`eval` block: `eval_return`, `loo_drop`, `role_similarity`,
`behavioral_sep`) and a `.gif`. Real science budgets are
`TOTAL_TIMESTEPS=5e6` — use the GPU + FarmShare for those.

---

## 4. FarmShare (the real runs)

### 4a. Get the code onto FarmShare

You work from a local copy and push code only; results stay on the
cluster. Edit `farmshare/sync_to_farmshare.sh` so `REMOTE` points at
**your** scratch (`<SUNET>@rice.stanford.edu:/scratch/users/<SUNET>/marc`),
then:

```bash
bash farmshare/sync_to_farmshare.sh    # rsyncs marc/ + farmshare/
```

On FarmShare, recreate the env once (§1) at
`/scratch/users/<SUNET>/marc/envs/jaxmarl` and put the code under
`/scratch/users/<SUNET>/marc/{code/marc,scripts,configs,results,gifs,logs}`.
(The `.sbatch` files assume that layout — adjust the `S=` path at the
top of each to your scratch.)

### 4b. Submit a sweep (SLURM job array)

Each `*.sbatch` reads a manifest by `$SLURM_ARRAY_TASK_ID`. Get the
array size from the manifest, then submit throttled:

```bash
cd /scratch/users/<SUNET>/marc/code/marc
N=$(python sweep_manifest.py --count)          # Overcooked vanilla/marc/abl
sbatch --array=0-$((N-1))%6 ../../scripts/marc_sweep.sbatch
```

Useful manifests/sbatch (Overcooked + MPE only):

| sweep | manifest | sbatch |
|---|---|---|
| Overcooked vanilla/marc + ablations | `sweep_manifest.py` | `marc_sweep.sbatch` |
| MPE N=3/6/9 scaling | `mpe_scaling_manifest.py` | `marc_mpe_scaling.sbatch` |
| MPE confirm (vanilla/arch/normgateup) | `mpe_confirm_manifest.py` | `mpe_confirm.sbatch` |
| Agent-ID + role-latent dumps | `baselines_manifest.py` | `baselines.sbatch` |
| MAPPO baseline | `mappo_manifest.py` | `mappo.sbatch` |
| CDS-style baseline | `cds_manifest.py` | `cds.sbatch` |

The gpu QOS cap is **32 submitted tasks / 4 concurrent**.
`farmshare/queue_filler.sh` (run detached:
`setsid nohup bash farmshare/queue_filler.sh >qf.log 2>&1 &`) drips
chunked sweeps in as headroom frees so you never hit the cap — edit its
`CHUNKS=()` list to your jobs.

Lean jobs (1 GPU / 32G / **6h walltime**) are backfill-friendly — keep
the walltime short or you wait much longer in the queue.

### 4c. Monitor

```bash
squeue -u <SUNET> -o "%.14i %.16j %.9T %.6M %R"
tail -f /scratch/users/<SUNET>/marc/logs/<job>_<id>.log
```

---

## 4b. Modal (alternative cloud GPU backend, SMAX sweep)

`modal_smax_scaling.py` at the repo root is a [Modal](https://modal.com/)
app that runs the SMAX scaling sweep
(`marc/smax_scaling_manifest.py`: vanilla IPPO vs MARC arch,
{`smax_3m`, `smax_8m`, `smax_10m_vs_11m`}, seeds 30..39 — **60 runs**)
without needing FarmShare. Same `marc/run.py` entry point per
container, same JSON schema out, so `marc/rliable_report.py` consumes
the results unchanged.

Setup (one-time):

```bash
pip install modal     # in any local Python env (no GPU/JAX needed locally)
modal token new       # browser-auth your account
```

Smoke first (one container, ~5 min on A10G, ~$0.10):

```bash
modal run modal_smax_scaling.py --smoke
```

Full sweep (60 runs, A10G default ≈ 120 GPU-hours @ $1.10/hr ≈ $130;
finish wall-clock ~6 h with `--max-containers 20`):

```bash
modal run modal_smax_scaling.py
modal run modal_smax_scaling.py --max-containers 16     # raise parallelism
modal run modal_smax_scaling.py --gpu A100              # heavier GPU
modal run modal_smax_scaling.py --indices 0,30          # subset
```

Pull results locally and analyze:

```bash
modal volume get marc-smax-results / ./modal_results
python marc/rliable_report.py ./modal_results
```

The image pins `jax==0.4.36` cuda12 + JaxMARL @ the same commit
`requirements.txt` does, so numerics match the FarmShare baselines.
The `marc/` source is mounted at runtime (no rebuild on code edits).
Per-cell `NUM_ENVS` overrides (e.g. `smax_10m_vs_11m` -> `NUM_ENVS=32`)
are baked into the manifest, so no Modal-side tuning is needed.

---

## 5. Analyze

```bash
cd marc
python aggregate.py /scratch/users/<SUNET>/marc/results        # mean±std table
python rliable_report.py /scratch/users/<SUNET>/marc/results   # IQM + 95% CI + P(improve)
```

`rliable_report.py` is pure-numpy (no GPU, runs anywhere the result
`.json`s are) and already groups vanilla / marc / marc-arch / agent-ID /
mappo / cds for Overcooked (pooled + per-layout) and MPE (per-N +
pooled).

Role-latent differentiation figure (after a run with
`--set DUMP_LATENTS=true`):

```bash
python ../farmshare/plot_role_latents.py <results>/<run>_zself.npz
```

---

## 6. Conventions (please keep)

- `snapshot.sh <file> "reason"` before any behaviour-changing edit
  (lightweight versioned backup; see `farmshare/snapshot.sh`).
- Results/gifs/logs live on FarmShare; only code is synced/committed.
