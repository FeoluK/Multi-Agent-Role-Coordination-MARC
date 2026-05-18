# MARC — Multi-Agent Role-latent Coordination (Overcooked + MPE checkpoint)

Cooperative multi-agent RL study of an **anti-redundancy auxiliary loss**
plus a **role-latent / teammate-inference architecture**, built on
[JaxMARL](https://github.com/FLAIROx/JaxMARL) IPPO. This is a teammate
checkpoint — **only the Overcooked + MPE (JAX) track** (the Minecraft /
Craftax / Craftium exploration is intentionally excluded).

> New here? Read **`SETUP.md`** — it takes you from `git clone` to a
> training run and a FarmShare sweep step by step. **`RESULTS.md`** has
> the full numbers. `architectures/*.pdf` explains the networks.

## What's in here

| path | what |
|---|---|
| `marc/` | the env-agnostic MARC core (nets, PPO trainer, adapters, eval, manifests) |
| `marc/configs/` | `marc_overcooked.yaml`, `marc_mpe.yaml` |
| `farmshare/` | SLURM `*.sbatch` arrays + helpers (`sync_to_farmshare.sh`, `queue_filler.sh`, `snapshot.sh`, `plot_role_latents.py`) |
| `architectures/` | 4 PDF explainers (vanilla IPPO → MARC → fixed-aux) |
| `design/` | design doc + frozen results writeups |
| `results_gifs/` | trained-policy rollouts (Overcooked layouts, MPE N=3/6) |
| `requirements.txt` | exact pinned env from the working FarmShare run |

## Key results so far (Overcooked + MPE only)

**Method.** `NETWORK_KIND ∈ {vanilla, marc_self, marc, mappo, cds}`.
`marc` = role-latent self-encoder + per-teammate inferencer + attention
pool, with aux losses `L_align` (self vs teammates' inference) and
`L_div` (gated cosine of role-latents). `LAMBDA_AUX=0` = architecture
only (aux off).

1. **MPE `simple_spread`, scaling N=3/6/9 — the headline.** The MARC
   architecture beats vanilla IPPO at **every** team size, and the gap
   **grows monotonically with N**. Under rliable (IQM + 95%
   stratified-bootstrap CI + probability of improvement over 182 runs):
   **P(MARC > vanilla) = 1.000 [1.000, 1.000]** on eval *and* final
   return at N=3, 6, and 9. The fixed aux (`normgateup`) adds a further,
   N-scaling improvement that is significant paired (p<0.01 at every N).
   This is the proposal's central hypothesis ("redundancy hurts more as
   the team grows; per-teammate inference helps"), and it holds.

2. **Overcooked (2-agent, 5 layouts) — architecture good, aux harmful.**
   Pooled IQM eval return: vanilla **271** ≈ **marc-architecture-only
   272** ≫ marc-full-aux **197**. Per layout the architecture-only
   config **wins the tightest-coordination layout** (`forced_coord`,
   P≈0.94–1.0), ties the symmetric ones, and — critically — the
   divergence aux is the *sole* cause of two failures it otherwise
   fixes: `counter_circuit` **0 → 159** (collapse rescued, beats vanilla
   146) and `asymm_advantages` **260 → 476** (regression recovered).
   **Takeaway: the MARC architecture delivers the predicted coordination
   gains; the divergence objective as formulated should be disabled or
   annealed on hard 2-agent layouts.**

3. **Mechanism is real, not assumed.** Role-latent cosine similarity
   (`role_similarity` of `z_self`) flips from ≈+0.95 (redundant) to
   ≈0/negative (differentiated) exactly where specialization should
   happen — and a t-SNE/PCA projection of `z_self` (tooling in
   `farmshare/plot_role_latents.py`) gives the qualitative companion.

Numbers, CIs and caveats in **`RESULTS.md`**. Rollout GIFs in
`results_gifs/` (Overcooked per layout; MPE vanilla vs arch vs
normgateup at N=3/6).

## Directions to go

- **Baselines (in progress, GPU-queued on FarmShare):** agent-ID
  parameter-sharing (the "is MARC just param-sharing?" control), MAPPO
  (centralized critic — the real partial-observability baseline on MPE),
  CDS-style identity-MI diversity (the apt peer-method comparison).
  Closes the "every result is MARC-vs-its-own-ablation" gap.
  `marc/rliable_report.py` already auto-analyzes them.
- **Uniform seeds** (N≥8) for the headline cells; **SMAX** (the
  strongest >2-agent setting); **scale the MPE budget**.
- **Redesign the divergence aux** so it stops collapsing hard Overcooked
  layouts while keeping the MPE N-scaling gain.

## Provenance / reproducibility

`requirements.txt` is the exact environment of the run that produced
these numbers (Python 3.10, `jax==0.4.36` cuda12, `flax==0.10.4`,
JaxMARL pinned to commit
`2cfd60092438b71b24eb35f5f3d9ddd75d37d0b1`). See `SETUP.md`.
