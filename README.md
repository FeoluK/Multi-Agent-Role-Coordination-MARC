# MARC — Multi-Agent Role-latent Coordination (MPE + Overcooked + SMAX + Hanabi)

Cooperative multi-agent RL study of an **anti-redundancy auxiliary loss**
plus a **role-latent / teammate-inference architecture**, built on
[JaxMARL](https://github.com/FLAIROx/JaxMARL) IPPO. The original
checkpoint covered MPE + Overcooked; this snapshot extends to a
~150-run cross-game study on SMAX (`3m / 8m / 10m_vs_11m`, plus a
budget ladder and a `2s3z` heterogeneous-units ablation) and on Hanabi
(`2p / 3p / 5p`) under the rliable bootstrap protocol. The headline
flips from "MARC always wins" to a sharper, more useful **scope
condition** — see `RESULTS.md` §5.

> New here? Read **`SETUP.md`** — it takes you from `git clone` to a
> training run and a FarmShare sweep step by step. **`RESULTS.md`** has
> the full numbers. `architectures/*.pdf` explains the networks.

## What's in here

| path | what |
|---|---|
| `marc/` | the env-agnostic MARC core (nets, PPO trainer, adapters, eval, manifests) |
| `marc/configs/` | `marc_overcooked.yaml`, `marc_mpe.yaml`, `marc_smax.yaml`, `marc_hanabi.yaml` |
| `marc/adapters.py` | env adapters: Overcooked, MPE (3/6/9), SMAX (3m/8m/25m/2s3z/etc.), Hanabi (2p/3p/4p/5p) |
| `farmshare/` | SLURM `*.sbatch` arrays + helpers (`sync_to_farmshare.sh`, `queue_filler.sh`, `snapshot.sh`, `plot_role_latents.py`) |
| `modal_smax_scaling.py` | Modal app for cloud-GPU sweeps; `--manifest` selects which sweep |
| `smax_pull/`, `smax_followup_pull/`, `hanabi_pull/` | consolidated result JSONs (single file each, npzs gitignored) |
| `architectures/` | 4 PDF explainers (vanilla IPPO → MARC → fixed-aux) |
| `design/` | design doc + frozen results writeups |
| `results_gifs/` | trained-policy rollouts (Overcooked layouts, MPE N=3/6) |
| `requirements.txt` | exact pinned env from the working FarmShare run |

## Key results

**Method.** `NETWORK_KIND ∈ {vanilla, marc_self, marc, mappo, cds}`.
`marc` = role-latent self-encoder + per-teammate inferencer + attention
pool, with aux losses `L_align` (self vs teammates' inference) and
`L_div` (gated cosine of role-latents). `LAMBDA_AUX=0` = architecture
only (aux off; the canonical "marc-arch" used in cross-game tests).

1. **MPE `simple_spread`, scaling N=3/6/9 — the headline.** MARC beats
   vanilla IPPO at **every** team size with the gap monotone in N.
   `P(MARC > vanilla) = 1.000` on both eval and final return at N=3, 6,
   and 9. `role_similarity` of `z_self` flips ≈+0.95 → ≈0/negative
   exactly where specialization should happen (mechanism firing).

2. **Overcooked (2-agent, 5 layouts) — architecture good, aux harmful.**
   Pooled IQM eval return: vanilla **271** ≈ **marc-arch 272** ≫
   marc-full-aux 197. Architecture wins the coordination-critical
   `forced_coord` (P≈0.94 \*); divergence aux *causes* two collapses
   (`counter_circuit` 0/146; `asymm` 260/490) that arch-only *fixes*.

3. **SMAX cross-game N-scaling — clean win at N=3, fails at N≥5.**
   `smax_3m`: MARC P = **0.97 \***, IQM ~2× vanilla (mechanism check
   fires: `loo_drop_min` 0.085 → 2.15). `smax_8m` and
   `smax_10m_vs_11m`: tied at 5e6, but a **budget ladder on 8m**
   (5e6 / 10e6 / 20e6) shows vanilla pulls progressively ahead while
   MARC plateaus — at 20e6, P(MARC > vanilla) = **0.111 \*** (vanilla
   wins). The N=8 tie was *not* concealing a MARC win.

4. **SMAX `2s3z` (heterogeneous units) — MARC's "home turf" falsified.**
   Hypothesis: MARC ties vanilla on homogeneous Marines because there's
   nothing to specialize on; once unit types differ, MARC will win.
   Result: P(marc > vanilla) = **0.19 \***, vanilla wins. `behavioral_sep`
   ~0.025 for both methods (vs ~0.001 on Marines) shows vanilla
   already differentiates units via the natural unit-type-bit features
   in the obs. MARC's specialization machinery is redundant.

5. **Hanabi cross-domain stress test (partial-info, turn-based) —
   tied at 2p/3p, vanilla wins at 5p.** With action masking matching
   JaxMARL's reference IPPO recipe and 5e7 timesteps: 2p eval ~22/25
   (P = 0.40, tied), 3p eval ~20/25 (P = 0.60, tied), 5p `final_return`
   vanilla 10.98 vs MARC 9.68 (P = **0.04 \***, vanilla wins).

6. **Scope condition** (the actual scientific contribution):
   MARC works when (a) redundancy is the binding constraint, (b) the
   obs doesn't already provide free positional / role-disambiguating
   features, and (c) the per-step compute budget is small enough that
   the heavier architecture isn't a meaningful drag. These conditions
   hold on MPE simple_spread and small-N coordination (smax_3m,
   Overcooked forced_coord); they don't hold on SMAX combat at scale
   or on Hanabi.

Full numbers, CIs and caveats in **`RESULTS.md`**. Result data:
`smax_pull/smax_results.json`, `smax_followup_pull/smax_followup_results.json`,
`hanabi_pull/hanabi_results.json` (consolidated; per-run JSONs collapsed
to single files). Rollout GIFs in `results_gifs/`.

## Directions to go

- **Established external baselines** (agent-ID parameter-sharing,
  MAPPO with centralized critic, CDS-style identity-MI diversity) are
  implemented in-tree and queued on FarmShare GPU; not yet folded into
  the SMAX / Hanabi cross-game cells. `marc/rliable_report.py` already
  auto-groups them when results land. Until those land, treat the
  headline as "MARC ≷ vanilla IPPO and its ablations under rigorous
  rliable testing across 4 envs" — not yet "beats all baselines on the
  positive cells".
- **Hanabi 5p MARC eval cell** is missing (training data only) due to
  Modal A100-80GB preemption + a hanabi_5 + GPU XLA bug under masked
  eval; see `RESULTS.md` §6 for the workaround.
- **Uniform seeds** (N≥10) on the SMAX budget-ladder and Hanabi cells
  (currently 3–5 seeds per cell there).
- **Redesign the divergence aux** so it stops collapsing hard Overcooked
  layouts while keeping the MPE N-scaling gain (still pending from the
  original checkpoint).

## Provenance / reproducibility

`requirements.txt` is the exact environment of the run that produced
these numbers (Python 3.10, `jax==0.4.36` cuda12, `flax==0.10.4`,
JaxMARL pinned to commit
`2cfd60092438b71b24eb35f5f3d9ddd75d37d0b1`). See `SETUP.md`.
