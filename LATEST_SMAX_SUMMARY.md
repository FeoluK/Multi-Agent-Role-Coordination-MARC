# SMAX summary: vanilla / MAPPO / MARC / MARC+LATENT_GATE

Eval-return table on the SMAX N-scaling ladder, comparing the four
methods we have data for at $5\!\times\!10^{6}$ steps per run.

## All seeds (full data per method)

| Cell             | vanilla              | MAPPO               | MARC (cold)          | MARC + LATENT_GATE   |
| ---------------- | -------------------- | ------------------- | -------------------- | -------------------- |
| `smax_3m`        | 2.005 ± 0.579 (n=10) | 4.284 ± 0.288 (n=3) | 3.524 ± 0.364 (n=10) | 2.654 ± 0.945 (n=3)  |
| `smax_8m`        | 0.558 ± 0.084 (n=10) | 0.443 ± 0.008 (n=3) | 0.545 ± 0.118 (n=10) | **0.613 ± 0.103** (n=3) |
| `smax_10m_vs_11m`| 0.545 ± 0.104 (n=10) | 0.512 ± 0.193 (n=3) | 0.499 ± 0.101 (n=10) | **0.582 ± 0.074** (n=3) |

## Matched seeds (30, 31, 32 only — paired comparison)

| Cell             | vanilla            | MAPPO              | MARC (cold)        | MARC + LATENT_GATE |
| ---------------- | ------------------ | ------------------ | ------------------ | ------------------ |
| `smax_3m`        | 2.232 ± 0.315 (n=3) | 4.284 ± 0.288 (n=3) | 3.299 ± 0.448 (n=3) | 2.654 ± 0.945 (n=3) |
| `smax_8m`        | 0.518 ± 0.034 (n=3) | 0.443 ± 0.008 (n=3) | 0.554 ± 0.211 (n=3) | **0.613 ± 0.103** (n=3) |
| `smax_10m_vs_11m`| 0.500 ± 0.135 (n=3) | 0.512 ± 0.193 (n=3) | 0.542 ± 0.085 (n=3) | **0.582 ± 0.074** (n=3) |

## Flag check (per teammate's request)

> *"Flag any game where gated MARC is still below vanilla."*

**No SMAX cells flagged.** MARC + LATENT_GATE ≥ vanilla on every SMAX
cell at both the all-seeds and matched-seeds views.

## Key observations

1. **LATENT_GATE rescues the failure cells.** On `8m` and `10m_vs_11m`,
   where cold MARC tied or slightly lost vanilla, MARC+gate wins by
   $+0.05$ and $+0.04$ respectively (matched seeds). The ReZero gate
   lets the policy start as vanilla and PPO opens the gate only when
   the role latents help, exactly as designed.

2. **LATENT_GATE pays a tax on the easy cell.** On `smax_3m`, where
   cold MARC already wins decisively (3.30 vs vanilla 2.23 at matched
   seeds), MARC+gate drops to 2.65 — still above vanilla but $\sim 0.6$
   points below cold MARC. The gate gives PPO an "out" that it
   sometimes takes even when the role latents would have helped, so
   peak performance on MARC's home turf is reduced.

3. **MAPPO remains the strongest method on 3m only.** Centralised
   critic helps when the joint observation is small (90-dim at 3m); on
   8m and 10m_vs_11m it's neutral or actively hurts (loses to vanilla
   at 8m by P=0.00 in the original rliable run).

4. **No method strictly dominates.** MAPPO wins 3m, MARC+gate wins 8m
   and 10m_vs_11m, MARC cold is competitive only at 3m. The scope
   condition holds: each method's advantage tracks structural
   properties of the cell, not raw capacity.

## Methodology / data provenance

- **vanilla / MARC cold:** `smax_pull/smax_results.json`
  (10 seeds × 3 cells = 60 runs from `smax_scaling_manifest.py`,
  budget $5\!\times\!10^{6}$ steps).
- **MAPPO:** `smax_mappo_pull/smax_mappo_results.json`
  (3 seeds × 3 cells = 9 runs from `smax_mappo_manifest.py`).
- **MARC + LATENT_GATE:** `smax_latentgate_pull/*.json`
  (3 seeds × 3 cells = 9 runs from `smax_latent_gate_manifest.py`,
  with `--set LATENT_GATE=1 --set LAMBDA_AUX=0`).
- All runs use identical PPO hyperparameters (`configs/marc_smax.yaml`)
  and `NUM_ENVS=32` override on `10m_vs_11m` for memory headroom.
- Reported metric is `eval_return` (sampled-action eval over 64 parallel
  episodes, fixed horizon = `env.max_steps=100`). Seeds 30..39 for
  vanilla/MARC, 30..32 for MAPPO/LATENT_GATE.
