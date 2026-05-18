> 📒 **LIVING LOG — APPEND-ONLY.** Update this file consistently. For every
> experiment/decision/finding append a dated entry `## [YYYY-MM-DD] …`; never
> overwrite or delete prior content (supersede with a new dated entry). This
> is part of the full chronological record of everything we tried. _Started
> 2026-05-15._

# Overcooked Baseline — Published Reference Numbers

## Key correction: IPPO, not MAPPO, for Overcooked

JaxMARL deliberately does **not** provide MAPPO for Overcooked. Overcooked is
fully observable (each agent sees the whole map), so a centralized critic adds
nothing — MAPPO ≈ IPPO here. JaxMARL paper App. F.1: *"we used MAPPO as the
baseline algorithm, except for Overcooked where we used IPPO."* No MAPPO
Overcooked config exists in the repo. **Our validated baseline = IPPO**
(`baselines/IPPO/ippo_cnn_overcooked.py` + `config/ippo_cnn_overcooked.yaml`).

Implication for the proposal: proposal names "vanilla MAPPO" as the baseline.
For Overcooked the apples-to-apples published baseline is IPPO; MAPPO with
shared obs is equivalent. Worth a sentence in the writeup.

## Setup that MUST match for numbers to be comparable

| Thing | Value |
|---|---|
| Delivery reward | **+20 per soup** (code: `DELIVERY_REWARD=20`). Docs saying "+1" are WRONG. |
| Shaped reward | +3 place-in-pot, +3 plate-pickup, +5 soup-pickup; annealed 1→0 over first **2.5e6** steps |
| Episode length | 400 steps |
| Training budget | **5e6** env timesteps, 64 envs × 256 rollout steps (CNN config) |
| Eval metric | **sparse game score only** (shaped reward removed) |

If logging return *including* shaped reward before anneal completes, early
numbers are inflated and not comparable — only compare post-2.5e6 sparse score.

## Target numbers (approx — papers only publish bar/curve plots, no tables)

| Layout | IPPO converged (sparse, +20/soup, 400-step ep) | Source |
|---|---|---|
| **cramped_room** | **~180–220** (best sanity check; ≈200 ≈ 10 soups/ep) | JaxMARL Fig 13 (precise anchor: ≈200 @ 5e6, overlays original Overcooked-AI) |
| asymm_advantages | ~200+ (highest-ceiling layout) | JaxMARL Fig 14 |
| coord_ring | ~150–200 | JaxMARL Fig 14 |
| forced_coord | ~100–200 (high variance, hardest coordination) | JaxMARL Fig 14 / Carroll Fig 4 |
| counter_circuit | ~100–150 (lowest, slowest to converge) | JaxMARL Fig 14 / Carroll Fig 4 |

**Primary go/no-go:** cramped_room must reach ~200. If not, something is wrong
(most likely cause: reward-scale confusion, +20 vs +1).

## Sources
- JaxMARL paper: arXiv:2311.10090 ; NeurIPS 2024 camera-ready (full appendix
  F.1, Table 6). Figs 13 (cramped_room IPPO≈200 @5e6, matches original), 14.
- Carroll et al. 2019: arXiv:1910.05789 — +20/soup, 400-step horizon, 100
  rollouts, 5 seeds; Fig 4 self-play PPO bars per layout.
- JaxMARL repo configs: `baselines/IPPO/config/ippo_cnn_overcooked.yaml`
  (`TOTAL_TIMESTEPS: 5e6`, `REW_SHAPING_HORIZON: 2.5e6`).
- Env source: `jaxmarl/environments/overcooked/overcooked.py`
  (`DELIVERY_REWARD=20` L65, `max_steps=400` L74, shaping L25-27).

## MAPPO vs IPPO decision (recorded)
- MAPPO = IPPO + centralized critic (critic-only diff, training-only). Actor &
  execution identical. They coincide in Overcooked because it's fully observable.
- MARC's additions are actor/representation-level (CTDE training inputs),
  orthogonal to the critic choice. -> Build MARC on IPPO backbone for
  Overcooked; compare MARC-IPPO vs vanilla IPPO. Note IPPO≡MAPPO (full obs) so
  proposal's "vs MAPPO" framing holds.
- General MARC (Task 3): backbone = config switch (PPO + centralized|decentralized
  critic). MAPPO-style centralized critic is the default for partially-observable
  envs (e.g. Minecraft stretch goal).
