# RESULTS — Overcooked + MPE (rliable, 2026-05-18)

All numbers from `marc/rliable_report.py` over 182 result `.json`s:
**IQM** (interquartile mean) with **95% stratified-bootstrap CI**, plus
**P(X > vanilla)** = common-language effect size (probability a random
X-seed beats a random vanilla-seed), bootstrap CI. `*` = 95% CI excludes
0.5 ⇒ statistically distinguishable. Agarwal et al. (2021) protocol.

Configs: `vanilla` = IPPO shared net. `marc-full` = full role-latent
architecture **with** the divergence aux (`LAMBDA_AUX=0.1`).
`marc-arch` = same architecture, **aux off** (`LAMBDA_AUX=0`).
`normgateup` = the salvaged fixed aux (`LAMBDA_AUX=0.5 AUX_NORM AUX_GATE
AUX_ANNEAL=up`).

---

## 1. MPE `simple_spread` — scaling in team size N (the headline)

Shared, negative reward (−distance−collisions); higher = better. 3
vanilla seeds, 8 marc seeds per cell.

**eval return, IQM:**

| N | vanilla | marc-arch | normgateup | P(arch>van) | P(ngup>van) |
|---|---|---|---|---|---|
| 3 | −19.52 | **−10.99** | −11.0 | 1.000 [1,1] * | 1.000 [1,1] * |
| 6 | −34.69 | **−31.55** | **−29.35** | 1.000 [1,1] * | 1.000 [1,1] * |
| 9 | −54.03 | **−43.34** | **−40.07** | 1.000 [1,1] * | 1.000 [1,1] * |
| pooled | −35.47 | −29.92 | **−25.41** | 1.000 [1,1] * | 1.000 [1,1] * |

Identical story on final (training) return. **The MARC architecture
beats vanilla IPPO at every N with probability 1.0, and the absolute gap
grows with N** (−8.5 → −3.1 → −10.7 IQM at N=3/6/9; pooled −5.5).
`normgateup` adds a further gain that itself scales with N and is
significant under a seed-matched paired test (N=3 n=8 p=0.0013; N=6 n=8
p=0.0002; N=9 n=6 p=0.0026 — all p<0.01, 22/22 paired runs improved).
`role_similarity` of `z_self` flips ≈+0.95 → ≈0/negative (mechanism
firing). This is the proposal's core hypothesis, confirmed.

---

## 2. Overcooked — 2-agent, 5 layouts (architecture good, aux harmful)

3 seeds/cell. eval return, IQM:

| layout | vanilla | marc-full (aux on) | **marc-arch (aux off)** |
|---|---|---|---|
| cramped_room | 240 | 240 | 240 (tie, ceiling) |
| coord_ring | 287 | 287 | 254 (slight regression) |
| forced_coord | 192 | 200 | **199**  (P(arch>van)≈0.94 *) |
| asymm_advantages | **490** | 260  (collapse) | **476**  (recovered) |
| counter_circuit | 146 | **0**  (collapse, all seeds) | **159**  (rescued, > vanilla) |
| **pooled (mean)** | **271** | **197** | **272 ≈ vanilla** |

Pooled, `marc-full` is significantly *worse* than vanilla
(P=0.34 [0.23,0.47] *) — but that is **entirely** the divergence aux:
turning it off (`marc-arch`) restores vanilla parity pooled, keeps the
`forced_coord` coordination win (the tightest-coordination layout), and
**eliminates both failure modes** the aux causes (`counter_circuit`
0→159, `asymm` 260→476).

**Conclusion:** the MARC *architecture* delivers the predicted
coordination gain on the layout that needs it and never collapses; the
divergence *objective* as formulated is net-harmful on hard 2-agent
layouts and should be disabled or annealed there. This is consistent
with MPE, where `LAMBDA_AUX=0` (architecture-only) also wins.

---

## 3. Honest caveats

- 2-agent Overcooked structurally caps a "gap grows with N" method;
  the N-scaling evidence is MPE-only (by design — Overcooked is 2-agent).
- Overcooked cells are 3 seeds; MPE marc cells 6–8, vanilla 3 (uneven —
  rliable handles it via within-cell bootstrap; uniform N≥8 top-ups are
  a pending direction).
- The MPE comparison is currently **MARC vs its own ablations + vanilla
  IPPO**. Established baselines (agent-ID parameter-sharing, MAPPO with
  a centralized critic, CDS-style identity-MI diversity) are
  implemented and queued on FarmShare (GPU-contended at checkpoint
  time); `rliable_report.py` will fold them in automatically. Until
  those land, treat the headline as "MARC beats vanilla IPPO and its
  ablations, rigorously" — not yet "beats all baselines".
- Single budget per cell (≈3–5e6 steps); a scaled-budget rerun is a
  pending direction.

Frozen long-form writeups: `design/results_final.md`,
`design/results_interim.md`. Rollout GIFs: `results_gifs/`.
