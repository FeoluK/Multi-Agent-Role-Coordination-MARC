> 📒 **LIVING LOG — APPEND-ONLY.** Update this file consistently. For every
> experiment/decision/finding append a dated entry `## [YYYY-MM-DD] …`; never
> overwrite or delete prior content (supersede with a new dated entry). This
> is part of the full chronological record of everything we tried. _Started
> 2026-05-15._

# MARC — General Architecture Design

Status: **design for review** (implement after sign-off).
Scope: env-agnostic core + thin per-env adapters. Backbone = PPO with
centralized|decentralized critic as a config switch. Validated IPPO harness
(`code/baselines/run_ippo_overcooked.py`) is the starting point.

---

## 1. Problem (recap) and what MARC adds

Shared-reward MARL → **lazy agents** (contribute nothing) and **redundancy**
(agents duplicate a sub-task). Cause: shared reward gives no per-agent
contribution signal, and a policy on its own obs can't tell what teammates do.

MARC adds three actor-side modules + two aux losses:
1. **Self-encoder** → agent's own role latent from its recent history.
2. **Per-teammate inferencer** → a *separate* latent per teammate, inferred
   from that teammate's observable behavior (the proposal's novel piece — not
   an aggregated summary).
3. **Contribution-conditioned policy** → policy conditioned on own obs + own
   role latent + the set of inferred teammate latents.
4. `L_align` (match self-latent to what teammates infer about you) +
   `L_div` (push self-latents apart → anti-redundancy).

All four are **critic-agnostic** and **backbone-agnostic** (sit on the actor /
representation path). See `baseline_reference_numbers.md` for the IPPO≡MAPPO
rationale.

---

## 2. Env-agnostic abstraction

Everything below operates on abstract tensors. Per env, only an **adapter**
(§6) differs. Notation: `N` agents, agent `i`, teammates `j≠i`, history
window `H`, feature dim `D`, latent dim `Z`.

| Symbol | Shape | Meaning |
|---|---|---|
| `o_i_t` | env-specific | raw per-agent observation |
| `f_i_t` | `(D,)` | adapter-encoded obs feature (`ObsAdapter(o_i_t)`) |
| `a_i_t` | `()` int (or `(A,)`) | agent action (one-hot/embed internally) |
| `h_i_t` | `(H, D + A)` | rolling history of `(f, a)` for agent i |
| `z_i^self` | `(Z,)` | self role latent |
| `z_{i←j}` | `(Z,)` | i's inferred latent for teammate j |
| `Z_i^team` | `(N-1, Z)` | stack of i's inferred teammate latents |

**Teammate visibility.** The proposal states teammate observations *and*
actions are visible at execution ("as in standard CTDE"). We follow that:
the inferencer consumes teammate `(f_j, a_j)` histories at both train and
execution. (Design switch `teammate_obs_visible`: if an env only exposes
teammate *actions*, the inferencer falls back to `(a_j)` history. Flag for
discussion — Overcooked is fully observable so this is moot there.)

---

## 3. Modules (interfaces + shapes)

### 3.1 ObsAdapter (env-specific, the ONLY env-coupled module)
```
ObsAdapter(o_i_t) -> f_i_t : (D,)
```
- Overcooked: the validated 3-layer CNN from `ippo_cnn_overcooked` (obs
  `(h,w,26)` → `(D=64,)`). Reuse exactly so single-agent capacity matches the
  validated baseline.
- Minecraft (later): a small conv/MLP over its obs. Same `D` out.

### 3.2 SelfEncoder (general)
```
SelfEncoder(h_i_t) -> z_i^self : (Z,)
```
Small Transformer encoder over the length-`H` sequence of `(f, a_embed)`;
take the last position (or a [CLS] token). LayerNorm; `Z` ~ 32–64.

### 3.3 TeammateInferencer (general) — novel piece
```
TeammateInferencer(h_j_t) -> z_{i←j} : (Z,)   # applied per teammate j
```
Same Transformer architecture as SelfEncoder but **separate weights**, applied
**independently per teammate** → `Z_i^team : (N-1, Z)`. Per-teammate, never
aggregated at inference time. Weight-shared across teammate slots ⇒
N-agnostic.

### 3.4 ConditionedPolicy (general, N-agnostic)
```
ConditionedPolicy(f_i_t, z_i^self, Z_i^team) -> pi_i : Categorical(A)
```
Consuming a *variable* number of teammate latents while staying per-teammate:
**attention pool** over `Z_i^team` with query = `[f_i_t ; z_i^self]`, →
context `c_i : (Z,)`. Policy MLP on `[f_i_t ; z_i^self ; c_i]`. Attention pool
(not concat) is what makes this work for N=2 (Overcooked) and N>2 (Minecraft /
scaling claim) with one architecture. Per-teammate inference is preserved; only
the policy's *consumption* is pooled.

### 3.5 Critic (backbone switch)
- `decentralized` (IPPO-style, default for Overcooked / full-obs): critic on
  `[f_i_t ; z_i^self ; c_i]`.
- `centralized` (MAPPO-style, default off-Overcooked): critic on global
  state / concat of all `f_j` (+ latents). Config flag `critic: ff|central`.

---

## 4. Losses

```
L_total = L_PPO + λ_aux ( L_align + β · L_div )
```

- **L_PPO**: unchanged clipped PPO actor + value (+ entropy), reusing the
  validated IPPO loss/GAE.
- **L_align** (alignment): MSE between agent i's self-latent and what teammate
  j infers about i:
  ```
  L_align = mean_{i, j≠i} || stopgrad? z_i^self  -  z_{j←i} ||^2
  ```
  Decision needed: stop-gradient on `z_i^self` (inferencer chases self-encoder,
  one-directional) vs no stopgrad (both move together). Proposal text implies
  symmetric ("match what others believe it contributes"); default = **no
  stopgrad**, ablatable.
- **L_div** (diversity / anti-redundancy): penalize pairwise cosine similarity
  of self-latents:
  ```
  L_div = mean_{i<k} cos( z_i^self , z_k^self )
  ```
- Aux losses use the **same rollout batch** as PPO (no extra env steps).
  `λ_aux`, `β` config. Ablation hooks: `λ_aux=0`, zero `Z_i^team` before
  policy, `β=0` (proposal's three ablations) — all single config flags.

---

## 5. Where it plugs into the harness

We fork the validated IPPO loop into `code/marc/train_marc.py` (NOT editing
`JaxMARL_ref`). Changes vs vanilla IPPO loop:
1. Rollout buffer additionally stores per-agent `f` and `a` so histories
   `h_i_t` can be assembled (sliding window of length `H`; pad at episode
   start, mask on `done`).
2. Network = `ObsAdapter → {SelfEncoder, TeammateInferencer} → ConditionedPolicy
   + Critic`, a single Flax module so PPO update flows through everything.
3. Loss = `L_PPO + λ_aux(L_align + β L_div)`; aux terms computed from the
   batch's latents.
4. Eval/metrics add: leave-one-out return degradation (lazy-agent measure) and
   self-latent role-similarity (proposal's dependent measures).

Everything else (GAE, clipping, anneal, env, +20 reward, GIF) stays identical
so MARC-vs-IPPO is a clean controlled comparison.

---

## 6. Env-adapter contract (what a new env must supply)

```python
class EnvAdapter(Protocol):
    obs_adapter: nn.Module          # raw obs -> (D,)
    action_dim: int                 # A
    num_agents: int                 # N
    teammate_obs_visible: bool      # see §2
    def make_env(**kwargs): ...     # returns a JaxMARL-style env
```
Overcooked adapter = {CNN, A=6, N=2, visible=True}. Minecraft later = swap the
module + dims. Core MARC code imports nothing env-specific.

---

## 7. Decisions (LOCKED)

1. **History window `H` = 16.**
2. **`L_align`: symmetric, no stop-gradient** (self-encoder and inferencer
   move toward each other). Ablatable.
3. **Latent dim `Z` = 32**, Transformer = 2 layers, 2 heads, ff 64 (tiny,
   matched to the small validated CNN capacity).
4. **Minecraft: typed adapter stub now**, Overcooked adapter built fully.
5. **Windowed Transformer** over the length-16 `(f,a)` window (no recurrence).

---

## 8. Build order (after sign-off)
1. Refactor IPPO harness → pluggable network + adapter contract (no behavior
   change; re-validate cramped_room ≈ 240 to prove the refactor is clean).
2. Add SelfEncoder + ConditionedPolicy (no teammate yet) — sanity: should ≈
   vanilla IPPO.
3. Add TeammateInferencer + attention pool.
4. Add `L_align`, `L_div`; wire ablation flags.
5. Add LOO-degradation + role-similarity metrics.
6. Full MARC-IPPO vs vanilla IPPO sweep on the 5 layouts.
