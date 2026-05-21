"""Hanabi N-scaling sweep: does MARC's anti-redundancy mechanism transfer
to a partial-info, turn-based, fundamentally different cooperative game?

vanilla IPPO vs MARC architecture (LAMBDA_AUX=0 — the winning Overcooked
+ MPE config) across N = 2 / 3 / 5 players. 5 seeds per cell,
2 kinds * 3 N * 5 seeds = 30 runs.

N=4 is skipped because Hanabi's hand_size drops from 5 (for N<=3) to 4
(for N>=4), which would confound a clean 'only N varies' ladder.

NB v2: this revision (commit b82f937 onwards) adds illegal-action
masking to train_marc.py / evaluate.py so the policy only samples
legal Hanabi moves. Previous Hanabi runs at 5e6 timesteps without
masking produced unreliable comparisons (both vanilla and marc trained
against ~50% wasted samples per step). Bumping TOTAL_TIMESTEPS to 1e8
(20x the previous Hanabi sweep, still 100x under JaxMARL reference's
1e10 but within Modal-budget reach). PER_CELL_SETS overrides keep the
rollout buffer fitting in GPU memory at N=5.

Caveat (worth flagging in the writeup): Hanabi's turn-based structure
naturally differentiates 'roles' by who's next to act, which dilutes
MARC's anti-redundancy mechanism a priori. This is a stress test of
cross-domain generalization, NOT a target environment.
"""
import sys

PER_CELL_SETS = {
    ("vanilla", "hanabi_5"): ["NUM_ENVS=32"],   # obs dim ~1280; t_obs heavy
    ("marc",    "hanabi_5"): ["NUM_ENVS=32"],
}
ADAPTERS = ["hanabi_2", "hanabi_3", "hanabi_5"]
SEEDS = list(range(30, 35))                     # 5 seeds per cell
# 5e7 timesteps: 10x the previous 5e6 Hanabi sweep, ~3050 PPO updates
# at NUM_ENVS=64/NUM_STEPS=256. Defensible: if MARC needs more budget
# than vanilla to converge, 10x is a strong test. If MARC still loses
# at 10x, that's consistent with the SMAX 8m budget-ladder finding
# (MARC's heavier architecture is a real efficiency tax, not just an
# undertraining artifact). JaxMARL's reference IPPO uses 1e10 at
# NUM_ENVS=1024 NUM_STEPS=128 (~76k updates); we're at ~25x fewer
# updates but with action masking now matching the reference recipe.
HANABI_BUDGET_SETS = ["TOTAL_TIMESTEPS=5e7"]
CONFIGS = [
    ("hanabi_vanilla", "vanilla", list(HANABI_BUDGET_SETS)),
    ("hanabi_marc",    "marc",    ["LAMBDA_AUX=0", *HANABI_BUDGET_SETS]),
]
RUNS = [(t, k, s_cfg + PER_CELL_SETS.get((k, ad), []), ad, sd)
        for (t, k, s_cfg) in CONFIGS
        for ad in ADAPTERS
        for sd in SEEDS]


def spec(idx):
    tag, kind, sets, adapter, seed = RUNS[idx]
    extra = " ".join(f"--set {s}" for s in sets)
    return (f"TAG={tag} KIND={kind} ADAPTER={adapter} SEED={seed} "
            f"SETS='--set ADAPTER={adapter} {extra}'")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--count":
        print(len(RUNS))
    else:
        print(spec(int(sys.argv[1])))
