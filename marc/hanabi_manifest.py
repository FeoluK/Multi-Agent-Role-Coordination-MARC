"""Hanabi N-scaling sweep: does MARC's anti-redundancy mechanism transfer
to a partial-info, turn-based, fundamentally different cooperative game?

vanilla IPPO vs MARC architecture (LAMBDA_AUX=0 — the winning Overcooked
+ MPE config) across N = 2 / 3 / 5 players. 5 seeds per cell,
2 kinds * 3 N * 5 seeds = 30 runs.

N=4 is skipped because Hanabi's hand_size drops from 5 (for N<=3) to 4
(for N>=4), which would confound a clean 'only N varies' ladder.

Per-cell NUM_ENVS overrides keep the rollout buffer in GPU memory.
Hanabi's obs is large (~660-dim @ 2p, ~1280-dim @ 5p) and the MARC
rollout buffer t_obs ~ NA*M*H*obs scales with N*N_obs(N), so 5p needs
NUM_ENVS=32 to fit on A10G; A100-40GB / A100-80GB clear it at 64.

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
CONFIGS = [
    ("hanabi_vanilla", "vanilla", []),
    ("hanabi_marc",    "marc",    ["LAMBDA_AUX=0"]),
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
