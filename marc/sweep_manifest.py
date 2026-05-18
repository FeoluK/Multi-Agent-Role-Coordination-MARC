"""Sweep manifest: maps a flat index (SLURM_ARRAY_TASK_ID) -> one run spec.

Primary scientific comparison + the proposal's 3 ablations, across the 5
standard Overcooked layouts, multi-seed. Printed one-line so the array
sbatch can `eval` it.
"""
import sys

LAYOUTS = ["cramped_room", "asymm_advantages", "coord_ring",
           "forced_coord", "counter_circuit"]
SEEDS = [30, 31, 32]

# (tag, network_kind, extra --set overrides)
CONFIGS = [
    ("vanilla",   "vanilla",   []),
    ("marc",      "marc",      []),
    ("abl_lam0",  "marc",      ["LAMBDA_AUX=0"]),      # no aux supervision
    ("abl_beta0", "marc",      ["BETA=0"]),            # no diversity term
    ("abl_zt",    "marc",      ["ZERO_TEAMMATE=true"]),  # no teammate latent
]

RUNS = [
    (tag, kind, sets, layout, seed)
    for (tag, kind, sets) in CONFIGS
    for layout in LAYOUTS
    for seed in SEEDS
]


def spec(idx):
    tag, kind, sets, layout, seed = RUNS[idx]
    sets_str = " ".join(f"--set {s}" for s in sets)
    return (f"TAG={tag} KIND={kind} LAYOUT={layout} SEED={seed} "
            f"SETS='{sets_str}'")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--count":
        print(len(RUNS))
    else:
        print(spec(int(sys.argv[1])))
