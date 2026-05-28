"""Hanabi MAPPO baseline sweep: centralised-critic CTDE (Yu et al. 2022)
on the same N-scaling ladder we used for vanilla IPPO and MARC. The
'is MARC's tie just because the centralised critic also struggles with
partial-info turn-based games?' control on Hanabi.

3 adapters * 3 seeds = 9 runs, matching the seed count of mappo_manifest.py
(MPE) and smax_mappo_manifest.py for cross-env consistency. Tag pattern
hanabi_mappo_{adapter} so rliable_report.py picks runs up alongside
hanabi_vanilla_* / hanabi_marc_*.

Same TOTAL_TIMESTEPS=5e7 budget as the existing Hanabi sweep so the
within-cell vanilla-vs-marc-vs-mappo comparison is at matched compute.
NUM_ENVS=32 override on hanabi_5 for memory headroom (matches existing
sweep). MAPPO's centralised critic adds a (NA, n_agents*obs_dim)
world-state buffer — at hanabi_5 that's 32 envs * 5 agents * 1280-dim
obs * 5 agents joined ≈ 1 GB, so 5p may need A100-80GB if A10G OOMs.
"""
import sys

PER_CELL_SETS = {
    ("mappo", "hanabi_5"): ["NUM_ENVS=32"],
}
ADAPTERS = ["hanabi_2", "hanabi_3", "hanabi_5"]
SEEDS = [30, 31, 32]                                  # match SMAX/MPE MAPPO seeds
HANABI_BUDGET_SETS = ["TOTAL_TIMESTEPS=5e7"]
CONFIGS = [
    ("hanabi_mappo", "mappo", list(HANABI_BUDGET_SETS)),
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
