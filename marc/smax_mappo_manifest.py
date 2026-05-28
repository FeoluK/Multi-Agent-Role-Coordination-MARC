"""SMAX MAPPO baseline sweep: centralised-critic CTDE (Yu et al. 2022)
on the same Marines ladder we used for vanilla IPPO and MARC. Adds the
canonical 'is MARC's gain just a centralised critic under partial obs?'
control on SMAX, mirroring the MPE MAPPO sweep in mappo_manifest.py.

3 adapters * 3 seeds = 9 runs. Tag pattern smax_mappo_{adapter} so
rliable_report.py picks the runs up alongside smax_vanilla_* /
smax_marc_*. Uses configs/marc_smax.yaml -> identical PPO hyperparams
to vanilla/marc, so the only systematic difference is MAPPO's
centralised critic input (joint observation).

Per-cell NUM_ENVS overrides match the vanilla/marc sweep so the
within-cell comparison is fair (same PPO batch size for all three
methods on each scenario).
"""
import sys

PER_CELL_SETS = {
    ("mappo", "smax_10m_vs_11m"): ["NUM_ENVS=32"],
}
ADAPTERS = ["smax_3m", "smax_8m", "smax_10m_vs_11m"]
SEEDS = [30, 31, 32]                                  # match MPE MAPPO seeds
CONFIGS = [
    ("smax_mappo", "mappo", []),
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
