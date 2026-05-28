"""Switch Riddle MAPPO baseline (does the MARC clean win survive
centralised value?). 3 sizes (N=3, 4, 5) x 3 seeds = 9 runs.

Mirrors smax_mappo_manifest / hanabi_mappo_manifest tag pattern so
rliable_report.py picks the runs up next to sr_vanilla_* / sr_marc_*.
"""
import sys

ADAPTERS = ["switch_riddle_3", "switch_riddle_4", "switch_riddle_5"]
SEEDS = [30, 31, 32]
CONFIGS = [
    ("sr_mappo", "mappo", []),
]
RUNS = [(t, k, s_cfg, ad, sd)
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
