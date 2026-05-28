"""Switch Riddle CDS-style baseline (identity-MI intrinsic reward;
Li et al. 2021). 3 N values x 3 seeds = 9 runs.

Closes the 'is MARC just a generic diversity bonus?' question on
Switch Riddle the same way smax_followup did for SMAX 2s3z.
"""
import sys

ADAPTERS = ["switch_riddle_3", "switch_riddle_4", "switch_riddle_5"]
SEEDS = [30, 31, 32]
CONFIGS = [
    ("sr_cds", "cds", []),
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
