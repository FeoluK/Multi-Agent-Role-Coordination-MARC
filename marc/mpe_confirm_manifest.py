"""Confirm the one positive aux finding with statistics: does the
norm+gate+anneal aux (normgateup) beat architecture-alone, and does that
gain grow with team size N? vanilla/arch/normgateup x N=3/6/9 x 3 seeds.
27 runs (MPE simple_spread, fast).

normgateup = LAMBDA_AUX=0.5 AUX_NORM AUX_GATE AUX_ANNEAL=up (best in battery).
"""
import sys

ADAPTERS = ["mpe_spread_3", "mpe_spread_6", "mpe_spread_9"]
SEEDS = [30, 31, 32]
CONFIGS = [
    ("cf_vanilla",    "vanilla", []),
    ("cf_arch",       "marc",    ["LAMBDA_AUX=0"]),
    ("cf_normgateup", "marc",    ["LAMBDA_AUX=0.5", "AUX_NORM=true",
                                  "AUX_GATE=true", "AUX_ANNEAL=up"]),
]
RUNS = [(t, k, s, ad, sd)
        for (t, k, s) in CONFIGS
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
