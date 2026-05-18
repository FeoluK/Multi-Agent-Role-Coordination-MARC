"""Power-up the confirmed positive aux finding: 5 MORE seeds (33-37) of
arch + normgateup x N=3/6/9, seed-matched for paired analysis. Pools with
the existing 3-seed mpe_confirm runs (same cf_arch/cf_normgateup tags;
filenames carry the seed so no collision) -> 8 seeds total.

Goal: per-N paired test was p(1-sided)~0.06 at n=3 (underpowered); n=8
should drive N=6/9 to p<0.01 and make N=3 assessable. Vanilla NOT rerun
(existing sd .12/.18/.41 already tight). 30 runs.

normgateup = LAMBDA_AUX=0.5 AUX_NORM AUX_GATE AUX_ANNEAL=up.
"""
import sys

ADAPTERS = ["mpe_spread_3", "mpe_spread_6", "mpe_spread_9"]
SEEDS = [33, 34, 35, 36, 37]
CONFIGS = [
    ("cf_arch",       "marc", ["LAMBDA_AUX=0"]),
    ("cf_normgateup", "marc", ["LAMBDA_AUX=0.5", "AUX_NORM=true",
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
