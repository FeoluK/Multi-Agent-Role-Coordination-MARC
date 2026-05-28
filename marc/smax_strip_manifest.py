"""SMAX scope-condition flip test (Aux. Loss paper, §scope condition).

Tests whether the `2s3z` MARC loss is *caused* by SMAX's per-unit
unit_type_bits one-hot (a free role-disambiguating feature) by zeroing
those bits out and re-comparing vanilla vs MARC.

Pairs cleanly with the existing un-stripped 2s3z runs already in
smax_followup_results.json (10 seeds vanilla/MARC). If the scope
condition holds, MARC's gap should reappear once the type bits go away.

Sweep: vanilla vs marc x smax_2s3z_stripped x 3 seeds = 6 runs.
Tag: smax_strip_<kind>. Same NUM_ENVS as the un-stripped 2s3z runs
so the comparison is apples-to-apples.
"""
import sys

PER_CELL_SETS = {}                                  # full default
ADAPTERS = ["smax_2s3z_stripped"]
SEEDS = [30, 31, 32]
CONFIGS = [
    ("smax_strip_vanilla", "vanilla", []),
    ("smax_strip_marc",    "marc",    ["LAMBDA_AUX=0"]),  # arch-only
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
