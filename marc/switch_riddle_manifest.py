"""Switch Riddle sweep manifest: vanilla IPPO vs MARC architecture across
N=3/4/5 agents, 3 seeds. Tests MARC's role-differentiation thesis on a
minimal cooperative signalling task (agents must specialise from identical
observations). 2 x 3 x 3 = 18 runs.
"""
import sys

ADAPTERS = ["switch_riddle_3", "switch_riddle_4", "switch_riddle_5"]
SEEDS = [30, 31, 32]
CONFIGS = [
    ("sr_vanilla", "vanilla", []),
    ("sr_marc",    "marc",    ["LAMBDA_AUX=0"]),  # architecture-only (winning config from MPE)
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
    if len(sys.argv) == 2 and sys.argv[1] == "--count":
        print(len(RUNS))
    elif len(sys.argv) == 2:
        print(spec(int(sys.argv[1])))
    else:
        for i in range(len(RUNS)):
            print(f"{i}: {spec(i)}")
