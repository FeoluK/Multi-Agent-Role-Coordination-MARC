"""MPE simple_spread scaling sweep: does MARC's advantage grow with team
size N? (The proposal's central hypothesis — untestable in 2-agent
Overcooked.) vanilla vs MARC-architecture (LAMBDA_AUX=0, the winning config)
across N = 3 / 6 / 9 agents, 3 seeds. 2 x 3 x 3 = 18 runs.

simple_spread is itself a coverage / anti-redundancy task (agents cover
landmarks, penalized for overlap+collision) → directly MARC's target failure
mode, and N is the knob. Reward is shared & negative (−distance−collisions);
higher (less negative) = better.
"""
import sys

ADAPTERS = ["mpe_spread_3", "mpe_spread_6", "mpe_spread_9"]
SEEDS = [30, 31, 32]
CONFIGS = [
    ("mpe_vanilla", "vanilla", []),
    ("mpe_marc",    "marc",    ["LAMBDA_AUX=0"]),   # winning config
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
