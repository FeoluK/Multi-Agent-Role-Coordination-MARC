"""Switch Riddle 4-way comparison: vanilla, marc, marc+latent_gate, mappo.
N=3/4/5 agents x 3 seeds = 36 runs total.
"""
import sys

ADAPTERS = ["switch_riddle_3", "switch_riddle_4", "switch_riddle_5"]
SEEDS = [30, 31, 32]
CONFIGS = [
    ("sr_vanilla",  "vanilla", []),
    ("sr_marc",     "marc",    ["LAMBDA_AUX=0"]),
    ("sr_marc_lg",  "marc",    ["LAMBDA_AUX=0", "LATENT_GATE=True"]),
    ("sr_mappo",    "mappo",   []),
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
