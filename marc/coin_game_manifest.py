"""Coin Game sweep manifest: vanilla IPPO vs MARC architecture, 3 seeds.
2-agent cooperative grid-world social dilemma. Tests whether MARC's
role-latent differentiation improves coordination when agents must split
coin-collection duties. 2 x 3 = 6 runs.
"""
import sys

SEEDS = [30, 31, 32]
CONFIGS = [
    ("cg_vanilla", "vanilla", []),
    ("cg_marc",    "marc",    ["LAMBDA_AUX=0"]),  # architecture-only
]
RUNS = [(t, k, s, sd)
        for (t, k, s) in CONFIGS
        for sd in SEEDS]


def spec(idx):
    tag, kind, sets, seed = RUNS[idx]
    extra = " ".join(f"--set {s}" for s in sets)
    return (f"TAG={tag} KIND={kind} ADAPTER=coin_game SEED={seed} "
            f"SETS='--set ADAPTER=coin_game {extra}'")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--count":
        print(len(RUNS))
    elif len(sys.argv) == 2:
        print(spec(int(sys.argv[1])))
    else:
        for i in range(len(RUNS)):
            print(f"{i}: {spec(i)}")
