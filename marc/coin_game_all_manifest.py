"""Coin Game 4-way comparison: vanilla, marc, marc+latent_gate, mappo.
N=2 (fixed by CoinGameAdapter), 3 seeds = 12 runs total.
"""
import sys

SEEDS = [30, 31, 32]
CONFIGS = [
    ("cg_vanilla",  "vanilla", []),
    ("cg_marc",     "marc",    ["LAMBDA_AUX=0"]),
    ("cg_marc_lg",  "marc",    ["LAMBDA_AUX=0", "LATENT_GATE=True"]),
    ("cg_mappo",    "mappo",   []),
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
