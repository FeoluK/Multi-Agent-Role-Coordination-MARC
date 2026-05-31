"""Hanabi LATENT_GATE sweep: matched-seed comparison for the third arm
of vanilla / marc / marc+LATENT_GATE on Hanabi N=2/3/5.

LATENT_GATE (nets.py f6f5a9f, ReZero-style) wraps z_self and ctx in
learnable scalars init=0; combined with LAMBDA_AUX=0 the only
difference vs hanabi_marc_* is the two scalar gates. Same 5e7 step
budget as the existing Hanabi sweep so within-cell comparisons are
matched-compute. NUM_ENVS=32 override on hanabi_5 mirrors
hanabi_manifest.py.

3 cells * 3 seeds = 9 runs at seeds 30/31/32 (paired with existing
hanabi_vanilla / hanabi_marc cells at the same seeds).
"""
import sys

PER_CELL_SETS = {
    ("marc", "hanabi_5"): ["NUM_ENVS=32"],
}
ADAPTERS = ["hanabi_2", "hanabi_3", "hanabi_5"]
SEEDS = [30, 31, 32]
HANABI_BUDGET_SETS = ["TOTAL_TIMESTEPS=5e7"]
CONFIGS = [
    ("hanabi_latentgate_marc", "marc",
     ["LAMBDA_AUX=0", "LATENT_GATE=1", *HANABI_BUDGET_SETS]),
]
RUNS = [(t, k, s_cfg + PER_CELL_SETS.get((k, ad), []), ad, sd)
        for ad in ADAPTERS
        for (t, k, s_cfg) in CONFIGS
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
