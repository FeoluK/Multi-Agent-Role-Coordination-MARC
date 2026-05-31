"""SMAX LATENT_GATE sweep: matched-seed comparison for the third arm
of vanilla / marc / marc+LATENT_GATE.

LATENT_GATE (nets.py f6f5a9f, ReZero-style) scales z_self and ctx by
learnable scalars init=0, so MARC starts equivalent to vanilla and
opens the gates only if the role latents reduce loss. Combined with
LAMBDA_AUX=0 this matches our existing marc_arch protocol -- the
only systematic difference vs smax_marc_* is the two scalar gates.

3 SMAX cells * 3 seeds = 9 runs at seeds 30/31/32 (matching the
existing vanilla / marc cells we already have at the same seeds, so
the comparison is paired). Tag pattern smax_latentgate_marc_{adapter}.
NUM_ENVS=32 override on 10m_vs_11m mirrors the smax_scaling_manifest.
"""
import sys

PER_CELL_SETS = {
    ("marc", "smax_10m_vs_11m"): ["NUM_ENVS=32"],
}
ADAPTERS = ["smax_3m", "smax_8m", "smax_10m_vs_11m"]
SEEDS = [30, 31, 32]
CONFIGS = [
    ("smax_latentgate_marc", "marc",
     ["LAMBDA_AUX=0", "LATENT_GATE=1"]),
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
