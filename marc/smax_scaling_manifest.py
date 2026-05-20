"""SMAX scaling sweep: does MARC's MPE 'gap grows with N' result transfer
to a richer (StarCraft-micro) game? vanilla IPPO vs MARC architecture
(LAMBDA_AUX=0 — the winning Overcooked+MPE config) across a homogeneous
Marines ladder. 2 kinds * 3 scenarios * 10 seeds = 60 runs.

Default ladder is 3m / 8m / 10m_vs_11m (N = 3 / 8 / 10) — pure-Marine,
single knob = team size, mirroring the MPE simple_spread N=3/6/9 design.

Per-cell NUM_ENVS overrides keep the rollout buffer fitting in GPU
memory. MARC's loss-update scan retains all NUM_MINIBATCHES (=16)
minibatches' worth of transformer activations simultaneously, so MARC
at N=8/10 needs an A100 (40 GB) — it will OOM on A10G (24 GB) regardless
of NUM_ENVS. With A100, vanilla and marc can use the SAME NUM_ENVS per
cell, so the comparison is apples-to-apples within a cell.
    smax_10m_vs_11m: NUM_ENVS=32 (both kinds — A10G/A100 both OK)
    smax_8m:         NUM_ENVS=64 (default — fits A10G for vanilla, A100 for marc)
    smax_3m:         NUM_ENVS=64 (default)
25m is the SMAC canon 'large' cell but needs NUM_ENVS<=8 — stretch
follow-up, not in v1.
"""
import sys

# (kind, adapter) -> per-cell --set overrides. Lookup miss -> [].
PER_CELL_SETS = {
    # vanilla and marc use the SAME NUM_ENVS per cell so the within-cell
    # comparison is fair (no smaller PPO batch handicapping either side).
    ("vanilla", "smax_10m_vs_11m"): ["NUM_ENVS=32"],
    ("marc",    "smax_10m_vs_11m"): ["NUM_ENVS=32"],
}
ADAPTERS = ["smax_3m", "smax_8m", "smax_10m_vs_11m"]
SEEDS = list(range(30, 40))                            # 10 seeds per cell
CONFIGS = [
    ("smax_vanilla", "vanilla", []),
    ("smax_marc",    "marc",    ["LAMBDA_AUX=0"]),     # winning MPE config
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
