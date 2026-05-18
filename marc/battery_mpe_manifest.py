"""Loss-fix battery on MPE simple_spread (fast, scalable, an anti-redundancy
coverage task — the ideal testbed). Tests whether ANY aux formulation beats
architecture-alone (LAMBDA_AUX=0) without over-forcing differentiation.

Diagnosis being tested: original L_align = sum over Z=32 of squared diffs
~O(30), so even LAMBDA_AUX=0.03 dominates the PPO gradient. Fixes:
  AUX_NORM  : L_align mean over Z (~O(1)) -> LAMBDA_AUX is meaningful
  AUX_GATE  : L_div penalises only positive cos (redundancy); never rewards
              over-separation -> "encourage difference only when redundant"
  AUX_ANNEAL=up : no aux early (let PPO find reward), ramp in -> dodge the
              exploration-trap collapse

N=3 for the full grid; N=6 on the most promising configs (scaling). 2 seeds.
"""
import sys

# (tag, kind, [set-overrides])
CONFIGS = [
    ("bat_vanilla",     "vanilla", []),
    ("bat_arch",        "marc",    ["LAMBDA_AUX=0"]),
    ("bat_sum01",       "marc",    ["LAMBDA_AUX=0.1"]),                    # old, reproduce+instrument
    ("bat_norm01",      "marc",    ["LAMBDA_AUX=0.1", "AUX_NORM=true"]),
    ("bat_norm03",      "marc",    ["LAMBDA_AUX=0.3", "AUX_NORM=true"]),
    ("bat_norm10",      "marc",    ["LAMBDA_AUX=1.0", "AUX_NORM=true"]),
    ("bat_normgate03",  "marc",    ["LAMBDA_AUX=0.3", "AUX_NORM=true", "AUX_GATE=true"]),
    ("bat_normgate10",  "marc",    ["LAMBDA_AUX=1.0", "AUX_NORM=true", "AUX_GATE=true"]),
    ("bat_normup10",    "marc",    ["LAMBDA_AUX=1.0", "AUX_NORM=true", "AUX_ANNEAL=up"]),
    ("bat_normgateup",  "marc",    ["LAMBDA_AUX=0.5", "AUX_NORM=true", "AUX_GATE=true", "AUX_ANNEAL=up"]),
]
# N=6 only on baselines + the two most promising "gated" ideas (scaling)
N6_TAGS = {"bat_vanilla", "bat_arch", "bat_normgate03", "bat_normgateup"}
SEEDS = [30, 31]

RUNS = []
for (tag, kind, sets) in CONFIGS:
    for adapter in ("mpe_spread_3", "mpe_spread_6"):
        if adapter == "mpe_spread_6" and tag not in N6_TAGS:
            continue
        for sd in SEEDS:
            RUNS.append((tag, kind, sets, adapter, sd))


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
