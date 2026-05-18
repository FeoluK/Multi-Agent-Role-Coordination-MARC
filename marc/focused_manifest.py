"""Focused diagnostic manifest (separate from sweep_manifest.py so the
broad arrays' index mapping is never disturbed).

Only the informative cells: the 3 ablations x the 3 layouts where marc
diverged from vanilla in the core sweep (asymm worse, counter_circuit
collapse, forced_coord the lone win). Uninformative cramped_room/coord_ring
ablations are dropped. 3 ablations x 3 layouts x 3 seeds = 27 runs.

Diagnoses:
  abl_lam0 (LAMBDA_AUX=0): does removing aux fix counter_circuit collapse /
                           recover asymm toward vanilla? (aux destabilizes?)
  abl_beta0 (BETA=0):      does dropping diversity recover asymm 260->~490?
                           (L_div over-forces differentiation?)
  abl_zt (ZERO_TEAMMATE):  architectural contribution isolation.
"""
import sys

LAYOUTS = ["asymm_advantages", "counter_circuit", "forced_coord"]
SEEDS = [30, 31, 32]
CONFIGS = [
    ("abl_lam0",  "marc", ["LAMBDA_AUX=0"]),
    ("abl_beta0", "marc", ["BETA=0"]),
    ("abl_zt",    "marc", ["ZERO_TEAMMATE=true"]),
]
RUNS = [(t, k, s, lay, sd)
        for (t, k, s) in CONFIGS
        for lay in LAYOUTS
        for sd in SEEDS]


def spec(idx):
    tag, kind, sets, layout, seed = RUNS[idx]
    sets_str = " ".join(f"--set {s}" for s in sets)
    return (f"TAG={tag} KIND={kind} LAYOUT={layout} SEED={seed} "
            f"SETS='{sets_str}'")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--count":
        print(len(RUNS))
    else:
        print(spec(int(sys.argv[1])))
