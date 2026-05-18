"""Tuning sweep: find a MARC setting that beats vanilla IPPO.

Diagnosis (abl_lam0): MARC architecture alone ≈ vanilla; aux loss at
LAMBDA_AUX=0.1 over-forces role separation (helps forced_coord, hurts
asymm, collapses counter_circuit). Goal: a setting that keeps the
forced_coord-style coordination gain WITHOUT the asymm/counter harm.

Screen two levers on the 4 decisive layouts, 2 seeds (fast screen;
winner reconfirmed at 3 seeds + cramped_room after):
  tune_la003 : LAMBDA_AUX=0.03            (lower overall aux)
  tune_la01  : LAMBDA_AUX=0.01            (lower still)
  tune_b025  : LAMBDA_AUX=0.1, BETA=0.25  (keep align, soften L_div)

Vanilla refs to beat (eval): asymm 490, counter_circuit 146,
forced_coord 192, coord_ring 287.
"""
import sys

LAYOUTS = ["asymm_advantages", "counter_circuit",
           "forced_coord", "coord_ring"]
SEEDS = [30, 31]
CONFIGS = [
    ("tune_la003", "marc", ["LAMBDA_AUX=0.03"]),
    ("tune_la01",  "marc", ["LAMBDA_AUX=0.01"]),
    ("tune_b025",  "marc", ["LAMBDA_AUX=0.1", "BETA=0.25"]),
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
