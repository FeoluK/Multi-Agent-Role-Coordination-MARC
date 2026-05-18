"""Overcooked confirmation battery — the decisive aux questions:
 (1) does correct gradient scaling (AUX_NORM) prevent the counter_circuit
     COLLAPSE and recover asymm (turn aux harmful -> salvageable)?
 (2) does AUX_ANNEAL=up (no aux during early sparse-reward discovery) dodge
     the exploration-trap collapse?
 (3) reconfirm MARC-arch >= vanilla on the decisive layouts.

Configs x 3 informative layouts x 2 seeds = 30 runs.
Vanilla refs (core sweep): counter_circuit 146, asymm 490, forced_coord 192.
"""
import sys

CONFIGS = [
    ("ocb_arch",      "marc", ["LAMBDA_AUX=0"]),
    ("ocb_sum01",     "marc", ["LAMBDA_AUX=0.1"]),                                   # old: reproduce collapse + instrument L_align~70
    ("ocb_norm01",    "marc", ["LAMBDA_AUX=0.1", "AUX_NORM=true"]),
    ("ocb_normgate03","marc", ["LAMBDA_AUX=0.3", "AUX_NORM=true", "AUX_GATE=true"]),
    ("ocb_normup10",  "marc", ["LAMBDA_AUX=1.0", "AUX_NORM=true", "AUX_ANNEAL=up"]),
]
LAYOUTS = ["counter_circuit", "asymm_advantages", "forced_coord"]
SEEDS = [30, 31]
RUNS = [(t, k, s, lay, sd)
        for (t, k, s) in CONFIGS
        for lay in LAYOUTS
        for sd in SEEDS]


def spec(idx):
    tag, kind, sets, layout, seed = RUNS[idx]
    extra = " ".join(f"--set {s}" for s in sets)
    return (f"TAG={tag} KIND={kind} LAYOUT={layout} SEED={seed} "
            f"SETS='{extra}'")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--count":
        print(len(RUNS))
    else:
        print(spec(int(sys.argv[1])))
