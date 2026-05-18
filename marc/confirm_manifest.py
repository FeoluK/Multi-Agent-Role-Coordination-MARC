"""No-regression confirmation: MARC architecture at LAMBDA_AUX=0 on the two
layouts not yet tested at aux=0 (cramped_room, coord_ring), 3 seeds.

Needed to claim "MARC architecture ≥ vanilla on ALL 5 layouts": we already
have abl_lam0 on asymm(474)/counter(159)/forced(199) vs vanilla
(490/146/192). This fills cramped_room + coord_ring (vanilla 240 / 287).
6 runs.
"""
import sys

LAYOUTS = ["cramped_room", "coord_ring"]
SEEDS = [30, 31, 32]
RUNS = [("conf_lam0", "marc", ["LAMBDA_AUX=0"], lay, sd)
        for lay in LAYOUTS for sd in SEEDS]


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
