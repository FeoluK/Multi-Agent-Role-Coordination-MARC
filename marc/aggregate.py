"""Aggregate sweep results -> MARC vs vanilla comparison table.

Reads results/sweep_*.json, groups by (config tag, layout), reports
mean +/- std over seeds for: final sparse return, eval return, min
leave-one-out drop (lazy-agent measure), role similarity.

Usage: python aggregate.py /scratch/users/flukol/marc/results
"""
import glob
import json
import os
import statistics as st
import sys

LAYOUTS = ["cramped_room", "asymm_advantages", "coord_ring",
           "forced_coord", "counter_circuit"]
ORDER = ["sweep_vanilla", "sweep_marc", "sweep_abl_lam0",
         "sweep_abl_beta0", "sweep_abl_zt"]


def ms(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "  -  "
    if len(xs) == 1:
        return f"{xs[0]:6.1f}"
    return f"{st.mean(xs):6.1f}±{st.pstdev(xs):4.1f}"


def main(results_dir):
    rows = {}
    for f in glob.glob(os.path.join(results_dir, "sweep_*.json")):
        d = json.load(open(f))
        tag, lay = d["tag"], d["layout"]
        rows.setdefault((tag, lay), []).append(d)

    for metric, getter in [
        ("final sparse return (train)",
         lambda d: d["final_sparse_return"]),
        ("eval return (sampled)",
         lambda d: d.get("eval", {}).get("eval_return")),
        ("min LOO drop (lazy-agent)",
         lambda d: d.get("eval", {}).get("loo_drop_min")),
        ("role similarity (lower=better)",
         lambda d: d.get("eval", {}).get("role_similarity")),
    ]:
        print(f"\n=== {metric} ===")
        print(f"{'config':16} " + " ".join(f"{l[:11]:>11}"
                                            for l in LAYOUTS))
        for tag in ORDER:
            cells = []
            for lay in LAYOUTS:
                ds = rows.get((tag, lay), [])
                cells.append(ms([getter(d) for d in ds]) if ds
                             else "  -  ")
            n = sum(len(rows.get((tag, l), [])) for l in LAYOUTS)
            if n:
                print(f"{tag:16} " + " ".join(f"{c:>11}"
                                              for c in cells))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else "/scratch/users/flukol/marc/results")
