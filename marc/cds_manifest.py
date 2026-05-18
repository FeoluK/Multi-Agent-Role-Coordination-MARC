"""CDS-style diversity baseline sweep (identity-MI; Li et al. 2021).

The real partial-observability baseline: MPE simple_spread is partially
observable, so CDS-style intrinsic diversity: the apt peer-method comparison.
vanilla IPPO. Also run on Overcooked (fully observable -> MAPPO should
~= IPPO; forecloses the "you didn't even try MAPPO" objection).

  - MPE  : cds   x N=3/6/9 x 3 seeds   (tag cds_mpe_spread_N)
  - OC   : cds   x 5 layouts x 3 seeds (tag cds_oc)

Tags chosen so rliable_report.py groups them next to cf_* / sweep_*.
"""
import sys

SEEDS = [30, 31, 32]
MPE = [(f"cds_mpe_spread_{n}", "marc_mpe.yaml", f"mpe_spread_{n}", "", sd)
       for n in (3, 6, 9) for sd in SEEDS]
OC_LAYOUTS = ["cramped_room", "asymm_advantages", "coord_ring",
              "forced_coord", "counter_circuit"]
OC = [("cds_oc", "marc_overcooked.yaml", "", lay, sd)
      for lay in OC_LAYOUTS for sd in SEEDS]
RUNS = MPE + OC


def spec(idx):
    tag, cfg, adapter, layout, seed = RUNS[idx]
    sets = f"--set ADAPTER={adapter}" if adapter else ""
    return (f"TAG={tag} CONFIG={cfg} KIND=cds SEED={seed} "
            f"LAYOUT={layout} SETS='{sets}'")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--count":
        print(len(RUNS))
    else:
        print(spec(int(sys.argv[1])))
