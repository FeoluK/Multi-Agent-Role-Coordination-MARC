"""SMAX follow-ups to the main scaling sweep (smax_scaling_manifest.py),
addressing the two open questions left by the headline run:

1. Budget ladder on smax_8m: did N=8 tie because both methods were
   undertrained at 5e6 timesteps? loo_drop_min went *negative* in the
   main sweep (knocking out a unit improved team return — signature of
   policies too weak to use the extra body), so 2x and 4x budget should
   show whether MARC's gap opens with convergence. Seeds 30..32 give a
   matched-seed extension to the existing 5e6 cell.
   -> 2 kinds (vanilla, marc) * 2 budgets (10e6, 20e6) * 3 seeds = 12 runs.

2. Compositional ablation on smax_2s3z (5 allies = 2 stalkers + 3 zealots
   vs identical enemy team): heterogeneous unit types make role-
   specialization a *structural* constraint, the regime where MARC's
   anti-redundancy mechanism should bite hardest. Homogeneous Marines
   in the main sweep gave units automatic positional differentiation
   already; 2s3z removes that confound. If MARC wins on 2s3z but tied on
   homogeneous Marines at the same N, the homogeneous result reads as
   "no redundancy to fix" rather than "method failure".
   -> 2 kinds * 10 seeds = 20 runs.

Total: 32 runs. Same modal_smax_scaling.py app drives this manifest via
--manifest smax_followup_manifest.py.
"""
import sys

RUNS = []

# (1) Budget ladder on smax_8m. Tag encodes the budget so the rliable
# cell maps don't pool different budgets together.
for budget_label, budget in [("10e6", 10_000_000), ("20e6", 20_000_000)]:
    for kind, kind_sets in [("vanilla", []),
                             ("marc",    ["LAMBDA_AUX=0"])]:
        tag = f"smax_{kind}_smax_8m_{budget_label}"
        for sd in range(30, 33):                          # 3 seeds
            sets = kind_sets + [f"TOTAL_TIMESTEPS={budget}"]
            RUNS.append((tag, kind, "smax_8m", sets, sd))

# (2) Compositional ablation on smax_2s3z. Default 5e6 budget — same as
# the main sweep so it's a fair drop-in for the 8m comparison.
for kind, kind_sets in [("vanilla", []),
                         ("marc",    ["LAMBDA_AUX=0"])]:
    tag = f"smax_{kind}_smax_2s3z"
    for sd in range(30, 40):                              # 10 seeds
        RUNS.append((tag, kind, "smax_2s3z", list(kind_sets), sd))


def spec(idx):
    tag, kind, adapter, sets, seed = RUNS[idx]
    extra = " ".join(f"--set {s}" for s in sets)
    return (f"TAG={tag} KIND={kind} ADAPTER={adapter} SEED={seed} "
            f"SETS='--set ADAPTER={adapter} {extra}'")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--count":
        print(len(RUNS))
    else:
        print(spec(int(sys.argv[1])))
