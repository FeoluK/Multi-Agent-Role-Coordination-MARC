"""rliable-style statistical report over existing MARC results.

Zero new compute: reads results/*.json and computes, for each headline
comparison, the Agarwal et al. (2021) robust statistics:
  - IQM (interquartile mean) with 95% stratified-bootstrap CI
  - mean with 95% bootstrap CI
  - probability of improvement P(MARC > vanilla) with 95% CI
    (Mann-Whitney / common-language effect size, averaged over tasks)

Pure numpy so it runs anywhere the result jsons live (FarmShare).
Uneven seed counts are handled (bootstrap resamples within each cell).

Usage: python rliable_report.py [results_dir]
"""
import glob
import json
import os
import sys

import numpy as np

RNG = np.random.default_rng(0)
N_BOOT = 5000

# ---- headline comparisons --------------------------------------------
# Overcooked: 5 layouts, vanilla vs marc (3 seeds each).
OC_LAYOUTS = ["cramped_room", "asymm_advantages", "coord_ring",
              "forced_coord", "counter_circuit"]
OC_GROUPS = {
    "vanilla": [("sweep_vanilla", l) for l in OC_LAYOUTS],
    "marc":    [("sweep_marc", l) for l in OC_LAYOUTS],
    "mappo":   [("mappo_oc", l) for l in OC_LAYOUTS],
    "cds":     [("cds_oc", l) for l in OC_LAYOUTS],
}
# marc-architecture-ONLY (LAMBDA_AUX=0 — the config that WINS on MPE).
# lam0 ablation runs are scattered across tags; pool them per layout.
OC_LAM0_TAGS = ["focus_abl_lam0", "conf_lam0", "sweep_abl_lam0"]
OC_GROUPS_ARCH = {
    "vanilla":   [("sweep_vanilla", l) for l in OC_LAYOUTS],
    "marc_full": [("sweep_marc", l) for l in OC_LAYOUTS],
    "marc_arch": [(t, l) for l in OC_LAYOUTS for t in OC_LAM0_TAGS],
}
# SMAX scaling (Marines ladder), the cross-game N-scaling test.
# (label, layout-key) pairs; layout-key matches what run.py writes.
SMAX_CELLS = [("3m",         "smax_3m"),
              ("8m",         "smax_8m"),
              ("10m_vs_11m", "smax_10m_vs_11m")]
SMAX_GROUPS = {
    "vanilla":   [(f"smax_vanilla_{lay}", lay) for (_, lay) in SMAX_CELLS],
    "marc_arch": [(f"smax_marc_{lay}",    lay) for (_, lay) in SMAX_CELLS],
}
# MPE simple_spread, scaling in team size N (the clean fixed-aux story).
MPE_NS = [3, 6, 9]
MPE_GROUPS = {
    "vanilla":    [(f"cf_vanilla_mpe_spread_{n}", f"mpe_spread_{n}")
                   for n in MPE_NS],
    "agentid":    [("aid", f"mpe_spread_{n}_id") for n in MPE_NS],
    "mappo":      [(f"mappo_mpe_spread_{n}", f"mpe_spread_{n}")
                   for n in MPE_NS],
    "cds":        [(f"cds_mpe_spread_{n}", f"mpe_spread_{n}")
                   for n in MPE_NS],
    "marc_arch":  [(f"cf_arch_mpe_spread_{n}", f"mpe_spread_{n}")
                   for n in MPE_NS],
    "marc_ngup":  [(f"cf_normgateup_mpe_spread_{n}", f"mpe_spread_{n}")
                   for n in MPE_NS],
}

METRICS = [
    ("eval_return",    lambda d: d.get("eval", {}).get("eval_return")),
    ("final_return",   lambda d: d.get("final_sparse_return")),
    ("behavioral_sep", lambda d: d.get("eval", {}).get("behavioral_sep")),
    ("role_similarity", lambda d: d.get("eval", {}).get("role_similarity")),
    ("loo_drop_min",   lambda d: d.get("eval", {}).get("loo_drop_min")),
]


def load(results_dir):
    """Load runs from a directory of *.json. Each .json is either a
    single run summary (the canonical one-file-per-run layout used on
    FarmShare) OR a list of run summaries (the consolidated format used
    in smax_pull/smax_results.json). Both paths produce the same
    cells = {(tag, layout): [run, ...]} mapping."""
    cells = {}
    for f in glob.glob(os.path.join(results_dir, "*.json")):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        runs = d if isinstance(d, list) else [d]
        for r in runs:
            k = (r.get("tag"), r.get("layout"))
            cells.setdefault(k, []).append(r)
    return cells


def iqm(x):
    x = np.sort(np.asarray(x, float))
    if x.size == 0:
        return np.nan
    lo, hi = int(np.floor(0.25 * x.size)), int(np.ceil(0.75 * x.size))
    seg = x[lo:hi] if hi > lo else x
    return float(seg.mean())


def boot_ci(per_task, stat, n=N_BOOT):
    """Stratified bootstrap: resample seeds *within* each task, pool,
    apply `stat`. per_task = list of 1-D arrays (one per task/cell)."""
    per_task = [np.asarray(t, float) for t in per_task if len(t)]
    if not per_task:
        return np.nan, np.nan, np.nan
    pooled = np.concatenate(per_task)
    point = stat(pooled)
    bs = np.empty(n)
    for i in range(n):
        res = [t[RNG.integers(0, len(t), len(t))] for t in per_task]
        bs[i] = stat(np.concatenate(res))
    return point, float(np.percentile(bs, 2.5)), float(
        np.percentile(bs, 97.5))


def prob_improvement(treat_tasks, base_tasks, n=N_BOOT):
    """P(X>Y) averaged over tasks (common-language effect size),
    with stratified-bootstrap 95% CI. Lists aligned by task."""
    def cles(tt, bb):
        vals = []
        for x, y in zip(tt, bb):
            x, y = np.asarray(x, float), np.asarray(y, float)
            if len(x) == 0 or len(y) == 0:
                continue
            gt = (x[:, None] > y[None, :]).mean()
            eq = (x[:, None] == y[None, :]).mean()
            vals.append(gt + 0.5 * eq)
        return float(np.mean(vals)) if vals else np.nan
    point = cles(treat_tasks, base_tasks)
    bs = np.empty(n)
    for i in range(n):
        tt = [np.asarray(t, float)[RNG.integers(0, len(t), len(t))]
              if len(t) else t for t in treat_tasks]
        bb = [np.asarray(b, float)[RNG.integers(0, len(b), len(b))]
              if len(b) else b for b in base_tasks]
        bs[i] = cles(tt, bb)
    return point, float(np.percentile(bs, 2.5)), float(
        np.percentile(bs, 97.5))


def collect(cells, group, getter):
    """group = list[(tag,layout)] -> list of per-task value arrays."""
    out = []
    for k in group:
        ds = cells.get(k, [])
        vals = [getter(d) for d in ds]
        vals = [v for v in vals if v is not None]
        out.append(np.array(vals, float))
    return out


def report_block(title, cells, groups, base_key):
    print(f"\n{'='*70}\n{title}\n{'='*70}")
    keys = list(groups)
    for mname, getter in METRICS:
        per = {k: collect(cells, groups[k], getter) for k in keys}
        ns = {k: [len(t) for t in per[k]] for k in keys}
        if all(sum(ns[k]) == 0 for k in keys):
            continue
        print(f"\n--- {mname}  (seeds/task: "
              + "; ".join(f"{k}={ns[k]}" for k in keys) + ")")
        for k in keys:
            pt, lo, hi = boot_ci(per[k], iqm)
            mpt, mlo, mhi = boot_ci(per[k], lambda z: float(z.mean()))
            print(f"  {k:11s}  IQM {pt:8.3f} [{lo:8.3f},{hi:8.3f}]"
                  f"   mean {mpt:8.3f} [{mlo:8.3f},{mhi:8.3f}]")
        for k in keys:
            if k == base_key:
                continue
            p, plo, phi = prob_improvement(per[k], per[base_key])
            flag = ""
            if not np.isnan(plo):
                flag = "  *" if (plo > 0.5 or phi < 0.5) else ""
            print(f"  P({k} > {base_key}) = {p:.3f} "
                  f"[{plo:.3f},{phi:.3f}]{flag}")


def main(results_dir):
    cells = load(results_dir)
    print(f"loaded {sum(len(v) for v in cells.values())} runs "
          f"across {len(cells)} cells from {results_dir}")
    report_block("OVERCOOKED  (5 layouts pooled): vanilla vs marc",
                 cells, OC_GROUPS, "vanilla")
    report_block("OVERCOOKED pooled: vanilla vs marc-full vs "
                 "marc-ARCH-ONLY (lam0, the MPE-winning config)",
                 cells, OC_GROUPS_ARCH, "vanilla")
    # per-layout Overcooked: the actual thesis claim is per-layout
    # (wins on coordination-critical layouts), not pooled.
    for lay in OC_LAYOUTS:
        g = {"vanilla": [("sweep_vanilla", lay)],
             "marc": [("sweep_marc", lay)],
             "marc_arch": [(t, lay) for t in OC_LAM0_TAGS],
             "mappo": [("mappo_oc", lay)],
             "cds": [("cds_oc", lay)]}
        report_block(f"OVERCOOKED  layout={lay}", cells, g, "vanilla")
    report_block("SMAX scaling (3m/8m/10m_vs_11m pooled): "
                 "vanilla vs marc-arch (the cross-game N-scaling test)",
                 cells, SMAX_GROUPS, "vanilla")
    for label, lay in SMAX_CELLS:
        g = {"vanilla":   [(f"smax_vanilla_{lay}", lay)],
             "marc_arch": [(f"smax_marc_{lay}",    lay)]}
        report_block(f"SMAX  scenario={label}", cells, g, "vanilla")
    report_block("MPE simple_spread (N=3,6,9 pooled): "
                 "vanilla vs marc-arch vs marc-normgateup",
                 cells, MPE_GROUPS, "vanilla")
    # per-N MPE breakdown (the monotone-in-N headline). Derive each
    # per-N group by keeping the entry whose layout matches this N
    # (mpe_spread_{n} or the agent-ID variant mpe_spread_{n}_id).
    for n in MPE_NS:
        want = {f"mpe_spread_{n}", f"mpe_spread_{n}_id"}
        g = {kk: [v for v in vv if v[1] in want]
             for kk, vv in MPE_GROUPS.items()}
        report_block(f"MPE simple_spread N={n}", cells, g, "vanilla")
    print("\n( * = 95% CI excludes 0.5 -> statistically distinguishable )")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else "/scratch/users/flukol/marc/results")
