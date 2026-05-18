"""Paper-gap battery (one SLURM array):

A) Agent-ID parameter-sharing BASELINE (the canonical "is MARC just doing
   param-sharing?" control) on MPE simple_spread N=3/6/9, 3 seeds. Same
   shared net as cf_vanilla but obs += one-hot(agent_id) -> can break
   symmetry WITHOUT any role/diversity machinery. Compares directly
   against existing cf_vanilla / cf_arch / cf_normgateup cells.

B) Role-latent DUMP runs for the t-SNE/PCA differentiation proof
   (DUMP_LATENTS): the differentiated vs collapsed contrast that pairs
   with the numeric role_similarity:
     - MPE spread_6 marc lam0  (arch-only; role_sim < 0  -> separated)
     - MPE spread_6 marc full  (aux on;   role_sim ~ .95 -> collapsed)
     - OC forced_coord marc lam0 (role_sim ~ -0.99 -> strongly separated)
     - OC cramped_room marc lam0 (symmetric; role_sim > 0 -> collapsed)

Emits a uniform line; the sbatch passes --layout only when LAYOUT set.
"""
import sys

# ---- A) agent-ID param-share baseline ----
AID = [("aid", "marc_mpe.yaml", "vanilla", f"mpe_spread_{n}_id", "", sd, [])
       for n in (3, 6, 9) for sd in (30, 31, 32)]

# ---- B) latent-dump proof runs (1 seed; viz needs a trained net) ----
ZL_SETS = ["DUMP_LATENTS=true", "LATENT_EPISODES=16"]
ZL = [
    ("zlsep", "marc_mpe.yaml", "marc", "mpe_spread_6", "", 30,
     ["LAMBDA_AUX=0"] + ZL_SETS),                       # separated
    ("zlcol", "marc_mpe.yaml", "marc", "mpe_spread_6", "", 30,
     ZL_SETS),                                          # collapsed (aux on)
    ("zlfc", "marc_overcooked.yaml", "marc", "", "forced_coord", 30,
     ["LAMBDA_AUX=0"] + ZL_SETS),                       # strongly separated
    ("zlcr", "marc_overcooked.yaml", "marc", "", "cramped_room", 30,
     ["LAMBDA_AUX=0"] + ZL_SETS),                       # collapsed (symm)
]

RUNS = AID + ZL


def spec(idx):
    tag, cfg, kind, adapter, layout, seed, sets = RUNS[idx]
    extra = list(sets)
    if adapter:
        extra = [f"ADAPTER={adapter}"] + extra
    sets_str = " ".join(f"--set {s}" for s in extra)
    return (f"TAG={tag} CONFIG={cfg} KIND={kind} SEED={seed} "
            f"LAYOUT={layout} SETS='{sets_str}'")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--count":
        print(len(RUNS))
    else:
        print(spec(int(sys.argv[1])))
