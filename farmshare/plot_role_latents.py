"""Role-latent differentiation proof: 2D projection of per-agent z_self.

Input: one or more *_zself.npz (key z_self, shape (n_agents, n_pts, Z))
pulled from FarmShare results/. For each: PCA(2) and t-SNE(2) scatter,
points coloured by agent identity. Separable per-agent clusters == real
role differentiation (works for CONTINUOUS roles — no discrete labels).
Pairs naturally with the numeric role_similarity (cosine of z_self):
collapsed blob <-> role_sim~1 ; separated <-> role_sim<0.

Usage: python scripts/plot_role_latents.py a_zself.npz [b_zself.npz ...]
Writes <stem>_roles.png next to each input.
"""
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from sklearn.manifold import TSNE
    HAVE_TSNE = True
except Exception:
    HAVE_TSNE = False


def pca2(x):
    xc = x - x.mean(0, keepdims=True)
    u, s, vt = np.linalg.svd(xc, full_matrices=False)
    ev = (s ** 2) / max(len(x) - 1, 1)
    return xc @ vt[:2].T, ev[:2] / ev.sum()


def mean_pairwise_cos(z):  # z: (n_agents, n_pts, Z) -> scalar like role_sim
    c = z.mean(1)
    c = c / (np.linalg.norm(c, axis=-1, keepdims=True) + 1e-8)
    n = len(c)
    vals = [float(c[i] @ c[j])
            for i in range(n) for j in range(i + 1, n)]
    return float(np.mean(vals)) if vals else float("nan")


def plot_one(npz_path, max_per_agent=400):
    d = np.load(npz_path)
    z = d["z_self"]                              # (n, P, Z)
    n, P, Z = z.shape
    rng = np.random.default_rng(0)
    idx = rng.choice(P, size=min(P, max_per_agent), replace=False)
    zs = z[:, idx, :]                            # (n, p, Z)
    flat = zs.reshape(n * len(idx), Z)
    lab = np.repeat(np.arange(n), len(idx))
    rsim = mean_pairwise_cos(z)

    panels = [("PCA", *pca2(flat))]
    if HAVE_TSNE and flat.shape[0] >= 10:
        ts = TSNE(n_components=2, perplexity=min(30, flat.shape[0] // 4),
                  init="pca", random_state=0).fit_transform(flat)
        panels.append(("t-SNE", ts, None))

    fig, axes = plt.subplots(1, len(panels),
                             figsize=(6 * len(panels), 5.2))
    if len(panels) == 1:
        axes = [axes]
    cmap = plt.get_cmap("tab10")
    stem = os.path.basename(npz_path).replace("_zself.npz", "")
    for ax, (name, xy, extra) in zip(axes, panels):
        for a in range(n):
            m = lab == a
            ax.scatter(xy[m, 0], xy[m, 1], s=8, alpha=0.45,
                       color=cmap(a % 10), label=f"agent {a}")
        sub = name
        if name == "PCA" and extra is not None:
            sub += f"  (var {extra[0]:.0%}/{extra[1]:.0%})"
        ax.set_title(sub)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{stem}   role_similarity(cos z_self) = {rsim:+.3f}"
                 f"   [{n} agents]", fontsize=12)
    axes[0].legend(loc="best", fontsize=8, framealpha=0.6)
    fig.tight_layout()
    out = npz_path.replace("_zself.npz", "_roles.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"{stem}: role_sim={rsim:+.3f}  n={n}  pts/agent={P}  -> {out}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: plot_role_latents.py <*_zself.npz> ...")
    for p in sys.argv[1:]:
        plot_one(p)
