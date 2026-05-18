"""CLI runner for the MARC harness.

Runs training (network_kind from config), persists the sparse-return curve +
summary JSON + a trained-agent GIF. Mirrors the validated baseline runner so
results are directly comparable. Sparse return = mean of last 10 updates
(past the 2.5e6 shaped-reward anneal -> pure +20/soup game score).

Usage:
  python run.py --config configs/marc_overcooked.yaml \
      --layout cramped_room --network-kind vanilla \
      --results-dir /scratch/users/flukol/marc/results \
      --gifs-dir /scratch/users/flukol/marc/gifs --tag refactor
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--layout", default=None)
    p.add_argument("--network-kind", default=None)
    p.add_argument("--total-timesteps", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--results-dir", required=True)
    p.add_argument("--gifs-dir", required=True)
    p.add_argument("--tag", default="marc",
                   help="experiment tag -> filename prefix / subdir")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VAL",
                   help="override any config key (e.g. --set LAMBDA_AUX=0 "
                        "--set ZERO_TEAMMATE=true)")
    args = p.parse_args()

    def _coerce(s):
        low = s.lower()
        if low in ("true", "false"):
            return low == "true"
        try:
            return int(s)
        except ValueError:
            pass
        try:
            return float(s)
        except ValueError:
            return s

    import jax
    from omegaconf import OmegaConf
    from train_marc import make_train, get_rollout

    config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    config.setdefault("ENV_KWARGS", {})
    if args.layout is not None:
        config["ENV_KWARGS"]["layout"] = args.layout
    if args.network_kind is not None:
        config["NETWORK_KIND"] = args.network_kind
    if args.total_timesteps is not None:
        config["TOTAL_TIMESTEPS"] = args.total_timesteps
    if args.seed is not None:
        config["SEED"] = args.seed
    for kv in args.set:
        k, v = kv.split("=", 1)
        config[k] = _coerce(v)
    config["NUM_SEEDS"] = 1
    # scenario name: Overcooked has a layout; MPE/others use the adapter name
    scenario = config["ENV_KWARGS"].get("layout", config["ADAPTER"])
    layout_name = scenario

    os.makedirs(args.results_dir, exist_ok=True)
    gifs_dir = os.path.join(args.gifs_dir, args.tag)
    os.makedirs(gifs_dir, exist_ok=True)

    print(f"[run] tag={args.tag} kind={config['NETWORK_KIND']} "
          f"layout={layout_name} jax={jax.devices()} "
          f"steps={config['TOTAL_TIMESTEPS']:.0e}", flush=True)

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    t0 = time.time()
    out = jax.vmap(jax.jit(make_train(config)))(rngs)
    jax.block_until_ready(out)
    train_min = (time.time() - t0) / 60.0

    m = out["metrics"]
    returns = np.asarray(m["returned_episode_returns"]).mean(axis=0)
    env_step = np.asarray(m["env_step"]).mean(axis=0)
    if returns.size == 0:
        raise SystemExit("No updates recorded: TOTAL_TIMESTEPS too small.")
    n_tail = min(10, returns.size)
    final_sparse = float(returns[-n_tail:].mean())

    base = f"{args.tag}_{config['NETWORK_KIND']}_{layout_name}_seed{config['SEED']}"

    # Aux instrumentation curves (present only for marc kind).
    def _curve(k):
        return (np.asarray(m[k]).mean(axis=0) if k in m else None)

    aux_curves = {k: _curve(k) for k in
                  ("m_l_align", "m_l_div", "m_actor_loss",
                   "m_value_loss", "m_aux_eff")}
    np.savez(os.path.join(args.results_dir, base + ".npz"),
             returns=returns, env_step=env_step,
             **{k: v for k, v in aux_curves.items() if v is not None})

    def _fin(v):
        return (round(float(np.asarray(v)[-n_tail:].mean()), 4)
                if v is not None else None)

    summary = {
        "tag": args.tag,
        "network_kind": config["NETWORK_KIND"],
        "layout": layout_name,
        "seed": config["SEED"],
        "total_timesteps": config["TOTAL_TIMESTEPS"],
        "final_sparse_return": final_sparse,
        "max_return": float(returns.max()),
        "train_minutes": round(train_min, 2),
        "device": str(jax.devices()[0]),
        "lambda_aux": config.get("LAMBDA_AUX"),
        "beta": config.get("BETA"),
        "aux_norm": bool(config.get("AUX_NORM", False)),
        "aux_anneal": config.get("AUX_ANNEAL", "none"),
        "aux_gate": bool(config.get("AUX_GATE", False)),
        # final-window aux magnitudes: lets us SEE scale mismatch
        "m_l_align": _fin(aux_curves["m_l_align"]),
        "m_l_div": _fin(aux_curves["m_l_div"]),
        "m_actor_loss": _fin(aux_curves["m_actor_loss"]),
        "m_value_loss": _fin(aux_curves["m_value_loss"]),
        "m_aux_eff": _fin(aux_curves["m_aux_eff"]),
    }
    with open(os.path.join(args.results_dir, base + ".json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("[summary]", json.dumps(summary), flush=True)

    train_state = jax.tree.map(lambda x: x[0], out["runner_state"][0])

    # Task 8 dependent measures (LOO degradation + role similarity)
    from evaluate import eval_metrics
    em = eval_metrics(train_state.params, config,
                      episodes=config.get("EVAL_EPISODES", 64))
    summary["eval"] = em
    with open(os.path.join(args.results_dir, base + ".json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("[eval]", json.dumps(em), flush=True)

    # Optional: dump per-agent role latents for the t-SNE/PCA proof.
    # Non-fatal (never blocks results/DONE); only marc/marc_self have z_self.
    if config.get("DUMP_LATENTS"):
        try:
            from evaluate import collect_role_latents
            zl = collect_role_latents(
                train_state.params, config,
                episodes=config.get("LATENT_EPISODES", 16))
            if zl is not None:
                zpath = os.path.join(args.results_dir, base + "_zself.npz")
                np.savez(zpath, z_self=zl)   # (n_agents, n_points, Z)
                print(f"[latents] saved {zl.shape} -> {zpath}", flush=True)
        except Exception as e:
            print(f"[latents] skipped: {e}", flush=True)

    # Episode GIF. ALWAYS non-fatal: results/eval are already saved above;
    # a render failure must never fail the run or block "DONE".
    adapter_name = config["ADAPTER"]
    # include network_kind so vanilla/marc rollout videos don't overwrite
    # each other (same tag/layout otherwise collides).
    gif_path = os.path.join(
        gifs_dir, f"{layout_name}_{config['NETWORK_KIND']}.gif")
    try:
        state_seq = get_rollout(train_state.params, config)
        if adapter_name == "overcooked":
            from jaxmarl.viz.overcooked_visualizer import OvercookedVisualizer
            OvercookedVisualizer().animate(state_seq, agent_view_size=5,
                                           filename=gif_path)
        elif adapter_name.startswith("mpe"):
            # Custom MPE renderer: state.p_pos = (n_agents+n_landmarks, 2),
            # agents first. JaxMARL's generic Visualizer lacks MPE support
            # (SimpleSpreadMPE has no init_render).
            import numpy as _np
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as _plt
            import matplotlib.animation as _anim
            from adapters import get_adapter
            na = get_adapter(adapter_name).num_agents
            pos = _np.stack([_np.asarray(s.p_pos) for s in state_seq])  # (T,E,2)
            fig, ax = _plt.subplots(figsize=(4, 4))

            def _draw(t):
                ax.clear()
                ax.set_xlim(-2, 2)
                ax.set_ylim(-2, 2)
                ax.set_aspect("equal")
                ax.set_title(f"{adapter_name}  t={t}")
                p = pos[t]
                ax.scatter(p[na:, 0], p[na:, 1], c="lightgray", s=420,
                           marker="o", label="landmarks")
                ax.scatter(p[:na, 0], p[:na, 1],
                           c=range(na), cmap="tab10", s=120,
                           marker="*", label="agents")

            a = _anim.FuncAnimation(fig, _draw, frames=len(state_seq),
                                    interval=120)
            a.save(gif_path, writer="pillow")
            _plt.close(fig)
        elif adapter_name == "craftax_coop":
            # Craftax-Coop: render each rollout state with the game's own
            # pixel renderer (whole shared world, all 3 agents visible) ->
            # mp4 + gif. Mirrors CraftaxCoopPixelsEnv.get_obs's render call.
            import numpy as _np
            import imageio
            from craftax_coop.constants import (
                BLOCK_PIXEL_SIZE_IMG, TEXTURES,
                load_player_specific_textures)
            from craftax_coop.renderer.renderer_pixels import (
                render_craftax_pixels)
            from adapters import get_adapter
            sp = get_adapter("craftax_coop").make_env().static_env_params
            ps = load_player_specific_textures(
                TEXTURES[BLOCK_PIXEL_SIZE_IMG], sp.player_count)
            # Craftax-Coop is partially observable: the renderer returns
            # (num_players, H, W, 3) — each agent's own POV. Stitch the
            # agent views side-by-side (white separators) -> one frame,
            # same-world per-agent POV proof (cf. the Mineflayer/MineLand
            # stitched videos, here for the trained MARC policy).
            def _stitch(f):
                if f.ndim == 4:                       # (P,H,W,3)
                    P, H, W, _ = f.shape
                    sep = _np.full((H, 3, 3), 255, _np.uint8)
                    cols = []
                    for p in range(P):
                        cols.append(f[p])
                        if p < P - 1:
                            cols.append(sep)
                    return _np.concatenate(cols, axis=1)
                return f                              # (H,W,3) fallback
            frames = [
                _stitch(_np.asarray(render_craftax_pixels(
                    s, BLOCK_PIXEL_SIZE_IMG, sp, ps)).astype("uint8"))
                for s in state_seq]
            mp4_path = gif_path[:-4] + ".mp4"
            imageio.mimsave(mp4_path, frames, fps=8,
                            codec="libx264", macro_block_size=None)
            imageio.mimsave(gif_path, frames[::2], fps=4)
            print(f"[vid] {mp4_path}  frame={frames[0].shape}", flush=True)
        print(f"[gif] {gif_path}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[gif-skip] render failed (non-fatal): {type(e).__name__}: "
              f"{e}", flush=True)


if __name__ == "__main__":
    main()
