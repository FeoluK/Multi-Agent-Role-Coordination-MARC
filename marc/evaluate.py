"""Evaluation metrics (design §5 dependent measures):

  * eval_return        — greedy sparse team return (no shaped reward)
  * loo_drop[k]        — leave-one-out: return lost when agent k is forced to
                         the no-op action (small drop => lazy / non-contributing
                         agent; the lazy-agent failure MARC targets)
  * role_similarity    — mean pairwise cosine similarity of agents' self-
                         latents over eval states (marc/marc_self only; low =>
                         differentiated roles => anti-redundancy working)

Greedy (argmax) actions, raw env (sparse +20/soup only), vmapped over
EVAL_EPISODES parallel envs, fixed horizon = env.max_steps.
"""
from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp

from adapters import get_adapter
from nets import (build_network, needs_history, needs_teammate,
                  needs_world_state, needs_discriminator)
from train_marc import (_pad_mask, _team_index, _world_state,
                        batchify, unbatchify)


def eval_metrics(params, config, episodes=64, seed=12345):
    adapter = get_adapter(config["ADAPTER"])
    env = adapter.make_env(**config["ENV_KWARGS"])
    network = build_network(config, adapter)
    HIST = needs_history(config["NETWORK_KIND"])
    TEAM = needs_teammate(config["NETWORK_KIND"])
    CENTRAL = needs_world_state(config["NETWORK_KIND"])
    CDS = needs_discriminator(config["NETWORK_KIND"])
    H = config.get("HISTORY", 16)
    obs_shape = adapter.obs_shape(env)
    n = env.num_agents
    E = episodes
    NA = n * E
    T = env.max_steps
    tidx = _team_index(n, E) if TEAM else None
    noop = adapter.noop_action
    A = adapter.action_dim
    MASK = bool(getattr(adapter, "has_action_mask", False))

    def _maybe_mask(pi, state):
        """If the env has action masking (Hanabi), zero-out illegal-action
        logits before sampling so eval rollouts don't waste on no-ops."""
        if not MASK:
            return pi
        import distrax
        avail = adapter.get_avail_actions(env, state)            # dict
        avail_b = batchify(avail, env.agents, NA).astype(jnp.float32)
        logits = pi.logits if hasattr(pi, "logits") else pi.distribution.logits
        return distrax.Categorical(logits=logits - (1.0 - avail_b) * 1e10)

    def policy(params, ob, obs_h, act_h, ep_len, akey, state):
        """-> (sampled_actions (NA,), z_self or None). Sampled, not greedy:
        deterministic Overcooked policies deadlock (0 return); the training /
        published return metric uses sampled actions."""
        if TEAM:
            ow = jnp.concatenate([obs_h[:, 1:], ob[:, None]], axis=1)
            m = _pad_mask(ep_len + 1, H)
            pi, _, aux = network.apply(
                params, ow, act_h, m,
                ow[tidx], act_h[tidx], m[tidx], compute_aux=False)
            return _maybe_mask(pi, state).sample(seed=akey), aux["z_self"]
        if HIST:
            ow = jnp.concatenate([obs_h[:, 1:], ob[:, None]], axis=1)
            m = _pad_mask(ep_len + 1, H)
            pi, _, aux = network.apply(params, ow, act_h, m)
            return _maybe_mask(pi, state).sample(seed=akey), aux["z_self"]
        if CENTRAL:
            pi, _ = network.apply(params, ob, _world_state(ob, n, E))
            return _maybe_mask(pi, state).sample(seed=akey), None
        if CDS:
            pi, _, _ = network.apply(params, ob)
            return _maybe_mask(pi, state).sample(seed=akey), None
        pi, _ = network.apply(params, ob)
        return _maybe_mask(pi, state).sample(seed=akey), None

    def run(params, drop_k):
        """One vmapped batch of E episodes. drop_k in [-1, n): force agent
        drop_k to no-op. Returns (mean_return, mean_role_sim_or_nan)."""
        rng = jax.random.PRNGKey(seed)
        rng, rr = jax.random.split(rng)
        obs, state = jax.vmap(env.reset)(jax.random.split(rr, E))
        obs_h = jnp.zeros((NA, H) + tuple(obs_shape))
        act_h = jnp.zeros((NA, H), jnp.int32)
        ep_len = jnp.zeros((NA,), jnp.int32)

        def step(carry, _):
            obs, state, obs_h, act_h, ep_len, rng, ret, sim_acc, ac = carry
            ob = jnp.stack([obs[a] for a in env.agents]).reshape(
                -1, *obs_shape)
            rng, ak, rs = jax.random.split(rng, 3)
            act, z_self = policy(params, ob, obs_h, act_h, ep_len, ak, state)
            # behavioral role separation: per-agent action histogram (normal
            # mode only) -> JS divergence between agents. Works for vanilla
            # AND marc (no latent needed).
            ar = act.reshape(n, E)
            ac = ac + jax.nn.one_hot(ar, A).sum(axis=1)   # (n, A)
            if drop_k >= 0:                       # leave-one-out
                act = act.reshape(n, E)
                act = act.at[drop_k].set(noop).reshape(NA)
            env_act = {k: v.flatten() for k, v in unbatchify(
                act, env.agents, E, n).items()}
            obs2, state2, reward, done, _ = jax.vmap(env.step)(
                jax.random.split(rs, E), state, env_act)
            r0 = reward[env.agents[0]]            # shared reward
            ret = ret + r0
            if z_self is not None and n >= 2:
                zs = z_self.reshape(n, E, -1)
                zs = zs / (jnp.linalg.norm(zs, axis=-1, keepdims=True) + 1e-8)
                pairs = jnp.stack([
                    jnp.sum(zs[i] * zs[j], -1)
                    for i, j in itertools.combinations(range(n), 2)])
                sim_acc = sim_acc + pairs.mean()
            obs_h = jnp.concatenate([obs_h[:, 1:], ob[:, None]], axis=1)
            act_h = jnp.concatenate(
                [act_h[:, 1:], act.reshape(NA, 1).astype(jnp.int32)], axis=1)
            db = batchify(done, env.agents, NA).squeeze().astype(bool)
            ep_len = jnp.where(db, 0, jnp.minimum(ep_len + 1, H)).astype(
                jnp.int32)
            return (obs2, state2, obs_h, act_h, ep_len, rng, ret,
                    sim_acc, ac), None

        init = (obs, state, obs_h, act_h, ep_len, rng,
                jnp.zeros((E,)), jnp.float32(0.0),
                jnp.zeros((n, A)))
        (_, _, _, _, _, _, ret, sim_acc, ac), _ = jax.lax.scan(
            step, init, None, T)
        # mean pairwise Jensen-Shannon divergence between agents' action
        # distributions (0 = identical behavior, ~0.69 = disjoint)
        p = ac / (ac.sum(axis=-1, keepdims=True) + 1e-8)

        def _kl(a, b):
            return jnp.sum(a * (jnp.log(a + 1e-8) - jnp.log(b + 1e-8)), -1)

        js = []
        for i, j in itertools.combinations(range(n), 2):
            m = 0.5 * (p[i] + p[j])
            js.append(0.5 * _kl(p[i], m) + 0.5 * _kl(p[j], m))
        jsd = jnp.stack(js).mean()
        return ret.mean(), sim_acc / T, jsd     # device arrays

    jrun = jax.jit(run, static_argnums=(1,))
    er, rs, jsd = jrun(params, -1)
    eval_return, role_sim = float(er), float(rs)
    behavioral_sep = float(jsd)
    loo = {}
    for k in range(n):
        rk, _, _ = jrun(params, k)
        loo[env.agents[k]] = round(eval_return - float(rk), 3)
    drops = list(loo.values())
    return {
        "eval_return": round(eval_return, 3),
        "loo_drop": loo,
        "loo_drop_min": round(min(drops), 3),
        "loo_drop_mean": round(sum(drops) / len(drops), 3),
        "role_similarity": (round(role_sim, 4)
                            if (HIST or TEAM) else None),
        "behavioral_sep": round(behavioral_sep, 4),  # JSD, both kinds
        "eval_episodes": E,
    }


def collect_role_latents(params, config, episodes=16, seed=777):
    """Roll out one batch and record every agent's self-latent z_self at
    every step -> array (n_agents, n_points, latent_dim). The qualitative
    companion to role_similarity: project to 2D (PCA/t-SNE) and colour by
    agent -> separable clusters == real role differentiation (no need for
    roles to be discrete). marc / marc_self only (vanilla has no z_self).
    """
    import numpy as np
    adapter = get_adapter(config["ADAPTER"])
    env = adapter.make_env(**config["ENV_KWARGS"])
    network = build_network(config, adapter)
    HIST = needs_history(config["NETWORK_KIND"])
    TEAM = needs_teammate(config["NETWORK_KIND"])
    if not (HIST or TEAM):
        return None                       # vanilla: no role latent
    H = config.get("HISTORY", 16)
    obs_shape = adapter.obs_shape(env)
    n = env.num_agents
    E = episodes
    NA = n * E
    T = env.max_steps
    tidx = _team_index(n, E) if TEAM else None

    MASK = bool(getattr(adapter, "has_action_mask", False))

    def _maybe_mask(pi, state):
        if not MASK:
            return pi
        import distrax
        avail = adapter.get_avail_actions(env, state)
        avail_b = batchify(avail, env.agents, NA).astype(jnp.float32)
        logits = pi.logits if hasattr(pi, "logits") else pi.distribution.logits
        return distrax.Categorical(logits=logits - (1.0 - avail_b) * 1e10)

    def policy(params, ob, obs_h, act_h, ep_len, akey, state):
        if TEAM:
            ow = jnp.concatenate([obs_h[:, 1:], ob[:, None]], axis=1)
            m = _pad_mask(ep_len + 1, H)
            pi, _, aux = network.apply(
                params, ow, act_h, m,
                ow[tidx], act_h[tidx], m[tidx], compute_aux=False)
            return _maybe_mask(pi, state).sample(seed=akey), aux["z_self"]
        ow = jnp.concatenate([obs_h[:, 1:], ob[:, None]], axis=1)
        m = _pad_mask(ep_len + 1, H)
        pi, _, aux = network.apply(params, ow, act_h, m)
        return _maybe_mask(pi, state).sample(seed=akey), aux["z_self"]

    def run(params):
        rng = jax.random.PRNGKey(seed)
        rng, rr = jax.random.split(rng)
        obs, state = jax.vmap(env.reset)(jax.random.split(rr, E))
        obs_h = jnp.zeros((NA, H) + tuple(obs_shape))
        act_h = jnp.zeros((NA, H), jnp.int32)
        ep_len = jnp.zeros((NA,), jnp.int32)

        def step(carry, _):
            obs, state, obs_h, act_h, ep_len, rng = carry
            ob = jnp.stack([obs[a] for a in env.agents]).reshape(
                -1, *obs_shape)
            rng, ak, rs = jax.random.split(rng, 3)
            act, z_self = policy(params, ob, obs_h, act_h, ep_len, ak, state)
            zs = z_self.reshape(n, E, -1)            # (n, E, Z)
            env_act = {k: v.flatten() for k, v in unbatchify(
                act, env.agents, E, n).items()}
            obs2, state2, _, done, _ = jax.vmap(env.step)(
                jax.random.split(rs, E), state, env_act)
            obs_h = jnp.concatenate([obs_h[:, 1:], ob[:, None]], axis=1)
            act_h = jnp.concatenate(
                [act_h[:, 1:], act.reshape(NA, 1).astype(jnp.int32)],
                axis=1)
            db = batchify(done, env.agents, NA).squeeze().astype(bool)
            ep_len = jnp.where(db, 0, jnp.minimum(ep_len + 1, H)).astype(
                jnp.int32)
            return (obs2, state2, obs_h, act_h, ep_len, rng), zs

        init = (obs, state, obs_h, act_h, ep_len, rng)
        _, zs_seq = jax.lax.scan(step, init, None, T)   # (T, n, E, Z)
        return zs_seq

    zs = np.asarray(jax.jit(run)(params))               # (T, n, E, Z)
    T_, n_, E_, Z = zs.shape
    # -> (n, T*E, Z): all of an agent's role-latent samples
    return np.transpose(zs, (1, 0, 2, 3)).reshape(n_, T_ * E_, Z)
