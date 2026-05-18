"""MARC training harness — forked from JaxMARL ippo_cnn_overcooked.

Two paths, selected by NETWORK_KIND:
  * vanilla    — behaviour-preserving refactor of the validated IPPO loop
                 (re-validates cramped_room ~240). Single-frame obs.
  * marc_self  — episode-aware (obs,action) history window (H) threaded
                 through the scan; network = self-encoder + conditioned
                 policy (Task 5). z_self is produced for the Task 7 losses
                 (unused by the loss until then).

History causality: at step t the window holds observations [t-H+1..t] and
the *previous* actions [t-H..t-1] (the action at t is not yet chosen). A
per-actor episode-step counter drives a pad mask so cross-episode / pre-roll
slots are attention-masked (no buffer zeroing needed).

Differences vs the reference (none numeric on the vanilla path): wandb
removed (metrics returned from the scan); trace-time debug print removed.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from jaxmarl.wrappers.baselines import LogWrapper

from adapters import get_adapter
from nets import (build_network, needs_history, needs_teammate,
                  needs_world_state, needs_discriminator)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray          # vanilla: (A,*obs); hist: (A,H,*obs)
    info: jnp.ndarray
    act_w: jnp.ndarray        # hist: (A,H) prev actions; else dummy
    mask: jnp.ndarray         # hist: (A,H) valid mask;   else dummy
    t_obs: jnp.ndarray        # marc: (A,M,H,*obs); else dummy
    t_act: jnp.ndarray        # marc: (A,M,H);      else dummy
    t_mask: jnp.ndarray       # marc: (A,M,H);      else dummy
    world_state: jnp.ndarray  # mappo: (A, n*flat); else dummy
    agent_id: jnp.ndarray     # cds: (A,) actor's agent index; else dummy


def _team_index(n, E):
    """Static (NA, n-1) map: actor a=(g*E+e) -> its teammates' actor indices
    (other agents g'!=g, same env e). Matches batchify's
    a = agent_idx*NUM_ENVS + env_idx layout."""
    rows = []
    for g in range(n):
        for e in range(E):
            rows.append([gp * E + e for gp in range(n) if gp != g])
    return jnp.asarray(rows, dtype=jnp.int32)            # (NA, n-1)


def batchify(x, agent_list, num_actors):
    return jnp.stack([x[a] for a in agent_list]).reshape((num_actors, -1))


def unbatchify(x, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def _pad_mask(ep_len, H):
    """valid = the most-recent min(ep_len, H) positions (1), else 0. (A,H)."""
    valid = jnp.minimum(ep_len, H)                       # (A,)
    idx = jnp.arange(H)[None, :]                          # (1,H)
    return (idx >= (H - valid[:, None])).astype(jnp.float32)


def _world_state(ob, n, E):
    """MAPPO joint observation. ob is (n*E, *obs) with the canonical
    a = agent_idx*E + env_idx layout. Returns (n*E, n*flat): every actor
    in env e gets the SAME concat of all n agents' obs in that env."""
    flat = ob.reshape(n, E, -1)                          # (n, E, F)
    joint = flat.transpose(1, 0, 2).reshape(E, -1)       # (E, n*F)
    ws = jnp.broadcast_to(joint[None], (n, E, joint.shape[-1]))
    return ws.reshape(n * E, -1)                          # (n*E, n*F)


def get_rollout(params, config):
    """One sampled episode -> state_seq for GIF rendering."""
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
    tidx = _team_index(n, 1) if TEAM else None     # (n, n-1), single env

    key = jax.random.PRNGKey(0)
    key, kr = jax.random.split(key)
    obs, state = env.reset(kr)
    state_seq = [state]
    if HIST:
        obs_h = jnp.zeros((n, H) + tuple(obs_shape))
        act_h = jnp.zeros((n, H), jnp.int32)
        ep_len = jnp.zeros((n,), jnp.int32)
    done = False
    # Cap the rollout so a surviving policy can't produce a giant state_seq
    # (Craftax-Coop episodes can run to 1e5 steps). env.max_steps exists for
    # every adapter (Overcooked/MPE natively; Craftax via the adapter).
    cap = int(getattr(env, "max_steps", 1000))
    steps = 0
    while not done and steps < cap:
        steps += 1
        key, ka, ks = jax.random.split(key, 3)
        ob = jnp.stack([obs[a] for a in env.agents]).reshape(-1, *obs_shape)
        if TEAM:
            ow = jnp.concatenate([obs_h[:, 1:], ob[:, None]], axis=1)
            m = _pad_mask(ep_len + 1, H)
            pi, _, _ = network.apply(
                params, ow, act_h, m,
                ow[tidx], act_h[tidx], m[tidx], compute_aux=False)
        elif HIST:
            ow = jnp.concatenate([obs_h[:, 1:], ob[:, None]], axis=1)
            m = _pad_mask(ep_len + 1, H)
            pi, _, _ = network.apply(params, ow, act_h, m)
        elif CENTRAL:
            pi, _ = network.apply(params, ob, _world_state(ob, n, 1))
        elif CDS:
            pi, _, _ = network.apply(params, ob)
        else:
            pi, _ = network.apply(params, ob)
        action = pi.sample(seed=ka)
        env_act = {k: v.squeeze() for k, v in
                   unbatchify(action, env.agents, 1, n).items()}
        obs, state, _, d, _ = env.step(ks, state, env_act)
        done = d["__all__"]
        if HIST:
            obs_h = jnp.concatenate([obs_h[:, 1:], ob[:, None]], axis=1)
            act_h = jnp.concatenate(
                [act_h[:, 1:], action.reshape(n, 1).astype(jnp.int32)], axis=1)
            ep_len = jnp.where(d["__all__"], 0, jnp.minimum(ep_len + 1, H))
        state_seq.append(state)
    return state_seq


def make_train(config):
    adapter = get_adapter(config["ADAPTER"])
    env = adapter.make_env(**config["ENV_KWARGS"])
    HIST = needs_history(config["NETWORK_KIND"])
    TEAM = needs_teammate(config["NETWORK_KIND"])
    CENTRAL = needs_world_state(config["NETWORK_KIND"])
    CDS = needs_discriminator(config["NETWORK_KIND"])
    CDS_ALPHA = float(config.get("CDS_ALPHA", 0.05))
    H = config.get("HISTORY", 16)
    n_ag = env.num_agents
    E = config["NUM_ENVS"]
    team_idx = _team_index(n_ag, E) if TEAM else None     # (NA, M)
    M = n_ag - 1

    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"])
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"])

    env = LogWrapper(env, replace_info=False)
    obs_shape = adapter.obs_shape(env)
    NA = config["NUM_ACTORS"]

    rew_shaping_anneal = optax.linear_schedule(
        init_value=1.0, end_value=0.0,
        transition_steps=config["REW_SHAPING_HORIZON"])

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"]
                                * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    def _gather_team(ow, aw, m):
        """ow/aw/m are per-actor windows (NA,...). Gather each actor's
        teammates -> (NA, M, ...)."""
        return ow[team_idx], aw[team_idx], m[team_idx]

    def fwd(params, ob, ow, aw, m, ws=None):
        """Unified forward -> (pi, value, aux). aux is {} for vanilla,
        {'z_self'} for marc_self, full dict for marc."""
        net = build_network(config, adapter)
        if TEAM:
            tow, taw, tm = _gather_team(ow, aw, m)
            pi, v, aux = net.apply(params, ow, aw, m, tow, taw, tm,
                                   compute_aux=False)  # rollout: skip aux
            return pi, v, aux
        if HIST:
            pi, v, aux = net.apply(params, ow, aw, m)
            return pi, v, aux
        if CENTRAL:
            pi, v = net.apply(params, ob, ws)
            return pi, v, {}
        if CDS:
            pi, v, disc = net.apply(params, ob)
            return pi, v, {"disc": disc}
        pi, v = net.apply(params, ob)
        return pi, v, {}

    def train(rng):
        network = build_network(config, adapter)
        rng, _rng = jax.random.split(rng)
        if TEAM:
            ow0 = jnp.zeros((1, H) + tuple(obs_shape))
            aw0 = jnp.zeros((1, H), jnp.int32)
            m0 = jnp.ones((1, H))
            tow0 = jnp.zeros((1, M, H) + tuple(obs_shape))
            taw0 = jnp.zeros((1, M, H), jnp.int32)
            tm0 = jnp.ones((1, M, H))
            network_params = network.init(_rng, ow0, aw0, m0,
                                          tow0, taw0, tm0)
        elif HIST:
            network_params = network.init(
                _rng, jnp.zeros((1, H) + tuple(obs_shape)),
                jnp.zeros((1, H), jnp.int32), jnp.ones((1, H)))
        elif CENTRAL:
            flat = int(np.prod(obs_shape))
            network_params = network.init(
                _rng, jnp.zeros((1, *obs_shape)),
                jnp.zeros((1, n_ag * flat)))
        else:
            network_params = network.init(_rng, jnp.zeros((1, *obs_shape)))

        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5))
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5))
        train_state = TrainState.create(
            apply_fn=network.apply, params=network_params, tx=tx)

        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

        obs_h0 = jnp.zeros((NA, H) + tuple(obs_shape))
        act_h0 = jnp.zeros((NA, H), jnp.int32)
        ep0 = jnp.zeros((NA,), jnp.int32)

        def _update_step(runner_state, unused):
            def _env_step(runner_state, unused):
                (train_state, env_state, last_obs, update_step,
                 obs_h, act_h, ep_len, rng) = runner_state
                rng, _rng = jax.random.split(rng)
                ob = jnp.stack([last_obs[a] for a in env.agents]).reshape(
                    -1, *obs_shape)

                if HIST:
                    ow = jnp.concatenate([obs_h[:, 1:], ob[:, None]], axis=1)
                    aw = act_h
                    m = _pad_mask(ep_len + 1, H)
                else:
                    ow = aw = m = jnp.zeros((NA, 1))  # dummies
                ws = (_world_state(ob, n_ag, config["NUM_ENVS"])
                      if CENTRAL else jnp.zeros((NA, 1)))
                pi, value, _aux = fwd(train_state.params, ob, ow, aw, m, ws)

                if TEAM:
                    tow, taw, tm = _gather_team(ow, aw, m)
                else:
                    tow = taw = tm = jnp.zeros((NA, 1))  # dummies

                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                env_act = {k: v.flatten() for k, v in unbatchify(
                    action, env.agents, config["NUM_ENVS"],
                    env.num_agents).items()}

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0))(rng_step, env_state, env_act)

                # Overcooked exposes a shaped reward to anneal; MPE/others
                # do not. Static (trace-time) presence check -> env-agnostic.
                if "shaped_reward" in info:
                    shaped_reward = info.pop("shaped_reward")
                    current_timestep = (update_step * config["NUM_STEPS"]
                                        * config["NUM_ENVS"])
                    reward = jax.tree.map(
                        lambda x, y: x + y
                        * rew_shaping_anneal(current_timestep),
                        reward, shaped_reward)
                info = jax.tree.map(
                    lambda x: x.reshape((NA,)), info)

                done_b = batchify(done, env.agents, NA).squeeze()
                stored_obs = ow if HIST else ob
                rew_b = batchify(reward, env.agents, NA).squeeze()
                # actor index per row (canonical a = agent_idx*E + env_idx)
                aid_b = (jnp.arange(NA) // config["NUM_ENVS"]).astype(
                    jnp.int32)
                if CDS:
                    # intrinsic diversity reward = alpha * variational MI
                    # bound: log q(self | o) - log(1/n)  (>0 when the
                    # discriminator can identify the agent from its state).
                    lp = jax.nn.log_softmax(_aux["disc"], axis=-1)
                    own = lp[jnp.arange(NA), aid_b]
                    r_int = CDS_ALPHA * (own + jnp.log(float(n_ag)))
                    rew_b = rew_b + jax.lax.stop_gradient(r_int)
                transition = Transition(
                    done_b, action, value, rew_b,
                    log_prob, stored_obs, info, aw, m, tow, taw, tm,
                    ws, aid_b)

                if HIST:
                    obs_h = jnp.concatenate(
                        [obs_h[:, 1:], ob[:, None]], axis=1)
                    act_h = jnp.concatenate(
                        [act_h[:, 1:],
                         action.reshape(NA, 1).astype(jnp.int32)], axis=1)
                    ep_len = jnp.where(
                        done_b.astype(bool), 0,
                        jnp.minimum(ep_len + 1, H)).astype(jnp.int32)

                runner_state = (train_state, env_state, obsv, update_step,
                                obs_h, act_h, ep_len, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"])

            (train_state, env_state, last_obs, update_step,
             obs_h, act_h, ep_len, rng) = runner_state

            # Aux-weight schedule (closure-visible to _loss_fn below).
            #   AUX_ANNEAL: none=const | up=warmup 0->1 (let PPO find reward
            #   first, then add aux) | down=1->0. aux_scale multiplies
            #   LAMBDA_AUX. Horizon defaults to REW_SHAPING_HORIZON.
            _ct = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
            _hz = float(config.get("AUX_HORIZON",
                                   config["REW_SHAPING_HORIZON"]))
            _mode = config.get("AUX_ANNEAL", "none")
            if _mode == "up":
                aux_scale = jnp.clip(_ct / _hz, 0.0, 1.0)
            elif _mode == "down":
                aux_scale = jnp.clip(1.0 - _ct / _hz, 0.0, 1.0)
            else:
                aux_scale = 1.0

            ob = jnp.stack([last_obs[a] for a in env.agents]).reshape(
                -1, *obs_shape)
            if HIST:
                ow = jnp.concatenate([obs_h[:, 1:], ob[:, None]], axis=1)
                _, last_val, _ = fwd(train_state.params, ob, ow,
                                     act_h, _pad_mask(ep_len + 1, H))
            elif CENTRAL:
                ow = aw = m = jnp.zeros((NA, 1))
                ws = _world_state(ob, n_ag, config["NUM_ENVS"])
                _, last_val, _ = fwd(train_state.params, ob, ow, aw, m, ws)
            else:
                ow = aw = m = jnp.zeros((NA, 1))
                _, last_val, _ = fwd(train_state.params, ob, ow, aw, m)

            def _calculate_gae(traj_batch, last_val):
                def _get_adv(carry, transition):
                    gae, next_value = carry
                    done, value, reward = (transition.done, transition.value,
                                           transition.reward)
                    delta = (reward + config["GAMMA"] * next_value
                             * (1 - done) - value)
                    gae = (delta + config["GAMMA"] * config["GAE_LAMBDA"]
                           * (1 - done) * gae)
                    return (gae, value), gae
                _, adv = jax.lax.scan(
                    _get_adv, (jnp.zeros_like(last_val), last_val),
                    traj_batch, reverse=True, unroll=16)
                return adv, adv + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, bi):
                    tb, gae, tgt = bi

                    def _loss_fn(params, tb, gae, tgt):
                        if TEAM:
                            pi, value, aux = network.apply(
                                params, tb.obs, tb.act_w, tb.mask,
                                tb.t_obs, tb.t_act, tb.t_mask,
                                compute_aux=True)  # loss: need aux latents
                        elif HIST:
                            pi, value, aux = network.apply(
                                params, tb.obs, tb.act_w, tb.mask)
                        elif CENTRAL:
                            pi, value = network.apply(
                                params, tb.obs, tb.world_state)
                            aux = {}
                        elif CDS:
                            pi, value, _disc = network.apply(
                                params, tb.obs)
                            aux = {"disc": _disc}
                        else:
                            pi, value = network.apply(params, tb.obs)
                            aux = {}
                        log_prob = pi.log_prob(tb.action)
                        vpc = tb.value + (value - tb.value).clip(
                            -config["CLIP_EPS"], config["CLIP_EPS"])
                        v_l = jnp.square(value - tgt)
                        v_lc = jnp.square(vpc - tgt)
                        value_loss = 0.5 * jnp.maximum(v_l, v_lc).mean()
                        ratio = jnp.exp(log_prob - tb.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        la1 = ratio * gae
                        la2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"],
                                       1.0 + config["CLIP_EPS"]) * gae
                        loss_actor = -jnp.minimum(la1, la2).mean()
                        entropy = pi.entropy().mean()
                        total = (loss_actor + config["VF_COEF"] * value_loss
                                 - config["ENT_COEF"] * entropy)

                        # CDS: train the identity discriminator (CE on the
                        # true agent index). Encoder is stop-grad inside the
                        # module, so this only fits the diversity head; the
                        # diversity pressure on the policy is delivered via
                        # the intrinsic reward added in the rollout.
                        if CDS:
                            dlp = jax.nn.log_softmax(aux["disc"], axis=-1)
                            disc_ce = -dlp[jnp.arange(dlp.shape[0]),
                                           tb.agent_id].mean()
                            total = total + float(
                                config.get("CDS_DISC_COEF", 1.0)) * disc_ce

                        # MARC aux losses (design §4). Only kind=marc has the
                        # required latents. Ablations via config:
                        #   LAMBDA_AUX=0 -> no aux; BETA=0 -> no diversity;
                        #   ZERO_TEAMMATE -> handled in the network.
                        l_align = 0.0
                        l_div = 0.0
                        if TEAM:
                            z_self = aux["z_self"]              # (b,Z)
                            z_si = aux["z_self_infer"]          # (b,Z)
                            z_ts = aux["z_team_self"]           # (b,M,Z)
                            # L_align: MSE(self-latent, teammate's inference
                            # of self). AUX_NORM -> mean over Z (~O(1)) so
                            # LAMBDA_AUX is on the same scale as the PPO
                            # losses; default sum (~O(Z), the original).
                            sq = (z_self - z_si) ** 2
                            l_align = jnp.mean(
                                sq.mean(-1) if config.get("AUX_NORM")
                                else sq.sum(-1))
                            zsn = z_self / (jnp.linalg.norm(
                                z_self, axis=-1, keepdims=True) + 1e-8)
                            ztn = z_ts / (jnp.linalg.norm(
                                z_ts, axis=-1, keepdims=True) + 1e-8)
                            cos = jnp.einsum("bz,bmz->bm", zsn, ztn)
                            # AUX_GATE: only penalise POSITIVE similarity
                            # (redundancy); don't reward over-separation
                            # (cos<0). "encourage difference only when
                            # redundant, don't over-force."
                            l_div = jnp.mean(
                                jnp.maximum(cos, 0.0)
                                if config.get("AUX_GATE") else cos)
                            total = total + (
                                config["LAMBDA_AUX"] * aux_scale * (
                                    l_align + config["BETA"] * l_div))
                        return total, (value_loss, loss_actor, entropy,
                                       l_align, l_div)

                    gfn = jax.value_and_grad(_loss_fn, has_aux=True)
                    tl, grads = gfn(train_state.params, tb, gae, tgt)
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, tl

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = (config["MINIBATCH_SIZE"]
                              * config["NUM_MINIBATCHES"])
                assert batch_size == config["NUM_STEPS"] * NA, \
                    "batch size must equal num_steps * num_actors"
                perm = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree.map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch)
                shuf = jax.tree.map(
                    lambda x: jnp.take(x, perm, axis=0), batch)
                mbs = jax.tree.map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1]
                        + list(x.shape[1:])), shuf)
                train_state, tl = jax.lax.scan(
                    _update_minbatch, train_state, mbs)
                return (train_state, traj_batch, advantages, targets,
                        rng), tl

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"])
            train_state = update_state[0]
            rng = update_state[-1]

            metric = jax.tree.map(lambda x: x.mean(), traj_batch.info)
            # Instrumentation: raw aux magnitudes vs PPO terms, so we can SEE
            # whether the aux gradient dominates (scale-mismatch diagnosis).
            _vl, _la_act, _ent, _lal, _ldv = loss_info[1]
            metric["m_value_loss"] = _vl.mean()
            metric["m_actor_loss"] = _la_act.mean()
            metric["m_l_align"] = jnp.asarray(_lal, jnp.float32).mean()
            metric["m_l_div"] = jnp.asarray(_ldv, jnp.float32).mean()
            metric["m_aux_eff"] = (config["LAMBDA_AUX"]
                                   * (jnp.asarray(_lal, jnp.float32).mean()
                                      + config["BETA"]
                                      * jnp.asarray(_ldv, jnp.float32).mean()))
            update_step = update_step + 1
            metric["update_step"] = update_step
            metric["env_step"] = (update_step * config["NUM_STEPS"]
                                  * config["NUM_ENVS"])
            runner_state = (train_state, env_state, last_obs, update_step,
                            obs_h, act_h, ep_len, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, 0,
                        obs_h0, act_h0, ep0, _rng)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"])
        return {"runner_state": runner_state, "metrics": metric}

    return train
