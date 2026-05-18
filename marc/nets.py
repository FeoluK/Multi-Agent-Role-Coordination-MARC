"""Pluggable networks for the MARC harness.

`VanillaActorCritic` is a faithful re-implementation of JaxMARL's validated
ippo_cnn_overcooked ActorCritic, with the obs encoder swapped for the
adapter-provided module. It MUST stay numerically equivalent so the refactor
re-validates at cramped_room ~240. MARC networks are added in later tasks.

Note the faithfully-preserved quirk from the reference: the actor computes a
hidden Dense(64) layer then *ignores it*, taking logits directly from the
encoder embedding. Replicated exactly (it consumes init RNG and so shifts
downstream param init; changing it would change the numbers).
"""
from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal


class VanillaActorCritic(nn.Module):
    obs_encoder: nn.Module
    action_dim: int
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        act = nn.relu if self.activation == "relu" else nn.tanh
        embedding = self.obs_encoder(x)

        # actor (quirk preserved: final Dense reads `embedding`, not hidden)
        actor_h = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)),
                           bias_init=constant(0.0))(embedding)
        actor_h = act(actor_h)
        logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01),
                          bias_init=constant(0.0))(embedding)
        import distrax
        pi = distrax.Categorical(logits=logits)

        critic = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)),
                          bias_init=constant(0.0))(embedding)
        critic = act(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0),
                          bias_init=constant(0.0))(critic)
        return pi, jnp.squeeze(critic, axis=-1)


class CdsActorCritic(nn.Module):
    """CDS-style diversity baseline (the identifiability / mutual-info
    core of Li et al. 2021, "Celebrating Diversity in Shared MARL").

    Shared actor-critic (== vanilla) PLUS an agent-identity discriminator
    q(agent | obs). The discriminator is trained to predict which agent
    produced a state; each agent gets an intrinsic reward proportional to
    log q(self | o) (a variational lower bound on I(agent ; state) ), so
    the shared policy is pushed toward behaviourally DIVERSE, mutually
    distinguishable per-agent trajectories — without any teammate
    inference / role-latent machinery. The apt peer-method comparison for
    MARC (both target redundancy via diversity, different mechanisms).
    apply(params, x) -> (pi, value, disc_logits[n_agents]).
    """

    obs_encoder: nn.Module
    action_dim: int
    n_agents: int
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        act = nn.relu if self.activation == "relu" else nn.tanh
        embedding = self.obs_encoder(x)
        actor_h = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)),
                           bias_init=constant(0.0))(embedding)
        actor_h = act(actor_h)
        logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01),
                          bias_init=constant(0.0))(embedding)
        import distrax
        pi = distrax.Categorical(logits=logits)

        critic = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)),
                          bias_init=constant(0.0))(embedding)
        critic = act(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0),
                          bias_init=constant(0.0))(critic)

        # identity discriminator q(agent | obs) — stop-grad the encoder so
        # it does not distort the shared representation (CDS keeps the
        # diversity head separate from the policy trunk).
        d = jax.lax.stop_gradient(embedding)
        d = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)),
                     bias_init=constant(0.0))(d)
        d = act(d)
        disc_logits = nn.Dense(self.n_agents, kernel_init=orthogonal(0.01),
                               bias_init=constant(0.0))(d)
        return pi, jnp.squeeze(critic, axis=-1), disc_logits


class MappoActorCritic(nn.Module):
    """MAPPO baseline: decentralised shared actor (local obs) + a
    CENTRALISED critic over the joint observation (concat of all agents'
    obs). The standard CTDE baseline (Yu et al. 2022). Distinct from
    vanilla IPPO only in the critic's input -> isolates "is MARC's gain
    just centralised value under partial observability?" (MPE is
    partially observable, so this is a real, non-trivial baseline).
    apply(params, local_obs, world_state) -> (pi, value).
    """

    obs_encoder: nn.Module        # actor encoder (local obs)
    action_dim: int
    activation: str = "relu"

    @nn.compact
    def __call__(self, x, world_state):
        act = nn.relu if self.activation == "relu" else nn.tanh
        embedding = self.obs_encoder(x)             # actor: local obs only
        actor_h = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)),
                           bias_init=constant(0.0))(embedding)
        actor_h = act(actor_h)
        logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01),
                          bias_init=constant(0.0))(embedding)
        import distrax
        pi = distrax.Categorical(logits=logits)

        # centralised critic: own MLP over the joint observation
        c = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)),
                     bias_init=constant(0.0))(world_state)
        c = act(c)
        c = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)),
                     bias_init=constant(0.0))(c)
        c = act(c)
        c = nn.Dense(1, kernel_init=orthogonal(1.0),
                     bias_init=constant(0.0))(c)
        return pi, jnp.squeeze(c, axis=-1)


# --------------------------------------------------------------------------- #
# MARC modules (design §3). Task 5: SelfEncoder + ConditionedPolicy (no        #
# teammate inferencer yet — added Task 6).                                     #
# --------------------------------------------------------------------------- #
class TinyTransformer(nn.Module):
    """2-layer / 2-head pre-norm Transformer encoder over a length-H token
    sequence; returns the last-position embedding. `pad_mask` (..., H) marks
    valid (1) vs padding/pre-episode (0) positions."""

    depth: int = 2
    heads: int = 2
    dim: int = 32
    ff: int = 64

    @nn.compact
    def __call__(self, tokens, pad_mask):
        x = nn.Dense(self.dim, kernel_init=orthogonal(np.sqrt(2)),
                     bias_init=constant(0.0))(tokens)
        H = x.shape[-2]
        x = x + self.param("pos", orthogonal(0.02), (H, self.dim))
        # (..., 1, H, H): keys masked on invalid positions
        attn_mask = pad_mask[..., None, None, :]
        for _ in range(self.depth):
            y = nn.LayerNorm()(x)
            y = nn.MultiHeadDotProductAttention(
                num_heads=self.heads, qkv_features=self.dim)(
                y, y, mask=attn_mask)
            x = x + y
            y = nn.LayerNorm()(x)
            y = nn.Dense(self.ff, kernel_init=orthogonal(np.sqrt(2)))(y)
            y = nn.relu(y)
            y = nn.Dense(self.dim, kernel_init=orthogonal(np.sqrt(2)))(y)
            x = x + y
        return nn.LayerNorm()(x)[..., -1, :]  # last-position embedding


class HistoryEncoder(nn.Module):
    """(obs_window, act_window) -> role latent z (Z,). obs_window:
    (..., H, *obs_shape); act_window: (..., H) int. Encodes each frame with
    the shared obs encoder, embeds actions, concatenates, runs TinyTransformer.
    Used for BOTH the self-encoder and (Task 6) the per-teammate inferencer
    (separate instances => separate weights)."""

    obs_encoder: nn.Module
    action_dim: int
    latent_dim: int = 32

    @nn.compact
    def __call__(self, obs_window, act_window, pad_mask):
        # obs_window: (B, H, *obs_shape); act/mask: (B, H).
        B, Hs = obs_window.shape[0], obs_window.shape[1]
        flat = obs_window.reshape((B * Hs,) + obs_window.shape[2:])
        feats = self.obs_encoder(flat).reshape(B, Hs, -1)
        a_emb = nn.Embed(self.action_dim + 1, 16)(act_window + 1)  # 0 = pad
        tokens = jnp.concatenate([feats, a_emb], axis=-1)
        z = TinyTransformer(dim=self.latent_dim)(tokens, pad_mask)
        cur_feat = feats[:, -1, :]
        return z, cur_feat


class MarcSelfActorCritic(nn.Module):
    """Task 5 network: self-encoder + policy/critic conditioned on
    [current_feat ; z_self]. No teammate inferencer yet."""

    obs_encoder: nn.Module
    action_dim: int
    latent_dim: int = 32
    activation: str = "relu"

    @nn.compact
    def __call__(self, obs_window, act_window, pad_mask):
        act = nn.relu if self.activation == "relu" else nn.tanh
        z_self, cur_feat = HistoryEncoder(
            obs_encoder=self.obs_encoder, action_dim=self.action_dim,
            latent_dim=self.latent_dim, name="self_enc")(
            obs_window, act_window, pad_mask)
        h = jnp.concatenate([cur_feat, z_self], axis=-1)

        a = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)),
                     bias_init=constant(0.0))(h)
        a = act(a)
        logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01),
                          bias_init=constant(0.0))(a)
        import distrax
        pi = distrax.Categorical(logits=logits)

        c = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)),
                     bias_init=constant(0.0))(h)
        c = act(c)
        v = nn.Dense(1, kernel_init=orthogonal(1.0),
                     bias_init=constant(0.0))(c)
        return pi, jnp.squeeze(v, axis=-1), {"z_self": z_self}


class AttentionPool(nn.Module):
    """N-agnostic pool over per-teammate latents. query = [cur_feat ; z_self],
    keys/values = z_team (..., M, Z). Single-head scaled dot-product over the
    M teammate slots -> context (..., Z). M=1 (Overcooked) -> softmax of one
    element = identity, so it degrades gracefully; M>1 (scaling claim) works
    unchanged. Preserves per-teammate inference (only the policy's
    *consumption* is pooled)."""

    dim: int = 32

    @nn.compact
    def __call__(self, query, z_team):
        q = nn.Dense(self.dim, kernel_init=orthogonal(np.sqrt(2)),
                     bias_init=constant(0.0))(query)            # (..., Z)
        k = nn.Dense(self.dim, kernel_init=orthogonal(np.sqrt(2)))(z_team)
        v = nn.Dense(self.dim, kernel_init=orthogonal(np.sqrt(2)))(z_team)
        scores = jnp.einsum("...z,...mz->...m", q, k) / np.sqrt(self.dim)
        w = nn.softmax(scores, axis=-1)                          # (..., M)
        return jnp.einsum("...m,...mz->...z", w, v)              # (..., Z)


class MarcActorCritic(nn.Module):
    """Full MARC network (Task 6): self-encoder + per-teammate inferencer
    (separate weights) + attention-pooled conditioned policy/critic.
    Returns (pi, value, z_self, z_team) — latents feed the Task 7 aux losses.

    v1 choice: self-encoder and teammate-inferencer each own their obs
    encoder (separate weights), per design §3.2/3.3 ("separate weights"); the
    obs encoder is not parameter-shared between them.
    """

    obs_encoder_self: nn.Module
    obs_encoder_team: nn.Module
    action_dim: int
    latent_dim: int = 32
    activation: str = "relu"
    zero_teammate: bool = False   # ablation: architectural contribution test

    @nn.compact
    def __call__(self, ow, aw, m, tow, taw, tm, compute_aux: bool = True):
        # compute_aux=False (rollout): skip the two aux-only encoder passes
        # (z_self_infer, z_team_self). They reuse self_enc/team_enc params so
        # no init/param mismatch; ~halves encoder cost during rollout.
        act = nn.relu if self.activation == "relu" else nn.tanh
        B, M = tow.shape[0], tow.shape[1]

        self_enc = HistoryEncoder(
            obs_encoder=self.obs_encoder_self, action_dim=self.action_dim,
            latent_dim=self.latent_dim, name="self_enc")
        infer = HistoryEncoder(
            obs_encoder=self.obs_encoder_team, action_dim=self.action_dim,
            latent_dim=self.latent_dim, name="team_enc")

        z_self, cur_feat = self_enc(ow, aw, m)

        def _flat(x):
            return x.reshape((B * M,) + x.shape[2:])

        # i's per-teammate inference -> z_team (B,M,Z)
        z_team, _ = infer(_flat(tow), _flat(taw), _flat(tm))
        z_team = z_team.reshape(B, M, self.latent_dim)

        # Aux-loss latents (weight-shared encoders => computable per sample):
        #  z_self_infer = inferencer on OWN window == what any teammate would
        #                 infer about me (shared inferencer weights) -> L_align
        #  z_team_self  = self-encoder on teammate windows == teammates'
        #                 self-latents -> L_div
        # Only needed by the loss; skipped during rollout (compute_aux=False).
        if compute_aux:
            z_self_infer, _ = infer(ow, aw, m)                      # (B,Z)
            z_team_self, _ = self_enc(_flat(tow), _flat(taw), _flat(tm))
            z_team_self = z_team_self.reshape(B, M, self.latent_dim)
        else:
            z_self_infer = jnp.zeros_like(z_self)
            z_team_self = jnp.zeros_like(z_team)

        if self.zero_teammate:
            z_team = jnp.zeros_like(z_team)

        query = jnp.concatenate([cur_feat, z_self], axis=-1)
        ctx = AttentionPool(dim=self.latent_dim, name="pool")(query, z_team)
        h = jnp.concatenate([cur_feat, z_self, ctx], axis=-1)

        a = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)),
                     bias_init=constant(0.0))(h)
        a = act(a)
        logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01),
                          bias_init=constant(0.0))(a)
        import distrax
        pi = distrax.Categorical(logits=logits)

        c = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)),
                     bias_init=constant(0.0))(h)
        c = act(c)
        val = nn.Dense(1, kernel_init=orthogonal(1.0),
                       bias_init=constant(0.0))(c)
        aux = {"z_self": z_self, "z_team": z_team,
               "z_self_infer": z_self_infer, "z_team_self": z_team_self}
        return pi, jnp.squeeze(val, axis=-1), aux


def needs_history(kind: str) -> bool:
    return kind in ("marc_self", "marc")


def needs_teammate(kind: str) -> bool:
    return kind == "marc"


def needs_world_state(kind: str) -> bool:
    """MAPPO: centralised critic consumes the joint observation."""
    return kind == "mappo"


def needs_discriminator(kind: str) -> bool:
    """CDS: identity discriminator + intrinsic diversity reward."""
    return kind == "cds"


def build_network(config, adapter):
    """Return a flax module.
      vanilla   : apply(params, obs)                 -> (pi, value)
      marc_self : apply(params, obs_w, act_w, mask)  -> (pi, value, z_self)
    """
    kind = config.get("NETWORK_KIND", "vanilla")
    if kind == "vanilla":
        return VanillaActorCritic(
            obs_encoder=adapter.obs_encoder(),
            action_dim=adapter.action_dim,
            activation=config["ACTIVATION"])
    if kind == "mappo":
        return MappoActorCritic(
            obs_encoder=adapter.obs_encoder(),
            action_dim=adapter.action_dim,
            activation=config["ACTIVATION"])
    if kind == "cds":
        return CdsActorCritic(
            obs_encoder=adapter.obs_encoder(),
            action_dim=adapter.action_dim,
            n_agents=adapter.num_agents,
            activation=config["ACTIVATION"])
    if kind == "marc_self":
        return MarcSelfActorCritic(
            obs_encoder=adapter.obs_encoder(),
            action_dim=adapter.action_dim,
            latent_dim=config.get("LATENT_DIM", 32),
            activation=config["ACTIVATION"])
    if kind == "marc":
        return MarcActorCritic(
            obs_encoder_self=adapter.obs_encoder(),
            obs_encoder_team=adapter.obs_encoder(),
            action_dim=adapter.action_dim,
            latent_dim=config.get("LATENT_DIM", 32),
            activation=config["ACTIVATION"],
            zero_teammate=bool(config.get("ZERO_TEAMMATE", False)))
    raise ValueError(f"unknown NETWORK_KIND {kind!r}")
