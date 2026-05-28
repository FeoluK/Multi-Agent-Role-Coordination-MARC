"""Env-adapter contract + Overcooked adapter (+ typed Minecraft stub).

The MARC core (nets/train) is env-agnostic: it only sees abstract per-agent
obs/action tensors and an ObsEncoder module. Each environment supplies a thin
adapter implementing `EnvAdapter`. See notes/marc_design.md §6.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal

import jaxmarl
from jaxmarl.environments.overcooked import overcooked_layouts


class _Box:
    """Minimal observation_space stand-in (adapter.obs_shape reads .shape)."""

    def __init__(self, shape):
        self.shape = tuple(shape)


class AgentIDWrapper:
    """Canonical shared-parameter + one-hot agent-ID baseline.

    Appends a fixed one-hot(agent_index, num_agents) to each agent's
    (vector) observation. This is the standard MARL parameter-sharing
    baseline (Gupta et al. 2017; Terry et al. 2020): identical to the
    vanilla shared net but agents can condition behaviour on their index,
    so it can break symmetry / specialise WITHOUT any role/diversity
    machinery — the right "is MARC just doing param-sharing?" control.
    Pure jnp + delegation -> safe under jax.vmap(reset/step).
    """

    def __init__(self, env):
        self._env = env
        self.agents = env.agents
        self.num_agents = env.num_agents
        self.max_steps = env.max_steps
        self._eye = jnp.eye(self.num_agents)

    def _aug(self, obs):
        return {a: jnp.concatenate([obs[a], self._eye[i]], axis=-1)
                for i, a in enumerate(self.agents)}

    def reset(self, key):
        obs, state = self._env.reset(key)
        return self._aug(obs), state

    def step(self, key, state, actions):
        obs, state, rew, done, info = self._env.step(key, state, actions)
        return self._aug(obs), state, rew, done, info

    def observation_space(self, agent=None):
        base = self._env.observation_space(
            agent if agent is not None else self.agents[0]).shape
        return _Box((base[0] + self.num_agents,))

    def __getattr__(self, name):           # delegate anything else
        return getattr(self._env, name)


class StripUnitTypeBitsWrapper:
    """SMAX scope-condition flip test. Zeros out the per-unit
    `unit_type_bits` slots in every agent's obs (the last
    `unit_type_bits` entries of each unit-feature block + of the own
    block), erasing the natural role-disambiguating feature SMAX hands
    vanilla. If the scope condition holds, MARC's gap over vanilla
    should reappear on `2s3z_stripped` because vanilla can no longer
    learn role-specialised behaviour directly from the obs.
    Pure jnp + delegation -> safe under jax.vmap(reset/step).
    """

    def __init__(self, env):
        self._env = env
        self.agents = env.agents
        self.num_agents = env.num_agents
        self.max_steps = env.max_steps
        # Compute the obs indices to zero out, statically.
        # Per smax_env.py: obs = [ally_blocks; enemy_blocks; own_block].
        # Each unit (ally/enemy) block has len(unit_features) entries
        # whose LAST `unit_type_bits` are the type one-hot. Own block
        # has len(own_features) entries with the LAST `unit_type_bits`
        # being the own-type one-hot.
        n_allies = int(env.num_allies)
        n_enemies = int(env.num_enemies)
        unit_dim = len(env.unit_features)
        own_dim = len(env.own_features)
        ut = int(env.unit_type_bits)
        idx = []
        # visible-ally blocks (num_allies - 1 of them per agent)
        for k in range(n_allies - 1):
            base = k * unit_dim
            for b in range(ut):
                idx.append(base + unit_dim - ut + b)
        # enemy blocks
        for k in range(n_enemies):
            base = (n_allies - 1) * unit_dim + k * unit_dim
            for b in range(ut):
                idx.append(base + unit_dim - ut + b)
        # own block
        own_base = (n_allies - 1) * unit_dim + n_enemies * unit_dim
        for b in range(ut):
            idx.append(own_base + own_dim - ut + b)
        self._strip_idx = jnp.asarray(idx, dtype=jnp.int32)

    def _strip(self, obs_vec):
        return obs_vec.at[..., self._strip_idx].set(0.0)

    def _aug(self, obs):
        return {a: self._strip(obs[a]) for a in self.agents}

    def reset(self, key):
        obs, state = self._env.reset(key)
        return self._aug(obs), state

    def step(self, key, state, actions):
        obs, state, rew, done, info = self._env.step(key, state, actions)
        return self._aug(obs), state, rew, done, info

    def observation_space(self, agent=None):
        return self._env.observation_space(
            agent if agent is not None else self.agents[0])

    def __getattr__(self, name):
        return getattr(self._env, name)


# --------------------------------------------------------------------------- #
# ObsEncoder modules (the ONLY env-coupled network piece)                      #
# --------------------------------------------------------------------------- #
class MLPEncoder(nn.Module):
    """Vector-obs encoder (MPE / non-image envs). raw obs (obs_dim,) ->
    feature (D=64,). Same output dim as OvercookedCNN so the MARC/vanilla
    heads are unchanged across envs (env-agnostic core)."""

    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        act = nn.relu if self.activation == "relu" else nn.tanh
        x = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)),
                     bias_init=constant(0.0))(x)
        x = act(x)
        x = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)),
                     bias_init=constant(0.0))(x)
        x = act(x)
        return x


class OvercookedCNN(nn.Module):
    """Exactly the CNN from JaxMARL's validated ippo_cnn_overcooked.

    Copied verbatim (layer order, inits, activation handling) so the vanilla
    path is numerically identical to the validated baseline (cramped_room
    ~240). raw obs (H,W,26) -> feature (D=64,).
    """

    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        act = nn.relu if self.activation == "relu" else nn.tanh
        x = nn.Conv(32, (5, 5), kernel_init=orthogonal(np.sqrt(2)),
                    bias_init=constant(0.0))(x)
        x = act(x)
        x = nn.Conv(32, (3, 3), kernel_init=orthogonal(np.sqrt(2)),
                    bias_init=constant(0.0))(x)
        x = act(x)
        x = nn.Conv(32, (3, 3), kernel_init=orthogonal(np.sqrt(2)),
                    bias_init=constant(0.0))(x)
        x = act(x)
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)),
                     bias_init=constant(0.0))(x)
        x = act(x)
        return x


class CraftaxMLPEncoder(nn.Module):
    """Wide MLP for Craftax symbolic obs (~8.7k-dim int vector). Casts to
    float, 512->256->64 so the 64-d feature matches OvercookedCNN/MLPEncoder
    — the MARC/vanilla heads stay env-agnostic and unchanged across envs.
    """

    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        act = nn.relu if self.activation == "relu" else nn.tanh
        x = x.astype(jnp.float32)
        x = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)),
                     bias_init=constant(0.0))(x)
        x = act(x)
        x = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)),
                     bias_init=constant(0.0))(x)
        x = act(x)
        x = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)),
                     bias_init=constant(0.0))(x)
        x = act(x)
        return x


# --------------------------------------------------------------------------- #
# Adapter contract                                                             #
# --------------------------------------------------------------------------- #
@runtime_checkable
class EnvAdapter(Protocol):
    name: str
    num_agents: int
    action_dim: int
    noop_action: int            # leave-one-out: action that "drops" an agent
    teammate_obs_visible: bool  # see design §2
    # If True, the trainer queries get_avail_actions() each step and masks
    # illegal-action logits before sample/log_prob/entropy. Default off so
    # existing-env (Overcooked / MPE / SMAX) results stay byte-reproducible.
    has_action_mask: bool = False

    def make_env(self, **env_kwargs): ...
    def obs_shape(self, env) -> tuple: ...
    def obs_encoder(self) -> nn.Module: ...
    def get_avail_actions(self, env, env_state) -> dict: ...   # optional


class OvercookedAdapter:
    name = "overcooked"
    num_agents = 2
    action_dim = 6
    noop_action = 4              # Actions.stay (for leave-one-out eval)
    teammate_obs_visible = True  # Overcooked is fully observable

    def make_env(self, **env_kwargs):
        kw = dict(env_kwargs)
        layout = kw.get("layout")
        if isinstance(layout, str):
            kw["layout"] = overcooked_layouts[layout]
        return jaxmarl.make("overcooked", **kw)

    def obs_shape(self, env):
        return env.observation_space("agent_0").shape

    def obs_encoder(self) -> nn.Module:
        return OvercookedCNN(activation="relu")


class MPEAdapter:
    """MPE simple_spread (cooperative coverage / anti-redundancy) with a
    configurable team size — a scalable-N ladder (N=3/6/9) to test MARC's
    core thesis (gap grows with team size; per-teammate inference) that
    2-agent Overcooked structurally cannot. Vector obs (6N,) -> MLPEncoder.

    Uses MPE_simple_spread_v3 with action_type='Discrete' (Discrete(5),
    action 0 = no-op for leave-one-out). The FACMAC *Na variants are
    CONTINUOUS-action (Box) and incompatible with our Categorical policy —
    do not use them. N is set via num_agents (=num_landmarks).
    """

    N_BY_NAME = {"mpe_spread": 3, "mpe_spread_3": 3,
                 "mpe_spread_6": 6, "mpe_spread_9": 9}

    def __init__(self, name="mpe_spread"):
        self.name = name
        # "<name>_id" -> same task wrapped in the agent-ID param-share
        # baseline (one-hot agent index appended to obs).
        self._agent_id = name.endswith("_id")
        base = name[:-3] if self._agent_id else name
        self._N = self.N_BY_NAME[base]
        e = self.make_env()
        self.num_agents = e.num_agents
        self.action_dim = e.action_space(e.agents[0]).n   # Discrete(5)
        self.noop_action = 0          # JaxMARL MPE discrete: 0 = no_op
        self.teammate_obs_visible = True

    def make_env(self, **env_kwargs):
        kw = dict(num_agents=self._N, num_landmarks=self._N,
                  action_type="Discrete")
        kw.update(env_kwargs)
        env = jaxmarl.make("MPE_simple_spread_v3", **kw)
        return AgentIDWrapper(env) if self._agent_id else env

    def obs_shape(self, env):
        return env.observation_space(env.agents[0]).shape

    def obs_encoder(self) -> nn.Module:
        return MLPEncoder(activation="relu")


class SMAXAdapter:
    """SMAX (StarCraft micro abstracted) — JaxMARL's >2-agent benchmark.
    The cleanest non-MPE test of MARC's central 'gap grows with N'
    hypothesis: pick a homogeneous symmetric Marines ladder so unit
    composition is fixed and only team size varies (mirrors MPE
    simple_spread N=3/6/9 in spirit).

    Wrapped via HeuristicEnemySMAX so the trainable side is the
    cooperative ally team (shared team reward) facing scripted enemy
    units — single-team API matches how Overcooked/MPE expose agents.
    Vector obs Box(-1,1,(obs_size,)) -> MLPEncoder (D=64), so MARC heads
    are unchanged. Discrete actions per ally:
        0..3  : 4 cardinal moves
        4     : stop / hold position (the leave-one-out 'drop' no-op)
        5..   : attack enemy_k  (one slot per enemy unit)
    => action_dim = num_movement_actions + num_enemies = 5 + n_enemies.
    """

    SCENARIOS = {
        # symmetric homogeneous (clean N-scaling ladder)
        "smax_3m":         "3m",          # N=3
        "smax_8m":         "8m",          # N=8
        "smax_25m":        "25m",         # N=25 (heavy: needs NUM_ENVS<=8)
        # asymmetric homogeneous (slightly off-spec; a harder middle cell)
        "smax_5m_vs_6m":   "5m_vs_6m",
        "smax_10m_vs_11m": "10m_vs_11m",
        "smax_27m_vs_30m": "27m_vs_30m",
        # mixed unit types (kept for completeness; not pure N-scaling)
        "smax_2s3z":       "2s3z",
        "smax_3s5z":       "3s5z",
        "smax_3s_vs_5z":   "3s_vs_5z",
        "smax_6h_vs_8z":   "6h_vs_8z",
    }

    def __init__(self, name="smax_3m"):
        self.name = name
        self._agent_id = name.endswith("_id")
        self._strip_unit_type = name.endswith("_stripped")
        base = name
        if self._agent_id:
            base = name[:-3]
        elif self._strip_unit_type:
            base = name[:-len("_stripped")]
        if base not in self.SCENARIOS:
            raise KeyError(f"unknown SMAX adapter {base!r}; "
                           f"have {sorted(self.SCENARIOS)}")
        self._scenario_name = self.SCENARIOS[base]
        e = self.make_env()
        self.num_agents = int(e.num_agents)         # = num_allies
        # All allies share the same action space; pick ally_0.
        self.action_dim = int(e.action_space(e.agents[0]).n)
        self.noop_action = 4                         # 'stop' (see docstring)
        self.teammate_obs_visible = True             # allies are in each obs

    def make_env(self, **env_kwargs):
        from jaxmarl.environments.smax import map_name_to_scenario
        scenario = map_name_to_scenario(self._scenario_name)
        env = jaxmarl.make("HeuristicEnemySMAX",
                           scenario=scenario, **env_kwargs)
        if self._strip_unit_type:
            env = StripUnitTypeBitsWrapper(env)
        if self._agent_id:
            env = AgentIDWrapper(env)
        return env

    def obs_shape(self, env):
        return env.observation_space(env.agents[0]).shape

    def obs_encoder(self) -> nn.Module:
        return MLPEncoder(activation="relu")


class HanabiAdapter:
    """Hanabi (cooperative imperfect-info card game) — JaxMARL's
    *partial-observability* benchmark, the most structurally different
    SMAC/MPE-alternative env we can throw at MARC.

    N players (2..5) take TURNS playing 1 of:
        play card_k    (hand_size options)
        discard card_k (hand_size options)
        hint color_c to player_p   ((N-1) * num_colors options)
        hint rank_r  to player_p   ((N-1) * num_ranks options)
        no-op (always last index)
    => action_dim = 2*hand_size + (N-1)*(colors+ranks) + 1.

    Crucial difference from Overcooked / MPE / SMAX: only the *current*
    player's action is used per step (HanabiEnv discards the rest), but
    JaxMARL's API still samples actions for all agents simultaneously,
    so MARC's existing simultaneous-action training loop works unchanged.
    Each agent acts ~1/N of the time, so per-agent sample efficiency is
    N times worse than MPE — expect Hanabi to need more timesteps to
    converge at higher N. Reward is shared (cooperative score in [0, 25]).

    Obs is a long flat float vector (660-dim @ 2p up to ~1280-dim @ 5p);
    routed through MLPEncoder (D=64) so MARC heads are unchanged.

    Caveat (worth noting in the writeup): Hanabi's turn-based structure
    naturally differentiates 'roles' by who's next to act, which dilutes
    MARC's anti-redundancy mechanism a priori. This is a stress test of
    cross-domain generalization, not a target environment.
    """

    N_BY_NAME = {
        "hanabi":   2,         # default alias
        "hanabi_2": 2,
        "hanabi_3": 3,
        "hanabi_4": 4,         # hand_size drops 5 -> 4 here vs <=3p
        "hanabi_5": 5,
    }

    # No natural max_steps in HanabiGame; ~80 turns is a safe upper
    # bound (50-card deck + 8 info tokens + 3 lives). evaluate.py reads
    # env.max_steps for the eval scan, so we attach one in make_env.
    EVAL_HORIZON = 100

    # Hanabi has illegal moves at most timesteps (e.g. hint with no info
    # tokens, play card not in hand). JaxMARL's reference IPPO masks
    # them; we follow suit. The trainer threads `avail` through the
    # rollout so log_prob / entropy / sample all use the masked dist.
    has_action_mask = True

    def __init__(self, name="hanabi_2"):
        self.name = name
        self._agent_id = name.endswith("_id")
        base = name[:-3] if self._agent_id else name
        if base not in self.N_BY_NAME:
            raise KeyError(f"unknown Hanabi adapter {base!r}; "
                           f"have {sorted(self.N_BY_NAME)}")
        self._N = self.N_BY_NAME[base]
        e = self.make_env()
        self.num_agents = int(e.num_agents)
        # All players share the same action space; pick agent_0.
        self.action_dim = int(e.action_space(e.agents[0]).n)
        # HanabiEnv's last action is the no-op (see hanabi.py: # noop).
        self.noop_action = self.action_dim - 1
        # Other players' hands are visible (canonical Hanabi rule), but
        # *your own* hand is hidden — partial-obs from each agent's view.
        # The MARC inferencer still has signal: it sees teammates' obs
        # histories, which include their visible hands and last actions.
        self.teammate_obs_visible = True

    @staticmethod
    def _unwrap(obj, attr):
        """Drill through wrappers (LogWrapper, AgentIDWrapper, ...) to find
        the deepest object that has `attr`."""
        cur = obj
        # try direct attr first
        while not hasattr(cur, attr) and hasattr(cur, "_env"):
            cur = cur._env
        return cur

    def get_avail_actions(self, env, env_state):
        """Returns Dict[agent_name, jnp.ndarray (NUM_ENVS, action_dim) {0,1}]
        of legal moves for the current game state, vmapped over the
        rollout's parallel-env batch dim. Drills through LogWrapper /
        AgentIDWrapper wrappers around both env and env_state.

        HanabiEnv.get_legal_moves expects a single state, so we vmap it
        — matching JaxMARL's reference ippo_ff_hanabi recipe."""
        # LogWrapper stores the underlying state in .env_state; nest if
        # multiple wrappers added it.
        st = env_state
        while hasattr(st, "env_state"):
            st = st.env_state
        e = self._unwrap(env, "get_legal_moves")
        import jax
        return jax.vmap(e.get_legal_moves)(st)

    def make_env(self, **env_kwargs):
        kw = dict(num_agents=self._N)
        kw.update(env_kwargs)
        env = jaxmarl.make("hanabi", **kw)
        # HanabiGame doesn't set max_steps; evaluate.py needs it for the
        # eval rollout's lax.scan length. Attach a tractable upper bound.
        if not hasattr(env, "max_steps"):
            env.max_steps = self.EVAL_HORIZON
        return AgentIDWrapper(env) if self._agent_id else env

    def obs_shape(self, env):
        # JaxMARL's HanabiEnv constructs its Box space with shape=obs_size
        # (a bare int) rather than (obs_size,), so .shape returns a
        # scalar np.int64 instead of a tuple. Normalize to a 1-D tuple so
        # train_marc.py's `tuple(obs_shape)` works.
        s = env.observation_space(env.agents[0]).shape
        return s if isinstance(s, tuple) else (int(s),)

    def obs_encoder(self) -> nn.Module:
        return MLPEncoder(activation="relu")


class CraftaxCoopAdapter:
    """MARC on Minecraft — the design §7.4 stub, realised.

    Substrate: Multi-Agent Craftax 'Craftax-Coop'
    (github.com/BaselOmari/MA-Craftax) — a JAX Minecraft-abstraction
    (Crafter mechanics: mine/craft/tech-tree/survive/combat) multi-agent env
    that subclasses JaxMARL's MultiAgentEnv (same base as our Overcooked/MPE
    adapters, so train/eval are unchanged). 3 agents with built-in roles
    (Forager / Warrior / Miner) + a resource-request/trade action set —
    cooperation-required and heterogeneous, maximally aligned with MARC's
    anti-redundancy / role-differentiation thesis.

    Symbolic flat-vector obs (~8.7k int: local map + teammate dashboard +
    inventory) -> CraftaxMLPEncoder. Discrete actions, NOOP=0 (the
    leave-one-out 'drop' action). Runs in MARC's native regime (GPU-vmapped,
    ~250M env-steps/hr/GPU) — the throughput wall that made MineLand
    infeasible is gone; no run-reduction needed.

    Package path setup (repo is not pip-installable; internal modules use
    bare `from craftax_coop...` imports) is handled by the FarmShare job
    script via PYTHONPATH (repo root + craftax/ dir).
    """

    name = "craftax_coop"

    def __init__(self):
        e = self.make_env()
        self.num_agents = int(e.num_agents)
        self.action_dim = int(e.action_space(e.agents[0]).n)
        self.noop_action = 0              # Action.NOOP (leave-one-out drop)
        self.teammate_obs_visible = True  # symbolic obs has a teammate dashboard

    # eval/rollout fixed horizon. Craftax-Coop's EnvParams.max_timesteps is
    # 100000 (survival cap) — absurd for eval_metrics' lax.scan(T) and the
    # rollout gif. evaluate.py reads env.max_steps (Overcooked ~400). Use a
    # tractable horizon that still captures the sparse achievement returns.
    EVAL_HORIZON = 1000

    def make_env(self, **env_kwargs):
        # make_craftax_env_from_name takes only `name`; ENV_KWARGS is {}.
        from craftax.craftax_env import make_craftax_env_from_name
        env = make_craftax_env_from_name("Craftax-Coop-Symbolic")
        # evaluate.py expects env.max_steps (JaxMARL envs expose it;
        # CraftaxCoopSymbolicEnv does not). Attach it here in the
        # env-coupling layer so the MARC core stays env-agnostic.
        if not hasattr(env, "max_steps"):
            env.max_steps = self.EVAL_HORIZON
        return env

    def obs_shape(self, env):
        return env.observation_space(env.agents[0]).shape

    def obs_encoder(self) -> nn.Module:
        return CraftaxMLPEncoder(activation="relu")


class CoinGameAdapter:
    """Coin Game (Lerer & Peysakhovich 2017) — a 2-agent grid-world social
    dilemma on a 3×3 grid. Red and blue players collect coins; picking up
    the other player's coin is penalised. With shared_rewards=True the task
    becomes fully cooperative: agents must coordinate collection to maximise
    joint reward, which requires role differentiation (who collects which
    coin). Always exactly 2 agents.

    Obs per agent: (36,) flat vector encoding both player positions and both
    coin positions on the 3×3 grid. Action space: Discrete(5) — right, left,
    up, down, stay. Noop = stay (action 4) for leave-one-out eval.
    Episodes are short (max_steps = num_inner_steps = 10).
    """

    name = "coin_game"
    num_agents = 2
    action_dim = 5    # right=0, left=1, up=2, down=3, stay=4
    noop_action = 4   # stay
    teammate_obs_visible = True  # obs encodes both agent positions

    def make_env(self, **env_kwargs):
        kw = dict(shared_rewards=True)
        kw.update(env_kwargs)
        env = jaxmarl.make("coin_game", **kw)
        if not hasattr(env, "max_steps"):
            # num_inner_steps is a closure var, not stored on the env object
            env.max_steps = kw.get("num_inner_steps", 10)
        return env

    def obs_shape(self, env):
        return env.observation_space(env.agents[0]).shape

    def obs_encoder(self) -> nn.Module:
        return MLPEncoder(activation="relu")


class SwitchRiddleAdapter:
    """Switch Riddle (Sukhbaatar et al. 2016) — a minimal cooperative
    signalling task. N agents take turns visiting a room with a light
    switch; exactly one must press it (to signal they've been) and then
    any agent can call TELL to claim all have visited. Reward is +1 if
    correct, -1 if wrong. Obs per agent: [am_i_in_room, bulb_state] (2,).

    Why it's interesting for MARC: agents must differentiate roles
    (who presses vs who waits) from *identical* observations, which is
    exactly the anti-redundancy / role-differentiation thesis. Scales
    cleanly with N — harder coordination at larger N.

    Action space: NOTHING=0, SWITCH_LIGHT=1, TELL=2 (3 actions).
    Noop = NOTHING (action 0) for leave-one-out eval.
    Episodes are very short (max_steps = 4N-6).
    """

    N_BY_NAME = {
        "switch_riddle":   3,
        "switch_riddle_3": 3,
        "switch_riddle_4": 4,
        "switch_riddle_5": 5,
    }

    def __init__(self, name="switch_riddle"):
        self.name = name
        self._N = self.N_BY_NAME[name]
        e = self.make_env()
        self.num_agents = e.num_agents
        self.action_dim = 3   # NOTHING, SWITCH_LIGHT, TELL
        self.noop_action = 0  # NOTHING (leave-one-out drop)
        self.teammate_obs_visible = False  # agents see only own room flag + bulb

    def make_env(self, **env_kwargs):
        kw = dict(num_agents=self._N)
        kw.update(env_kwargs)
        return jaxmarl.make("switch_riddle", **kw)

    def obs_shape(self, env):
        return env.observation_space(env.agents[0]).shape

    def obs_encoder(self) -> nn.Module:
        return MLPEncoder(activation="relu")


class MinecraftAdapter:
    """Typed stub (proposal stretch goal). Proves the abstraction is general;
    intentionally not implemented until MARC works on Overcooked."""

    name = "minecraft"
    num_agents = 2
    action_dim = -1
    noop_action = -1
    teammate_obs_visible = True

    def make_env(self, **env_kwargs):
        raise NotImplementedError("Minecraft adapter is a stub (design §7.4).")

    def obs_shape(self, env):
        raise NotImplementedError

    def obs_encoder(self) -> nn.Module:
        raise NotImplementedError


# name -> zero-arg factory returning an adapter instance
ADAPTERS = {
    "overcooked": OvercookedAdapter,
    "coin_game": CoinGameAdapter,
    "craftax_coop": CraftaxCoopAdapter,
    "minecraft": MinecraftAdapter,
}
for _mpe in MPEAdapter.N_BY_NAME:
    ADAPTERS[_mpe] = (lambda n=_mpe: MPEAdapter(n))
    ADAPTERS[_mpe + "_id"] = (lambda n=_mpe: MPEAdapter(n + "_id"))
for _smax in SMAXAdapter.SCENARIOS:
    ADAPTERS[_smax] = (lambda n=_smax: SMAXAdapter(n))
    ADAPTERS[_smax + "_id"] = (lambda n=_smax: SMAXAdapter(n + "_id"))
    ADAPTERS[_smax + "_stripped"] = (
        lambda n=_smax: SMAXAdapter(n + "_stripped"))
for _han in HanabiAdapter.N_BY_NAME:
    ADAPTERS[_han] = (lambda n=_han: HanabiAdapter(n))
    ADAPTERS[_han + "_id"] = (lambda n=_han: HanabiAdapter(n + "_id"))
for _sr in SwitchRiddleAdapter.N_BY_NAME:
    ADAPTERS[_sr] = (lambda n=_sr: SwitchRiddleAdapter(n))


def get_adapter(name: str) -> EnvAdapter:
    if name not in ADAPTERS:
        raise KeyError(f"unknown adapter {name!r}; have {list(ADAPTERS)}")
    return ADAPTERS[name]()
