import jax
import jax.numpy as jnp
import chex
from flax import struct
from gymnax.environments import environment
from typing import Tuple


# --------------------------------------------------------------------------- #
#  LogWrapper – tracks per-episode returns and lengths                        #
# --------------------------------------------------------------------------- #

@struct.dataclass
class LogEnvState:
    env_state: chex.PyTreeDef
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int
    returned_episode: bool


class LogWrapper(environment.Environment):

    def __init__(self, env: environment.Environment):
        super().__init__()
        self._env = env

    @property
    def default_params(self):
        return self._env.default_params

    def reset_env(self, key: chex.PRNGKey, params) -> Tuple[chex.Array, LogEnvState]:
        obs, env_state = self._env.reset_env(key, params)
        state = LogEnvState(
            env_state=env_state,
            episode_returns=0.0,
            episode_lengths=0,
            returned_episode_returns=0.0,
            returned_episode_lengths=0,
            returned_episode=False,
        )
        return obs, state

    def step_env(
        self, key: chex.PRNGKey, state: LogEnvState, action: chex.Array, params
    ) -> Tuple[chex.Array, LogEnvState, float, bool, dict]:
        obs, env_state, reward, done, info = self._env.step_env(
            key, state.env_state, action, params)

        new_returns = state.episode_returns + reward
        new_lengths = state.episode_lengths + 1

        # Auto-reset: when the episode is done, replace env_state/obs with a
        # fresh reset so the next scan step begins a new episode.
        key, reset_key = jax.random.split(key)
        reset_obs, reset_env_state = self._env.reset_env(reset_key, params)
        env_state = jax.tree_util.tree_map(
            lambda r, s: jnp.where(done, r, s), reset_env_state, env_state)
        obs = jnp.where(done, reset_obs, obs)

        new_state = LogEnvState(
            env_state=env_state,
            episode_returns=new_returns * (1 - done),
            episode_lengths=new_lengths * (1 - done),
            returned_episode_returns=jnp.where(done, new_returns, state.returned_episode_returns),
            returned_episode_lengths=jnp.where(done, new_lengths, state.returned_episode_lengths),
            returned_episode=done,
        )

        info["returned_episode"] = done
        info["returned_episode_returns"] = new_state.returned_episode_returns
        info["returned_episode_lengths"] = new_state.returned_episode_lengths
        return obs, new_state, reward, done, info

    def action_space(self, params=None):
        return self._env.action_space(params)

    def observation_space(self, params):
        return self._env.observation_space(params)

    @property
    def name(self) -> str:
        return self._env.name


# --------------------------------------------------------------------------- #
#  VecEnv – vmap over num_envs parallel environments                          #
# --------------------------------------------------------------------------- #

class VecEnv(environment.Environment):

    def __init__(self, env: environment.Environment):
        super().__init__()
        self._env = env
        self.reset = jax.vmap(self._env.reset_env, in_axes=(0, None))
        self.step  = jax.vmap(self._env.step_env,  in_axes=(0, 0, 0, None))

    @property
    def default_params(self):
        return self._env.default_params

    def reset_env(self, key, params):
        return self._env.reset_env(key, params)

    def step_env(self, key, state, action, params):
        return self._env.step_env(key, state, action, params)

    def action_space(self, params=None):
        return self._env.action_space(params)

    def observation_space(self, params):
        return self._env.observation_space(params)

    @property
    def name(self) -> str:
        return self._env.name
