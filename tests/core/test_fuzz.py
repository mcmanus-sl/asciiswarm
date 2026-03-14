"""Fuzz test: random actions on Game 01, no NaNs, all terminate."""

import jax
import jax.numpy as jnp
from jaxswarm.games import game_01_empty_exit as game


NUM_EPISODES = 1000


class TestFuzz:
    def test_random_agent_no_nans(self):
        """Run 1000 random-action episodes, assert no NaN and all terminate."""
        config = game.CONFIG

        def run_episode(key):
            k_reset, k_actions = jax.random.split(key)
            state, obs = game.reset(k_reset)

            def step_fn(carry, action_key):
                state, obs, done_flag, cum_reward = carry
                action = jax.random.randint(action_key, (), 0, config.num_actions)
                new_state, new_obs, reward, done = game.step(state, action)

                state = jax.tree.map(
                    lambda n, o: jnp.where(done_flag, o, n), new_state, state
                )
                obs = jax.tree.map(
                    lambda n, o: jnp.where(done_flag, o, n), new_obs, obs
                )
                cum_reward = jnp.where(done_flag, cum_reward, cum_reward + reward)
                done_flag = done_flag | done
                return (state, obs, done_flag, cum_reward), None

            action_keys = jax.random.split(k_actions, config.max_turns)
            init = (state, obs, jnp.bool_(False), jnp.float32(0.0))
            (final_state, _, terminated, cum_reward), _ = jax.lax.scan(
                step_fn, init, action_keys
            )

            return terminated, cum_reward, final_state.turn_number

        keys = jax.random.split(jax.random.PRNGKey(0), NUM_EPISODES)
        run_batch = jax.jit(jax.vmap(run_episode))
        terminated, rewards, lengths = run_batch(keys)

        # No NaN rewards
        assert not jnp.isnan(rewards).any(), "NaN found in rewards"

        # All episodes should terminate (either won or hit max_turns)
        assert terminated.all(), f"Not all episodes terminated: {terminated.sum()}/{NUM_EPISODES}"
