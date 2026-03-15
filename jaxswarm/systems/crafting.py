"""Crafting system — workbench interaction for combining resources."""

import jax
import jax.numpy as jnp
from jaxswarm.core.state import EnvState, EnvConfig
from jaxswarm.core.grid_ops import get_entities_at


# Adjacent directions: N, S, E, W
ADJ_DX = jnp.array([0, 0, 1, -1], dtype=jnp.int32)
ADJ_DY = jnp.array([-1, 1, 0, 0], dtype=jnp.int32)


def crafting_system(
    state: EnvState, action: jnp.int32, config: EnvConfig,
    workbench_type: int = 6,
    wood_prop: int = 0,
    ore_prop: int = 1,
    result_prop: int = 2,
    wood_cost: int = 2,
    ore_cost: int = 2,
    reward: float = 0.3,
    gs_idx: int = 2,
) -> EnvState:
    """
    On interact (action 4), check adjacent cells for workbench.
    If player has enough resources, craft item.
    """
    is_interact = (action == 4)
    pidx = state.player_idx
    px, py = state.x[pidx], state.y[pidx]

    has_wood = state.properties[pidx, wood_prop] >= wood_cost
    has_ore = state.properties[pidx, ore_prop] >= ore_cost
    can_craft = is_interact & has_wood & has_ore & (state.properties[pidx, result_prop] < 1.0)

    # Check 4 cardinal neighbors for workbench
    def check_dir(d, carry):
        state, found = carry
        nx = px + ADJ_DX[d]
        ny = py + ADJ_DY[d]
        in_bounds = (nx >= 0) & (nx < config.grid_w) & (ny >= 0) & (ny < config.grid_h)
        safe_nx = jnp.clip(nx, 0, config.grid_w - 1)
        safe_ny = jnp.clip(ny, 0, config.grid_h - 1)

        slots, count = get_entities_at(state, safe_nx, safe_ny)

        def check_slot(j, inner):
            st, fd = inner
            slot = slots[j]
            is_wb = (j < count) & in_bounds & st.alive[slot] & (st.entity_type[slot] == workbench_type)
            do_craft = is_wb & can_craft & ~fd

            new_st = st.replace(
                properties=st.properties
                    .at[pidx, wood_prop].set(st.properties[pidx, wood_prop] - wood_cost)
                    .at[pidx, ore_prop].set(st.properties[pidx, ore_prop] - ore_cost)
                    .at[pidx, result_prop].set(1.0),
                reward_acc=st.reward_acc + reward,
                game_state=st.game_state.at[gs_idx].set(1.0),
            )
            st = jax.tree.map(lambda n, o: jnp.where(do_craft, n, o), new_st, st)
            fd = fd | do_craft
            return (st, fd)

        state, found = jax.lax.fori_loop(0, config.max_stack, check_slot, (state, found))
        return (state, found)

    state, _ = jax.lax.fori_loop(0, 4, check_dir, (state, jnp.bool_(False)))
    return state
