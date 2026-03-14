"""Tests for entity CRUD, movement, grid consistency."""

import jax
import jax.numpy as jnp
from jaxswarm.core.state import init_state
from jaxswarm.core.grid_ops import (
    create_entity,
    destroy_entity,
    move_entity,
    get_entities_at,
    find_by_tag,
    find_by_type,
    rebuild_grid,
)


def _make_tags(config, *indices):
    t = jnp.zeros(config.num_tags, dtype=jnp.bool_)
    for i in indices:
        t = t.at[i].set(True)
    return t


def _make_props(config, *values):
    p = jnp.zeros(config.num_props, dtype=jnp.float32)
    for i, v in enumerate(values):
        p = p.at[i].set(v)
    return p


class TestCreateEntity:
    def test_basic_create(self, config, state):
        tags = _make_tags(config, 0, 2)
        props = _make_props(config, 1.0, 2.0)
        new_state, slot = create_entity(state, config, jnp.int32(1), jnp.int32(3), jnp.int32(4), tags, props)

        assert slot == 0
        assert new_state.alive[slot]
        assert new_state.entity_type[slot] == 1
        assert new_state.x[slot] == 3
        assert new_state.y[slot] == 4
        assert new_state.tags[slot, 0]
        assert not new_state.tags[slot, 1]
        assert new_state.tags[slot, 2]
        assert new_state.properties[slot, 0] == 1.0
        assert new_state.properties[slot, 1] == 2.0

    def test_grid_updated(self, config, state):
        tags = _make_tags(config)
        props = _make_props(config)
        new_state, slot = create_entity(state, config, jnp.int32(1), jnp.int32(2), jnp.int32(3), tags, props)

        assert new_state.grid[3, 2, 0] == slot
        assert new_state.grid_count[3, 2] == 1

    def test_full_slots(self, config, rng):
        state = init_state(config, rng)
        tags = _make_tags(config)
        props = _make_props(config)

        # Fill all slots
        for i in range(config.max_entities):
            state, slot = create_entity(state, config, jnp.int32(1), jnp.int32(i % config.grid_w), jnp.int32(0), tags, props)
            assert slot == i

        # Next create should fail
        state2, slot = create_entity(state, config, jnp.int32(1), jnp.int32(0), jnp.int32(0), tags, props)
        assert slot == -1
        # State unchanged
        assert jnp.array_equal(state2.alive, state.alive)

    def test_multiple_same_cell(self, config, state):
        tags = _make_tags(config)
        props = _make_props(config)
        state, s0 = create_entity(state, config, jnp.int32(1), jnp.int32(0), jnp.int32(0), tags, props)
        state, s1 = create_entity(state, config, jnp.int32(2), jnp.int32(0), jnp.int32(0), tags, props)

        assert state.grid[0, 0, 0] == s0
        assert state.grid[0, 0, 1] == s1
        assert state.grid_count[0, 0] == 2


class TestDestroyEntity:
    def test_basic_destroy(self, config, state):
        tags = _make_tags(config, 1)
        props = _make_props(config, 5.0)
        state, slot = create_entity(state, config, jnp.int32(2), jnp.int32(1), jnp.int32(1), tags, props)
        state = destroy_entity(state, config, slot)

        assert not state.alive[slot]
        assert state.entity_type[slot] == 0
        assert (state.properties[slot] == 0).all()
        assert (state.tags[slot] == False).all()

    def test_grid_updated(self, config, state):
        tags = _make_tags(config)
        props = _make_props(config)
        state, slot = create_entity(state, config, jnp.int32(1), jnp.int32(2), jnp.int32(3), tags, props)
        state = destroy_entity(state, config, slot)

        assert state.grid_count[3, 2] == 0
        assert state.grid[3, 2, 0] == -1

    def test_stack_compaction(self, config, state):
        tags = _make_tags(config)
        props = _make_props(config)
        state, s0 = create_entity(state, config, jnp.int32(1), jnp.int32(0), jnp.int32(0), tags, props)
        state, s1 = create_entity(state, config, jnp.int32(1), jnp.int32(0), jnp.int32(0), tags, props)

        # Destroy first entity — second should compact to position 0
        state = destroy_entity(state, config, s0)
        assert state.grid_count[0, 0] == 1
        assert state.grid[0, 0, 0] == s1
        assert state.grid[0, 0, 1] == -1

    def test_create_destroy_reuse(self, config, state):
        tags = _make_tags(config)
        props = _make_props(config)
        state, s0 = create_entity(state, config, jnp.int32(1), jnp.int32(0), jnp.int32(0), tags, props)
        state = destroy_entity(state, config, s0)
        state, s1 = create_entity(state, config, jnp.int32(2), jnp.int32(1), jnp.int32(1), tags, props)
        # Should reuse slot 0
        assert s1 == 0
        assert state.alive[s1]
        assert state.entity_type[s1] == 2


class TestMoveEntity:
    def test_basic_move(self, config, state):
        tags = _make_tags(config)
        props = _make_props(config)
        state, slot = create_entity(state, config, jnp.int32(1), jnp.int32(0), jnp.int32(0), tags, props)

        state, moved = move_entity(state, config, slot, jnp.int32(1), jnp.int32(0))
        assert moved
        assert state.x[slot] == 1
        assert state.y[slot] == 0
        # Old cell empty
        assert state.grid_count[0, 0] == 0
        assert state.grid[0, 0, 0] == -1
        # New cell has entity
        assert state.grid_count[0, 1] == 1
        assert state.grid[0, 1, 0] == slot

    def test_out_of_bounds(self, config, state):
        tags = _make_tags(config)
        props = _make_props(config)
        state, slot = create_entity(state, config, jnp.int32(1), jnp.int32(0), jnp.int32(0), tags, props)

        state2, moved = move_entity(state, config, slot, jnp.int32(-1), jnp.int32(0))
        assert not moved
        assert state2.x[slot] == 0
        assert state2.y[slot] == 0

    def test_full_cell(self, config, state):
        tags = _make_tags(config)
        props = _make_props(config)
        # Fill cell (1,0) to max_stack
        for _ in range(config.max_stack):
            state, _ = create_entity(state, config, jnp.int32(1), jnp.int32(1), jnp.int32(0), tags, props)

        # Create entity at (0,0) and try to move to full cell
        state, slot = create_entity(state, config, jnp.int32(1), jnp.int32(0), jnp.int32(0), tags, props)
        state2, moved = move_entity(state, config, slot, jnp.int32(1), jnp.int32(0))
        assert not moved
        assert state2.x[slot] == 0


class TestQueries:
    def test_get_entities_at(self, config, state):
        tags = _make_tags(config)
        props = _make_props(config)
        state, s0 = create_entity(state, config, jnp.int32(1), jnp.int32(2), jnp.int32(3), tags, props)
        slots, count = get_entities_at(state, jnp.int32(2), jnp.int32(3))
        assert count == 1
        assert slots[0] == s0

    def test_find_by_tag(self, config, state):
        tags_a = _make_tags(config, 0)
        tags_b = _make_tags(config, 1)
        props = _make_props(config)
        state, s0 = create_entity(state, config, jnp.int32(1), jnp.int32(0), jnp.int32(0), tags_a, props)
        state, s1 = create_entity(state, config, jnp.int32(1), jnp.int32(1), jnp.int32(0), tags_b, props)

        mask = find_by_tag(state, 0)
        assert mask[s0]
        assert not mask[s1]

    def test_find_by_tag_only_alive(self, config, state):
        tags = _make_tags(config, 0)
        props = _make_props(config)
        state, s0 = create_entity(state, config, jnp.int32(1), jnp.int32(0), jnp.int32(0), tags, props)
        state = destroy_entity(state, config, s0)
        mask = find_by_tag(state, 0)
        assert not mask.any()

    def test_find_by_type(self, config, state):
        tags = _make_tags(config)
        props = _make_props(config)
        state, s0 = create_entity(state, config, jnp.int32(1), jnp.int32(0), jnp.int32(0), tags, props)
        state, s1 = create_entity(state, config, jnp.int32(2), jnp.int32(1), jnp.int32(0), tags, props)

        mask = find_by_type(state, 1)
        assert mask[s0]
        assert not mask[s1]


class TestRebuildGrid:
    def test_matches_incremental(self, config, state):
        tags = _make_tags(config)
        props = _make_props(config)
        state, _ = create_entity(state, config, jnp.int32(1), jnp.int32(2), jnp.int32(3), tags, props)
        state, _ = create_entity(state, config, jnp.int32(2), jnp.int32(5), jnp.int32(1), tags, props)
        state, _ = create_entity(state, config, jnp.int32(1), jnp.int32(2), jnp.int32(3), tags, props)

        # Clear grid and rebuild
        rebuilt = rebuild_grid(state, config)
        assert jnp.array_equal(rebuilt.grid, state.grid)
        assert jnp.array_equal(rebuilt.grid_count, state.grid_count)

    def test_after_destroy_and_rebuild(self, config, state):
        tags = _make_tags(config)
        props = _make_props(config)
        state, s0 = create_entity(state, config, jnp.int32(1), jnp.int32(0), jnp.int32(0), tags, props)
        state, s1 = create_entity(state, config, jnp.int32(1), jnp.int32(1), jnp.int32(1), tags, props)
        state = destroy_entity(state, config, s0)

        rebuilt = rebuild_grid(state, config)
        assert rebuilt.grid_count[0, 0] == 0
        assert rebuilt.grid_count[1, 1] == 1
        assert rebuilt.grid[1, 1, 0] == s1
