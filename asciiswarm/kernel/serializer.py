import json


def serialize_state(env):
    """Serialize the environment state to a JSON-serializable dict.

    Includes: grid dimensions, all entities, PRNG state, entity ID counter,
    turn number, game status.
    """
    entities = []
    for entity in env.get_all_entities():
        entity.validate_properties_serializable()
        entities.append({
            'id': entity.id,
            'type': entity.type,
            'x': entity.x,
            'y': entity.y,
            'glyph': entity.glyph,
            'tags': list(entity.tags),
            'z_order': entity.z_order,
            'properties': dict(entity.properties),
        })

    # Sort entities by numeric ID for deterministic output
    entities.sort(key=lambda e: int(e['id'][1:]))

    # Sort property keys alphabetically within each entity
    for ent_data in entities:
        ent_data['properties'] = dict(sorted(ent_data['properties'].items()))

    return {
        'grid': list(env.config['grid']),
        'entities': entities,
        'prng_state': env._rng.getstate(),
        'next_entity_id': env._next_entity_id,
        'turn_number': env.turn_number,
        'status': env.status,
    }


def load_state(env, state):
    """Restore environment state from a serialized dict.

    Does NOT call setup(). Re-instantiates Entity class instances from the dict data.
    Overwrites turn number, ID counter, status, and PRNG state.
    """
    from asciiswarm.kernel.entity import Entity

    # Clear current entities
    env._entities.clear()
    env._grid = [[[] for _ in range(env._width)] for _ in range(env._height)]

    # Re-instantiate entities
    for ent_data in state['entities']:
        entity = Entity(
            id=ent_data['id'],
            type=ent_data['type'],
            x=ent_data['x'],
            y=ent_data['y'],
            glyph=ent_data['glyph'],
            tags=ent_data['tags'],
            z_order=ent_data['z_order'],
            properties=dict(ent_data['properties']),
        )
        env._entities[entity.id] = entity
        env._grid[entity.y][entity.x].append(entity)

    # Restore PRNG state — convert lists back to tuples
    prng_state = state['prng_state']
    # random.getstate() returns (version, internalstate_tuple, gauss_next)
    # JSON converts tuples to lists, so we must convert back
    restored_state = (
        prng_state[0],
        tuple(prng_state[1]),
        prng_state[2],
    )
    env._rng.setstate(restored_state)

    env._next_entity_id = state['next_entity_id']
    env.turn_number = state['turn_number']
    env.status = state['status']
