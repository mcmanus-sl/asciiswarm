import random as random_module
from collections import OrderedDict

import gymnasium
import numpy as np

from asciiswarm.kernel.entity import Entity
from asciiswarm.kernel.events import EventSystem
from asciiswarm.kernel.renderer import render_ascii
from asciiswarm.kernel.serializer import serialize_state, load_state
from asciiswarm.kernel.types import DEFAULT_GAME_CONFIG


class GridGameEnv(gymnasium.Env):
    """Turn-based ASCII game engine as a Gymnasium environment.

    The engine IS the Gym env. No wrapper needed.
    """

    metadata = {'render_modes': ['ansi'], 'render_fps': 4}

    def __init__(self, game_module, seed=42, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        # Validate game module
        if not hasattr(game_module, 'GAME_CONFIG'):
            raise ValueError("Game module must export GAME_CONFIG dict")
        if not isinstance(game_module.GAME_CONFIG, dict):
            raise ValueError("GAME_CONFIG must be a dict")
        if not hasattr(game_module, 'setup') or not callable(game_module.setup):
            raise ValueError("Game module must export a callable setup(env)")

        self._game_module = game_module

        # Merge config with defaults
        self.config = dict(DEFAULT_GAME_CONFIG)
        self.config.update(game_module.GAME_CONFIG)

        # Grid dimensions
        self._width, self._height = self.config['grid']

        # Build action space and map
        actions = self.config['actions']
        self.action_space = gymnasium.spaces.Discrete(len(actions))
        self.ACTION_MAP = {i: a for i, a in enumerate(actions)}
        self._action_set = set(actions)

        # Build observation space
        tags = self.config['tags']
        self._tag_indices = {tag: i for i, tag in enumerate(tags)}
        player_props = self.config['player_properties']

        self.observation_space = gymnasium.spaces.Dict({
            'grid': gymnasium.spaces.Box(
                low=0.0, high=1.0,
                shape=(len(tags), self._height, self._width),
                dtype=np.float32,
            ),
            'scalars': gymnasium.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(3 + len(player_props),),
                dtype=np.float32,
            ),
        })

        # PRNG
        self._seed = seed
        self._rng = random_module.Random(seed)

        # Event system
        self.events = EventSystem()

        # Entity storage
        self._entities = OrderedDict()  # id -> Entity, preserves creation order
        self._grid = [[[] for _ in range(self._width)] for _ in range(self._height)]
        self._next_entity_id = 1
        self._behaviors = {}  # entity_type -> handler

        # Game state
        self.turn_number = 0
        self.status = 'playing'
        self._current_step_reward = 0.0

        # Built-in reward event listener (registered fresh each reset)
        self._reward_unsub = None

    # ---- Gymnasium interface ----

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = random_module.Random(seed)

        # Clear all state
        self._entities.clear()
        self._grid = [[[] for _ in range(self._width)] for _ in range(self._height)]
        self._next_entity_id = 1
        self._behaviors.clear()
        self.turn_number = 0
        self.status = 'playing'
        self._current_step_reward = 0.0

        # Clear all event handlers to prevent accumulation
        self.events.clear()

        # Register built-in reward listener
        self._reward_unsub = self.events.on('reward', self._on_reward)

        # Let the game module set up the world
        self._game_module.setup(self)

        # Validate player singleton
        players = self.get_entities_by_tag('player')
        if len(players) != 1:
            raise ValueError(
                f"Exactly one entity tagged 'player' must exist after setup(). "
                f"Found {len(players)}."
            )

        obs = self._get_obs()
        info = {'turn': self.turn_number, 'status': self.status}
        return obs, info

    def step(self, action):
        # Early exit guard: game already over before this call
        if self.status != 'playing':
            obs = self._get_obs()
            info = {'turn': self.turn_number, 'status': self.status}
            return obs, 0.0, True, False, info

        # Reset per-step reward accumulator
        self._current_step_reward = 0.0

        # Translate action
        action_str = self.ACTION_MAP.get(action)
        if action_str is None:
            action_str = 'wait'  # fallback for invalid int

        # Execute the turn
        self.handle_input(action_str)

        # Compute reward additively
        reward = self.config['step_penalty'] + self._current_step_reward
        if self.status == 'won':
            reward += 10.0
        elif self.status == 'lost':
            reward += -10.0

        terminated = self.status != 'playing'
        truncated = self.turn_number >= self.config['max_turns']

        obs = self._get_obs()
        info = {'turn': self.turn_number, 'status': self.status}
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == 'ansi':
            return self.render_ascii()
        return None

    def _on_reward(self, event):
        self._current_step_reward += event.payload.get('amount', 0.0)

    def _get_obs(self):
        tags = self.config['tags']
        player_props = self.config['player_properties']

        # Grid observation: channel-first tag map
        grid_obs = np.zeros(
            (len(tags), self._height, self._width), dtype=np.float32
        )
        # O(entities) approach: iterate entities once
        for entity in self._entities.values():
            for tag in entity.tags:
                idx = self._tag_indices.get(tag)
                if idx is not None:
                    grid_obs[idx, entity.y, entity.x] = 1.0

        # Scalars observation
        scalars = np.zeros(3 + len(player_props), dtype=np.float32)
        scalars[0] = self._width / 100.0
        scalars[1] = self._height / 100.0
        scalars[2] = self.turn_number / self.config['max_turns']

        # Player properties
        players = self.get_entities_by_tag('player')
        if players:
            player = players[0]
            for i, prop_def in enumerate(player_props):
                key = prop_def['key']
                max_val = prop_def['max']
                raw = player.get(key, 0.0)
                normalized = raw / max_val if max_val != 0 else 0.0
                scalars[3 + i] = np.clip(normalized, 0.0, 1.0)

        return {'grid': grid_obs, 'scalars': scalars}

    # ---- Entity management ----

    def create_entity(self, type, x, y, glyph, tags, z_order=0, properties=None):
        if not tags:
            raise ValueError("Entity must have at least one tag")

        valid_tags = set(self.config['tags'])
        for tag in tags:
            if tag not in valid_tags:
                raise ValueError(
                    f"Unknown tag {tag!r}. Valid tags: {sorted(valid_tags)}"
                )

        entity_id = f"e{self._next_entity_id}"
        self._next_entity_id += 1

        entity = Entity(
            id=entity_id, type=type, x=x, y=y,
            glyph=glyph, tags=tags, z_order=z_order,
            properties=dict(properties) if properties else {},
        )
        self._entities[entity_id] = entity
        self._grid[y][x].append(entity)

        self.events.emit('entity_created', {'entity': entity})
        return entity

    def destroy_entity(self, entity_id):
        entity = self._entities.get(entity_id)
        if entity is None:
            return

        self._grid[entity.y][entity.x].remove(entity)
        del self._entities[entity_id]
        self.events.emit('entity_destroyed', {'entity': entity})

    def move_entity(self, entity_id, x, y):
        entity = self._entities.get(entity_id)
        if entity is None:
            return False

        # Bounds check
        if x < 0 or x >= self._width or y < 0 or y >= self._height:
            return False

        old_x, old_y = entity.x, entity.y

        # Emit before_move (cancellable)
        event = self.events.emit('before_move', {
            'entity': entity,
            'from_x': old_x, 'from_y': old_y,
            'to_x': x, 'to_y': y,
        })
        if event.cancelled:
            return False

        # Check for occupants at target cell
        occupants = self._grid[y][x]
        if occupants:
            # Emit collision (cancellable)
            coll_event = self.events.emit('collision', {
                'mover': entity,
                'occupants': list(occupants),
                'x': x, 'y': y,
            })
            if coll_event.cancelled:
                return False

        # Perform the move
        self._grid[old_y][old_x].remove(entity)
        entity.x = x
        entity.y = y
        self._grid[y][x].append(entity)
        return True

    def get_entities_at(self, x, y):
        if 0 <= x < self._width and 0 <= y < self._height:
            return list(self._grid[y][x])
        return []

    def get_entities_by_type(self, type):
        return [e for e in self._entities.values() if e.type == type]

    def get_entities_by_tag(self, tag):
        return [e for e in self._entities.values() if tag in e.tags]

    def get_entity(self, entity_id):
        return self._entities.get(entity_id)

    def get_all_entities(self):
        return list(self._entities.values())

    # ---- Behaviors ----

    def register_behavior(self, entity_type, handler):
        self._behaviors[entity_type] = handler

    # ---- Event system shortcuts ----

    def on(self, event_name, handler):
        return self.events.on(event_name, handler)

    def emit(self, event_name, payload=None):
        return self.events.emit(event_name, payload)

    # ---- Turn loop ----

    def handle_input(self, action, payload=None):
        # Validate action
        if action not in self._action_set:
            return

        # No-op if game is over
        if self.status != 'playing':
            return

        # Increment turn
        self.turn_number += 1

        # Emit turn_start
        self.events.emit('turn_start', {})
        if self.status != 'playing':
            self.events.emit('turn_end', {})
            return

        # Emit input
        self.events.emit('input', {'action': action, 'payload': payload})
        if self.status != 'playing':
            self.events.emit('turn_end', {})
            return

        # Run behaviors: snapshot iteration
        entity_snapshot = list(self._entities.values())
        for entity in entity_snapshot:
            if self.status != 'playing':
                break
            # Verify entity still exists
            if self.get_entity(entity.id) is None:
                continue
            behavior = self._behaviors.get(entity.type)
            if behavior:
                behavior(entity, self)

        # Emit turn_end
        self.events.emit('turn_end', {})

    # ---- Game control ----

    def end_game(self, status):
        if status not in ('won', 'lost'):
            raise ValueError(f"Status must be 'won' or 'lost', got {status!r}")
        self.status = status

    # ---- PRNG ----

    def random(self):
        return self._rng.random()

    # ---- Renderer ----

    def render_ascii(self):
        return render_ascii(self._grid, self._width, self._height)

    # ---- Serialization ----

    def serialize_state(self):
        return serialize_state(self)

    def load_state(self, state):
        load_state(self, state)
