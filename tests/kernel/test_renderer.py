from asciiswarm.kernel.entity import Entity
from asciiswarm.kernel.renderer import render_ascii


def test_empty_grid():
    grid = [[[] for _ in range(3)] for _ in range(3)]
    result = render_ascii(grid, 3, 3)
    assert result == '...\n...\n...'


def test_single_entity():
    grid = [[[] for _ in range(3)] for _ in range(3)]
    e = Entity('e1', 'player', 1, 1, '@', ['player'])
    grid[1][1].append(e)
    result = render_ascii(grid, 3, 3)
    lines = result.split('\n')
    assert lines[0] == '...'
    assert lines[1] == '.@.'
    assert lines[2] == '...'


def test_z_order_rendering():
    """Highest z-order entity's glyph is shown."""
    grid = [[[] for _ in range(3)] for _ in range(3)]
    low = Entity('e1', 'floor', 1, 1, '.', ['npc'], z_order=0)
    high = Entity('e2', 'player', 1, 1, '@', ['player'], z_order=10)
    grid[1][1].extend([low, high])
    result = render_ascii(grid, 3, 3)
    assert result.split('\n')[1] == '.@.'


def test_multiple_entities_different_cells():
    grid = [[[] for _ in range(4)] for _ in range(4)]
    p = Entity('e1', 'player', 0, 0, '@', ['player'], z_order=10)
    e = Entity('e2', 'exit', 3, 3, '>', ['exit'], z_order=5)
    grid[0][0].append(p)
    grid[3][3].append(e)
    result = render_ascii(grid, 4, 4)
    lines = result.split('\n')
    assert lines[0][0] == '@'
    assert lines[3][3] == '>'


def test_exact_dimensions():
    grid = [[[] for _ in range(5)] for _ in range(3)]
    result = render_ascii(grid, 5, 3)
    lines = result.split('\n')
    assert len(lines) == 3
    for line in lines:
        assert len(line) == 5
