def render_ascii(grid, width, height):
    """Render the grid as an ASCII string.

    Each cell shows the glyph of the highest z-order entity, or '.' for empty cells.
    Returns exactly `height` lines of `width` characters.

    Args:
        grid: 2D list [y][x] of lists of Entity objects.
        width: Grid width.
        height: Grid height.
    """
    lines = []
    for y in range(height):
        row = []
        for x in range(width):
            entities = grid[y][x]
            if entities:
                top = max(entities, key=lambda e: e.z_order)
                row.append(top.glyph)
            else:
                row.append('.')
        lines.append(''.join(row))
    return '\n'.join(lines)
