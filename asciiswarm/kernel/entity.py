import json


class Entity:
    """A game entity on the grid."""

    def __init__(self, id, type, x, y, glyph, tags, z_order=0, properties=None):
        self.id = id
        self.type = type
        self.x = x
        self.y = y
        self.glyph = glyph
        self.tags = list(tags)
        self.z_order = z_order
        self.properties = dict(properties) if properties else {}

    def set(self, key, value):
        self.properties[key] = value

    def get(self, key, default=None):
        return self.properties.get(key, default)

    def has_tag(self, tag):
        return tag in self.tags

    def validate_properties_serializable(self):
        """Check that all properties are JSON-serializable primitives."""
        try:
            json.dumps(self.properties)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Entity {self.id} has non-JSON-serializable properties: {e}"
            )

    def __repr__(self):
        return f"Entity({self.id!r}, type={self.type!r}, pos=({self.x},{self.y}), tags={self.tags})"
