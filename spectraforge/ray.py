"""
Ray class for representing rays in 3D space.

A ray is defined by an origin point and a direction vector.
Ray(t) = origin + t * direction
"""

from __future__ import annotations
from .vec3 import Vec3, Point3


class Ray:
    """A ray with origin and direction.

    The parametric form is: P(t) = origin + t * direction
    where t >= 0 represents points along the ray.
    """

    __slots__ = ('origin', 'direction', 'time')

    def __init__(self, origin: Point3, direction: Vec3, time: float = 0.0):
        """Create a ray with given origin and direction.

        Args:
            origin: The starting point of the ray
            direction: The direction vector (should be normalized for most uses)
            time: Time value for motion blur (default 0)
        """
        self.origin = origin
        self.direction = direction
        self.time = time

    def at(self, t: float) -> Point3:
        """Get the point along the ray at parameter t.

        Args:
            t: The parameter value (distance if direction is normalized)

        Returns:
            The point at origin + t * direction
        """
        return self.origin + self.direction * t

    def __repr__(self) -> str:
        return f"Ray(origin={self.origin}, direction={self.direction})"
