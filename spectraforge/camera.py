"""
Camera module for generating primary rays.

Supports:
- Perspective projection
- Depth of field (defocus blur)
- Configurable field of view
- Arbitrary positioning via look-at
- Motion blur (shutter time range)
"""

from __future__ import annotations
import math
import random
from .vec3 import Vec3, Point3
from .ray import Ray


class Camera:
    """A camera with perspective projection, depth of field, and motion blur."""

    def __init__(
        self,
        look_from: Point3,
        look_at: Point3,
        vup: Vec3 = Vec3(0, 1, 0),
        vfov: float = 90.0,
        aspect_ratio: float = 16.0 / 9.0,
        aperture: float = 0.0,
        focus_dist: float = 1.0,
        shutter_open: float = 0.0,
        shutter_close: float = 0.0
    ):
        """Create a camera.

        Args:
            look_from: Camera position in world space
            look_at: Point the camera is looking at
            vup: World up vector (usually (0, 1, 0))
            vfov: Vertical field of view in degrees
            aspect_ratio: Width / Height ratio
            aperture: Lens aperture for depth of field (0 = pinhole)
            focus_dist: Distance to the focus plane
            shutter_open: Time when shutter opens (for motion blur)
            shutter_close: Time when shutter closes (for motion blur)
        """
        theta = math.radians(vfov)
        h = math.tan(theta / 2)
        viewport_height = 2.0 * h
        viewport_width = aspect_ratio * viewport_height

        # Compute orthonormal camera basis
        self.w = (look_from - look_at).normalize()  # Points backward from camera
        self.u = vup.cross(self.w).normalize()       # Points right
        self.v = self.w.cross(self.u)                # Points up

        self.origin = look_from
        self.horizontal = self.u * viewport_width * focus_dist
        self.vertical = self.v * viewport_height * focus_dist
        self.lower_left_corner = (
            self.origin
            - self.horizontal / 2
            - self.vertical / 2
            - self.w * focus_dist
        )

        self.lens_radius = aperture / 2
        self.shutter_open = shutter_open
        self.shutter_close = shutter_close

    def get_ray(self, s: float, t: float) -> Ray:
        """Generate a ray for the given UV coordinates on the image plane.

        Args:
            s: Horizontal coordinate [0, 1] (0 = left, 1 = right)
            t: Vertical coordinate [0, 1] (0 = bottom, 1 = top)

        Returns:
            A ray from the camera through the specified pixel
        """
        # Depth of field: random point on lens
        if self.lens_radius > 0:
            rd = Vec3.random_in_unit_disk() * self.lens_radius
            offset = self.u * rd.x + self.v * rd.y
        else:
            offset = Vec3(0, 0, 0)

        direction = (
            self.lower_left_corner
            + self.horizontal * s
            + self.vertical * t
            - self.origin
            - offset
        )

        # Motion blur: random time within shutter interval
        if self.shutter_close > self.shutter_open:
            time = self.shutter_open + random.random() * (self.shutter_close - self.shutter_open)
        else:
            time = self.shutter_open

        return Ray(self.origin + offset, direction.normalize(), time)

    def __repr__(self) -> str:
        return f"Camera(origin={self.origin}, looking_at={self.lower_left_corner + self.horizontal/2 + self.vertical/2})"
