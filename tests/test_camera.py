"""Tests for Camera class."""

import pytest
import math
from spectraforge.vec3 import Vec3, Point3
from spectraforge.camera import Camera


class TestCameraCreation:
    """Test Camera construction."""

    def test_default_camera(self):
        cam = Camera(
            look_from=Point3(0, 0, 0),
            look_at=Point3(0, 0, -1),
            vup=Vec3(0, 1, 0),
            vfov=90,
            aspect_ratio=16/9
        )
        assert cam.origin == Point3(0, 0, 0)

    def test_camera_basis_vectors(self):
        cam = Camera(
            look_from=Point3(0, 0, 0),
            look_at=Point3(0, 0, -1),
            vup=Vec3(0, 1, 0),
            vfov=90,
            aspect_ratio=1.0
        )
        # w should point backward (opposite of look direction)
        assert cam.w.z > 0
        # u should point right
        assert abs(cam.u.x - 1.0) < 1e-6
        # v should point up
        assert abs(cam.v.y - 1.0) < 1e-6


class TestCameraRays:
    """Test Camera.get_ray() method."""

    def test_center_ray(self):
        cam = Camera(
            look_from=Point3(0, 0, 0),
            look_at=Point3(0, 0, -1),
            vup=Vec3(0, 1, 0),
            vfov=90,
            aspect_ratio=1.0,
            aperture=0.0
        )
        ray = cam.get_ray(0.5, 0.5)

        # Center ray should go straight forward
        assert abs(ray.direction.x) < 0.1
        assert abs(ray.direction.y) < 0.1
        assert ray.direction.z < 0  # Forward is -Z

    def test_corner_rays(self):
        cam = Camera(
            look_from=Point3(0, 0, 0),
            look_at=Point3(0, 0, -1),
            vup=Vec3(0, 1, 0),
            vfov=90,
            aspect_ratio=1.0,
            aperture=0.0
        )

        # Bottom-left
        bl = cam.get_ray(0, 0)
        assert bl.direction.x < 0
        assert bl.direction.y < 0

        # Top-right
        tr = cam.get_ray(1, 1)
        assert tr.direction.x > 0
        assert tr.direction.y > 0

    def test_ray_origin_without_dof(self):
        cam = Camera(
            look_from=Point3(1, 2, 3),
            look_at=Point3(0, 0, 0),
            vup=Vec3(0, 1, 0),
            vfov=90,
            aspect_ratio=1.0,
            aperture=0.0
        )
        ray = cam.get_ray(0.5, 0.5)
        assert ray.origin == cam.origin


class TestDepthOfField:
    """Test Camera depth of field."""

    def test_dof_varies_origin(self):
        cam = Camera(
            look_from=Point3(0, 0, 0),
            look_at=Point3(0, 0, -10),
            vup=Vec3(0, 1, 0),
            vfov=90,
            aspect_ratio=1.0,
            aperture=2.0,  # Large aperture
            focus_dist=10.0
        )

        origins = [cam.get_ray(0.5, 0.5).origin for _ in range(100)]

        # With DOF, origins should vary
        xs = [o.x for o in origins]
        assert max(xs) - min(xs) > 0.1

    def test_no_dof_fixed_origin(self):
        cam = Camera(
            look_from=Point3(0, 0, 0),
            look_at=Point3(0, 0, -10),
            vup=Vec3(0, 1, 0),
            vfov=90,
            aspect_ratio=1.0,
            aperture=0.0,  # Pinhole
            focus_dist=10.0
        )

        origins = [cam.get_ray(0.5, 0.5).origin for _ in range(10)]

        # Without DOF, all origins should be the same
        for o in origins:
            assert o == cam.origin


class TestFieldOfView:
    """Test Camera field of view."""

    def test_narrow_fov(self):
        cam_narrow = Camera(
            look_from=Point3(0, 0, 0),
            look_at=Point3(0, 0, -1),
            vup=Vec3(0, 1, 0),
            vfov=20,
            aspect_ratio=1.0,
            aperture=0.0
        )

        cam_wide = Camera(
            look_from=Point3(0, 0, 0),
            look_at=Point3(0, 0, -1),
            vup=Vec3(0, 1, 0),
            vfov=90,
            aspect_ratio=1.0,
            aperture=0.0
        )

        # Corner ray should have smaller angle for narrow FOV
        ray_narrow = cam_narrow.get_ray(1, 1)
        ray_wide = cam_wide.get_ray(1, 1)

        # Narrow should be more aligned with center
        center = Vec3(0, 0, -1)
        assert ray_narrow.direction.dot(center) > ray_wide.direction.dot(center)


class TestCameraPositioning:
    """Test various camera positions."""

    def test_looking_down(self):
        cam = Camera(
            look_from=Point3(0, 10, 0),
            look_at=Point3(0, 0, 0),
            vup=Vec3(0, 0, -1),  # Z-up would cause issues, use something else
            vfov=90,
            aspect_ratio=1.0
        )

        ray = cam.get_ray(0.5, 0.5)
        # Should be looking down
        assert ray.direction.y < 0

    def test_angled_camera(self):
        cam = Camera(
            look_from=Point3(5, 5, 5),
            look_at=Point3(0, 0, 0),
            vup=Vec3(0, 1, 0),
            vfov=60,
            aspect_ratio=1.0
        )

        ray = cam.get_ray(0.5, 0.5)
        # Should point toward origin
        target = Point3(0, 0, 0) - cam.origin
        assert ray.direction.dot(target.normalize()) > 0.9
