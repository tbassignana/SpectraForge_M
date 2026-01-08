"""Tests for motion blur functionality."""

import pytest
import math
from spectraforge.vec3 import Vec3, Point3, Color
from spectraforge.ray import Ray
from spectraforge.shapes import MovingSphere, HittableList
from spectraforge.camera import Camera
from spectraforge.materials import Lambertian


class TestMovingSphere:
    """Test MovingSphere class."""

    def test_creation(self):
        sphere = MovingSphere(
            center0=Point3(0, 0, 0),
            center1=Point3(1, 0, 0),
            time0=0.0,
            time1=1.0,
            radius=0.5
        )
        assert sphere.center0 == Point3(0, 0, 0)
        assert sphere.center1 == Point3(1, 0, 0)
        assert sphere.radius == 0.5

    def test_center_interpolation(self):
        sphere = MovingSphere(
            center0=Point3(0, 0, 0),
            center1=Point3(2, 0, 0),
            time0=0.0,
            time1=1.0,
            radius=0.5
        )

        # At t=0, should be at center0
        c0 = sphere.center(0.0)
        assert abs(c0.x) < 1e-6

        # At t=1, should be at center1
        c1 = sphere.center(1.0)
        assert abs(c1.x - 2.0) < 1e-6

        # At t=0.5, should be halfway
        c_mid = sphere.center(0.5)
        assert abs(c_mid.x - 1.0) < 1e-6

    def test_hit_at_time0(self):
        sphere = MovingSphere(
            center0=Point3(0, 0, 0),
            center1=Point3(2, 0, 0),
            time0=0.0,
            time1=1.0,
            radius=0.5
        )

        # Ray at time=0 should hit sphere at center0
        ray = Ray(Point3(0, 0, -5), Vec3(0, 0, 1), time=0.0)
        hit = sphere.hit(ray, 0.001, float('inf'))

        assert hit is not None
        assert abs(hit.point.x) < 0.6  # Near center0.x = 0

    def test_hit_at_time1(self):
        sphere = MovingSphere(
            center0=Point3(0, 0, 0),
            center1=Point3(2, 0, 0),
            time0=0.0,
            time1=1.0,
            radius=0.5
        )

        # Ray at time=1 should hit sphere at center1
        ray = Ray(Point3(2, 0, -5), Vec3(0, 0, 1), time=1.0)
        hit = sphere.hit(ray, 0.001, float('inf'))

        assert hit is not None
        assert abs(hit.point.x - 2.0) < 0.6  # Near center1.x = 2

    def test_miss_wrong_time(self):
        sphere = MovingSphere(
            center0=Point3(0, 0, 0),
            center1=Point3(5, 0, 0),  # Moves far
            time0=0.0,
            time1=1.0,
            radius=0.5
        )

        # Ray at time=0 aimed at center1 position should miss
        ray = Ray(Point3(5, 0, -5), Vec3(0, 0, 1), time=0.0)
        hit = sphere.hit(ray, 0.001, float('inf'))

        assert hit is None  # Sphere is at (0,0,0) at time=0

    def test_hit_at_intermediate_time(self):
        sphere = MovingSphere(
            center0=Point3(0, 0, 0),
            center1=Point3(4, 0, 0),
            time0=0.0,
            time1=1.0,
            radius=0.5
        )

        # At t=0.5, sphere is at (2, 0, 0)
        ray = Ray(Point3(2, 0, -5), Vec3(0, 0, 1), time=0.5)
        hit = sphere.hit(ray, 0.001, float('inf'))

        assert hit is not None
        assert abs(hit.point.x - 2.0) < 0.6

    def test_bounding_box_contains_both_positions(self):
        sphere = MovingSphere(
            center0=Point3(0, 0, 0),
            center1=Point3(3, 0, 0),
            time0=0.0,
            time1=1.0,
            radius=0.5
        )

        bbox = sphere.bounding_box()
        assert bbox is not None

        # Should contain center0 with radius
        assert bbox.minimum.x <= -0.5
        # Should contain center1 with radius
        assert bbox.maximum.x >= 3.5

    def test_with_material(self):
        material = Lambertian(Color(1, 0, 0))
        sphere = MovingSphere(
            center0=Point3(0, 0, 0),
            center1=Point3(1, 0, 0),
            time0=0.0,
            time1=1.0,
            radius=0.5,
            material=material
        )

        ray = Ray(Point3(0, 0, -5), Vec3(0, 0, 1), time=0.0)
        hit = sphere.hit(ray, 0.001, float('inf'))

        assert hit is not None
        assert hit.material is material

    def test_uv_coordinates(self):
        sphere = MovingSphere(
            center0=Point3(0, 0, 0),
            center1=Point3(1, 0, 0),
            time0=0.0,
            time1=1.0,
            radius=0.5
        )

        ray = Ray(Point3(0, 0, -5), Vec3(0, 0, 1), time=0.0)
        hit = sphere.hit(ray, 0.001, float('inf'))

        assert hit is not None
        assert 0 <= hit.u <= 1
        assert 0 <= hit.v <= 1


class TestCameraMotionBlur:
    """Test camera shutter time functionality."""

    def test_default_no_motion_blur(self):
        camera = Camera(
            look_from=Point3(0, 0, 5),
            look_at=Point3(0, 0, 0)
        )

        ray = camera.get_ray(0.5, 0.5)
        assert ray.time == 0.0  # Default shutter_open

    def test_shutter_time_range(self):
        camera = Camera(
            look_from=Point3(0, 0, 5),
            look_at=Point3(0, 0, 0),
            shutter_open=0.0,
            shutter_close=1.0
        )

        times = set()
        for _ in range(100):
            ray = camera.get_ray(0.5, 0.5)
            times.add(round(ray.time, 2))

        # Should have variation in times
        assert len(times) > 1
        # All times should be in range
        for t in times:
            assert 0.0 <= t <= 1.0

    def test_fixed_shutter(self):
        camera = Camera(
            look_from=Point3(0, 0, 5),
            look_at=Point3(0, 0, 0),
            shutter_open=0.5,
            shutter_close=0.5  # Same as open = no blur
        )

        ray = camera.get_ray(0.5, 0.5)
        assert ray.time == 0.5


class TestMotionBlurIntegration:
    """Integration tests for motion blur rendering."""

    def test_moving_sphere_in_scene(self):
        """Test that a moving sphere can be hit at different times."""
        world = HittableList()
        world.add(MovingSphere(
            center0=Point3(-1, 0, 0),
            center1=Point3(1, 0, 0),
            time0=0.0,
            time1=1.0,
            radius=0.5
        ))

        # At time=0, sphere is at x=-1
        ray0 = Ray(Point3(-1, 0, -5), Vec3(0, 0, 1), time=0.0)
        hit0 = world.hit(ray0, 0.001, float('inf'))
        assert hit0 is not None

        # At time=1, sphere is at x=1
        ray1 = Ray(Point3(1, 0, -5), Vec3(0, 0, 1), time=1.0)
        hit1 = world.hit(ray1, 0.001, float('inf'))
        assert hit1 is not None

        # At time=0, x=1 position should miss
        ray_miss = Ray(Point3(1, 0, -5), Vec3(0, 0, 1), time=0.0)
        hit_miss = world.hit(ray_miss, 0.001, float('inf'))
        assert hit_miss is None

    def test_static_and_moving_spheres(self):
        """Test scene with both static and moving objects."""
        world = HittableList()

        # Static sphere
        from spectraforge.shapes import Sphere
        world.add(Sphere(Point3(0, 0, 0), 0.5))

        # Moving sphere
        world.add(MovingSphere(
            center0=Point3(2, 0, 0),
            center1=Point3(4, 0, 0),
            time0=0.0,
            time1=1.0,
            radius=0.5
        ))

        # Static sphere should be hit regardless of time
        ray0 = Ray(Point3(0, 0, -5), Vec3(0, 0, 1), time=0.0)
        ray1 = Ray(Point3(0, 0, -5), Vec3(0, 0, 1), time=1.0)

        assert world.hit(ray0, 0.001, float('inf')) is not None
        assert world.hit(ray1, 0.001, float('inf')) is not None


class TestEdgeCases:
    """Test edge cases for motion blur."""

    def test_zero_time_range(self):
        """MovingSphere with same start/end time behaves like static sphere."""
        sphere = MovingSphere(
            center0=Point3(0, 0, 0),
            center1=Point3(1, 0, 0),  # Different positions
            time0=0.5,
            time1=0.5,  # Same time = static
            radius=0.5
        )

        # Should always be at center0
        c = sphere.center(0.5)
        assert c == Point3(0, 0, 0)

    def test_time_outside_range(self):
        """Test behavior with time outside defined range."""
        sphere = MovingSphere(
            center0=Point3(0, 0, 0),
            center1=Point3(2, 0, 0),
            time0=0.0,
            time1=1.0,
            radius=0.5
        )

        # Time=2 extrapolates beyond center1
        c = sphere.center(2.0)
        assert c.x > 2.0  # Extrapolated

        # Time=-1 extrapolates before center0
        c_neg = sphere.center(-1.0)
        assert c_neg.x < 0.0

    def test_very_fast_motion(self):
        """Test sphere moving very fast."""
        sphere = MovingSphere(
            center0=Point3(0, 0, 0),
            center1=Point3(100, 0, 0),
            time0=0.0,
            time1=1.0,
            radius=0.5
        )

        bbox = sphere.bounding_box()
        # Bounding box should be very wide
        assert bbox.maximum.x - bbox.minimum.x > 99
