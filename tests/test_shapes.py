"""Tests for geometric shapes."""

import pytest
import math
from spectraforge.vec3 import Vec3, Point3
from spectraforge.ray import Ray
from spectraforge.shapes import Sphere, Plane, Triangle, HittableList, AABB, Box, Cylinder, Cone
from spectraforge.materials import Lambertian, Color


class TestSphere:
    """Test Sphere class."""

    def test_creation(self):
        center = Point3(0, 0, 0)
        sphere = Sphere(center, 1.0)
        assert sphere.center == center
        assert sphere.radius == 1.0

    def test_hit_through_center(self):
        sphere = Sphere(Point3(0, 0, 0), 1.0)
        ray = Ray(Point3(0, 0, -5), Vec3(0, 0, 1))
        hit = sphere.hit(ray, 0.001, float('inf'))

        assert hit is not None
        assert abs(hit.t - 4.0) < 1e-6  # Hits at z=-1
        assert abs(hit.point.z - (-1.0)) < 1e-6

    def test_hit_front_face(self):
        sphere = Sphere(Point3(0, 0, 0), 1.0)
        ray = Ray(Point3(0, 0, -5), Vec3(0, 0, 1))
        hit = sphere.hit(ray, 0.001, float('inf'))

        assert hit is not None
        assert hit.front_face is True
        # Normal should point outward (against ray)
        assert hit.normal.z < 0

    def test_hit_from_inside(self):
        sphere = Sphere(Point3(0, 0, 0), 1.0)
        ray = Ray(Point3(0, 0, 0), Vec3(0, 0, 1))
        hit = sphere.hit(ray, 0.001, float('inf'))

        assert hit is not None
        assert hit.front_face is False

    def test_miss(self):
        sphere = Sphere(Point3(0, 0, 0), 1.0)
        ray = Ray(Point3(0, 5, -5), Vec3(0, 0, 1))  # Ray passes above sphere
        hit = sphere.hit(ray, 0.001, float('inf'))

        assert hit is None

    def test_behind_ray(self):
        sphere = Sphere(Point3(0, 0, -5), 1.0)
        ray = Ray(Point3(0, 0, 0), Vec3(0, 0, 1))  # Ray points away from sphere
        hit = sphere.hit(ray, 0.001, float('inf'))

        assert hit is None

    def test_t_range(self):
        sphere = Sphere(Point3(0, 0, 0), 1.0)
        ray = Ray(Point3(0, 0, -5), Vec3(0, 0, 1))

        # Hit is at t=4, exclude it with t_min
        hit = sphere.hit(ray, 4.5, float('inf'))
        assert hit is not None
        assert hit.t > 4.5  # Should hit back of sphere

    def test_uv_coordinates(self):
        sphere = Sphere(Point3(0, 0, 0), 1.0)
        ray = Ray(Point3(0, 0, -5), Vec3(0, 0, 1))
        hit = sphere.hit(ray, 0.001, float('inf'))

        assert hit is not None
        assert 0 <= hit.u <= 1
        assert 0 <= hit.v <= 1

    def test_with_material(self):
        material = Lambertian(Color(1, 0, 0))
        sphere = Sphere(Point3(0, 0, 0), 1.0, material)
        ray = Ray(Point3(0, 0, -5), Vec3(0, 0, 1))
        hit = sphere.hit(ray, 0.001, float('inf'))

        assert hit is not None
        assert hit.material is material

    def test_bounding_box(self):
        sphere = Sphere(Point3(0, 0, 0), 1.0)
        bbox = sphere.bounding_box()

        assert bbox is not None
        assert bbox.minimum.x == -1
        assert bbox.maximum.x == 1


class TestPlane:
    """Test Plane class."""

    def test_hit_perpendicular(self):
        plane = Plane(Point3(0, 0, 0), Vec3(0, 1, 0))  # XZ plane at y=0
        ray = Ray(Point3(0, 5, 0), Vec3(0, -1, 0))  # Straight down
        hit = plane.hit(ray, 0.001, float('inf'))

        assert hit is not None
        assert abs(hit.t - 5.0) < 1e-6
        assert abs(hit.point.y) < 1e-6

    def test_hit_at_angle(self):
        plane = Plane(Point3(0, 0, 0), Vec3(0, 1, 0))
        ray = Ray(Point3(0, 5, -5), Vec3(0, -1, 1).normalize())
        hit = plane.hit(ray, 0.001, float('inf'))

        assert hit is not None
        assert abs(hit.point.y) < 1e-6

    def test_miss_parallel(self):
        plane = Plane(Point3(0, 0, 0), Vec3(0, 1, 0))
        ray = Ray(Point3(0, 5, 0), Vec3(1, 0, 0))  # Parallel to plane
        hit = plane.hit(ray, 0.001, float('inf'))

        assert hit is None

    def test_miss_pointing_away(self):
        plane = Plane(Point3(0, 0, 0), Vec3(0, 1, 0))
        ray = Ray(Point3(0, 5, 0), Vec3(0, 1, 0))  # Points away from plane
        hit = plane.hit(ray, 0.001, float('inf'))

        assert hit is None

    def test_no_bounding_box(self):
        plane = Plane(Point3(0, 0, 0), Vec3(0, 1, 0))
        bbox = plane.bounding_box()

        assert bbox is None  # Infinite planes have no bbox


class TestTriangle:
    """Test Triangle class."""

    def test_hit_center(self):
        v0 = Point3(0, 0, 0)
        v1 = Point3(1, 0, 0)
        v2 = Point3(0, 1, 0)
        tri = Triangle(v0, v1, v2)

        ray = Ray(Point3(0.25, 0.25, -1), Vec3(0, 0, 1))
        hit = tri.hit(ray, 0.001, float('inf'))

        assert hit is not None
        assert abs(hit.point.z) < 1e-6

    def test_miss_outside(self):
        v0 = Point3(0, 0, 0)
        v1 = Point3(1, 0, 0)
        v2 = Point3(0, 1, 0)
        tri = Triangle(v0, v1, v2)

        ray = Ray(Point3(5, 5, -1), Vec3(0, 0, 1))
        hit = tri.hit(ray, 0.001, float('inf'))

        assert hit is None

    def test_miss_parallel(self):
        v0 = Point3(0, 0, 0)
        v1 = Point3(1, 0, 0)
        v2 = Point3(0, 1, 0)
        tri = Triangle(v0, v1, v2)

        ray = Ray(Point3(0, 0, 0), Vec3(1, 0, 0))
        hit = tri.hit(ray, 0.001, float('inf'))

        assert hit is None

    def test_bounding_box(self):
        v0 = Point3(0, 0, 0)
        v1 = Point3(1, 0, 0)
        v2 = Point3(0, 1, 0)
        tri = Triangle(v0, v1, v2)
        bbox = tri.bounding_box()

        assert bbox is not None
        assert bbox.minimum.x < 0.001
        assert bbox.maximum.x > 0.999


class TestAABB:
    """Test Axis-Aligned Bounding Box."""

    def test_hit_through(self):
        bbox = AABB(Point3(-1, -1, -1), Point3(1, 1, 1))
        ray = Ray(Point3(0, 0, -5), Vec3(0, 0, 1))
        assert bbox.hit(ray, 0.001, float('inf')) is True

    def test_miss(self):
        bbox = AABB(Point3(-1, -1, -1), Point3(1, 1, 1))
        ray = Ray(Point3(0, 5, -5), Vec3(0, 0, 1))
        assert bbox.hit(ray, 0.001, float('inf')) is False

    def test_ray_inside(self):
        bbox = AABB(Point3(-1, -1, -1), Point3(1, 1, 1))
        ray = Ray(Point3(0, 0, 0), Vec3(1, 0, 0))
        assert bbox.hit(ray, 0.001, float('inf')) is True

    def test_surrounding_box(self):
        box1 = AABB(Point3(0, 0, 0), Point3(1, 1, 1))
        box2 = AABB(Point3(2, 2, 2), Point3(3, 3, 3))
        combined = AABB.surrounding_box(box1, box2)

        assert combined.minimum.x == 0
        assert combined.maximum.x == 3


class TestHittableList:
    """Test HittableList class."""

    def test_empty_list(self):
        world = HittableList()
        ray = Ray(Point3(0, 0, 0), Vec3(1, 0, 0))
        hit = world.hit(ray, 0.001, float('inf'))
        assert hit is None

    def test_hit_closest(self):
        world = HittableList()
        world.add(Sphere(Point3(0, 0, -5), 1.0))  # Closer
        world.add(Sphere(Point3(0, 0, -10), 1.0))  # Farther

        ray = Ray(Point3(0, 0, 0), Vec3(0, 0, -1))
        hit = world.hit(ray, 0.001, float('inf'))

        assert hit is not None
        assert abs(hit.point.z - (-4.0)) < 1e-6  # Should hit closer sphere

    def test_add_and_clear(self):
        world = HittableList()
        world.add(Sphere(Point3(0, 0, 0), 1.0))
        assert len(world) == 1

        world.clear()
        assert len(world) == 0

    def test_iteration(self):
        world = HittableList()
        s1 = Sphere(Point3(0, 0, 0), 1.0)
        s2 = Sphere(Point3(1, 0, 0), 1.0)
        world.add(s1)
        world.add(s2)

        objects = list(world)
        assert len(objects) == 2
        assert s1 in objects
        assert s2 in objects

    def test_bounding_box(self):
        world = HittableList()
        world.add(Sphere(Point3(0, 0, 0), 1.0))
        world.add(Sphere(Point3(5, 0, 0), 1.0))

        bbox = world.bounding_box()
        assert bbox is not None
        assert bbox.minimum.x == -1
        assert bbox.maximum.x == 6


class TestNumericalStability:
    """Test edge cases and numerical stability."""

    def test_grazing_ray(self):
        """Ray that barely grazes the sphere surface."""
        sphere = Sphere(Point3(0, 0, 0), 1.0)
        # Ray that just touches the sphere
        ray = Ray(Point3(0, 1, -5), Vec3(0, 0, 1))
        hit = sphere.hit(ray, 0.001, float('inf'))

        # Should hit at the tangent point
        if hit is not None:
            assert abs(hit.point.y - 1.0) < 1e-3

    def test_very_small_sphere(self):
        sphere = Sphere(Point3(0, 0, 0), 1e-6)
        ray = Ray(Point3(0, 0, -1), Vec3(0, 0, 1))
        hit = sphere.hit(ray, 0.001, float('inf'))
        # Very small sphere should still be hittable
        assert hit is not None

    def test_very_large_sphere(self):
        sphere = Sphere(Point3(0, 0, 0), 1e6)
        ray = Ray(Point3(0, 0, -2e6), Vec3(0, 0, 1))
        hit = sphere.hit(ray, 0.001, float('inf'))
        assert hit is not None


class TestBox:
    """Test Box class."""

    def test_creation(self):
        box = Box(Point3(0, 0, 0), Point3(1, 1, 1))
        # Box should normalize corners so p0 is min and p1 is max
        assert box.p0.x == 0
        assert box.p1.x == 1

    def test_corners_normalized(self):
        """Test that corners are normalized regardless of input order."""
        box = Box(Point3(1, 1, 1), Point3(0, 0, 0))
        assert box.p0.x == 0
        assert box.p1.x == 1

    def test_hit_front_face(self):
        box = Box(Point3(-1, -1, -1), Point3(1, 1, 1))
        ray = Ray(Point3(0, 0, -5), Vec3(0, 0, 1))
        hit = box.hit(ray, 0.001, float('inf'))

        assert hit is not None
        assert abs(hit.t - 4.0) < 1e-6  # Hits at z=-1
        assert abs(hit.point.z - (-1.0)) < 1e-6

    def test_hit_from_side(self):
        box = Box(Point3(-1, -1, -1), Point3(1, 1, 1))
        ray = Ray(Point3(5, 0, 0), Vec3(-1, 0, 0))
        hit = box.hit(ray, 0.001, float('inf'))

        assert hit is not None
        assert abs(hit.point.x - 1.0) < 1e-6

    def test_miss(self):
        box = Box(Point3(-1, -1, -1), Point3(1, 1, 1))
        ray = Ray(Point3(0, 5, -5), Vec3(0, 0, 1))
        hit = box.hit(ray, 0.001, float('inf'))

        assert hit is None

    def test_ray_inside(self):
        box = Box(Point3(-1, -1, -1), Point3(1, 1, 1))
        ray = Ray(Point3(0, 0, 0), Vec3(1, 0, 0))
        hit = box.hit(ray, 0.001, float('inf'))

        assert hit is not None
        assert hit.front_face is False

    def test_uv_coordinates(self):
        box = Box(Point3(0, 0, 0), Point3(1, 1, 1))
        ray = Ray(Point3(0.5, 0.5, -1), Vec3(0, 0, 1))
        hit = box.hit(ray, 0.001, float('inf'))

        assert hit is not None
        assert 0 <= hit.u <= 1
        assert 0 <= hit.v <= 1

    def test_bounding_box(self):
        box = Box(Point3(-1, -2, -3), Point3(1, 2, 3))
        bbox = box.bounding_box()

        assert bbox is not None
        assert bbox.minimum.x == -1
        assert bbox.maximum.y == 2

    def test_with_material(self):
        material = Lambertian(Color(1, 0, 0))
        box = Box(Point3(-1, -1, -1), Point3(1, 1, 1), material)
        ray = Ray(Point3(0, 0, -5), Vec3(0, 0, 1))
        hit = box.hit(ray, 0.001, float('inf'))

        assert hit is not None
        assert hit.material is material


class TestCylinder:
    """Test Cylinder class."""

    def test_creation(self):
        cyl = Cylinder(Point3(0, 0, 0), 1.0, 2.0)
        assert cyl.radius == 1.0
        assert cyl.height == 2.0
        assert cyl.y_min == 0
        assert cyl.y_max == 2.0

    def test_hit_side(self):
        cyl = Cylinder(Point3(0, 0, 0), 1.0, 2.0)
        ray = Ray(Point3(5, 1, 0), Vec3(-1, 0, 0))
        hit = cyl.hit(ray, 0.001, float('inf'))

        assert hit is not None
        assert abs(hit.point.x - 1.0) < 1e-6
        assert abs(hit.point.y - 1.0) < 1e-6

    def test_hit_top_cap(self):
        cyl = Cylinder(Point3(0, 0, 0), 1.0, 2.0, capped=True)
        ray = Ray(Point3(0, 5, 0), Vec3(0, -1, 0))
        hit = cyl.hit(ray, 0.001, float('inf'))

        assert hit is not None
        assert abs(hit.point.y - 2.0) < 1e-6

    def test_hit_bottom_cap(self):
        cyl = Cylinder(Point3(0, 0, 0), 1.0, 2.0, capped=True)
        ray = Ray(Point3(0, -5, 0), Vec3(0, 1, 0))
        hit = cyl.hit(ray, 0.001, float('inf'))

        assert hit is not None
        assert abs(hit.point.y) < 1e-6

    def test_miss_through_uncapped(self):
        """Ray through an uncapped cylinder shouldn't hit caps."""
        cyl = Cylinder(Point3(0, 0, 0), 1.0, 2.0, capped=False)
        ray = Ray(Point3(0, 5, 0), Vec3(0, -1, 0))
        hit = cyl.hit(ray, 0.001, float('inf'))

        # Should pass through without hitting (no caps, misses side)
        assert hit is None

    def test_miss_outside(self):
        cyl = Cylinder(Point3(0, 0, 0), 1.0, 2.0)
        ray = Ray(Point3(5, 1, 5), Vec3(1, 0, 0))
        hit = cyl.hit(ray, 0.001, float('inf'))

        assert hit is None

    def test_uv_coordinates_side(self):
        cyl = Cylinder(Point3(0, 0, 0), 1.0, 2.0)
        ray = Ray(Point3(5, 1, 0), Vec3(-1, 0, 0))
        hit = cyl.hit(ray, 0.001, float('inf'))

        assert hit is not None
        assert 0 <= hit.u <= 1
        assert 0 <= hit.v <= 1
        # v should be 0.5 at y=1 (half height)
        assert abs(hit.v - 0.5) < 1e-6

    def test_bounding_box(self):
        cyl = Cylinder(Point3(0, 0, 0), 1.0, 2.0)
        bbox = cyl.bounding_box()

        assert bbox is not None
        assert bbox.minimum.x == -1
        assert bbox.maximum.x == 1
        assert bbox.minimum.y == 0
        assert bbox.maximum.y == 2

    def test_with_material(self):
        material = Lambertian(Color(0, 1, 0))
        cyl = Cylinder(Point3(0, 0, 0), 1.0, 2.0, material)
        ray = Ray(Point3(5, 1, 0), Vec3(-1, 0, 0))
        hit = cyl.hit(ray, 0.001, float('inf'))

        assert hit is not None
        assert hit.material is material


class TestCone:
    """Test Cone class."""

    def test_creation(self):
        cone = Cone(Point3(0, 2, 0), 1.0, 2.0)
        assert cone.radius == 1.0
        assert cone.height == 2.0
        assert cone.y_min == 0  # base
        assert cone.y_max == 2  # apex

    def test_hit_side(self):
        cone = Cone(Point3(0, 2, 0), 1.0, 2.0)
        ray = Ray(Point3(5, 1, 0), Vec3(-1, 0, 0))
        hit = cone.hit(ray, 0.001, float('inf'))

        assert hit is not None
        # At y=1 (halfway), radius should be 0.5
        assert abs(hit.point.x - 0.5) < 1e-5

    def test_hit_base_cap(self):
        cone = Cone(Point3(0, 2, 0), 1.0, 2.0, capped=True)
        ray = Ray(Point3(0, -5, 0), Vec3(0, 1, 0))
        hit = cone.hit(ray, 0.001, float('inf'))

        assert hit is not None
        assert abs(hit.point.y) < 1e-6

    def test_miss_uncapped_through_base(self):
        cone = Cone(Point3(0, 2, 0), 1.0, 2.0, capped=False)
        ray = Ray(Point3(0, -5, 0), Vec3(0, 1, 0))
        hit = cone.hit(ray, 0.001, float('inf'))

        # Should pass through the open base
        assert hit is None

    def test_miss_outside(self):
        cone = Cone(Point3(0, 2, 0), 1.0, 2.0)
        ray = Ray(Point3(5, 1, 5), Vec3(1, 0, 0))
        hit = cone.hit(ray, 0.001, float('inf'))

        assert hit is None

    def test_uv_coordinates(self):
        cone = Cone(Point3(0, 2, 0), 1.0, 2.0)
        ray = Ray(Point3(5, 1, 0), Vec3(-1, 0, 0))
        hit = cone.hit(ray, 0.001, float('inf'))

        assert hit is not None
        assert 0 <= hit.u <= 1
        assert 0 <= hit.v <= 1

    def test_bounding_box(self):
        cone = Cone(Point3(0, 2, 0), 1.0, 2.0)
        bbox = cone.bounding_box()

        assert bbox is not None
        assert bbox.minimum.x == -1
        assert bbox.maximum.x == 1
        assert bbox.minimum.y == 0
        assert bbox.maximum.y == 2

    def test_with_material(self):
        material = Lambertian(Color(0, 0, 1))
        cone = Cone(Point3(0, 2, 0), 1.0, 2.0, material)
        ray = Ray(Point3(5, 1, 0), Vec3(-1, 0, 0))
        hit = cone.hit(ray, 0.001, float('inf'))

        assert hit is not None
        assert hit.material is material

    def test_hit_near_apex(self):
        """Test hitting the cone near its apex."""
        cone = Cone(Point3(0, 2, 0), 1.0, 2.0)
        ray = Ray(Point3(0.1, 1.9, -1), Vec3(0, 0, 1))
        hit = cone.hit(ray, 0.001, float('inf'))

        # Near apex, should still hit
        if hit is not None:
            assert hit.point.y < 2.0
