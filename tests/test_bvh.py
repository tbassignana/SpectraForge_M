"""Tests for BVH acceleration structure."""

import pytest
import random
from spectraforge.vec3 import Vec3, Point3, Color
from spectraforge.ray import Ray
from spectraforge.shapes import Sphere, HittableList
from spectraforge.materials import Lambertian
from spectraforge.bvh import BVH, BVHNode, build_bvh


class TestBVHNode:
    """Test BVHNode class."""

    def test_single_object(self):
        sphere = Sphere(Point3(0, 0, 0), 1.0)
        node = BVHNode([sphere], 0, 1)

        assert node.left is sphere
        assert node.right is None
        assert node.bbox is not None

    def test_two_objects(self):
        s1 = Sphere(Point3(-2, 0, 0), 1.0)
        s2 = Sphere(Point3(2, 0, 0), 1.0)
        node = BVHNode([s1, s2], 0, 2)

        assert node.left is not None
        assert node.right is not None
        assert node.bbox is not None

    def test_many_objects(self):
        spheres = [Sphere(Point3(i, 0, 0), 0.5) for i in range(100)]
        node = BVHNode(spheres, 0, 100)

        assert node.bbox is not None
        # Should have split recursively
        assert node.left is not None


class TestBVH:
    """Test BVH class."""

    def test_empty_scene(self):
        bvh = BVH([])
        ray = Ray(Point3(0, 0, 0), Vec3(1, 0, 0))
        hit = bvh.hit(ray, 0.001, float('inf'))
        assert hit is None

    def test_single_sphere(self):
        sphere = Sphere(Point3(0, 0, -5), 1.0)
        bvh = BVH([sphere])

        ray = Ray(Point3(0, 0, 0), Vec3(0, 0, -1))
        hit = bvh.hit(ray, 0.001, float('inf'))

        assert hit is not None
        assert abs(hit.t - 4.0) < 1e-6

    def test_multiple_spheres_finds_closest(self):
        spheres = [
            Sphere(Point3(0, 0, -5), 1.0),
            Sphere(Point3(0, 0, -10), 1.0),
            Sphere(Point3(0, 0, -15), 1.0),
        ]
        bvh = BVH(spheres)

        ray = Ray(Point3(0, 0, 0), Vec3(0, 0, -1))
        hit = bvh.hit(ray, 0.001, float('inf'))

        assert hit is not None
        assert abs(hit.t - 4.0) < 1e-6  # Closest sphere

    def test_miss(self):
        spheres = [
            Sphere(Point3(0, 0, -5), 1.0),
            Sphere(Point3(5, 0, -5), 1.0),
        ]
        bvh = BVH(spheres)

        ray = Ray(Point3(0, 10, 0), Vec3(1, 0, 0))  # Ray above all spheres
        hit = bvh.hit(ray, 0.001, float('inf'))

        assert hit is None

    def test_bvh_matches_hittable_list(self):
        """BVH should produce same results as linear search."""
        random.seed(42)
        spheres = [
            Sphere(Point3(random.uniform(-10, 10),
                          random.uniform(-10, 10),
                          random.uniform(-10, 10)), 0.5)
            for _ in range(50)
        ]

        bvh = BVH(spheres)
        linear = HittableList(spheres)

        # Test many random rays
        for _ in range(100):
            origin = Point3(0, 0, 20)
            direction = Vec3(
                random.uniform(-1, 1),
                random.uniform(-1, 1),
                random.uniform(-2, -0.5)
            ).normalize()
            ray = Ray(origin, direction)

            bvh_hit = bvh.hit(ray, 0.001, float('inf'))
            linear_hit = linear.hit(ray, 0.001, float('inf'))

            if bvh_hit is None:
                assert linear_hit is None
            else:
                assert linear_hit is not None
                assert abs(bvh_hit.t - linear_hit.t) < 1e-6

    def test_bounding_box(self):
        spheres = [
            Sphere(Point3(-5, 0, 0), 1.0),
            Sphere(Point3(5, 0, 0), 1.0),
        ]
        bvh = BVH(spheres)

        bbox = bvh.bounding_box()
        assert bbox is not None
        assert bbox.minimum.x == -6
        assert bbox.maximum.x == 6

    def test_len(self):
        spheres = [Sphere(Point3(i, 0, 0), 0.5) for i in range(10)]
        bvh = BVH(spheres)
        assert len(bvh) == 10


class TestBuildBVH:
    """Test build_bvh convenience function."""

    def test_build_from_hittable_list(self):
        world = HittableList()
        world.add(Sphere(Point3(0, 0, -5), 1.0))
        world.add(Sphere(Point3(0, 0, -10), 1.0))

        bvh = build_bvh(world)

        ray = Ray(Point3(0, 0, 0), Vec3(0, 0, -1))
        hit = bvh.hit(ray, 0.001, float('inf'))

        assert hit is not None


class TestBVHPerformance:
    """Test BVH performance characteristics."""

    def test_large_scene(self):
        """Test BVH handles large scenes."""
        random.seed(42)
        spheres = [
            Sphere(Point3(
                random.uniform(-50, 50),
                random.uniform(-50, 50),
                random.uniform(-100, -10)
            ), 0.5)
            for _ in range(1000)
        ]

        # Should build without error
        bvh = BVH(spheres, use_parallel=False)

        # Should be able to trace rays
        ray = Ray(Point3(0, 0, 0), Vec3(0, 0, -1))
        hit = bvh.hit(ray, 0.001, float('inf'))
        # May or may not hit depending on random positions

    def test_parallel_build(self):
        """Test parallel BVH construction."""
        random.seed(42)
        spheres = [
            Sphere(Point3(
                random.uniform(-50, 50),
                random.uniform(-50, 50),
                random.uniform(-100, -10)
            ), 0.5)
            for _ in range(2000)
        ]

        # Build with parallel
        bvh_parallel = BVH(spheres.copy(), use_parallel=True)

        # Build without parallel
        bvh_serial = BVH(spheres.copy(), use_parallel=False)

        # Both should work
        ray = Ray(Point3(0, 0, 0), Vec3(0, 0, -1))
        hit1 = bvh_parallel.hit(ray, 0.001, float('inf'))
        hit2 = bvh_serial.hit(ray, 0.001, float('inf'))

        if hit1 is not None and hit2 is not None:
            assert abs(hit1.t - hit2.t) < 1e-6


class TestBVHWithMaterials:
    """Test BVH preserves material information."""

    def test_material_preserved(self):
        mat = Lambertian(Color(1, 0, 0))
        sphere = Sphere(Point3(0, 0, -5), 1.0, mat)
        bvh = BVH([sphere])

        ray = Ray(Point3(0, 0, 0), Vec3(0, 0, -1))
        hit = bvh.hit(ray, 0.001, float('inf'))

        assert hit is not None
        assert hit.material is mat
