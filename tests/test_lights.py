"""Tests for lighting system."""

import pytest
import math
from spectraforge.vec3 import Vec3, Point3, Color
from spectraforge.lights import (
    PointLight, DirectionalLight, AreaLight, SphereLight, LightList
)


class TestPointLight:
    """Test PointLight class."""

    def test_sample_direction(self):
        light = PointLight(Point3(0, 10, 0), Color(1, 1, 1), 1.0)
        hit_point = Point3(0, 0, 0)
        sample = light.sample(hit_point)

        # Direction should point up
        assert sample.direction.y > 0.9
        assert abs(sample.direction.x) < 0.1
        assert abs(sample.direction.z) < 0.1

    def test_sample_distance(self):
        light = PointLight(Point3(0, 10, 0), Color(1, 1, 1), 1.0)
        hit_point = Point3(0, 0, 0)
        sample = light.sample(hit_point)

        assert abs(sample.distance - 10.0) < 1e-6

    def test_inverse_square_falloff(self):
        light = PointLight(Point3(0, 0, 0), Color(1, 1, 1), 100.0)

        sample1 = light.sample(Point3(1, 0, 0))
        sample2 = light.sample(Point3(2, 0, 0))

        # At 2x distance, intensity should be 1/4
        ratio = sample1.intensity.r / sample2.intensity.r
        assert abs(ratio - 4.0) < 1e-6

    def test_power(self):
        light = PointLight(Point3(0, 0, 0), Color(1, 0.5, 0.5), 10.0)
        assert light.power() > 0


class TestDirectionalLight:
    """Test DirectionalLight class."""

    def test_direction_is_inverted(self):
        # Light coming from above
        light = DirectionalLight(Vec3(0, -1, 0), Color(1, 1, 1), 1.0)
        hit_point = Point3(0, 0, 0)
        sample = light.sample(hit_point)

        # Sample direction should point UP (toward light)
        assert sample.direction.y > 0.9

    def test_infinite_distance(self):
        light = DirectionalLight(Vec3(0, -1, 0), Color(1, 1, 1), 1.0)
        sample = light.sample(Point3(0, 0, 0))
        assert sample.distance == float('inf')

    def test_no_falloff(self):
        light = DirectionalLight(Vec3(0, -1, 0), Color(1, 1, 1), 5.0)

        sample1 = light.sample(Point3(0, 0, 0))
        sample2 = light.sample(Point3(1000, 0, 0))

        # Same intensity regardless of position
        assert sample1.intensity.r == sample2.intensity.r


class TestAreaLight:
    """Test AreaLight class."""

    def test_sample_within_rectangle(self):
        corner = Point3(0, 5, -1)
        edge1 = Vec3(2, 0, 0)
        edge2 = Vec3(0, 0, 2)
        light = AreaLight(corner, edge1, edge2, Color(1, 1, 1), 1.0)

        hit_point = Point3(1, 0, 0)

        # Sample many times
        for _ in range(100):
            sample = light.sample(hit_point)
            # Direction should generally point up
            assert sample.direction.y > 0

    def test_normal_computed(self):
        corner = Point3(0, 5, 0)
        edge1 = Vec3(1, 0, 0)
        edge2 = Vec3(0, 0, 1)
        light = AreaLight(corner, edge1, edge2, Color(1, 1, 1), 1.0)

        # Normal should point down (cross product of edges)
        assert abs(light.normal.y - (-1)) < 1e-6 or abs(light.normal.y - 1) < 1e-6

    def test_area_computed(self):
        corner = Point3(0, 0, 0)
        edge1 = Vec3(2, 0, 0)
        edge2 = Vec3(0, 0, 3)
        light = AreaLight(corner, edge1, edge2, Color(1, 1, 1), 1.0)

        assert abs(light.area - 6.0) < 1e-6


class TestSphereLight:
    """Test SphereLight class."""

    def test_sample_toward_center(self):
        light = SphereLight(Point3(0, 10, 0), 1.0, Color(1, 1, 1), 1.0)
        hit_point = Point3(0, 0, 0)

        for _ in range(100):
            sample = light.sample(hit_point)
            # Direction should point generally toward light
            assert sample.direction.y > 0

    def test_power_scales_with_area(self):
        light1 = SphereLight(Point3(0, 0, 0), 1.0, Color(1, 1, 1), 1.0)
        light2 = SphereLight(Point3(0, 0, 0), 2.0, Color(1, 1, 1), 1.0)

        # 2x radius = 4x area = 4x power
        ratio = light2.power() / light1.power()
        assert abs(ratio - 4.0) < 1e-6


class TestLightList:
    """Test LightList class."""

    def test_empty_list(self):
        lights = LightList()
        sample, idx = lights.sample(Point3(0, 0, 0))
        assert idx == -1

    def test_single_light(self):
        lights = LightList()
        lights.add(PointLight(Point3(0, 10, 0), Color(1, 1, 1), 1.0))

        sample, idx = lights.sample(Point3(0, 0, 0))
        assert idx == 0
        assert sample.direction.y > 0

    def test_power_weighted_sampling(self):
        lights = LightList()
        # Add weak light
        lights.add(PointLight(Point3(-10, 5, 0), Color(1, 1, 1), 1.0))
        # Add strong light
        lights.add(PointLight(Point3(10, 5, 0), Color(1, 1, 1), 100.0))

        # Sample many times
        selections = [0, 0]
        for _ in range(1000):
            _, idx = lights.sample(Point3(0, 0, 0))
            selections[idx] += 1

        # Strong light should be selected more often
        assert selections[1] > selections[0] * 5

    def test_len(self):
        lights = LightList()
        assert len(lights) == 0

        lights.add(PointLight(Point3(0, 0, 0), Color(1, 1, 1), 1.0))
        lights.add(PointLight(Point3(1, 0, 0), Color(1, 1, 1), 1.0))
        assert len(lights) == 2

    def test_iteration(self):
        lights = LightList()
        l1 = PointLight(Point3(0, 0, 0), Color(1, 1, 1), 1.0)
        l2 = PointLight(Point3(1, 0, 0), Color(1, 1, 1), 1.0)
        lights.add(l1)
        lights.add(l2)

        light_list = list(lights)
        assert l1 in light_list
        assert l2 in light_list
