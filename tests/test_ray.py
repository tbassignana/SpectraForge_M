"""Tests for Ray class."""

import pytest
from spectraforge.vec3 import Vec3, Point3
from spectraforge.ray import Ray


class TestRayCreation:
    """Test Ray construction."""

    def test_default_time(self):
        origin = Point3(0, 0, 0)
        direction = Vec3(1, 0, 0)
        ray = Ray(origin, direction)
        assert ray.time == 0.0

    def test_with_time(self):
        origin = Point3(0, 0, 0)
        direction = Vec3(1, 0, 0)
        ray = Ray(origin, direction, 0.5)
        assert ray.time == 0.5

    def test_stores_origin(self):
        origin = Point3(1, 2, 3)
        direction = Vec3(1, 0, 0)
        ray = Ray(origin, direction)
        assert ray.origin == origin

    def test_stores_direction(self):
        origin = Point3(0, 0, 0)
        direction = Vec3(1, 2, 3)
        ray = Ray(origin, direction)
        assert ray.direction == direction


class TestRayAt:
    """Test Ray.at() method."""

    def test_at_zero(self):
        origin = Point3(1, 2, 3)
        direction = Vec3(1, 0, 0)
        ray = Ray(origin, direction)
        point = ray.at(0)
        assert point == origin

    def test_at_positive(self):
        origin = Point3(0, 0, 0)
        direction = Vec3(1, 0, 0)
        ray = Ray(origin, direction)
        point = ray.at(5)
        assert point.x == 5
        assert point.y == 0
        assert point.z == 0

    def test_at_negative(self):
        origin = Point3(0, 0, 0)
        direction = Vec3(1, 0, 0)
        ray = Ray(origin, direction)
        point = ray.at(-5)
        assert point.x == -5

    def test_at_with_diagonal(self):
        origin = Point3(0, 0, 0)
        direction = Vec3(1, 1, 1)
        ray = Ray(origin, direction)
        point = ray.at(2)
        assert point.x == 2
        assert point.y == 2
        assert point.z == 2


class TestRayRepr:
    """Test Ray string representation."""

    def test_repr(self):
        origin = Point3(1, 2, 3)
        direction = Vec3(0, 1, 0)
        ray = Ray(origin, direction)
        s = repr(ray)
        assert "Ray" in s
        assert "origin" in s
        assert "direction" in s
