"""Tests for Vec3 class."""

import pytest
import math
import numpy as np

from spectraforge.vec3 import Vec3, Point3, Color


class TestVec3Creation:
    """Test Vec3 construction."""

    def test_default_constructor(self):
        v = Vec3()
        assert v.x == 0.0
        assert v.y == 0.0
        assert v.z == 0.0

    def test_value_constructor(self):
        v = Vec3(1.0, 2.0, 3.0)
        assert v.x == 1.0
        assert v.y == 2.0
        assert v.z == 3.0

    def test_from_array(self):
        arr = np.array([1.0, 2.0, 3.0])
        v = Vec3.from_array(arr)
        assert v.x == 1.0
        assert v.y == 2.0
        assert v.z == 3.0

    def test_color_aliases(self):
        c = Color(0.5, 0.6, 0.7)
        assert c.r == 0.5
        assert c.g == 0.6
        assert c.b == 0.7


class TestVec3Arithmetic:
    """Test Vec3 arithmetic operations."""

    def test_negation(self):
        v = Vec3(1, 2, 3)
        neg = -v
        assert neg.x == -1
        assert neg.y == -2
        assert neg.z == -3

    def test_addition(self):
        v1 = Vec3(1, 2, 3)
        v2 = Vec3(4, 5, 6)
        result = v1 + v2
        assert result.x == 5
        assert result.y == 7
        assert result.z == 9

    def test_addition_scalar(self):
        v = Vec3(1, 2, 3)
        result = v + 10
        assert result.x == 11
        assert result.y == 12
        assert result.z == 13

    def test_subtraction(self):
        v1 = Vec3(4, 5, 6)
        v2 = Vec3(1, 2, 3)
        result = v1 - v2
        assert result.x == 3
        assert result.y == 3
        assert result.z == 3

    def test_multiplication(self):
        v = Vec3(1, 2, 3)
        result = v * 2
        assert result.x == 2
        assert result.y == 4
        assert result.z == 6

    def test_multiplication_vector(self):
        v1 = Vec3(1, 2, 3)
        v2 = Vec3(2, 3, 4)
        result = v1 * v2
        assert result.x == 2
        assert result.y == 6
        assert result.z == 12

    def test_division(self):
        v = Vec3(2, 4, 6)
        result = v / 2
        assert result.x == 1
        assert result.y == 2
        assert result.z == 3


class TestVec3VectorOps:
    """Test Vec3 vector operations."""

    def test_length(self):
        v = Vec3(3, 4, 0)
        assert v.length() == 5.0

    def test_length_squared(self):
        v = Vec3(3, 4, 0)
        assert v.length_squared() == 25.0

    def test_normalize(self):
        v = Vec3(3, 4, 0)
        n = v.normalize()
        assert abs(n.length() - 1.0) < 1e-10

    def test_normalize_zero_vector(self):
        v = Vec3(0, 0, 0)
        n = v.normalize()
        assert n.length() == 0.0

    def test_dot_product(self):
        v1 = Vec3(1, 0, 0)
        v2 = Vec3(0, 1, 0)
        assert v1.dot(v2) == 0.0

        v3 = Vec3(1, 2, 3)
        v4 = Vec3(4, 5, 6)
        assert v3.dot(v4) == 32.0  # 1*4 + 2*5 + 3*6

    def test_cross_product(self):
        v1 = Vec3(1, 0, 0)
        v2 = Vec3(0, 1, 0)
        cross = v1.cross(v2)
        assert cross.x == 0
        assert cross.y == 0
        assert cross.z == 1

    def test_reflect(self):
        # Ray coming in at 45 degrees
        incoming = Vec3(1, -1, 0).normalize()
        normal = Vec3(0, 1, 0)
        reflected = incoming.reflect(normal)
        expected = Vec3(1, 1, 0).normalize()
        assert abs(reflected.x - expected.x) < 1e-10
        assert abs(reflected.y - expected.y) < 1e-10


class TestVec3Refract:
    """Test Vec3 refraction."""

    def test_refract_air_to_glass(self):
        incoming = Vec3(0, -1, 0)  # Straight down
        normal = Vec3(0, 1, 0)
        eta_ratio = 1.0 / 1.5  # Air to glass
        refracted = incoming.refract(normal, eta_ratio)
        # Should bend towards normal
        assert refracted.y < 0

    def test_total_internal_reflection(self):
        # Coming from glass at steep angle
        incoming = Vec3(0.9, -0.1, 0).normalize()
        normal = Vec3(0, 1, 0)
        eta_ratio = 1.5  # Glass to air at steep angle
        refracted = incoming.refract(normal, eta_ratio)
        # Should return zero (TIR)
        assert refracted.length() < 1e-10


class TestVec3Utility:
    """Test Vec3 utility methods."""

    def test_near_zero(self):
        v1 = Vec3(1e-10, 1e-10, 1e-10)
        assert v1.near_zero()

        v2 = Vec3(1, 0, 0)
        assert not v2.near_zero()

    def test_clamp(self):
        v = Vec3(-0.5, 0.5, 1.5)
        clamped = v.clamp(0, 1)
        assert clamped.x == 0
        assert clamped.y == 0.5
        assert clamped.z == 1

    def test_gamma_correct(self):
        v = Vec3(0.5, 0.5, 0.5)
        corrected = v.gamma_correct(2.2)
        # sqrt(0.5) ≈ 0.707
        assert corrected.x > 0.7

    def test_to_array(self):
        v = Vec3(1, 2, 3)
        arr = v.to_array()
        assert isinstance(arr, np.ndarray)
        assert arr[0] == 1
        assert arr[1] == 2
        assert arr[2] == 3


class TestVec3Random:
    """Test Vec3 random generation."""

    def test_random(self):
        v = Vec3.random(0, 1)
        assert 0 <= v.x <= 1
        assert 0 <= v.y <= 1
        assert 0 <= v.z <= 1

    def test_random_in_unit_sphere(self):
        for _ in range(100):
            v = Vec3.random_in_unit_sphere()
            assert v.length_squared() < 1

    def test_random_unit_vector(self):
        for _ in range(100):
            v = Vec3.random_unit_vector()
            assert abs(v.length() - 1.0) < 1e-10

    def test_random_in_hemisphere(self):
        normal = Vec3(0, 1, 0)
        for _ in range(100):
            v = Vec3.random_in_hemisphere(normal)
            assert v.dot(normal) >= 0

    def test_random_in_unit_disk(self):
        for _ in range(100):
            v = Vec3.random_in_unit_disk()
            assert v.z == 0
            assert v.length_squared() < 1


class TestVec3Comparison:
    """Test Vec3 comparison operations."""

    def test_equality(self):
        v1 = Vec3(1, 2, 3)
        v2 = Vec3(1, 2, 3)
        assert v1 == v2

    def test_inequality(self):
        v1 = Vec3(1, 2, 3)
        v2 = Vec3(1, 2, 4)
        assert v1 != v2

    def test_approximate_equality(self):
        v1 = Vec3(1, 2, 3)
        v2 = Vec3(1 + 1e-12, 2, 3)
        assert v1 == v2  # Should be equal due to allclose


class TestVec3Indexing:
    """Test Vec3 indexing."""

    def test_getitem(self):
        v = Vec3(1, 2, 3)
        assert v[0] == 1
        assert v[1] == 2
        assert v[2] == 3

    def test_setitem(self):
        v = Vec3(1, 2, 3)
        v[0] = 10
        assert v.x == 10
