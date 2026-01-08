"""
Vector3 class for 3D math operations.

This is the fundamental building block of the ray tracer, used for:
- Points in 3D space
- Direction vectors
- RGB color values
"""

from __future__ import annotations
import math
from typing import Union
import numpy as np


class Vec3:
    """A 3D vector class supporting common vector operations.

    Uses numpy internally for efficient computation while providing
    a clean, Pythonic API.
    """

    __slots__ = ('_data',)

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self._data = np.array([x, y, z], dtype=np.float64)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> Vec3:
        """Create Vec3 from numpy array."""
        v = cls.__new__(cls)
        v._data = np.asarray(arr, dtype=np.float64)
        return v

    @property
    def x(self) -> float:
        return float(self._data[0])

    @property
    def y(self) -> float:
        return float(self._data[1])

    @property
    def z(self) -> float:
        return float(self._data[2])

    @x.setter
    def x(self, value: float):
        self._data[0] = value

    @y.setter
    def y(self, value: float):
        self._data[1] = value

    @z.setter
    def z(self, value: float):
        self._data[2] = value

    # Aliases for color operations
    @property
    def r(self) -> float:
        return self.x

    @property
    def g(self) -> float:
        return self.y

    @property
    def b(self) -> float:
        return self.z

    def __repr__(self) -> str:
        return f"Vec3({self.x:.4f}, {self.y:.4f}, {self.z:.4f})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vec3):
            return NotImplemented
        return np.allclose(self._data, other._data)

    def __hash__(self) -> int:
        return hash(tuple(self._data))

    def __neg__(self) -> Vec3:
        return Vec3.from_array(-self._data)

    def __add__(self, other: Union[Vec3, float]) -> Vec3:
        if isinstance(other, Vec3):
            return Vec3.from_array(self._data + other._data)
        return Vec3.from_array(self._data + other)

    def __radd__(self, other: float) -> Vec3:
        return Vec3.from_array(other + self._data)

    def __sub__(self, other: Union[Vec3, float]) -> Vec3:
        if isinstance(other, Vec3):
            return Vec3.from_array(self._data - other._data)
        return Vec3.from_array(self._data - other)

    def __rsub__(self, other: float) -> Vec3:
        return Vec3.from_array(other - self._data)

    def __mul__(self, other: Union[Vec3, float]) -> Vec3:
        if isinstance(other, Vec3):
            return Vec3.from_array(self._data * other._data)
        return Vec3.from_array(self._data * other)

    def __rmul__(self, other: float) -> Vec3:
        return Vec3.from_array(other * self._data)

    def __truediv__(self, other: Union[Vec3, float]) -> Vec3:
        if isinstance(other, Vec3):
            return Vec3.from_array(self._data / other._data)
        return Vec3.from_array(self._data / other)

    def __getitem__(self, index: int) -> float:
        return float(self._data[index])

    def __setitem__(self, index: int, value: float):
        self._data[index] = value

    def length(self) -> float:
        """Return the magnitude (length) of the vector."""
        return float(np.linalg.norm(self._data))

    def length_squared(self) -> float:
        """Return the squared magnitude (avoids sqrt for comparisons)."""
        return float(np.dot(self._data, self._data))

    def normalize(self) -> Vec3:
        """Return a unit vector in the same direction."""
        length = self.length()
        if length == 0:
            return Vec3(0, 0, 0)
        return Vec3.from_array(self._data / length)

    def dot(self, other: Vec3) -> float:
        """Compute dot product with another vector."""
        return float(np.dot(self._data, other._data))

    def cross(self, other: Vec3) -> Vec3:
        """Compute cross product with another vector."""
        return Vec3.from_array(np.cross(self._data, other._data))

    def reflect(self, normal: Vec3) -> Vec3:
        """Reflect this vector around the given normal."""
        return self - normal * 2 * self.dot(normal)

    def refract(self, normal: Vec3, eta_ratio: float) -> Vec3:
        """Refract this vector through surface with given normal and eta ratio.

        Args:
            normal: Surface normal (pointing outward)
            eta_ratio: Ratio of refractive indices (n1/n2)

        Returns:
            Refracted direction vector, or zero vector if total internal reflection
        """
        cos_theta = min(-self.dot(normal), 1.0)
        r_out_perp = (self + normal * cos_theta) * eta_ratio
        perp_len_sq = r_out_perp.length_squared()

        if perp_len_sq > 1.0:
            # Total internal reflection
            return Vec3(0, 0, 0)

        r_out_parallel = normal * (-math.sqrt(abs(1.0 - perp_len_sq)))
        return r_out_perp + r_out_parallel

    def near_zero(self, epsilon: float = 1e-8) -> bool:
        """Check if vector is close to zero in all dimensions."""
        return all(abs(c) < epsilon for c in self._data)

    def to_array(self) -> np.ndarray:
        """Return the underlying numpy array (copy)."""
        return self._data.copy()

    def clamp(self, min_val: float = 0.0, max_val: float = 1.0) -> Vec3:
        """Clamp all components to the given range."""
        return Vec3.from_array(np.clip(self._data, min_val, max_val))

    def gamma_correct(self, gamma: float = 2.2) -> Vec3:
        """Apply gamma correction (for converting linear to sRGB)."""
        inv_gamma = 1.0 / gamma
        return Vec3(
            self.x ** inv_gamma if self.x > 0 else 0,
            self.y ** inv_gamma if self.y > 0 else 0,
            self.z ** inv_gamma if self.z > 0 else 0
        )

    @staticmethod
    def random(min_val: float = 0.0, max_val: float = 1.0) -> Vec3:
        """Generate a random vector with components in [min_val, max_val]."""
        return Vec3(
            np.random.uniform(min_val, max_val),
            np.random.uniform(min_val, max_val),
            np.random.uniform(min_val, max_val)
        )

    @staticmethod
    def random_in_unit_sphere() -> Vec3:
        """Generate a random point inside the unit sphere."""
        while True:
            p = Vec3.random(-1, 1)
            if p.length_squared() < 1:
                return p

    @staticmethod
    def random_unit_vector() -> Vec3:
        """Generate a random unit vector (uniform on sphere surface)."""
        return Vec3.random_in_unit_sphere().normalize()

    @staticmethod
    def random_in_hemisphere(normal: Vec3) -> Vec3:
        """Generate a random vector in the hemisphere defined by normal."""
        in_unit_sphere = Vec3.random_in_unit_sphere()
        if in_unit_sphere.dot(normal) > 0.0:
            return in_unit_sphere
        return -in_unit_sphere

    @staticmethod
    def random_in_unit_disk() -> Vec3:
        """Generate a random point inside the unit disk (z=0)."""
        while True:
            p = Vec3(np.random.uniform(-1, 1), np.random.uniform(-1, 1), 0)
            if p.length_squared() < 1:
                return p


# Convenience type aliases
Point3 = Vec3
Color = Vec3
