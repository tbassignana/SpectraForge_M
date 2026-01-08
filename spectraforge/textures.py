"""
Texture system for the ray tracer.

Implements:
- Solid color textures
- Image textures (from files)
- Procedural textures (checker, noise, marble)
- Normal maps
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple
import math

import numpy as np
from PIL import Image

from .vec3 import Vec3, Color, Point3


class Texture(ABC):
    """Abstract base class for textures."""

    @abstractmethod
    def value(self, u: float, v: float, point: Point3) -> Color:
        """Get the texture color at the given UV coordinates.

        Args:
            u: Horizontal texture coordinate [0, 1]
            v: Vertical texture coordinate [0, 1]
            point: 3D point in world space (for procedural textures)

        Returns:
            Color at this location
        """
        pass


class SolidColor(Texture):
    """A solid color texture."""

    def __init__(self, color: Color):
        self.color = color

    @classmethod
    def from_rgb(cls, r: float, g: float, b: float) -> 'SolidColor':
        return cls(Color(r, g, b))

    def value(self, u: float, v: float, point: Point3) -> Color:
        return self.color


class ImageTexture(Texture):
    """A texture loaded from an image file."""

    def __init__(self, filename: str, gamma: float = 2.2):
        """Load a texture from an image file.

        Args:
            filename: Path to the image file
            gamma: Gamma value for converting sRGB to linear (2.2 for sRGB images)
        """
        self.filename = filename
        self.gamma = gamma
        self._data: Optional[np.ndarray] = None
        self._width = 0
        self._height = 0
        self._load_image()

    def _load_image(self) -> None:
        """Load and convert the image to linear color space."""
        path = Path(self.filename)
        if not path.exists():
            raise FileNotFoundError(f"Texture file not found: {self.filename}")

        img = Image.open(self.filename)
        img = img.convert('RGB')

        # Convert to numpy array and normalize to [0, 1]
        self._data = np.array(img, dtype=np.float64) / 255.0

        # Convert from sRGB to linear
        if self.gamma != 1.0:
            self._data = np.power(self._data, self.gamma)

        self._height, self._width = self._data.shape[:2]

    def value(self, u: float, v: float, point: Point3) -> Color:
        if self._data is None:
            return Color(1, 0, 1)  # Magenta for missing texture

        # Clamp UV coordinates
        u = max(0.0, min(1.0, u))
        v = max(0.0, min(1.0, v))

        # Flip v to match image coordinates (image y=0 is top)
        v = 1.0 - v

        # Convert to pixel coordinates
        i = int(u * (self._width - 1))
        j = int(v * (self._height - 1))

        pixel = self._data[j, i]
        return Color(pixel[0], pixel[1], pixel[2])

    def value_bilinear(self, u: float, v: float, point: Point3) -> Color:
        """Get texture value with bilinear interpolation."""
        if self._data is None:
            return Color(1, 0, 1)

        u = max(0.0, min(1.0, u))
        v = 1.0 - max(0.0, min(1.0, v))

        # Pixel coordinates (continuous)
        x = u * (self._width - 1)
        y = v * (self._height - 1)

        # Integer and fractional parts
        x0 = int(x)
        y0 = int(y)
        x1 = min(x0 + 1, self._width - 1)
        y1 = min(y0 + 1, self._height - 1)

        fx = x - x0
        fy = y - y0

        # Sample four pixels
        c00 = self._data[y0, x0]
        c10 = self._data[y0, x1]
        c01 = self._data[y1, x0]
        c11 = self._data[y1, x1]

        # Bilinear interpolation
        c0 = c00 * (1 - fx) + c10 * fx
        c1 = c01 * (1 - fx) + c11 * fx
        c = c0 * (1 - fy) + c1 * fy

        return Color(c[0], c[1], c[2])


class CheckerTexture(Texture):
    """A 3D checker pattern texture."""

    def __init__(self, scale: float, even: Texture, odd: Texture):
        """Create a checker texture.

        Args:
            scale: Size of each checker square
            even: Texture for even squares
            odd: Texture for odd squares
        """
        self.scale = scale
        self.even = even
        self.odd = odd

    @classmethod
    def from_colors(cls, scale: float, c1: Color, c2: Color) -> 'CheckerTexture':
        return cls(scale, SolidColor(c1), SolidColor(c2))

    def value(self, u: float, v: float, point: Point3) -> Color:
        # Use 3D position for the checker pattern
        x = int(math.floor(point.x * self.scale))
        y = int(math.floor(point.y * self.scale))
        z = int(math.floor(point.z * self.scale))

        if (x + y + z) % 2 == 0:
            return self.even.value(u, v, point)
        else:
            return self.odd.value(u, v, point)


class NoiseTexture(Texture):
    """Perlin noise texture."""

    def __init__(self, scale: float = 1.0, color: Color = None):
        """Create a noise texture.

        Args:
            scale: Scale of the noise pattern
            color: Base color (noise modulates intensity)
        """
        self.scale = scale
        self.color = color if color else Color(1, 1, 1)
        self._perm = self._generate_permutation()

    def _generate_permutation(self) -> np.ndarray:
        """Generate permutation table for Perlin noise."""
        p = np.arange(256, dtype=np.int32)
        np.random.shuffle(p)
        return np.concatenate([p, p])

    def _fade(self, t: float) -> float:
        """Fade function for smooth interpolation."""
        return t * t * t * (t * (t * 6 - 15) + 10)

    def _lerp(self, t: float, a: float, b: float) -> float:
        return a + t * (b - a)

    def _grad(self, hash_val: int, x: float, y: float, z: float) -> float:
        """Gradient function."""
        h = hash_val & 15
        u = x if h < 8 else y
        v = y if h < 4 else (z if h == 12 or h == 14 else x)
        return (u if (h & 1) == 0 else -u) + (v if (h & 2) == 0 else -v)

    def noise(self, x: float, y: float, z: float) -> float:
        """Compute Perlin noise at a point."""
        # Find unit cube containing point
        X = int(math.floor(x)) & 255
        Y = int(math.floor(y)) & 255
        Z = int(math.floor(z)) & 255

        # Find relative position in cube
        x -= math.floor(x)
        y -= math.floor(y)
        z -= math.floor(z)

        # Compute fade curves
        u = self._fade(x)
        v = self._fade(y)
        w = self._fade(z)

        # Hash coordinates of cube corners
        p = self._perm
        A = p[X] + Y
        AA = p[A] + Z
        AB = p[A + 1] + Z
        B = p[X + 1] + Y
        BA = p[B] + Z
        BB = p[B + 1] + Z

        # Blend results from 8 corners
        return self._lerp(w,
            self._lerp(v,
                self._lerp(u, self._grad(p[AA], x, y, z), self._grad(p[BA], x - 1, y, z)),
                self._lerp(u, self._grad(p[AB], x, y - 1, z), self._grad(p[BB], x - 1, y - 1, z))
            ),
            self._lerp(v,
                self._lerp(u, self._grad(p[AA + 1], x, y, z - 1), self._grad(p[BA + 1], x - 1, y, z - 1)),
                self._lerp(u, self._grad(p[AB + 1], x, y - 1, z - 1), self._grad(p[BB + 1], x - 1, y - 1, z - 1))
            )
        )

    def turbulence(self, point: Point3, depth: int = 7) -> float:
        """Multi-octave noise (turbulence)."""
        accum = 0.0
        weight = 1.0
        p = point

        for _ in range(depth):
            accum += weight * self.noise(p.x, p.y, p.z)
            weight *= 0.5
            p = p * 2

        return abs(accum)

    def value(self, u: float, v: float, point: Point3) -> Color:
        p = point * self.scale
        noise_val = 0.5 * (1 + self.noise(p.x, p.y, p.z))
        return self.color * noise_val


class MarbleTexture(Texture):
    """Marble-like procedural texture using turbulence."""

    def __init__(self, scale: float = 1.0, color1: Color = None, color2: Color = None):
        self.scale = scale
        self.color1 = color1 if color1 else Color(1, 1, 1)
        self.color2 = color2 if color2 else Color(0.2, 0.2, 0.2)
        self._noise = NoiseTexture(1.0)

    def value(self, u: float, v: float, point: Point3) -> Color:
        # Use sine wave modulated by turbulence
        t = 0.5 * (1 + math.sin(self.scale * point.z + 10 * self._noise.turbulence(point)))
        return self.color1 * t + self.color2 * (1 - t)


class NormalMap:
    """Normal map for perturbing surface normals."""

    def __init__(self, filename: str, strength: float = 1.0):
        """Load a normal map from an image file.

        Args:
            filename: Path to the normal map image
            strength: Strength of the normal perturbation
        """
        self.filename = filename
        self.strength = strength
        self._data: Optional[np.ndarray] = None
        self._width = 0
        self._height = 0
        self._load_image()

    def _load_image(self) -> None:
        """Load the normal map image."""
        path = Path(self.filename)
        if not path.exists():
            raise FileNotFoundError(f"Normal map not found: {self.filename}")

        img = Image.open(self.filename)
        img = img.convert('RGB')

        # Convert to numpy array and normalize to [-1, 1]
        self._data = np.array(img, dtype=np.float64) / 255.0 * 2.0 - 1.0
        self._height, self._width = self._data.shape[:2]

    def perturb_normal(self, u: float, v: float, normal: Vec3, tangent: Vec3, bitangent: Vec3) -> Vec3:
        """Perturb the surface normal using the normal map.

        Args:
            u, v: Texture coordinates
            normal: Original surface normal
            tangent: Surface tangent vector
            bitangent: Surface bitangent vector

        Returns:
            Perturbed normal vector
        """
        if self._data is None:
            return normal

        # Clamp and flip UV
        u = max(0.0, min(1.0, u))
        v = 1.0 - max(0.0, min(1.0, v))

        i = int(u * (self._width - 1))
        j = int(v * (self._height - 1))

        # Read normal from map (in tangent space)
        map_normal = self._data[j, i]

        # Apply strength
        map_normal[0] *= self.strength
        map_normal[1] *= self.strength

        # Transform from tangent space to world space
        perturbed = (
            tangent * map_normal[0] +
            bitangent * map_normal[1] +
            normal * map_normal[2]
        )

        return perturbed.normalize()


class UVMapping:
    """UV coordinate transformation utilities."""

    @staticmethod
    def spherical(point: Point3, center: Point3 = None) -> Tuple[float, float]:
        """Compute spherical UV coordinates for a point.

        Args:
            point: Point on the sphere surface
            center: Center of the sphere (default origin)

        Returns:
            (u, v) coordinates in [0, 1]
        """
        if center:
            p = (point - center).normalize()
        else:
            p = point.normalize()

        theta = math.acos(-p.y)
        phi = math.atan2(-p.z, p.x) + math.pi

        u = phi / (2 * math.pi)
        v = theta / math.pi
        return u, v

    @staticmethod
    def planar(point: Point3, scale: float = 1.0) -> Tuple[float, float]:
        """Compute planar UV coordinates (using x, z as u, v).

        Args:
            point: Point in 3D space
            scale: Scale factor for UV

        Returns:
            (u, v) coordinates
        """
        u = (point.x * scale) % 1.0
        v = (point.z * scale) % 1.0
        return u, v

    @staticmethod
    def cylindrical(point: Point3, axis: int = 1) -> Tuple[float, float]:
        """Compute cylindrical UV coordinates.

        Args:
            point: Point in 3D space
            axis: Axis of the cylinder (0=x, 1=y, 2=z)

        Returns:
            (u, v) coordinates
        """
        if axis == 0:
            u = math.atan2(point.z, point.y) / (2 * math.pi) + 0.5
            v = point.x
        elif axis == 1:
            u = math.atan2(point.x, point.z) / (2 * math.pi) + 0.5
            v = point.y
        else:
            u = math.atan2(point.y, point.x) / (2 * math.pi) + 0.5
            v = point.z
        return u, v % 1.0
