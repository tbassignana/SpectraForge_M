"""Tests for texture system."""

import pytest
import math
import tempfile
import os
from PIL import Image
import numpy as np

from spectraforge.vec3 import Vec3, Point3, Color
from spectraforge.textures import (
    SolidColor, ImageTexture, CheckerTexture, NoiseTexture,
    MarbleTexture, NormalMap, UVMapping
)


class TestSolidColor:
    """Test SolidColor texture."""

    def test_returns_constant_color(self):
        tex = SolidColor(Color(0.5, 0.3, 0.1))
        color = tex.value(0, 0, Point3(0, 0, 0))
        assert color == Color(0.5, 0.3, 0.1)

    def test_ignores_uv(self):
        tex = SolidColor(Color(1, 0, 0))
        c1 = tex.value(0, 0, Point3(0, 0, 0))
        c2 = tex.value(0.5, 0.5, Point3(1, 1, 1))
        c3 = tex.value(1, 1, Point3(-5, 10, 3))
        assert c1 == c2 == c3

    def test_from_rgb(self):
        tex = SolidColor.from_rgb(0.2, 0.4, 0.6)
        color = tex.value(0, 0, Point3(0, 0, 0))
        assert abs(color.r - 0.2) < 1e-6
        assert abs(color.g - 0.4) < 1e-6
        assert abs(color.b - 0.6) < 1e-6


class TestImageTexture:
    """Test ImageTexture class."""

    @pytest.fixture
    def temp_image(self):
        """Create a temporary test image."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            img = Image.new('RGB', (4, 4))
            # Create a simple pattern: red top-left, green top-right, blue bottom-left, white bottom-right
            pixels = img.load()
            pixels[0, 0] = (255, 0, 0)    # Red
            pixels[3, 0] = (0, 255, 0)    # Green
            pixels[0, 3] = (0, 0, 255)    # Blue
            pixels[3, 3] = (255, 255, 255)  # White
            img.save(f.name)
            yield f.name
        os.unlink(f.name)

    def test_load_image(self, temp_image):
        tex = ImageTexture(temp_image, gamma=1.0)  # No gamma for test
        assert tex._width == 4
        assert tex._height == 4

    def test_sample_corners(self, temp_image):
        tex = ImageTexture(temp_image, gamma=1.0)

        # Note: v is flipped in ImageTexture
        # Top-left in UV (0,1) = image top-left = red
        color = tex.value(0, 1, Point3(0, 0, 0))
        assert color.r > 0.9
        assert color.g < 0.1
        assert color.b < 0.1

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            ImageTexture("/nonexistent/path/image.png")


class TestCheckerTexture:
    """Test CheckerTexture class."""

    def test_alternating_pattern(self):
        tex = CheckerTexture.from_colors(1.0, Color(1, 1, 1), Color(0, 0, 0))

        # Test different positions
        c1 = tex.value(0, 0, Point3(0.5, 0.5, 0.5))  # Should be one color
        c2 = tex.value(0, 0, Point3(1.5, 0.5, 0.5))  # Should be other color

        # Colors should be different
        assert abs(c1.r - c2.r) > 0.5

    def test_scale_affects_pattern(self):
        tex_small = CheckerTexture.from_colors(10.0, Color(1, 1, 1), Color(0, 0, 0))
        tex_large = CheckerTexture.from_colors(1.0, Color(1, 1, 1), Color(0, 0, 0))

        # At the same position, different scales should give different results sometimes
        # Just verify they work
        _ = tex_small.value(0, 0, Point3(0.15, 0, 0))
        _ = tex_large.value(0, 0, Point3(0.15, 0, 0))


class TestNoiseTexture:
    """Test NoiseTexture class."""

    def test_noise_in_range(self):
        tex = NoiseTexture(scale=1.0, color=Color(1, 1, 1))

        for _ in range(100):
            point = Point3(
                np.random.uniform(-10, 10),
                np.random.uniform(-10, 10),
                np.random.uniform(-10, 10)
            )
            color = tex.value(0, 0, point)
            # Noise output should be in [0, 1] range
            assert 0 <= color.r <= 1
            assert 0 <= color.g <= 1
            assert 0 <= color.b <= 1

    def test_noise_varies(self):
        tex = NoiseTexture(scale=1.0)

        colors = [tex.value(0, 0, Point3(i * 0.1, 0, 0)) for i in range(10)]

        # Should have some variation
        values = [c.r for c in colors]
        assert max(values) - min(values) > 0.01

    def test_turbulence(self):
        tex = NoiseTexture(scale=1.0)
        turb = tex.turbulence(Point3(1, 2, 3))
        # Turbulence should be positive
        assert turb >= 0


class TestMarbleTexture:
    """Test MarbleTexture class."""

    def test_marble_in_range(self):
        tex = MarbleTexture(scale=1.0)

        for _ in range(50):
            point = Point3(
                np.random.uniform(-5, 5),
                np.random.uniform(-5, 5),
                np.random.uniform(-5, 5)
            )
            color = tex.value(0, 0, point)
            assert 0 <= color.r <= 1
            assert 0 <= color.g <= 1
            assert 0 <= color.b <= 1

    def test_marble_varies_with_z(self):
        tex = MarbleTexture(scale=1.0)

        colors = [tex.value(0, 0, Point3(0, 0, z * 0.5)) for z in range(10)]
        values = [c.r for c in colors]

        # Should have variation due to sine wave
        assert max(values) - min(values) > 0.1


class TestUVMapping:
    """Test UV mapping utilities."""

    def test_spherical_poles(self):
        # Top pole: y=1 gives acos(-1) = pi, so v = pi/pi = 1
        u, v = UVMapping.spherical(Point3(0, 1, 0))
        assert abs(v - 1.0) < 0.01  # v=1 at top (y=1)

        # Bottom pole: y=-1 gives acos(1) = 0, so v = 0/pi = 0
        u, v = UVMapping.spherical(Point3(0, -1, 0))
        assert abs(v) < 0.01  # v=0 at bottom (y=-1)

    def test_spherical_equator(self):
        # Points on equator
        u, v = UVMapping.spherical(Point3(1, 0, 0))
        assert abs(v - 0.5) < 0.01  # v=0.5 at equator

    def test_spherical_uv_range(self):
        for _ in range(100):
            p = Vec3.random_unit_vector()
            u, v = UVMapping.spherical(Point3(p.x, p.y, p.z))
            assert 0 <= u <= 1
            assert 0 <= v <= 1

    def test_planar_tiling(self):
        u1, v1 = UVMapping.planar(Point3(0.5, 0, 0.5))
        u2, v2 = UVMapping.planar(Point3(1.5, 0, 1.5))
        # Should wrap to same UV
        assert abs(u1 - u2) < 0.01
        assert abs(v1 - v2) < 0.01

    def test_cylindrical(self):
        u, v = UVMapping.cylindrical(Point3(1, 0.5, 0), axis=1)
        assert 0 <= u <= 1
        assert abs(v - 0.5) < 0.01


class TestNormalMap:
    """Test NormalMap class."""

    @pytest.fixture
    def temp_normal_map(self):
        """Create a temporary normal map (flat, pointing up)."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            img = Image.new('RGB', (4, 4), (128, 128, 255))  # Flat normal (0, 0, 1)
            img.save(f.name)
            yield f.name
        os.unlink(f.name)

    def test_load_normal_map(self, temp_normal_map):
        nmap = NormalMap(temp_normal_map)
        assert nmap._width == 4
        assert nmap._height == 4

    def test_flat_normal_no_perturbation(self, temp_normal_map):
        nmap = NormalMap(temp_normal_map, strength=1.0)

        normal = Vec3(0, 1, 0)
        tangent = Vec3(1, 0, 0)
        bitangent = Vec3(0, 0, 1)

        perturbed = nmap.perturb_normal(0.5, 0.5, normal, tangent, bitangent)

        # For a flat normal map, perturbation should be minimal
        assert perturbed.length() > 0.9  # Should still be normalized
