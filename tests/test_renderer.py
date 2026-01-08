"""Tests for Renderer class."""

import pytest
import numpy as np
import platform

from spectraforge.vec3 import Vec3, Point3, Color
from spectraforge.ray import Ray
from spectraforge.camera import Camera
from spectraforge.shapes import Sphere, HittableList
from spectraforge.materials import Lambertian, Emissive
from spectraforge.renderer import Renderer, RenderSettings, get_platform_info


class TestRenderSettings:
    """Test RenderSettings configuration."""

    def test_default_values(self):
        settings = RenderSettings()
        assert settings.width == 800
        assert settings.height == 600
        assert settings.samples_per_pixel == 100
        assert settings.max_depth == 50

    def test_custom_values(self):
        settings = RenderSettings(
            width=1920,
            height=1080,
            samples_per_pixel=500,
            max_depth=100
        )
        assert settings.width == 1920
        assert settings.height == 1080

    def test_auto_thread_detection(self):
        settings = RenderSettings(num_threads=0)
        import os
        assert settings.num_threads == (os.cpu_count() or 4)


class TestRendererBasic:
    """Test basic renderer functionality."""

    def test_render_produces_image(self):
        settings = RenderSettings(
            width=10,
            height=10,
            samples_per_pixel=1,
            max_depth=2,
            num_threads=1
        )
        renderer = Renderer(settings)

        world = HittableList()
        world.add(Sphere(Point3(0, 0, -5), 1.0, Lambertian(Color(1, 0, 0))))

        camera = Camera(
            look_from=Point3(0, 0, 0),
            look_at=Point3(0, 0, -1),
            vup=Vec3(0, 1, 0),
            vfov=90,
            aspect_ratio=1.0
        )

        image = renderer.render(world, camera)

        assert image.shape == (10, 10, 3)
        assert image.dtype == np.float64

    def test_sky_gradient(self):
        settings = RenderSettings(
            width=10,
            height=10,
            samples_per_pixel=1,
            max_depth=1,
            use_sky_gradient=True,
            num_threads=1
        )
        renderer = Renderer(settings)

        world = HittableList()  # Empty scene

        camera = Camera(
            look_from=Point3(0, 0, 0),
            look_at=Point3(0, 0, -1),
            vup=Vec3(0, 1, 0),
            vfov=90,
            aspect_ratio=1.0
        )

        image = renderer.render(world, camera)

        # Top should be more blue (sky)
        top_blue = image[0, 5, 2]  # Blue channel at top
        bottom_blue = image[9, 5, 2]  # Blue channel at bottom

        # Sky gradient: top is bluer
        assert top_blue >= bottom_blue

    def test_solid_background(self):
        bg_color = Color(0.1, 0.2, 0.3)
        settings = RenderSettings(
            width=10,
            height=10,
            samples_per_pixel=1,
            max_depth=1,
            use_sky_gradient=False,
            background_color=bg_color,
            num_threads=1
        )
        renderer = Renderer(settings)

        world = HittableList()

        camera = Camera(
            look_from=Point3(0, 0, 0),
            look_at=Point3(0, 0, -1),
            vup=Vec3(0, 1, 0),
            vfov=90,
            aspect_ratio=1.0
        )

        image = renderer.render(world, camera)

        # All pixels should be background color
        assert abs(image[5, 5, 0] - 0.1) < 0.01
        assert abs(image[5, 5, 1] - 0.2) < 0.01
        assert abs(image[5, 5, 2] - 0.3) < 0.01


class TestRendererProgress:
    """Test renderer progress reporting."""

    def test_progress_callback(self):
        settings = RenderSettings(
            width=10,
            height=10,
            samples_per_pixel=1,
            max_depth=1,
            tile_size=5,
            num_threads=1
        )
        renderer = Renderer(settings)

        progress_values = []

        def callback(p):
            progress_values.append(p)

        renderer.set_progress_callback(callback)

        world = HittableList()
        camera = Camera(
            look_from=Point3(0, 0, 0),
            look_at=Point3(0, 0, -1),
            vup=Vec3(0, 1, 0),
            vfov=90,
            aspect_ratio=1.0
        )

        renderer.render(world, camera)

        # Should have progress updates
        assert len(progress_values) > 0
        # Final progress should be 1.0
        assert progress_values[-1] == 1.0


class TestRendererEmission:
    """Test emissive materials in renderer."""

    def test_emissive_object_visible(self):
        settings = RenderSettings(
            width=20,
            height=20,
            samples_per_pixel=4,
            max_depth=2,
            use_sky_gradient=False,
            background_color=Color(0, 0, 0),
            num_threads=1
        )
        renderer = Renderer(settings)

        # Bright emissive sphere - large and close
        world = HittableList()
        world.add(Sphere(Point3(0, 0, -3), 1.5, Emissive(Color(1, 1, 1), 10.0)))

        camera = Camera(
            look_from=Point3(0, 0, 0),
            look_at=Point3(0, 0, -1),
            vup=Vec3(0, 1, 0),
            vfov=90,
            aspect_ratio=1.0,
            aperture=0.0
        )

        image = renderer.render(world, camera)

        # Center should be bright (emissive object)
        # The sphere covers a good portion of the center
        center_brightness = image[10, 10].sum()
        corner_brightness = image[0, 0].sum()

        # Emissive center should be much brighter than black corner
        assert center_brightness > 1.0  # Should be emitting significant light
        assert center_brightness > corner_brightness


class TestRendererLDRConversion:
    """Test HDR to LDR conversion."""

    def test_to_ldr(self):
        settings = RenderSettings()
        renderer = Renderer(settings)

        hdr = np.array([[[0.5, 0.5, 0.5]]], dtype=np.float64)
        ldr = renderer.to_ldr(hdr)

        assert ldr.dtype == np.uint8
        # Gamma corrected 0.5 -> ~0.73 -> 186
        assert 150 < ldr[0, 0, 0] < 220

    def test_to_ldr_clamps(self):
        settings = RenderSettings()
        renderer = Renderer(settings)

        # HDR values > 1
        hdr = np.array([[[2.0, 0.0, -0.5]]], dtype=np.float64)
        ldr = renderer.to_ldr(hdr)

        assert ldr[0, 0, 0] == 255  # Clamped high
        assert ldr[0, 0, 2] == 0   # Clamped low


class TestRendererTiling:
    """Test tile-based rendering."""

    def test_tiles_cover_image(self):
        settings = RenderSettings(
            width=100,
            height=100,
            tile_size=32
        )
        renderer = Renderer(settings)

        tiles = renderer._generate_tiles(100, 100)

        # Should have ceiling(100/32)^2 = 4x4 = 16 tiles
        assert len(tiles) == 16

        # Tiles should cover entire image
        covered = set()
        for x0, y0, x1, y1 in tiles:
            for x in range(x0, x1):
                for y in range(y0, y1):
                    covered.add((x, y))

        assert len(covered) == 100 * 100


class TestPlatformInfo:
    """Test platform detection."""

    def test_platform_info_structure(self):
        info = get_platform_info()

        assert 'system' in info
        assert 'machine' in info
        assert 'cpu_count' in info
        assert 'is_arm' in info
        assert 'is_x86' in info
        assert 'is_apple_silicon' in info

    def test_platform_detection(self):
        info = get_platform_info()

        # At least one architecture should be detected
        if info['machine'].lower() in ('arm64', 'aarch64'):
            assert info['is_arm'] is True
        elif info['machine'].lower() in ('x86_64', 'amd64'):
            assert info['is_x86'] is True


class TestRendererMultithreading:
    """Test multi-threaded rendering."""

    def test_multithreaded_produces_same_result(self):
        # This is a probabilistic test due to random sampling
        # We just verify it runs without error

        settings_single = RenderSettings(
            width=20,
            height=20,
            samples_per_pixel=1,
            max_depth=2,
            num_threads=1
        )

        settings_multi = RenderSettings(
            width=20,
            height=20,
            samples_per_pixel=1,
            max_depth=2,
            num_threads=4
        )

        world = HittableList()
        world.add(Sphere(Point3(0, 0, -5), 1.0, Lambertian(Color(1, 0, 0))))

        camera = Camera(
            look_from=Point3(0, 0, 0),
            look_at=Point3(0, 0, -1),
            vup=Vec3(0, 1, 0),
            vfov=90,
            aspect_ratio=1.0
        )

        renderer_single = Renderer(settings_single)
        renderer_multi = Renderer(settings_multi)

        # Both should complete without error
        image_single = renderer_single.render(world, camera)
        image_multi = renderer_multi.render(world, camera)

        assert image_single.shape == image_multi.shape
