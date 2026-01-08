"""Tests for light importance sampling."""

import pytest
import math
from spectraforge.vec3 import Vec3, Point3, Color
from spectraforge.ray import Ray
from spectraforge.shapes import Sphere, HittableList, Plane
from spectraforge.lights import (
    PointLight, DirectionalLight, AreaLight, SphereLight, LightList,
    LightSample, compute_direct_lighting, sample_light_direction
)
from spectraforge.materials import Lambertian


class TestPointLight:
    """Test PointLight sampling."""

    def test_sample_direction(self):
        light = PointLight(Point3(0, 5, 0), Color(1, 1, 1), intensity=10.0)
        sample = light.sample(Point3(0, 0, 0))

        # Direction should point toward light
        assert sample.direction.y > 0.9
        assert abs(sample.distance - 5.0) < 1e-6

    def test_inverse_square_falloff(self):
        light = PointLight(Point3(0, 1, 0), Color(1, 1, 1), intensity=1.0)

        sample1 = light.sample(Point3(0, 0, 0))  # Distance 1
        sample2 = light.sample(Point3(0, -1, 0))  # Distance 2

        # Intensity should follow inverse square law
        ratio = sample1.intensity.r / sample2.intensity.r
        assert abs(ratio - 4.0) < 0.1  # 2^2 = 4

    def test_power(self):
        light = PointLight(Point3(0, 0, 0), Color(1, 1, 1), intensity=10.0)
        assert light.power() == 10.0


class TestDirectionalLight:
    """Test DirectionalLight sampling."""

    def test_sample_direction(self):
        light = DirectionalLight(Vec3(0, -1, 0), Color(1, 1, 1))  # Light coming from above
        sample = light.sample(Point3(0, 0, 0))

        # Direction TO light is opposite of light direction
        assert sample.direction.y > 0.9

    def test_infinite_distance(self):
        light = DirectionalLight(Vec3(0, -1, 0), Color(1, 1, 1))
        sample = light.sample(Point3(0, 0, 0))

        assert sample.distance == float('inf')

    def test_no_falloff(self):
        light = DirectionalLight(Vec3(0, -1, 0), Color(1, 1, 1), intensity=5.0)

        sample1 = light.sample(Point3(0, 0, 0))
        sample2 = light.sample(Point3(0, -100, 0))

        # Intensity should be the same regardless of position
        assert sample1.intensity.r == sample2.intensity.r


class TestAreaLight:
    """Test AreaLight sampling."""

    def test_sample_on_rectangle(self):
        light = AreaLight(
            corner=Point3(-1, 5, -1),
            edge1=Vec3(2, 0, 0),
            edge2=Vec3(0, 0, 2),
            color=Color(1, 1, 1),
            intensity=1.0
        )

        sample = light.sample(Point3(0, 0, 0))

        # Direction should point upward (toward the light)
        assert sample.direction.y > 0
        assert sample.distance > 0

    def test_pdf_correct(self):
        light = AreaLight(
            corner=Point3(-1, 5, -1),
            edge1=Vec3(2, 0, 0),
            edge2=Vec3(0, 0, 2),
            color=Color(1, 1, 1)
        )

        sample = light.sample(Point3(0, 0, 0))

        # PDF should be positive
        assert sample.pdf > 0

    def test_power_proportional_to_area(self):
        light_small = AreaLight(
            corner=Point3(0, 0, 0),
            edge1=Vec3(1, 0, 0),
            edge2=Vec3(0, 0, 1),
            color=Color(1, 1, 1),
            intensity=1.0
        )

        light_large = AreaLight(
            corner=Point3(0, 0, 0),
            edge1=Vec3(2, 0, 0),
            edge2=Vec3(0, 0, 2),
            color=Color(1, 1, 1),
            intensity=1.0
        )

        # Power should scale with area
        assert abs(light_large.power() / light_small.power() - 4.0) < 0.01

    def test_hit_intersection(self):
        light = AreaLight(
            corner=Point3(-1, 5, -1),
            edge1=Vec3(2, 0, 0),
            edge2=Vec3(0, 0, 2),
            color=Color(1, 1, 1)
        )

        ray = Ray(Point3(0, 0, 0), Vec3(0, 1, 0))
        hit = light.hit(ray, 0.001, float('inf'))

        assert hit is not None
        assert abs(hit.point.y - 5.0) < 1e-6


class TestSphereLight:
    """Test SphereLight sampling."""

    def test_sample_toward_sphere(self):
        light = SphereLight(Point3(0, 5, 0), 1.0, Color(1, 1, 1))
        sample = light.sample(Point3(0, 0, 0))

        # Should sample toward the sphere
        assert sample.direction.y > 0

    def test_pdf_solid_angle(self):
        light = SphereLight(Point3(0, 10, 0), 1.0, Color(1, 1, 1))
        sample = light.sample(Point3(0, 0, 0))

        # PDF should be positive
        assert sample.pdf > 0

    def test_power_proportional_to_surface_area(self):
        light_small = SphereLight(Point3(0, 0, 0), 1.0, Color(1, 1, 1), intensity=1.0)
        light_large = SphereLight(Point3(0, 0, 0), 2.0, Color(1, 1, 1), intensity=1.0)

        # Surface area scales with r^2
        ratio = light_large.power() / light_small.power()
        assert abs(ratio - 4.0) < 0.01


class TestLightList:
    """Test LightList importance sampling."""

    def test_add_lights(self):
        lights = LightList()
        lights.add(PointLight(Point3(0, 5, 0), Color(1, 1, 1)))
        lights.add(PointLight(Point3(5, 5, 0), Color(1, 1, 1)))

        assert len(lights) == 2

    def test_clear_lights(self):
        lights = LightList()
        lights.add(PointLight(Point3(0, 5, 0), Color(1, 1, 1)))
        lights.clear()

        assert len(lights) == 0
        assert lights.total_power() == 0

    def test_sample_single_light(self):
        lights = LightList()
        lights.add(PointLight(Point3(0, 5, 0), Color(1, 1, 1)))

        sample, idx = lights.sample(Point3(0, 0, 0))
        assert idx == 0

    def test_power_weighted_sampling(self):
        """Test that brighter lights are sampled more often."""
        lights = LightList()
        lights.add(PointLight(Point3(0, 5, 0), Color(1, 1, 1), intensity=10.0))  # Bright
        lights.add(PointLight(Point3(5, 5, 0), Color(1, 1, 1), intensity=1.0))   # Dim

        counts = [0, 0]
        for _ in range(1000):
            _, idx = lights.sample(Point3(0, 0, 0))
            counts[idx] += 1

        # Bright light should be sampled ~10x more often
        ratio = counts[0] / counts[1]
        assert 5 < ratio < 20  # Roughly 10:1

    def test_pdf_for_light(self):
        lights = LightList()
        lights.add(PointLight(Point3(0, 5, 0), Color(1, 1, 1), intensity=3.0))
        lights.add(PointLight(Point3(5, 5, 0), Color(1, 1, 1), intensity=1.0))

        # First light has 3/4 of total power
        assert abs(lights.pdf(0) - 0.75) < 0.01
        assert abs(lights.pdf(1) - 0.25) < 0.01

    def test_sample_all(self):
        lights = LightList()
        lights.add(PointLight(Point3(0, 5, 0), Color(1, 0, 0)))
        lights.add(PointLight(Point3(5, 5, 0), Color(0, 1, 0)))

        samples = lights.sample_all(Point3(0, 0, 0))
        assert len(samples) == 2

    def test_empty_list(self):
        lights = LightList()
        sample, idx = lights.sample(Point3(0, 0, 0))

        assert idx == -1
        assert sample.intensity == Color(0, 0, 0)

    def test_indexing(self):
        lights = LightList()
        light1 = PointLight(Point3(0, 5, 0), Color(1, 1, 1))
        light2 = PointLight(Point3(5, 5, 0), Color(1, 1, 1))
        lights.add(light1)
        lights.add(light2)

        assert lights[0] is light1
        assert lights[1] is light2


class TestComputeDirectLighting:
    """Test direct lighting computation."""

    def test_lit_surface(self):
        # Simple scene with a point above a surface point
        world = HittableList()
        lights = LightList()
        lights.add(PointLight(Point3(0, 5, 0), Color(1, 1, 1), intensity=25.0))

        # Surface point looking up
        hit_point = Point3(0, 0, 0)
        normal = Vec3(0, 1, 0)

        result = compute_direct_lighting(hit_point, normal, lights, world)

        # Should receive light
        assert result.r > 0

    def test_shadowed_surface(self):
        # Scene with an occluder between light and surface
        world = HittableList()
        world.add(Sphere(Point3(0, 2.5, 0), 0.5))  # Blocker

        lights = LightList()
        lights.add(PointLight(Point3(0, 5, 0), Color(1, 1, 1), intensity=10.0))

        hit_point = Point3(0, 0, 0)
        normal = Vec3(0, 1, 0)

        result = compute_direct_lighting(hit_point, normal, lights, world)

        # Should be in shadow
        assert result.r < 0.1

    def test_backfacing_ignored(self):
        world = HittableList()
        lights = LightList()
        lights.add(PointLight(Point3(0, 5, 0), Color(1, 1, 1), intensity=10.0))

        # Surface facing away from light
        hit_point = Point3(0, 0, 0)
        normal = Vec3(0, -1, 0)  # Facing down

        result = compute_direct_lighting(hit_point, normal, lights, world)

        # Should not receive light
        assert result.r == 0


class TestSampleLightDirection:
    """Test light direction sampling for path tracing."""

    def test_returns_direction_toward_light(self):
        lights = LightList()
        lights.add(PointLight(Point3(0, 10, 0), Color(1, 1, 1)))

        direction, color, pdf = sample_light_direction(Point3(0, 0, 0), lights)

        assert direction.y > 0  # Should point toward light
        assert pdf > 0

    def test_empty_lights_uniform_hemisphere(self):
        lights = LightList()

        direction, color, pdf = sample_light_direction(Point3(0, 0, 0), lights)

        # Should return hemisphere sample
        assert direction.y >= 0  # Upper hemisphere
        assert abs(pdf - 1.0 / (2.0 * math.pi)) < 0.01


class TestIntegration:
    """Integration tests for light sampling."""

    def test_multiple_light_types(self):
        lights = LightList()
        lights.add(PointLight(Point3(0, 5, 0), Color(1, 0, 0), intensity=5.0))
        lights.add(DirectionalLight(Vec3(-1, -1, 0), Color(0, 1, 0), intensity=1.0))
        lights.add(SphereLight(Point3(5, 5, 5), 0.5, Color(0, 0, 1), intensity=3.0))

        # Should be able to sample from mixed light types
        for _ in range(100):
            sample, idx = lights.sample(Point3(0, 0, 0))
            assert 0 <= idx < 3
            assert sample.pdf > 0

    def test_cdf_binary_search(self):
        """Test that binary search CDF works correctly."""
        lights = LightList()
        for i in range(10):
            lights.add(PointLight(Point3(i, 5, 0), Color(1, 1, 1), intensity=1.0))

        # Sample many times, all indices should be valid
        indices = set()
        for _ in range(500):
            _, idx = lights.sample(Point3(0, 0, 0))
            indices.add(idx)

        # Should have sampled all lights at some point
        assert len(indices) == 10
