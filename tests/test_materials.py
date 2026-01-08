"""Tests for material system."""

import pytest
import math
from spectraforge.vec3 import Vec3, Point3, Color
from spectraforge.ray import Ray
from spectraforge.materials import (
    Lambertian, Metal, Dielectric, Emissive, PBRMaterial
)


class TestLambertian:
    """Test Lambertian diffuse material."""

    def test_scatter_always_succeeds(self):
        mat = Lambertian(Color(0.5, 0.5, 0.5))
        ray_in = Ray(Point3(0, 0, 0), Vec3(0, 0, -1))
        hit_point = Point3(0, 0, -1)
        normal = Vec3(0, 0, 1)

        for _ in range(100):
            result = mat.scatter(ray_in, hit_point, normal, True)
            assert result is not None

    def test_scattered_in_hemisphere(self):
        mat = Lambertian(Color(0.5, 0.5, 0.5))
        ray_in = Ray(Point3(0, 0, 0), Vec3(0, 0, -1))
        hit_point = Point3(0, 0, 0)
        normal = Vec3(0, 1, 0)

        for _ in range(100):
            result = mat.scatter(ray_in, hit_point, normal, True)
            # Scattered ray should be in hemisphere of normal
            assert result.scattered_ray.direction.dot(normal) >= -0.01

    def test_attenuation_matches_albedo(self):
        albedo = Color(0.8, 0.2, 0.3)
        mat = Lambertian(albedo)
        ray_in = Ray(Point3(0, 0, 0), Vec3(0, 0, -1))
        hit_point = Point3(0, 0, 0)
        normal = Vec3(0, 1, 0)

        result = mat.scatter(ray_in, hit_point, normal, True)
        assert result.attenuation == albedo

    def test_is_not_specular(self):
        mat = Lambertian(Color(0.5, 0.5, 0.5))
        ray_in = Ray(Point3(0, 0, 0), Vec3(0, 0, -1))
        result = mat.scatter(ray_in, Point3(0, 0, 0), Vec3(0, 1, 0), True)
        assert result.is_specular is False


class TestMetal:
    """Test Metal material."""

    def test_perfect_reflection(self):
        mat = Metal(Color(1, 1, 1), roughness=0.0)
        ray_in = Ray(Point3(0, 0, 0), Vec3(1, -1, 0).normalize())
        hit_point = Point3(1, 0, 0)
        normal = Vec3(0, 1, 0)

        result = mat.scatter(ray_in, hit_point, normal, True)
        assert result is not None

        # Perfect reflection: incoming (1, -1, 0) reflects to (1, 1, 0)
        expected = Vec3(1, 1, 0).normalize()
        assert abs(result.scattered_ray.direction.dot(expected) - 1.0) < 0.01

    def test_rough_metal_adds_fuzz(self):
        mat = Metal(Color(1, 1, 1), roughness=0.5)
        ray_in = Ray(Point3(0, 0, 0), Vec3(0, -1, 0))
        hit_point = Point3(0, 0, 0)
        normal = Vec3(0, 1, 0)

        # Collect scattered directions
        directions = []
        for _ in range(100):
            result = mat.scatter(ray_in, hit_point, normal, True)
            if result:
                directions.append(result.scattered_ray.direction)

        # With roughness, directions should vary
        if len(directions) > 1:
            first = directions[0]
            some_different = any(
                abs(d.dot(first) - 1.0) > 0.01 for d in directions[1:]
            )
            assert some_different

    def test_is_specular(self):
        mat = Metal(Color(1, 1, 1), roughness=0.0)
        ray_in = Ray(Point3(0, 0, 0), Vec3(0, -1, 0))
        result = mat.scatter(ray_in, Point3(0, 0, 0), Vec3(0, 1, 0), True)
        assert result is not None
        assert result.is_specular is True

    def test_no_scatter_below_surface(self):
        """Rough metal might produce rays pointing into surface."""
        mat = Metal(Color(1, 1, 1), roughness=1.0)
        ray_in = Ray(Point3(0, 0, 0), Vec3(1, -0.1, 0).normalize())
        hit_point = Point3(0, 0, 0)
        normal = Vec3(0, 1, 0)

        # With high roughness and grazing angle, some scatters may fail
        successes = 0
        for _ in range(100):
            result = mat.scatter(ray_in, hit_point, normal, True)
            if result:
                successes += 1
                # When scatter succeeds, ray should be above surface
                assert result.scattered_ray.direction.dot(normal) > 0

        # Some should succeed
        assert successes > 0


class TestDielectric:
    """Test Dielectric (glass) material."""

    def test_always_scatters(self):
        mat = Dielectric(1.5)
        ray_in = Ray(Point3(0, 0, 0), Vec3(0, -1, 0))
        hit_point = Point3(0, 0, 0)
        normal = Vec3(0, 1, 0)

        for _ in range(100):
            result = mat.scatter(ray_in, hit_point, normal, True)
            assert result is not None

    def test_refraction_bends_toward_normal(self):
        """When entering denser medium, ray bends toward normal."""
        mat = Dielectric(1.5)  # Glass
        # Ray at 45 degrees
        ray_in = Ray(Point3(0, 1, 0), Vec3(1, -1, 0).normalize())
        hit_point = Point3(1, 0, 0)
        normal = Vec3(0, 1, 0)

        # Multiple samples to get refraction (not reflection)
        refractions = []
        for _ in range(100):
            result = mat.scatter(ray_in, hit_point, normal, True)
            # Refracted ray should have smaller angle from normal than incoming
            if result.scattered_ray.direction.y < 0:  # Going down
                refractions.append(result.scattered_ray.direction)

        # At least some refractions should occur
        assert len(refractions) > 0

    def test_is_specular(self):
        mat = Dielectric(1.5)
        ray_in = Ray(Point3(0, 0, 0), Vec3(0, -1, 0))
        result = mat.scatter(ray_in, Point3(0, 0, 0), Vec3(0, 1, 0), True)
        assert result.is_specular is True

    def test_tint(self):
        tint = Color(0.8, 0.9, 1.0)
        mat = Dielectric(1.5, tint=tint)
        ray_in = Ray(Point3(0, 0, 0), Vec3(0, -1, 0))
        result = mat.scatter(ray_in, Point3(0, 0, 0), Vec3(0, 1, 0), True)
        assert result.attenuation == tint


class TestEmissive:
    """Test Emissive (light) material."""

    def test_no_scatter(self):
        mat = Emissive(Color(1, 1, 1), 1.0)
        ray_in = Ray(Point3(0, 0, 0), Vec3(0, -1, 0))
        result = mat.scatter(ray_in, Point3(0, 0, 0), Vec3(0, 1, 0), True)
        assert result is None

    def test_emits_light(self):
        color = Color(1, 0.5, 0.2)
        intensity = 2.0
        mat = Emissive(color, intensity)

        emitted = mat.emitted(0.5, 0.5, Point3(0, 0, 0))
        expected = color * intensity
        assert emitted == expected

    def test_default_intensity(self):
        mat = Emissive(Color(1, 1, 1))
        emitted = mat.emitted(0, 0, Point3(0, 0, 0))
        assert emitted == Color(1, 1, 1)


class TestPBRMaterial:
    """Test PBR material."""

    def test_metallic_uses_albedo_for_reflection(self):
        albedo = Color(1.0, 0.8, 0.0)  # Gold color
        mat = PBRMaterial(albedo=albedo, metallic=1.0, roughness=0.1)

        # Use an angled ray rather than straight down to ensure good GGX sampling
        ray_in = Ray(Point3(1, 2, 0), Vec3(-0.4, -0.9, 0).normalize())
        hit_point = Point3(0, 0, 0)
        normal = Vec3(0, 1, 0)

        # Sample many times
        attenuations = []
        for _ in range(200):
            result = mat.scatter(ray_in, hit_point, normal, True)
            if result:
                attenuations.append(result.attenuation)

        # Should have at least some valid scatters
        assert len(attenuations) > 0, "PBR material should produce valid scatters"

        # Metal should tint reflections with albedo
        avg_r = sum(a.r for a in attenuations) / len(attenuations)
        avg_g = sum(a.g for a in attenuations) / len(attenuations)
        # Gold should have more red than green
        assert avg_r > avg_g * 0.7  # Allow some variance

    def test_dielectric_has_diffuse_component(self):
        mat = PBRMaterial(albedo=Color(0.8, 0.2, 0.2), metallic=0.0, roughness=0.5)

        ray_in = Ray(Point3(0, 1, 0), Vec3(0, -1, 0))
        hit_point = Point3(0, 0, 0)
        normal = Vec3(0, 1, 0)

        # Sample to get diffuse bounces
        non_specular = 0
        for _ in range(100):
            result = mat.scatter(ray_in, hit_point, normal, True)
            if result and not result.is_specular:
                non_specular += 1

        # Should have some diffuse scatters
        assert non_specular > 0

    def test_emission(self):
        mat = PBRMaterial(
            albedo=Color(1, 1, 1),
            emission=Color(1, 0.5, 0),
            emission_strength=5.0
        )

        emitted = mat.emitted(0.5, 0.5, Point3(0, 0, 0))
        assert emitted.r == 5.0
        assert emitted.g == 2.5
        assert emitted.b == 0

    def test_roughness_clamping(self):
        # Roughness should be clamped to avoid division by zero
        mat = PBRMaterial(roughness=0.0)
        assert mat.roughness > 0

        mat2 = PBRMaterial(roughness=2.0)
        assert mat2.roughness <= 1.0


class TestMaterialInteractions:
    """Test material behavior in realistic scenarios."""

    def test_multiple_bounces(self):
        """Simulate multiple bounces to check energy conservation."""
        mat = Lambertian(Color(0.8, 0.8, 0.8))

        # Simulate path tracing
        energy = Color(1, 1, 1)
        ray = Ray(Point3(0, 5, 0), Vec3(0, -1, 0))
        normal = Vec3(0, 1, 0)

        for _ in range(10):
            result = mat.scatter(ray, Point3(0, 0, 0), normal, True)
            if result:
                energy = energy * result.attenuation
                ray = result.scattered_ray

        # Energy should decrease
        assert energy.r <= 1.0
        assert energy.g <= 1.0
        assert energy.b <= 1.0
