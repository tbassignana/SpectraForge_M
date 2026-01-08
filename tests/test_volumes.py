"""Tests for volumetric effects."""

import pytest
import math
from spectraforge.vec3 import Vec3, Point3, Color
from spectraforge.ray import Ray
from spectraforge.shapes import Sphere
from spectraforge.volumes import (
    IsotropicPhase, HenyeyGreensteinPhase,
    ConstantMedium, SubsurfaceScatteringMaterial,
    create_fog, create_smoke
)


class TestIsotropicPhase:
    """Test IsotropicPhase class."""

    def test_uniform_distribution(self):
        phase = IsotropicPhase()
        incoming = Vec3(0, 0, -1)

        # Sample many directions
        directions = [phase.sample(incoming) for _ in range(1000)]

        # Should have samples in all hemispheres
        positive_y = sum(1 for d in directions if d.y > 0)
        negative_y = sum(1 for d in directions if d.y < 0)

        # Should be roughly equal
        ratio = positive_y / negative_y
        assert 0.7 < ratio < 1.4

    def test_pdf_constant(self):
        phase = IsotropicPhase()
        incoming = Vec3(0, 0, -1)
        scattered = Vec3(1, 0, 0)

        pdf = phase.pdf(incoming, scattered)
        expected = 1.0 / (4.0 * math.pi)
        assert abs(pdf - expected) < 1e-6


class TestHenyeyGreensteinPhase:
    """Test HenyeyGreensteinPhase class."""

    def test_isotropic_when_g_zero(self):
        phase = HenyeyGreensteinPhase(0.0)
        incoming = Vec3(0, 0, -1)

        directions = [phase.sample(incoming) for _ in range(1000)]

        positive_y = sum(1 for d in directions if d.y > 0)
        negative_y = sum(1 for d in directions if d.y < 0)

        ratio = positive_y / negative_y
        assert 0.7 < ratio < 1.4

    def test_forward_scattering(self):
        phase = HenyeyGreensteinPhase(0.9)  # Strong forward
        incoming = Vec3(0, 0, -1)  # Light traveling in -z direction

        directions = [phase.sample(incoming) for _ in range(1000)]

        # Forward scattering: light continues roughly in same direction
        # The "forward" direction is -incoming, i.e., +z
        # So scattered light should mostly go in +z (same as light propagation)
        forward = sum(1 for d in directions if d.z > 0)
        backward = sum(1 for d in directions if d.z < 0)

        assert forward > backward * 3

    def test_backward_scattering(self):
        phase = HenyeyGreensteinPhase(-0.9)  # Strong backward
        incoming = Vec3(0, 0, -1)

        directions = [phase.sample(incoming) for _ in range(1000)]

        # Backward scattering: light reverses direction
        forward = sum(1 for d in directions if d.z > 0)
        backward = sum(1 for d in directions if d.z < 0)

        assert backward > forward * 3

    def test_pdf_integrates_to_one(self):
        """PDF should integrate to 1 over the sphere."""
        phase = HenyeyGreensteinPhase(0.5)
        incoming = Vec3(0, 0, -1)

        # Monte Carlo integration
        n_samples = 10000
        total = 0
        for _ in range(n_samples):
            scattered = Vec3.random_unit_vector()
            total += phase.pdf(incoming, scattered) * 4 * math.pi / n_samples

        # Should be close to 1
        assert 0.9 < total < 1.1


class TestConstantMedium:
    """Test ConstantMedium class."""

    def test_hit_inside_boundary(self):
        boundary = Sphere(Point3(0, 0, 0), 2.0)
        medium = ConstantMedium(boundary, 0.5, Color(1, 1, 1))

        # Ray through center
        ray = Ray(Point3(0, 0, -5), Vec3(0, 0, 1))

        # Hit should occur somewhere inside
        hit = medium.hit(ray, 0.001, float('inf'))

        # May or may not hit depending on random sampling
        if hit is not None:
            # Hit should be inside the sphere
            assert abs(hit.point.length()) < 2.0

    def test_high_density_almost_always_hits(self):
        boundary = Sphere(Point3(0, 0, 0), 2.0)
        medium = ConstantMedium(boundary, 100.0, Color(1, 1, 1))  # Very dense

        ray = Ray(Point3(0, 0, -5), Vec3(0, 0, 1))

        hits = sum(1 for _ in range(100) if medium.hit(ray, 0.001, float('inf')) is not None)

        # Should hit almost every time
        assert hits > 95

    def test_low_density_rarely_hits(self):
        boundary = Sphere(Point3(0, 0, 0), 2.0)
        medium = ConstantMedium(boundary, 0.01, Color(1, 1, 1))  # Very sparse

        ray = Ray(Point3(0, 0, -5), Vec3(0, 0, 1))

        hits = sum(1 for _ in range(100) if medium.hit(ray, 0.001, float('inf')) is not None)

        # Should rarely hit
        assert hits < 20

    def test_miss_outside_boundary(self):
        boundary = Sphere(Point3(0, 0, 0), 2.0)
        medium = ConstantMedium(boundary, 1.0, Color(1, 1, 1))

        # Ray that misses the sphere
        ray = Ray(Point3(10, 0, 0), Vec3(0, 1, 0))

        hit = medium.hit(ray, 0.001, float('inf'))
        assert hit is None

    def test_bounding_box(self):
        boundary = Sphere(Point3(0, 0, 0), 2.0)
        medium = ConstantMedium(boundary, 1.0, Color(1, 1, 1))

        bbox = medium.bounding_box()
        assert bbox is not None
        assert bbox.minimum.x == -2
        assert bbox.maximum.x == 2


class TestSubsurfaceScatteringMaterial:
    """Test SubsurfaceScatteringMaterial class."""

    def test_scatter_succeeds(self):
        mat = SubsurfaceScatteringMaterial(Color(0.8, 0.5, 0.5), scatter_distance=0.5)

        ray_in = Ray(Point3(0, 1, 0), Vec3(0, -1, 0))
        hit_point = Point3(0, 0, 0)
        normal = Vec3(0, 1, 0)

        for _ in range(100):
            result = mat.scatter(ray_in, hit_point, normal, True)
            assert result is not None

    def test_has_specular_component(self):
        mat = SubsurfaceScatteringMaterial(Color(0.8, 0.5, 0.5), roughness=0.0)

        ray_in = Ray(Point3(0, 1, 0), Vec3(0, -1, 0))
        hit_point = Point3(0, 0, 0)
        normal = Vec3(0, 1, 0)

        specular_count = 0
        for _ in range(100):
            result = mat.scatter(ray_in, hit_point, normal, True)
            if result.is_specular:
                specular_count += 1

        # Should have some specular reflections (Fresnel)
        assert specular_count > 0


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_create_fog(self):
        boundary = Sphere(Point3(0, 0, 0), 10.0)
        fog = create_fog(boundary, density=0.1, color=Color(1, 1, 1))

        assert fog is not None
        assert fog.neg_inv_density == -10.0  # 1/0.1

    def test_create_smoke(self):
        boundary = Sphere(Point3(0, 0, 0), 5.0)
        smoke = create_smoke(boundary, density=0.5)

        assert smoke is not None
        assert smoke.neg_inv_density == -2.0  # 1/0.5
