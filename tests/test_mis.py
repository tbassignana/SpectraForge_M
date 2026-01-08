"""Tests for Multiple Importance Sampling (MIS)."""

import pytest
import math
from spectraforge.vec3 import Vec3, Point3, Color
from spectraforge.ray import Ray
from spectraforge.shapes import Sphere, HittableList
from spectraforge.lights import PointLight, AreaLight, LightList, LightSample
from spectraforge.materials import Lambertian, Emissive, ScatterResult
from spectraforge.mis import (
    balance_heuristic, power_heuristic, multi_power_heuristic,
    MISIntegrator, one_sample_mis, NextEventEstimation
)


class TestBalanceHeuristic:
    """Test balance heuristic weights."""

    def test_equal_pdfs(self):
        w1, w2 = balance_heuristic(1.0, 1.0)
        assert abs(w1 - 0.5) < 1e-6
        assert abs(w2 - 0.5) < 1e-6

    def test_weights_sum_to_one(self):
        w1, w2 = balance_heuristic(3.0, 7.0)
        assert abs(w1 + w2 - 1.0) < 1e-6

    def test_higher_pdf_gets_higher_weight(self):
        w1, w2 = balance_heuristic(0.8, 0.2)
        assert w1 > w2
        assert abs(w1 - 0.8) < 1e-6
        assert abs(w2 - 0.2) < 1e-6

    def test_zero_pdfs(self):
        w1, w2 = balance_heuristic(0.0, 0.0)
        assert w1 == 0.5
        assert w2 == 0.5

    def test_one_zero_pdf(self):
        w1, w2 = balance_heuristic(1.0, 0.0)
        assert abs(w1 - 1.0) < 1e-6
        assert abs(w2) < 1e-6


class TestPowerHeuristic:
    """Test power heuristic weights."""

    def test_equal_pdfs(self):
        w1, w2 = power_heuristic(1.0, 1.0)
        assert abs(w1 - 0.5) < 1e-6
        assert abs(w2 - 0.5) < 1e-6

    def test_weights_sum_to_one(self):
        w1, w2 = power_heuristic(0.3, 0.7)
        assert abs(w1 + w2 - 1.0) < 1e-6

    def test_higher_pdf_gets_more_weight_than_balance(self):
        """Power heuristic should give even more weight to higher PDF."""
        bal_w1, bal_w2 = balance_heuristic(0.9, 0.1)
        pow_w1, pow_w2 = power_heuristic(0.9, 0.1)

        # Power heuristic should favor the higher PDF more
        assert pow_w1 > bal_w1
        assert pow_w2 < bal_w2

    def test_different_beta(self):
        """Higher beta = more extreme weighting."""
        w1_low, w2_low = power_heuristic(0.8, 0.2, beta=1.0)  # Same as balance
        w1_high, w2_high = power_heuristic(0.8, 0.2, beta=4.0)

        assert abs(w1_low - 0.8) < 1e-6  # Beta=1 is balance heuristic
        assert w1_high > w1_low  # Higher beta = more extreme

    def test_zero_pdfs(self):
        w1, w2 = power_heuristic(0.0, 0.0)
        assert w1 == 0.5
        assert w2 == 0.5


class TestMultiPowerHeuristic:
    """Test multi-strategy power heuristic."""

    def test_three_strategies(self):
        weights = multi_power_heuristic([1.0, 1.0, 1.0])
        assert len(weights) == 3
        assert abs(sum(weights) - 1.0) < 1e-6
        for w in weights:
            assert abs(w - 1/3) < 1e-6

    def test_varying_pdfs(self):
        weights = multi_power_heuristic([0.5, 0.3, 0.2])
        assert weights[0] > weights[1] > weights[2]
        assert abs(sum(weights) - 1.0) < 1e-6

    def test_empty_list(self):
        weights = multi_power_heuristic([])
        assert weights == []

    def test_single_pdf(self):
        weights = multi_power_heuristic([1.0])
        assert len(weights) == 1
        assert abs(weights[0] - 1.0) < 1e-6

    def test_all_zeros(self):
        weights = multi_power_heuristic([0.0, 0.0, 0.0])
        for w in weights:
            assert abs(w - 1/3) < 1e-6


class TestMISIntegrator:
    """Test MIS integrator functionality."""

    def test_creation(self):
        integrator = MISIntegrator()
        assert integrator.use_power_heuristic is True
        assert integrator.beta == 2.0

    def test_balance_mode(self):
        integrator = MISIntegrator(use_power_heuristic=False)
        w1, w2 = integrator.compute_weight(0.7, 0.3)
        bal_w1, bal_w2 = balance_heuristic(0.7, 0.3)
        assert abs(w1 - bal_w1) < 1e-6
        assert abs(w2 - bal_w2) < 1e-6

    def test_power_mode(self):
        integrator = MISIntegrator(use_power_heuristic=True)
        w1, w2 = integrator.compute_weight(0.7, 0.3)
        pow_w1, pow_w2 = power_heuristic(0.7, 0.3)
        assert abs(w1 - pow_w1) < 1e-6
        assert abs(w2 - pow_w2) < 1e-6

    def test_direct_lighting_with_lights(self):
        integrator = MISIntegrator()
        world = HittableList()
        lights = LightList()
        lights.add(PointLight(Point3(0, 5, 0), Color(1, 1, 1), intensity=10.0))

        hit_point = Point3(0, 0, 0)
        normal = Vec3(0, 1, 0)
        view_dir = Vec3(0, 1, 1).normalize()
        material = Lambertian(Color(1, 1, 1))

        result = integrator.sample_direct_lighting(
            hit_point, normal, view_dir, material, lights, world
        )

        # Should compute some lighting
        assert result.r >= 0

    def test_direct_lighting_in_shadow(self):
        integrator = MISIntegrator()

        # Create occluder
        world = HittableList()
        world.add(Sphere(Point3(0, 2.5, 0), 0.5))

        lights = LightList()
        lights.add(PointLight(Point3(0, 5, 0), Color(1, 1, 1), intensity=10.0))

        hit_point = Point3(0, 0, 0)
        normal = Vec3(0, 1, 0)
        view_dir = Vec3(0, 1, 1).normalize()
        material = Lambertian(Color(1, 1, 1))

        result = integrator.sample_direct_lighting(
            hit_point, normal, view_dir, material, lights, world
        )

        # Should be shadowed (or very dim)
        assert result.r < 0.5

    def test_empty_lights(self):
        integrator = MISIntegrator()
        world = HittableList()
        lights = LightList()

        hit_point = Point3(0, 0, 0)
        normal = Vec3(0, 1, 0)
        view_dir = Vec3(0, 1, 1).normalize()
        material = Lambertian(Color(1, 1, 1))

        result = integrator.sample_direct_lighting(
            hit_point, normal, view_dir, material, lights, world
        )

        assert result == Color(0, 0, 0)


class TestOneSampleMIS:
    """Test one-sample MIS estimator."""

    def test_produces_color(self):
        # Create sample data
        light = PointLight(Point3(0, 5, 0), Color(1, 1, 1), intensity=5.0)
        material = Lambertian(Color(1, 1, 1))

        hit_point = Point3(0, 0, 0)
        normal = Vec3(0, 1, 0)

        light_sample = light.sample(hit_point)

        # Create a BRDF sample
        incoming_ray = Ray(hit_point + Vec3(0, 1, 0), Vec3(0, -1, 0))
        brdf_sample = material.scatter(incoming_ray, hit_point, normal, True)

        # Run multiple times (random selection)
        for _ in range(10):
            result = one_sample_mis(
                hit_point, normal, material, light,
                brdf_sample, light_sample
            )
            # Result should be non-negative
            assert result.r >= 0

    def test_power_vs_balance(self):
        light = PointLight(Point3(0, 5, 0), Color(1, 1, 1), intensity=5.0)
        material = Lambertian(Color(1, 1, 1))

        hit_point = Point3(0, 0, 0)
        normal = Vec3(0, 1, 0)

        light_sample = light.sample(hit_point)
        incoming_ray = Ray(hit_point + Vec3(0, 1, 0), Vec3(0, -1, 0))
        brdf_sample = material.scatter(incoming_ray, hit_point, normal, True)

        # Both should work
        result_power = one_sample_mis(
            hit_point, normal, material, light,
            brdf_sample, light_sample, use_power_heuristic=True
        )
        result_balance = one_sample_mis(
            hit_point, normal, material, light,
            brdf_sample, light_sample, use_power_heuristic=False
        )

        assert result_power.r >= 0
        assert result_balance.r >= 0


class TestNextEventEstimation:
    """Test Next Event Estimation integrator."""

    def test_creation(self):
        nee = NextEventEstimation()
        assert nee.mis is not None

    def test_custom_mis(self):
        custom_mis = MISIntegrator(use_power_heuristic=False)
        nee = NextEventEstimation(mis_integrator=custom_mis)
        assert nee.mis is custom_mis

    def test_evaluate_with_material(self):
        from spectraforge.shapes import HitRecord

        nee = NextEventEstimation()
        world = HittableList()
        lights = LightList()
        lights.add(PointLight(Point3(0, 5, 0), Color(1, 1, 1), intensity=10.0))

        # Create hit record
        hit_record = HitRecord(
            point=Point3(0, 0, 0),
            normal=Vec3(0, 1, 0),
            t=1.0,
            front_face=True,
            material=Lambertian(Color(0.8, 0.8, 0.8)),
            u=0.5,
            v=0.5
        )

        incoming_ray = Ray(Point3(0, 1, 1), Vec3(0, -1, -1).normalize())
        throughput = Color(1, 1, 1)

        direct, next_ray, next_throughput = nee.evaluate(
            hit_record, incoming_ray, lights, world, throughput
        )

        # Should get some direct lighting
        assert direct.r >= 0
        # Should continue path
        assert next_ray is not None or next_throughput == Color(0, 0, 0)

    def test_evaluate_no_material(self):
        from spectraforge.shapes import HitRecord

        nee = NextEventEstimation()
        world = HittableList()
        lights = LightList()

        hit_record = HitRecord(
            point=Point3(0, 0, 0),
            normal=Vec3(0, 1, 0),
            t=1.0,
            front_face=True,
            material=None
        )

        incoming_ray = Ray(Point3(0, 1, 1), Vec3(0, -1, -1).normalize())
        throughput = Color(1, 1, 1)

        direct, next_ray, next_throughput = nee.evaluate(
            hit_record, incoming_ray, lights, world, throughput
        )

        # No material = no contribution
        assert direct == Color(0, 0, 0)
        assert next_ray is None


class TestMISProperties:
    """Test mathematical properties of MIS."""

    def test_weights_non_negative(self):
        """MIS weights should always be non-negative."""
        for pdf1 in [0.0, 0.1, 0.5, 1.0, 10.0]:
            for pdf2 in [0.0, 0.1, 0.5, 1.0, 10.0]:
                w1, w2 = balance_heuristic(pdf1, pdf2)
                assert w1 >= 0
                assert w2 >= 0

                w1, w2 = power_heuristic(pdf1, pdf2)
                assert w1 >= 0
                assert w2 >= 0

    def test_weights_bounded(self):
        """MIS weights should be in [0, 1]."""
        for pdf1 in [0.1, 0.5, 1.0, 5.0]:
            for pdf2 in [0.1, 0.5, 1.0, 5.0]:
                w1, w2 = power_heuristic(pdf1, pdf2)
                assert 0 <= w1 <= 1
                assert 0 <= w2 <= 1

    def test_symmetry(self):
        """Swapping PDFs should swap weights."""
        w1a, w2a = balance_heuristic(0.3, 0.7)
        w1b, w2b = balance_heuristic(0.7, 0.3)

        assert abs(w1a - w2b) < 1e-6
        assert abs(w2a - w1b) < 1e-6
