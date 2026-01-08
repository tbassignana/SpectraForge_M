"""
Multiple Importance Sampling (MIS) for variance reduction.

Implements:
- Balance heuristic (optimal for two strategies)
- Power heuristic (more robust for many strategies)
- Combined light and BRDF sampling
- One-sample MIS estimator
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Callable
import math
import random

from .vec3 import Vec3, Point3, Color
from .ray import Ray
from .shapes import Hittable, HitRecord
from .materials import Material, ScatterResult
from .lights import LightList, Light, LightSample


def balance_heuristic(pdf1: float, pdf2: float) -> Tuple[float, float]:
    """Compute balance heuristic weights for two sampling strategies.

    The balance heuristic is optimal when both strategies have similar
    variance characteristics.

    Args:
        pdf1: PDF of first strategy
        pdf2: PDF of second strategy

    Returns:
        Tuple of (weight1, weight2) that sum to 1
    """
    if pdf1 + pdf2 == 0:
        return 0.5, 0.5

    total = pdf1 + pdf2
    return pdf1 / total, pdf2 / total


def power_heuristic(pdf1: float, pdf2: float, beta: float = 2.0) -> Tuple[float, float]:
    """Compute power heuristic weights for two sampling strategies.

    The power heuristic with beta=2 is often more robust than the
    balance heuristic, especially when one strategy is much better
    in certain regions.

    Args:
        pdf1: PDF of first strategy
        pdf2: PDF of second strategy
        beta: Power exponent (default 2.0, Veach's recommendation)

    Returns:
        Tuple of (weight1, weight2) that sum to 1
    """
    if pdf1 + pdf2 == 0:
        return 0.5, 0.5

    p1 = pdf1 ** beta
    p2 = pdf2 ** beta
    total = p1 + p2

    if total == 0:
        return 0.5, 0.5

    return p1 / total, p2 / total


def multi_power_heuristic(pdfs: list[float], beta: float = 2.0) -> list[float]:
    """Compute power heuristic weights for multiple sampling strategies.

    Args:
        pdfs: List of PDFs from different strategies
        beta: Power exponent

    Returns:
        List of weights that sum to 1
    """
    if not pdfs:
        return []

    powers = [p ** beta for p in pdfs]
    total = sum(powers)

    if total == 0:
        return [1.0 / len(pdfs)] * len(pdfs)

    return [p / total for p in powers]


@dataclass
class MISSample:
    """Result of an MIS-weighted sample."""
    direction: Vec3
    color: Color
    pdf: float
    weight: float  # MIS weight
    is_light_sample: bool


class MISIntegrator:
    """Integrator using Multiple Importance Sampling.

    Combines BRDF sampling (material scattering) with direct light
    sampling for reduced variance in scenes with area lights.
    """

    def __init__(
        self,
        use_power_heuristic: bool = True,
        beta: float = 2.0
    ):
        """Create an MIS integrator.

        Args:
            use_power_heuristic: Use power heuristic (True) or balance heuristic (False)
            beta: Power exponent for power heuristic
        """
        self.use_power_heuristic = use_power_heuristic
        self.beta = beta

    def compute_weight(self, pdf1: float, pdf2: float) -> Tuple[float, float]:
        """Compute MIS weights for two strategies."""
        if self.use_power_heuristic:
            return power_heuristic(pdf1, pdf2, self.beta)
        else:
            return balance_heuristic(pdf1, pdf2)

    def sample_direct_lighting(
        self,
        hit_point: Point3,
        normal: Vec3,
        view_dir: Vec3,
        material: Material,
        lights: LightList,
        world: Hittable,
        ray_time: float = 0.0
    ) -> Color:
        """Compute direct lighting using MIS.

        This uses the one-sample MIS estimator, sampling either the light
        or the BRDF with 50% probability each.

        Args:
            hit_point: Surface point being shaded
            normal: Surface normal
            view_dir: Direction toward the camera
            material: Surface material
            lights: Available light sources
            world: Scene geometry for shadow rays
            ray_time: Time for motion blur

        Returns:
            MIS-weighted direct lighting contribution
        """
        if len(lights) == 0:
            return Color(0, 0, 0)

        result = Color(0, 0, 0)

        # Strategy 1: Sample the light
        light_sample, light_idx = lights.sample(hit_point)

        if light_sample.intensity.r > 0 or light_sample.intensity.g > 0 or light_sample.intensity.b > 0:
            # Check if light is on correct side
            n_dot_l = normal.dot(light_sample.direction)
            if n_dot_l > 0:
                # Shadow ray
                shadow_origin = hit_point + normal * 0.001
                shadow_ray = Ray(shadow_origin, light_sample.direction, ray_time)
                shadow_hit = world.hit(shadow_ray, 0.001, light_sample.distance - 0.001)

                if shadow_hit is None:
                    # Not in shadow - evaluate BRDF
                    # Create a "fake" incoming ray for the material
                    incoming_ray = Ray(hit_point + light_sample.direction, -light_sample.direction, ray_time)

                    # Get BRDF PDF for this direction
                    brdf_pdf = self._get_brdf_pdf(material, incoming_ray, hit_point, normal, light_sample.direction)

                    # MIS weight
                    w_light, _ = self.compute_weight(light_sample.pdf, brdf_pdf)

                    # Add contribution with MIS weight
                    contribution = light_sample.intensity * n_dot_l * w_light / light_sample.pdf
                    result = result + contribution

        # Strategy 2: Sample the BRDF
        # Create incoming ray (from camera direction)
        incoming_ray = Ray(hit_point + view_dir, -view_dir, ray_time)
        scatter_result = material.scatter(incoming_ray, hit_point, normal, True)

        if scatter_result is not None:
            scatter_dir = scatter_result.scattered_ray.direction

            # Check if we hit a light
            light_ray = Ray(hit_point + normal * 0.001, scatter_dir, ray_time)
            light_hit = world.hit(light_ray, 0.001, float('inf'))

            if light_hit is not None and light_hit.material is not None:
                emission = light_hit.material.emitted(light_hit.u, light_hit.v, light_hit.point)

                if emission.r > 0 or emission.g > 0 or emission.b > 0:
                    # We hit an emissive surface
                    brdf_pdf = self._get_brdf_pdf(material, incoming_ray, hit_point, normal, scatter_dir)

                    # Get light PDF for this direction
                    light_pdf = self._get_light_pdf(lights, hit_point, scatter_dir, light_hit.t)

                    # MIS weight
                    _, w_brdf = self.compute_weight(light_pdf, brdf_pdf)

                    # BRDF contribution
                    n_dot_l = max(0, normal.dot(scatter_dir))
                    contribution = emission * scatter_result.attenuation * n_dot_l * w_brdf / brdf_pdf

                    result = result + contribution

        return result

    def _get_brdf_pdf(
        self,
        material: Material,
        ray_in: Ray,
        hit_point: Point3,
        normal: Vec3,
        direction: Vec3
    ) -> float:
        """Estimate the BRDF PDF for a given direction.

        For Lambertian materials, this is cos(theta)/pi.
        For specular materials, this is a delta function.
        """
        # Cosine-weighted hemisphere PDF as approximation
        cos_theta = max(0, normal.dot(direction))
        return cos_theta / math.pi if cos_theta > 0 else 1e-8

    def _get_light_pdf(
        self,
        lights: LightList,
        hit_point: Point3,
        direction: Vec3,
        distance: float
    ) -> float:
        """Get the combined light PDF for sampling a given direction."""
        # This is an approximation - ideally each light would report its PDF
        # for the given direction
        total_pdf = 0.0

        for i, light in enumerate(lights):
            sample = light.sample(hit_point)
            # Check if this direction roughly matches the sample direction
            dot = direction.dot(sample.direction)
            if dot > 0.99:  # Similar direction
                selection_prob = lights.pdf(i)
                total_pdf += sample.pdf * selection_prob

        return max(total_pdf, 1e-8)


def one_sample_mis(
    hit_point: Point3,
    normal: Vec3,
    material: Material,
    light: Light,
    brdf_sample: ScatterResult,
    light_sample: LightSample,
    use_power_heuristic: bool = True
) -> Color:
    """Compute one-sample MIS estimator for a single light.

    Randomly choose between light sampling and BRDF sampling,
    apply MIS weight to reduce variance.

    Args:
        hit_point: Surface point
        normal: Surface normal
        material: Surface material
        light: Light source
        brdf_sample: BRDF scatter result
        light_sample: Light sample result
        use_power_heuristic: Use power (True) or balance (False) heuristic

    Returns:
        MIS-weighted contribution
    """
    if random.random() < 0.5:
        # Sample light
        n_dot_l = normal.dot(light_sample.direction)
        if n_dot_l <= 0:
            return Color(0, 0, 0)

        # BRDF value for light direction (simplified Lambertian)
        brdf_pdf = n_dot_l / math.pi

        if use_power_heuristic:
            w, _ = power_heuristic(light_sample.pdf, brdf_pdf)
        else:
            w, _ = balance_heuristic(light_sample.pdf, brdf_pdf)

        # Divide by 0.5 because we sampled with 50% probability
        return light_sample.intensity * n_dot_l * w / (light_sample.pdf * 0.5)
    else:
        # Sample BRDF
        n_dot_l = normal.dot(brdf_sample.scattered_ray.direction)
        if n_dot_l <= 0:
            return Color(0, 0, 0)

        brdf_pdf = n_dot_l / math.pi

        if use_power_heuristic:
            _, w = power_heuristic(light_sample.pdf, brdf_pdf)
        else:
            _, w = balance_heuristic(light_sample.pdf, brdf_pdf)

        return brdf_sample.attenuation * n_dot_l * w / (brdf_pdf * 0.5)


class NextEventEstimation:
    """Next Event Estimation (NEE) with MIS.

    Combines implicit path tracing with explicit light sampling.
    """

    def __init__(self, mis_integrator: MISIntegrator = None):
        """Create NEE integrator.

        Args:
            mis_integrator: MIS integrator to use (creates default if None)
        """
        self.mis = mis_integrator if mis_integrator else MISIntegrator()

    def evaluate(
        self,
        hit_record: HitRecord,
        incoming_ray: Ray,
        lights: LightList,
        world: Hittable,
        throughput: Color
    ) -> Tuple[Color, Optional[Ray], Color]:
        """Evaluate NEE at a hit point.

        Args:
            hit_record: Current hit information
            incoming_ray: Ray that hit the surface
            lights: Scene lights
            world: Scene geometry
            throughput: Current path throughput

        Returns:
            Tuple of (direct_contribution, next_ray, next_throughput)
        """
        if hit_record.material is None:
            return Color(0, 0, 0), None, Color(0, 0, 0)

        # Direct lighting with MIS
        view_dir = -incoming_ray.direction.normalize()
        direct = self.mis.sample_direct_lighting(
            hit_record.point,
            hit_record.normal,
            view_dir,
            hit_record.material,
            lights,
            world,
            incoming_ray.time
        )

        # Apply current throughput
        direct_contribution = throughput * direct

        # Continue path - sample BRDF
        scatter = hit_record.material.scatter(
            incoming_ray,
            hit_record.point,
            hit_record.normal,
            hit_record.front_face
        )

        if scatter is None:
            return direct_contribution, None, Color(0, 0, 0)

        next_ray = scatter.scattered_ray
        next_throughput = throughput * scatter.attenuation

        return direct_contribution, next_ray, next_throughput
