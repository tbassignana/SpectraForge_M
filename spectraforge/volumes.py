"""
Volumetric effects for the ray tracer.

Implements:
- Constant density volumes (fog, smoke)
- Subsurface scattering (SSS)
- Isotropic and anisotropic phase functions
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
import math
import random

from .vec3 import Vec3, Point3, Color
from .ray import Ray
from .shapes import Hittable, HitRecord, AABB
from .materials import Material, ScatterResult


class PhaseFunction(ABC):
    """Abstract base class for phase functions.

    Phase functions describe the angular distribution of scattered light
    in a participating medium.
    """

    @abstractmethod
    def sample(self, incoming: Vec3) -> Vec3:
        """Sample a scattered direction.

        Args:
            incoming: The incoming ray direction

        Returns:
            The scattered direction
        """
        pass

    @abstractmethod
    def pdf(self, incoming: Vec3, scattered: Vec3) -> float:
        """Get the probability density for the scattered direction.

        Args:
            incoming: The incoming ray direction
            scattered: The scattered direction

        Returns:
            The probability density
        """
        pass


class IsotropicPhase(PhaseFunction):
    """Isotropic phase function - scatters equally in all directions.

    Good for fog, smoke, and other diffuse participating media.
    """

    def sample(self, incoming: Vec3) -> Vec3:
        return Vec3.random_unit_vector()

    def pdf(self, incoming: Vec3, scattered: Vec3) -> float:
        return 1.0 / (4.0 * math.pi)


class HenyeyGreensteinPhase(PhaseFunction):
    """Henyey-Greenstein phase function.

    Provides control over forward/backward scattering with the g parameter:
    - g = 0: Isotropic
    - g > 0: Forward scattering (like fog)
    - g < 0: Backward scattering
    """

    def __init__(self, g: float = 0.0):
        """Create a Henyey-Greenstein phase function.

        Args:
            g: Asymmetry parameter (-1 to 1)
        """
        self.g = max(-0.999, min(0.999, g))

    def sample(self, incoming: Vec3) -> Vec3:
        g = self.g
        incoming_dir = incoming.normalize()

        if abs(g) < 1e-3:
            # Nearly isotropic
            return Vec3.random_unit_vector()

        # Sample cos(theta) using inverse CDF of Henyey-Greenstein
        u = random.random()
        # Correct formula for HG inverse CDF
        if abs(g) > 1e-3:
            sqr_term = (1 - g * g) / (1 - g + 2 * g * u)
            cos_theta = (1 + g * g - sqr_term * sqr_term) / (2 * g)
        else:
            cos_theta = 1 - 2 * u

        cos_theta = max(-1.0, min(1.0, cos_theta))
        sin_theta = math.sqrt(max(0, 1 - cos_theta * cos_theta))

        # Random phi
        phi = 2 * math.pi * random.random()

        # Build coordinate frame aligned with incoming direction
        # We want to scatter relative to -incoming (the direction light came from)
        forward = -incoming_dir

        # Find perpendicular vectors
        up = Vec3(0, 1, 0) if abs(forward.y) < 0.999 else Vec3(1, 0, 0)
        right = up.cross(forward).normalize()
        real_up = forward.cross(right)

        # Construct scattered direction in local frame, then transform
        local_x = sin_theta * math.cos(phi)
        local_y = sin_theta * math.sin(phi)
        local_z = cos_theta

        scattered = right * local_x + real_up * local_y + forward * local_z

        return scattered.normalize()

    def pdf(self, incoming: Vec3, scattered: Vec3) -> float:
        g = self.g
        cos_theta = -incoming.dot(scattered)

        denom = 1 + g * g - 2 * g * cos_theta
        if denom <= 0:
            return 0

        return (1 - g * g) / (4 * math.pi * math.pow(denom, 1.5))


class IsotropicMaterial(Material):
    """Material for volumes that scatters isotropically."""

    def __init__(self, albedo: Color):
        self.albedo = albedo
        self.phase = IsotropicPhase()

    def scatter(self, ray_in: Ray, hit_point: Vec3, normal: Vec3, front_face: bool) -> Optional[ScatterResult]:
        scattered_dir = self.phase.sample(ray_in.direction)
        scattered = Ray(hit_point, scattered_dir, ray_in.time)
        return ScatterResult(
            scattered_ray=scattered,
            attenuation=self.albedo,
            is_specular=False
        )


class ConstantMedium(Hittable):
    """A constant density participating medium.

    Can be used for fog, smoke, clouds, etc.
    The medium is defined by a boundary shape and a density.
    """

    def __init__(
        self,
        boundary: Hittable,
        density: float,
        color: Color,
        phase_g: float = 0.0
    ):
        """Create a constant density medium.

        Args:
            boundary: The shape that defines the medium's boundary
            density: The density of the medium (higher = more opaque)
            color: The albedo/color of the medium
            phase_g: Asymmetry parameter for Henyey-Greenstein (-1 to 1)
        """
        self.boundary = boundary
        self.neg_inv_density = -1.0 / density
        self.color = color

        if abs(phase_g) < 1e-3:
            self.phase = IsotropicPhase()
        else:
            self.phase = HenyeyGreensteinPhase(phase_g)

        # Material for scattering
        self.phase_material = IsotropicMaterial(color)

    def hit(self, ray: Ray, t_min: float, t_max: float) -> Optional[HitRecord]:
        """Sample the participating medium using delta tracking."""

        # Find entry and exit points
        hit1 = self.boundary.hit(ray, float('-inf'), float('inf'))
        if hit1 is None:
            return None

        hit2 = self.boundary.hit(ray, hit1.t + 0.0001, float('inf'))
        if hit2 is None:
            return None

        # Clamp to ray bounds
        if hit1.t < t_min:
            hit1.t = t_min
        if hit2.t > t_max:
            hit2.t = t_max

        if hit1.t >= hit2.t:
            return None

        if hit1.t < 0:
            hit1.t = 0

        # Sample distance using exponential distribution
        ray_length = ray.direction.length()
        distance_inside_boundary = (hit2.t - hit1.t) * ray_length
        hit_distance = self.neg_inv_density * math.log(random.random())

        if hit_distance > distance_inside_boundary:
            return None

        t = hit1.t + hit_distance / ray_length

        hit_record = HitRecord(
            point=ray.at(t),
            normal=Vec3(1, 0, 0),  # Arbitrary, not used for volumes
            t=t,
            front_face=True,
            material=self.phase_material,
            u=0,
            v=0
        )

        return hit_record

    def bounding_box(self) -> Optional[AABB]:
        return self.boundary.bounding_box()


class SubsurfaceScatteringMaterial(Material):
    """Subsurface scattering (SSS) material.

    Simulates light that penetrates the surface and scatters inside
    before exiting. Good for skin, wax, marble, jade, etc.

    Uses a simplified dipole approximation.
    """

    def __init__(
        self,
        color: Color,
        scatter_distance: float = 0.5,
        scatter_color: Color = None,
        ior: float = 1.5,
        roughness: float = 0.3
    ):
        """Create a subsurface scattering material.

        Args:
            color: Surface albedo
            scatter_distance: Mean free path length (how far light travels inside)
            scatter_color: Color of scattered light (defaults to color)
            ior: Index of refraction
            roughness: Surface roughness for specular component
        """
        self.color = color
        self.scatter_distance = scatter_distance
        self.scatter_color = scatter_color if scatter_color else color
        self.ior = ior
        self.roughness = roughness

        # Precompute extinction coefficient
        self.sigma_t = 1.0 / scatter_distance

    def scatter(self, ray_in: Ray, hit_point: Vec3, normal: Vec3, front_face: bool) -> Optional[ScatterResult]:
        """Scatter using a simplified SSS model."""
        view = -ray_in.direction.normalize()

        # Fresnel for specular reflection
        cos_theta = max(view.dot(normal), 0.0)
        fresnel = self._schlick(cos_theta)

        if random.random() < fresnel:
            # Specular reflection with roughness
            reflected = ray_in.direction.reflect(normal)
            if self.roughness > 0:
                reflected = reflected + Vec3.random_in_unit_sphere() * self.roughness
            scattered = Ray(hit_point, reflected.normalize(), ray_in.time)
            return ScatterResult(
                scattered_ray=scattered,
                attenuation=Color(1, 1, 1),
                is_specular=True
            )
        else:
            # Diffuse transmission with subsurface scattering approximation
            # Sample an exit point based on scatter distance
            scatter_dir = Vec3.random_in_hemisphere(-normal)

            # Exponential falloff based on scatter distance
            exit_distance = -self.scatter_distance * math.log(random.random() + 0.001)

            # Exit point is approximately where light would emerge
            # This is a simplified model; true SSS would trace through the object
            exit_point = hit_point + scatter_dir * exit_distance * 0.1

            # Final scattered direction
            final_dir = (normal + Vec3.random_unit_vector()).normalize()

            # Attenuation based on path length
            attenuation = self.scatter_color * math.exp(-exit_distance * self.sigma_t * 0.1)

            scattered = Ray(exit_point, final_dir, ray_in.time)
            return ScatterResult(
                scattered_ray=scattered,
                attenuation=attenuation,
                is_specular=False
            )

    def _schlick(self, cosine: float) -> float:
        """Schlick's approximation for Fresnel."""
        r0 = (1 - self.ior) / (1 + self.ior)
        r0 = r0 * r0
        return r0 + (1 - r0) * pow(1 - cosine, 5)


def create_fog(
    boundary: Hittable,
    density: float = 0.1,
    color: Color = Color(1, 1, 1)
) -> ConstantMedium:
    """Create a fog volume.

    Args:
        boundary: The shape defining the fog region
        density: Fog density (higher = more opaque)
        color: Fog color

    Returns:
        A ConstantMedium configured as fog
    """
    return ConstantMedium(boundary, density, color, phase_g=0.3)


def create_smoke(
    boundary: Hittable,
    density: float = 0.5,
    color: Color = Color(0.1, 0.1, 0.1)
) -> ConstantMedium:
    """Create a smoke volume.

    Args:
        boundary: The shape defining the smoke region
        density: Smoke density
        color: Smoke color (dark for realistic smoke)

    Returns:
        A ConstantMedium configured as smoke
    """
    return ConstantMedium(boundary, density, color, phase_g=-0.2)
