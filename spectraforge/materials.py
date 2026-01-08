"""
Materials system with PBR (Physically Based Rendering) support.

Implements:
- Lambertian diffuse
- Metal (specular reflection with roughness)
- Dielectric (glass, water - with refraction)
- PBR material (GGX/Cook-Torrance BRDF)
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple
import math

from .vec3 import Vec3, Color
from .ray import Ray


@dataclass
class ScatterResult:
    """Result of a material scatter operation."""
    scattered_ray: Ray
    attenuation: Color
    is_specular: bool = False


class Material(ABC):
    """Abstract base class for materials."""

    @abstractmethod
    def scatter(self, ray_in: Ray, hit_point: Vec3, normal: Vec3, front_face: bool) -> Optional[ScatterResult]:
        """Compute the scattered ray and attenuation.

        Args:
            ray_in: The incoming ray
            hit_point: Point of intersection
            normal: Surface normal at hit point
            front_face: Whether ray hit from outside

        Returns:
            ScatterResult if ray scatters, None if absorbed
        """
        pass

    def emitted(self, u: float, v: float, point: Vec3) -> Color:
        """Return emitted light color. Default is no emission."""
        return Color(0, 0, 0)


class Lambertian(Material):
    """Diffuse material with Lambertian (ideal matte) scattering."""

    def __init__(self, albedo: Color):
        """Create a Lambertian material.

        Args:
            albedo: The base color (RGB, each component 0-1)
        """
        self.albedo = albedo
        self.texture = None

    def scatter(self, ray_in: Ray, hit_point: Vec3, normal: Vec3, front_face: bool, u: float = 0, v: float = 0) -> Optional[ScatterResult]:
        scatter_direction = normal + Vec3.random_unit_vector()

        # Catch degenerate scatter direction
        if scatter_direction.near_zero():
            scatter_direction = normal

        scattered = Ray(hit_point, scatter_direction.normalize(), ray_in.time)

        # Use texture if available
        if self.texture is not None:
            attenuation = self.texture.value(u, v, hit_point)
        else:
            attenuation = self.albedo

        return ScatterResult(
            scattered_ray=scattered,
            attenuation=attenuation,
            is_specular=False
        )


class TexturedLambertian(Material):
    """Diffuse material with texture support."""

    def __init__(self, texture):
        """Create a textured Lambertian material.

        Args:
            texture: A Texture object for the albedo
        """
        from .textures import Texture
        self.texture = texture

    def scatter(self, ray_in: Ray, hit_point: Vec3, normal: Vec3, front_face: bool, u: float = 0, v: float = 0) -> Optional[ScatterResult]:
        scatter_direction = normal + Vec3.random_unit_vector()

        if scatter_direction.near_zero():
            scatter_direction = normal

        scattered = Ray(hit_point, scatter_direction.normalize(), ray_in.time)
        attenuation = self.texture.value(u, v, hit_point)

        return ScatterResult(
            scattered_ray=scattered,
            attenuation=attenuation,
            is_specular=False
        )


class Metal(Material):
    """Metallic material with specular reflection."""

    def __init__(self, albedo: Color, roughness: float = 0.0):
        """Create a metal material.

        Args:
            albedo: The reflection color
            roughness: Surface roughness (0 = mirror, 1 = very rough)
        """
        self.albedo = albedo
        self.roughness = min(roughness, 1.0)

    def scatter(self, ray_in: Ray, hit_point: Vec3, normal: Vec3, front_face: bool) -> Optional[ScatterResult]:
        reflected = ray_in.direction.normalize().reflect(normal)

        # Add roughness as random perturbation
        if self.roughness > 0:
            reflected = reflected + Vec3.random_in_unit_sphere() * self.roughness

        scattered = Ray(hit_point, reflected.normalize(), ray_in.time)

        # Only scatter if reflection is in the correct hemisphere
        if scattered.direction.dot(normal) > 0:
            return ScatterResult(
                scattered_ray=scattered,
                attenuation=self.albedo,
                is_specular=True
            )
        return None


class Dielectric(Material):
    """Dielectric (glass-like) material with refraction."""

    def __init__(self, ior: float = 1.5, tint: Color = None):
        """Create a dielectric material.

        Args:
            ior: Index of refraction (1.0 = air, 1.5 = glass, 2.4 = diamond)
            tint: Optional color tint for the glass
        """
        self.ior = ior
        self.tint = tint if tint else Color(1, 1, 1)

    def scatter(self, ray_in: Ray, hit_point: Vec3, normal: Vec3, front_face: bool) -> Optional[ScatterResult]:
        attenuation = self.tint

        # Determine refraction ratio based on whether we're entering or exiting
        refraction_ratio = 1.0 / self.ior if front_face else self.ior

        unit_direction = ray_in.direction.normalize()
        cos_theta = min(-unit_direction.dot(normal), 1.0)
        sin_theta = math.sqrt(1.0 - cos_theta * cos_theta)

        cannot_refract = refraction_ratio * sin_theta > 1.0

        # Use Schlick's approximation for reflectance
        if cannot_refract or self._reflectance(cos_theta, refraction_ratio) > Vec3.random().x:
            direction = unit_direction.reflect(normal)
        else:
            direction = unit_direction.refract(normal, refraction_ratio)

        scattered = Ray(hit_point, direction.normalize(), ray_in.time)
        return ScatterResult(
            scattered_ray=scattered,
            attenuation=attenuation,
            is_specular=True
        )

    @staticmethod
    def _reflectance(cosine: float, ref_idx: float) -> float:
        """Schlick's approximation for reflectance."""
        r0 = (1 - ref_idx) / (1 + ref_idx)
        r0 = r0 * r0
        return r0 + (1 - r0) * pow(1 - cosine, 5)


class Emissive(Material):
    """Light-emitting material."""

    def __init__(self, color: Color, intensity: float = 1.0):
        """Create an emissive material.

        Args:
            color: The emission color
            intensity: Emission intensity multiplier
        """
        self.color = color
        self.intensity = intensity

    def scatter(self, ray_in: Ray, hit_point: Vec3, normal: Vec3, front_face: bool) -> Optional[ScatterResult]:
        return None

    def emitted(self, u: float, v: float, point: Vec3) -> Color:
        return self.color * self.intensity


class PBRMaterial(Material):
    """Physically Based Rendering material using Cook-Torrance BRDF.

    This implements the industry-standard PBR workflow with:
    - Metallic-roughness parameterization
    - GGX/Trowbridge-Reitz normal distribution
    - Smith geometry function
    - Fresnel-Schlick approximation
    """

    def __init__(
        self,
        albedo: Color = Color(0.8, 0.8, 0.8),
        metallic: float = 0.0,
        roughness: float = 0.5,
        ior: float = 1.5,
        emission: Color = None,
        emission_strength: float = 0.0
    ):
        """Create a PBR material.

        Args:
            albedo: Base color
            metallic: Metalness (0 = dielectric, 1 = metal)
            roughness: Surface roughness (0 = smooth, 1 = rough)
            ior: Index of refraction for dielectrics
            emission: Emission color
            emission_strength: Emission intensity
        """
        self.albedo = albedo
        self.metallic = max(0.0, min(1.0, metallic))
        self.roughness = max(0.01, min(1.0, roughness))  # Clamp to avoid div by zero
        self.ior = ior
        self.emission = emission if emission else Color(0, 0, 0)
        self.emission_strength = emission_strength

        # Precompute F0 (Fresnel at normal incidence)
        # For dielectrics, use IOR-based formula; for metals, use albedo
        f0_dielectric = ((ior - 1) / (ior + 1)) ** 2
        self.f0 = Color(f0_dielectric, f0_dielectric, f0_dielectric) * (1 - self.metallic) + self.albedo * self.metallic

    def scatter(self, ray_in: Ray, hit_point: Vec3, normal: Vec3, front_face: bool) -> Optional[ScatterResult]:
        """Sample the PBR BRDF using importance sampling."""
        # For simplicity, we use a hybrid approach:
        # - Sample either diffuse or specular based on Fresnel term
        # - This is a Monte Carlo approximation of the full BRDF

        view = -ray_in.direction.normalize()
        cos_theta = max(view.dot(normal), 0.0)

        # Fresnel term determines reflection probability
        fresnel = self._fresnel_schlick(cos_theta, self.f0)
        reflect_prob = (fresnel.x + fresnel.y + fresnel.z) / 3.0

        # Metals don't have diffuse, increase reflect probability
        reflect_prob = reflect_prob * (1 - self.metallic) + self.metallic

        import random
        if random.random() < reflect_prob:
            # Specular reflection with GGX importance sampling
            direction = self._sample_ggx(normal, view)
            if direction.dot(normal) <= 0:
                return None

            scattered = Ray(hit_point, direction.normalize(), ray_in.time)

            # For metals, tint reflection with albedo
            attenuation = fresnel * (1 - self.metallic) + self.albedo * fresnel * self.metallic

            return ScatterResult(
                scattered_ray=scattered,
                attenuation=attenuation,
                is_specular=True
            )
        else:
            # Diffuse (Lambertian)
            scatter_direction = normal + Vec3.random_unit_vector()
            if scatter_direction.near_zero():
                scatter_direction = normal

            scattered = Ray(hit_point, scatter_direction.normalize(), ray_in.time)

            # Diffuse uses albedo, weighted by inverse Fresnel
            attenuation = self.albedo * (1 - fresnel.x) * (1 - self.metallic)

            return ScatterResult(
                scattered_ray=scattered,
                attenuation=attenuation,
                is_specular=False
            )

    def _fresnel_schlick(self, cos_theta: float, f0: Color) -> Color:
        """Fresnel-Schlick approximation."""
        return f0 + (Color(1, 1, 1) - f0) * pow(1.0 - cos_theta, 5)

    def _sample_ggx(self, normal: Vec3, view: Vec3) -> Vec3:
        """Sample the GGX distribution for importance sampling."""
        import random

        alpha = self.roughness * self.roughness
        alpha2 = alpha * alpha

        # Sample spherical coordinates
        u1 = random.random()
        u2 = random.random()

        phi = 2.0 * math.pi * u1
        cos_theta = math.sqrt((1.0 - u2) / (1.0 + (alpha2 - 1.0) * u2))
        sin_theta = math.sqrt(1.0 - cos_theta * cos_theta)

        # Convert to Cartesian in tangent space
        h_tangent = Vec3(
            sin_theta * math.cos(phi),
            sin_theta * math.sin(phi),
            cos_theta
        )

        # Build tangent space basis
        up = Vec3(0, 1, 0) if abs(normal.y) < 0.999 else Vec3(1, 0, 0)
        tangent = up.cross(normal).normalize()
        bitangent = normal.cross(tangent)

        # Transform to world space (h is the microfacet normal / half vector)
        h = tangent * h_tangent.x + bitangent * h_tangent.y + normal * h_tangent.z
        h = h.normalize()

        # Reflect view around half vector to get light direction (outgoing)
        # The reflected direction is: 2*(V·H)*H - V
        v_dot_h = view.dot(h)
        reflected = h * 2.0 * v_dot_h - view
        return reflected.normalize()

    def emitted(self, u: float, v: float, point: Vec3) -> Color:
        return self.emission * self.emission_strength
