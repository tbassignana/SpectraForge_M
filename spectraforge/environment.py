"""
Environment map lighting (HDRI) for image-based lighting.

Implements:
- HDR image loading (.hdr, .exr formats via imageio)
- Equirectangular environment map sampling
- Importance sampling based on luminance
- Solid color and gradient backgrounds
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple, List
import math

import numpy as np

from .vec3 import Vec3, Color, Point3
from .ray import Ray


class Environment(ABC):
    """Abstract base class for environment lighting."""

    @abstractmethod
    def sample(self, direction: Vec3) -> Color:
        """Get the environment color for a given direction.

        Args:
            direction: The direction to sample (normalized)

        Returns:
            Color value from the environment
        """
        pass

    def importance_sample(self) -> Tuple[Vec3, Color, float]:
        """Sample a direction weighted by luminance.

        Returns:
            Tuple of (direction, color, pdf)
        """
        # Default: uniform sampling over sphere
        import random
        u = random.random()
        v = random.random()

        phi = 2.0 * math.pi * u
        cos_theta = 1.0 - 2.0 * v
        sin_theta = math.sqrt(1.0 - cos_theta * cos_theta)

        direction = Vec3(
            sin_theta * math.cos(phi),
            cos_theta,
            sin_theta * math.sin(phi)
        )

        color = self.sample(direction)
        pdf = 1.0 / (4.0 * math.pi)  # Uniform sphere PDF

        return direction, color, pdf


class SolidColorEnvironment(Environment):
    """A solid color environment (uniform sky)."""

    def __init__(self, color: Color = Color(0, 0, 0)):
        """Create a solid color environment.

        Args:
            color: The background color
        """
        self.color = color

    def sample(self, direction: Vec3) -> Color:
        return self.color


class GradientEnvironment(Environment):
    """A vertical gradient environment (simple sky)."""

    def __init__(
        self,
        horizon_color: Color = Color(1, 1, 1),
        zenith_color: Color = Color(0.5, 0.7, 1.0),
        ground_color: Optional[Color] = None
    ):
        """Create a gradient environment.

        Args:
            horizon_color: Color at the horizon
            zenith_color: Color at the top of the sky
            ground_color: Color below horizon (defaults to darker horizon)
        """
        self.horizon_color = horizon_color
        self.zenith_color = zenith_color
        self.ground_color = ground_color if ground_color else horizon_color * 0.5

    def sample(self, direction: Vec3) -> Color:
        # Y-up: positive y is up
        t = direction.y

        if t >= 0:
            # Above horizon: blend from horizon to zenith
            return self.horizon_color * (1 - t) + self.zenith_color * t
        else:
            # Below horizon: blend from horizon to ground
            t = -t
            return self.horizon_color * (1 - t) + self.ground_color * t


class HDRIEnvironment(Environment):
    """High Dynamic Range Image environment map.

    Uses equirectangular projection for spherical mapping.
    """

    def __init__(
        self,
        filename: str,
        intensity: float = 1.0,
        rotation: float = 0.0
    ):
        """Load an HDRI environment map.

        Args:
            filename: Path to the HDR image (.hdr, .exr, .png, .jpg)
            intensity: Intensity multiplier for the lighting
            rotation: Horizontal rotation in degrees
        """
        self.filename = filename
        self.intensity = intensity
        self.rotation = math.radians(rotation)
        self._data: Optional[np.ndarray] = None
        self._width = 0
        self._height = 0
        self._cdf: Optional[np.ndarray] = None
        self._marginal_cdf: Optional[np.ndarray] = None
        self._load_image()
        self._build_sampling_distribution()

    def _load_image(self) -> None:
        """Load the HDR image."""
        path = Path(self.filename)
        if not path.exists():
            raise FileNotFoundError(f"HDRI file not found: {self.filename}")

        # Try to load with imageio (supports .hdr, .exr)
        try:
            import imageio.v3 as iio
            self._data = iio.imread(self.filename).astype(np.float64)
        except ImportError:
            # Fallback to PIL for regular images
            from PIL import Image
            img = Image.open(self.filename)
            img = img.convert('RGB')
            self._data = np.array(img, dtype=np.float64) / 255.0
            # Apply rough sRGB to linear conversion
            self._data = np.power(self._data, 2.2)

        self._height, self._width = self._data.shape[:2]

        # Ensure 3 channels
        if len(self._data.shape) == 2:
            self._data = np.stack([self._data] * 3, axis=-1)
        elif self._data.shape[2] == 4:
            self._data = self._data[:, :, :3]

    def _build_sampling_distribution(self) -> None:
        """Build CDF for importance sampling based on luminance."""
        # Compute luminance for each pixel
        luminance = (
            0.2126 * self._data[:, :, 0] +
            0.7152 * self._data[:, :, 1] +
            0.0722 * self._data[:, :, 2]
        )

        # Apply solid angle correction (pixels near poles cover less area)
        # For equirectangular: sin(theta) where theta is latitude from pole
        theta = np.linspace(0, np.pi, self._height)
        sin_theta = np.sin(theta)
        luminance = luminance * sin_theta[:, np.newaxis]

        # Build 2D CDF
        # First, conditional CDF for each row (u given v)
        row_sums = luminance.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)  # Avoid division by zero
        conditional_pdf = luminance / row_sums

        self._conditional_cdf = np.cumsum(conditional_pdf, axis=1)
        # Normalize each row (avoid division by zero)
        row_max = self._conditional_cdf[:, -1:]
        row_max = np.where(row_max == 0, 1, row_max)
        self._conditional_cdf = self._conditional_cdf / row_max

        # Marginal CDF for v
        marginal_pdf = row_sums.flatten()
        total = marginal_pdf.sum()
        if total > 0:
            marginal_pdf = marginal_pdf / total
        self._marginal_cdf = np.cumsum(marginal_pdf)

        # Store total for PDF computation
        self._total_luminance = total

    def sample(self, direction: Vec3) -> Color:
        """Sample the environment map at a given direction."""
        if self._data is None:
            return Color(0, 0, 0)

        # Convert direction to spherical coordinates
        # Phi is azimuth (around Y axis), theta is elevation from +Y
        d = direction.normalize()

        # Apply rotation
        cos_rot = math.cos(self.rotation)
        sin_rot = math.sin(self.rotation)
        x_rot = d.x * cos_rot - d.z * sin_rot
        z_rot = d.x * sin_rot + d.z * cos_rot

        # Spherical coordinates
        theta = math.acos(max(-1, min(1, d.y)))  # [0, pi] from +Y
        phi = math.atan2(z_rot, x_rot)  # [-pi, pi]

        # Convert to UV coordinates
        u = (phi + math.pi) / (2 * math.pi)  # [0, 1]
        v = theta / math.pi  # [0, 1]

        # Sample the image
        i = int(u * (self._width - 1))
        j = int(v * (self._height - 1))

        pixel = self._data[j, i]
        return Color(pixel[0], pixel[1], pixel[2]) * self.intensity

    def importance_sample(self) -> Tuple[Vec3, Color, float]:
        """Sample a direction weighted by luminance for efficient sampling."""
        import random

        # Sample v from marginal CDF
        rv = random.random()
        j = np.searchsorted(self._marginal_cdf, rv)
        j = min(j, self._height - 1)

        # Sample u from conditional CDF at row j
        ru = random.random()
        i = np.searchsorted(self._conditional_cdf[j], ru)
        i = min(i, self._width - 1)

        # Convert pixel coordinates to UV
        u = (i + 0.5) / self._width
        v = (j + 0.5) / self._height

        # Convert UV to direction
        phi = 2 * math.pi * u - math.pi - self.rotation
        theta = math.pi * v

        sin_theta = math.sin(theta)
        direction = Vec3(
            sin_theta * math.cos(phi),
            math.cos(theta),
            sin_theta * math.sin(phi)
        )

        # Get color at this direction
        color = self.sample(direction)

        # Compute PDF
        # PDF = luminance(pixel) * sin(theta) / (sum of all weighted luminances)
        # Normalized by the solid angle element
        pixel = self._data[j, i]
        luminance = 0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2]

        if self._total_luminance > 0 and sin_theta > 1e-8:
            pdf = (luminance * sin_theta * self._width * self._height) / (
                self._total_luminance * 2 * math.pi * math.pi * sin_theta
            )
            pdf = max(pdf, 1e-8)
        else:
            pdf = 1.0 / (4.0 * math.pi)

        return direction, color, pdf

    def get_pdf(self, direction: Vec3) -> float:
        """Get the PDF for sampling a specific direction."""
        if self._data is None:
            return 1.0 / (4.0 * math.pi)

        d = direction.normalize()

        # Apply rotation
        cos_rot = math.cos(self.rotation)
        sin_rot = math.sin(self.rotation)
        x_rot = d.x * cos_rot - d.z * sin_rot
        z_rot = d.x * sin_rot + d.z * cos_rot

        theta = math.acos(max(-1, min(1, d.y)))
        phi = math.atan2(z_rot, x_rot)

        u = (phi + math.pi) / (2 * math.pi)
        v = theta / math.pi

        i = int(u * (self._width - 1))
        j = int(v * (self._height - 1))

        pixel = self._data[j, i]
        luminance = 0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2]
        sin_theta = math.sin(theta)

        if self._total_luminance > 0 and sin_theta > 1e-8:
            return (luminance * self._width * self._height) / (
                self._total_luminance * 2 * math.pi * math.pi
            )
        return 1.0 / (4.0 * math.pi)


class ProceduralSky(Environment):
    """Procedural physical sky model.

    A simple Preetham-style sky model based on sun position.
    """

    def __init__(
        self,
        sun_direction: Vec3 = Vec3(0, 1, 0),
        turbidity: float = 2.0,
        sun_intensity: float = 20.0,
        sun_color: Color = Color(1, 0.9, 0.8),
        sky_intensity: float = 1.0
    ):
        """Create a procedural sky.

        Args:
            sun_direction: Direction to the sun (normalized)
            turbidity: Atmospheric turbidity (2.0 = clear, 10.0 = hazy)
            sun_intensity: Intensity of the sun disk
            sun_color: Color of the sun
            sky_intensity: Overall sky brightness
        """
        self.sun_direction = sun_direction.normalize()
        self.turbidity = turbidity
        self.sun_intensity = sun_intensity
        self.sun_color = sun_color
        self.sky_intensity = sky_intensity

        # Sun angular radius (about 0.5 degrees)
        self.sun_angular_radius = math.radians(0.53)

    def sample(self, direction: Vec3) -> Color:
        d = direction.normalize()

        # Check if we're looking at the sun
        cos_angle = d.dot(self.sun_direction)
        if cos_angle > math.cos(self.sun_angular_radius):
            return self.sun_color * self.sun_intensity

        # Simple atmospheric scattering approximation
        # Based on height in sky and angle to sun

        # Height factor (y component)
        height = max(d.y, 0.01)

        # Angle to sun
        sun_angle = max(0, cos_angle)

        # Rayleigh scattering (blue at zenith)
        rayleigh = Color(0.3, 0.5, 0.8) * (1.0 - 0.5 * sun_angle)

        # Mie scattering (bright haze near sun)
        mie_strength = pow(max(0, cos_angle), 8) * (1.0 / self.turbidity)
        mie = self.sun_color * mie_strength

        # Height-based color shift
        horizon_blend = math.exp(-height * 3)
        horizon_color = Color(1, 0.7, 0.5) * horizon_blend

        # Combine
        sky = (rayleigh + mie + horizon_color) * self.sky_intensity

        # Darken below horizon
        if d.y < 0:
            sky = sky * max(0, 1 + d.y * 3)

        return sky

    def importance_sample(self) -> Tuple[Vec3, Color, float]:
        """Sample with bias toward the sun."""
        import random

        # 50% chance to sample toward sun, 50% uniform
        if random.random() < 0.5:
            # Cosine-weighted hemisphere around sun direction
            # Build tangent frame
            up = Vec3(0, 1, 0) if abs(self.sun_direction.y) < 0.999 else Vec3(1, 0, 0)
            tangent = up.cross(self.sun_direction).normalize()
            bitangent = self.sun_direction.cross(tangent)

            # Sample in cone around sun
            r1 = random.random()
            r2 = random.random()
            cone_angle = math.pi / 8  # 22.5 degree cone

            cos_theta = 1 - r1 * (1 - math.cos(cone_angle))
            sin_theta = math.sqrt(1 - cos_theta * cos_theta)
            phi = 2 * math.pi * r2

            local_dir = Vec3(
                sin_theta * math.cos(phi),
                sin_theta * math.sin(phi),
                cos_theta
            )

            direction = (
                tangent * local_dir.x +
                bitangent * local_dir.y +
                self.sun_direction * local_dir.z
            ).normalize()

            # Compute PDF for this sampling strategy
            solid_angle = 2 * math.pi * (1 - math.cos(cone_angle))
            pdf = 0.5 / solid_angle + 0.5 / (4 * math.pi)
        else:
            # Uniform sphere sampling
            u = random.random()
            v = random.random()
            phi = 2 * math.pi * u
            cos_theta = 1 - 2 * v
            sin_theta = math.sqrt(1 - cos_theta * cos_theta)

            direction = Vec3(
                sin_theta * math.cos(phi),
                cos_theta,
                sin_theta * math.sin(phi)
            )
            pdf = 1.0 / (4.0 * math.pi)

        color = self.sample(direction)
        return direction, color, pdf


def load_hdri(filename: str, intensity: float = 1.0, rotation: float = 0.0) -> HDRIEnvironment:
    """Convenience function to load an HDRI environment map.

    Args:
        filename: Path to the HDR image
        intensity: Brightness multiplier
        rotation: Horizontal rotation in degrees

    Returns:
        HDRIEnvironment instance
    """
    return HDRIEnvironment(filename, intensity, rotation)


def create_simple_sky(
    sun_elevation: float = 45.0,
    sun_azimuth: float = 0.0,
    turbidity: float = 2.0
) -> ProceduralSky:
    """Create a procedural sky with sun position from angles.

    Args:
        sun_elevation: Sun elevation in degrees (0 = horizon, 90 = zenith)
        sun_azimuth: Sun azimuth in degrees (0 = north, 90 = east)
        turbidity: Atmospheric haziness

    Returns:
        ProceduralSky instance
    """
    elev = math.radians(sun_elevation)
    azim = math.radians(sun_azimuth)

    sun_dir = Vec3(
        math.cos(elev) * math.sin(azim),
        math.sin(elev),
        math.cos(elev) * math.cos(azim)
    )

    return ProceduralSky(sun_direction=sun_dir, turbidity=turbidity)
