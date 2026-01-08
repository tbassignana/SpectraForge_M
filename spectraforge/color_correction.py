"""
Color correction and grading for rendered images.

Implements:
- Exposure adjustment
- Contrast control
- Saturation adjustment
- Color temperature/tint
- Shadow/midtone/highlight control
- LUT (Look-Up Table) support
- Vignette effect
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple
import math

import numpy as np


@dataclass
class ColorCorrectionSettings:
    """Settings for color correction pipeline."""

    # Basic adjustments
    exposure: float = 0.0  # EV stops (-5 to +5)
    contrast: float = 1.0  # Multiplier (0.5 to 2.0)
    saturation: float = 1.0  # Multiplier (0 to 2.0, 0 = grayscale)

    # Color temperature
    temperature: float = 0.0  # Kelvin offset (-100 to +100, - = cooler, + = warmer)
    tint: float = 0.0  # Green-magenta (-100 to +100)

    # Tone adjustments
    shadows: float = 0.0  # Shadow lift (-1 to +1)
    midtones: float = 0.0  # Midtone adjustment (-1 to +1)
    highlights: float = 0.0  # Highlight adjustment (-1 to +1)

    # Color channel adjustments
    red_gain: float = 1.0
    green_gain: float = 1.0
    blue_gain: float = 1.0

    # Vignette
    vignette_intensity: float = 0.0  # 0 = none, 1 = strong
    vignette_softness: float = 0.5  # Edge softness


@dataclass
class ColorCorrectionResult:
    """Result of color correction."""

    image: np.ndarray  # Corrected image (H, W, 3)
    settings: ColorCorrectionSettings


class ColorCorrector:
    """Color correction and grading processor.

    Applies various color adjustments in a specific order for
    predictable, high-quality results.
    """

    def __init__(self, settings: Optional[ColorCorrectionSettings] = None):
        """Initialize color corrector.

        Args:
            settings: Color correction settings (uses defaults if None)
        """
        self.settings = settings or ColorCorrectionSettings()

    def _rgb_to_luminance(self, rgb: np.ndarray) -> np.ndarray:
        """Convert RGB to luminance using Rec. 709."""
        return 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]

    def _apply_exposure(self, image: np.ndarray) -> np.ndarray:
        """Apply exposure adjustment.

        Args:
            image: Input image (H, W, 3)

        Returns:
            Exposure-adjusted image
        """
        if self.settings.exposure == 0.0:
            return image

        # Exposure in EV stops: multiply by 2^EV
        multiplier = 2.0 ** self.settings.exposure
        return image * multiplier

    def _apply_contrast(self, image: np.ndarray) -> np.ndarray:
        """Apply contrast adjustment.

        Uses pivot point at middle gray (0.18 in linear space).
        """
        if self.settings.contrast == 1.0:
            return image

        # Pivot at middle gray
        pivot = 0.18
        result = pivot + (image - pivot) * self.settings.contrast
        return np.maximum(result, 0.0)

    def _apply_saturation(self, image: np.ndarray) -> np.ndarray:
        """Apply saturation adjustment.

        Interpolates between grayscale and original color.
        """
        if self.settings.saturation == 1.0:
            return image

        # Compute luminance
        luminance = self._rgb_to_luminance(image)
        luminance = luminance[:, :, np.newaxis]

        # Interpolate between grayscale and color
        # sat=0: grayscale, sat=1: original, sat>1: oversaturated
        result = luminance + self.settings.saturation * (image - luminance)
        return np.maximum(result, 0.0)

    def _apply_temperature_tint(self, image: np.ndarray) -> np.ndarray:
        """Apply color temperature and tint adjustment.

        Temperature affects blue-orange axis.
        Tint affects green-magenta axis.
        """
        if self.settings.temperature == 0.0 and self.settings.tint == 0.0:
            return image

        result = image.copy()

        # Temperature adjustment (simplified model)
        # Positive = warmer (more orange/yellow)
        # Negative = cooler (more blue)
        temp_scale = self.settings.temperature / 100.0

        if temp_scale > 0:
            # Warmer: increase red/yellow, decrease blue
            result[:, :, 0] *= 1.0 + temp_scale * 0.3  # Red
            result[:, :, 2] *= 1.0 - temp_scale * 0.2  # Blue
        else:
            # Cooler: increase blue, decrease red
            result[:, :, 0] *= 1.0 + temp_scale * 0.2  # Red (temp_scale is negative)
            result[:, :, 2] *= 1.0 - temp_scale * 0.3  # Blue

        # Tint adjustment
        # Positive = more magenta
        # Negative = more green
        tint_scale = self.settings.tint / 100.0
        result[:, :, 1] *= 1.0 - tint_scale * 0.2  # Green

        return np.maximum(result, 0.0)

    def _apply_shadows_midtones_highlights(self, image: np.ndarray) -> np.ndarray:
        """Apply shadow/midtone/highlight adjustments.

        Uses luminance-based weighting to target specific tonal ranges.
        """
        if (self.settings.shadows == 0.0 and
            self.settings.midtones == 0.0 and
            self.settings.highlights == 0.0):
            return image

        luminance = self._rgb_to_luminance(image)

        # Create weight masks for each tonal range
        # Shadows: low luminance (centered around 0.1)
        shadow_weight = 1.0 - self._smoothstep(0.0, 0.3, luminance)

        # Highlights: high luminance (centered around 0.8)
        highlight_weight = self._smoothstep(0.5, 1.0, luminance)

        # Midtones: middle luminance (1 - shadows - highlights)
        midtone_weight = 1.0 - shadow_weight - highlight_weight
        midtone_weight = np.maximum(midtone_weight, 0.0)

        # Apply adjustments
        result = image.copy()

        # Shadow lift/lower
        if self.settings.shadows != 0.0:
            adjustment = self.settings.shadows * 0.5  # Scale factor
            for i in range(3):
                result[:, :, i] += shadow_weight * adjustment

        # Midtone adjustment
        if self.settings.midtones != 0.0:
            adjustment = self.settings.midtones * 0.3
            for i in range(3):
                result[:, :, i] *= 1.0 + midtone_weight * adjustment

        # Highlight adjustment
        if self.settings.highlights != 0.0:
            adjustment = self.settings.highlights * 0.3
            for i in range(3):
                result[:, :, i] *= 1.0 + highlight_weight * adjustment

        return np.maximum(result, 0.0)

    def _smoothstep(self, edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
        """Smooth hermite interpolation."""
        t = np.clip((x - edge0) / (edge1 - edge0 + 1e-8), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    def _apply_channel_gains(self, image: np.ndarray) -> np.ndarray:
        """Apply per-channel gain adjustments."""
        if (self.settings.red_gain == 1.0 and
            self.settings.green_gain == 1.0 and
            self.settings.blue_gain == 1.0):
            return image

        result = image.copy()
        result[:, :, 0] *= self.settings.red_gain
        result[:, :, 1] *= self.settings.green_gain
        result[:, :, 2] *= self.settings.blue_gain

        return np.maximum(result, 0.0)

    def _apply_vignette(self, image: np.ndarray) -> np.ndarray:
        """Apply vignette effect (darkening at edges).

        Args:
            image: Input image (H, W, 3)

        Returns:
            Image with vignette applied
        """
        if self.settings.vignette_intensity == 0.0:
            return image

        h, w = image.shape[:2]

        # Create distance map from center
        y_coords = np.linspace(-1, 1, h)
        x_coords = np.linspace(-1, 1, w)
        xx, yy = np.meshgrid(x_coords, y_coords)
        distance = np.sqrt(xx ** 2 + yy ** 2)

        # Normalize to [0, 1] range
        distance = distance / math.sqrt(2)  # Max distance is sqrt(2)

        # Apply softness
        # Higher softness = larger bright center
        vignette = 1.0 - self._smoothstep(
            self.settings.vignette_softness,
            1.0,
            distance
        )

        # Scale by intensity
        vignette = 1.0 - (1.0 - vignette) * self.settings.vignette_intensity

        # Apply to image
        result = image.copy()
        for i in range(3):
            result[:, :, i] *= vignette

        return result

    def apply(self, image: np.ndarray) -> ColorCorrectionResult:
        """Apply full color correction pipeline.

        Order of operations:
        1. Exposure
        2. Temperature/Tint
        3. Contrast
        4. Shadows/Midtones/Highlights
        5. Saturation
        6. Channel gains
        7. Vignette

        Args:
            image: Input image (H, W, 3), linear HDR

        Returns:
            ColorCorrectionResult with corrected image
        """
        result = image.copy()

        # Apply corrections in order
        result = self._apply_exposure(result)
        result = self._apply_temperature_tint(result)
        result = self._apply_contrast(result)
        result = self._apply_shadows_midtones_highlights(result)
        result = self._apply_saturation(result)
        result = self._apply_channel_gains(result)
        result = self._apply_vignette(result)

        return ColorCorrectionResult(
            image=result,
            settings=self.settings
        )


class LUT:
    """3D Look-Up Table for color grading.

    Supports loading and applying 3D LUTs in .cube format,
    commonly used for cinematic color grading.
    """

    def __init__(self, data: np.ndarray, size: int):
        """Initialize LUT.

        Args:
            data: 3D LUT data (size, size, size, 3)
            size: LUT size (typically 17, 33, or 65)
        """
        self.data = data
        self.size = size

    @classmethod
    def from_cube_file(cls, filepath: str) -> "LUT":
        """Load LUT from .cube file.

        Args:
            filepath: Path to .cube file

        Returns:
            LUT instance
        """
        with open(filepath, 'r') as f:
            lines = f.readlines()

        size = 0
        data_lines = []

        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            if line.startswith('LUT_3D_SIZE'):
                size = int(line.split()[1])
            elif line.startswith('TITLE') or line.startswith('DOMAIN'):
                continue
            else:
                # Data line
                try:
                    values = [float(x) for x in line.split()]
                    if len(values) == 3:
                        data_lines.append(values)
                except ValueError:
                    continue

        if size == 0:
            raise ValueError("Could not find LUT_3D_SIZE in cube file")

        if len(data_lines) != size ** 3:
            raise ValueError(
                f"Expected {size**3} data lines, got {len(data_lines)}"
            )

        # Reshape data to 3D LUT
        data = np.array(data_lines).reshape((size, size, size, 3))

        return cls(data, size)

    @classmethod
    def identity(cls, size: int = 33) -> "LUT":
        """Create identity LUT (no color change).

        Args:
            size: LUT size

        Returns:
            Identity LUT
        """
        data = np.zeros((size, size, size, 3))
        for r in range(size):
            for g in range(size):
                for b in range(size):
                    data[r, g, b] = [
                        r / (size - 1),
                        g / (size - 1),
                        b / (size - 1)
                    ]
        return cls(data, size)

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply LUT to image using trilinear interpolation.

        Args:
            image: Input image (H, W, 3), values in [0, 1]

        Returns:
            Color-graded image (H, W, 3)
        """
        # Clamp input to [0, 1]
        img = np.clip(image, 0.0, 1.0)

        # Scale to LUT indices
        scale = self.size - 1
        scaled = img * scale

        # Get integer and fractional parts
        indices = np.floor(scaled).astype(np.int32)
        fractions = scaled - indices

        # Clamp indices
        i0 = np.clip(indices, 0, self.size - 2)
        i1 = i0 + 1

        # Get fractional weights
        fr = fractions[:, :, 0:1]
        fg = fractions[:, :, 1:2]
        fb = fractions[:, :, 2:3]

        # Trilinear interpolation
        h, w = image.shape[:2]
        result = np.zeros_like(image)

        for y in range(h):
            for x in range(w):
                r0, g0, b0 = i0[y, x]
                r1, g1, b1 = i1[y, x]
                rf, gf, bf = fractions[y, x]

                # 8 corner values
                c000 = self.data[r0, g0, b0]
                c001 = self.data[r0, g0, b1]
                c010 = self.data[r0, g1, b0]
                c011 = self.data[r0, g1, b1]
                c100 = self.data[r1, g0, b0]
                c101 = self.data[r1, g0, b1]
                c110 = self.data[r1, g1, b0]
                c111 = self.data[r1, g1, b1]

                # Interpolate along R axis
                c00 = c000 * (1 - rf) + c100 * rf
                c01 = c001 * (1 - rf) + c101 * rf
                c10 = c010 * (1 - rf) + c110 * rf
                c11 = c011 * (1 - rf) + c111 * rf

                # Interpolate along G axis
                c0 = c00 * (1 - gf) + c10 * gf
                c1 = c01 * (1 - gf) + c11 * gf

                # Interpolate along B axis
                result[y, x] = c0 * (1 - bf) + c1 * bf

        return result


class Vignette:
    """Standalone vignette effect.

    Creates darkening at image edges, simulating camera lens vignetting.
    """

    def __init__(
        self,
        intensity: float = 0.5,
        softness: float = 0.5,
        roundness: float = 1.0,
        center: Tuple[float, float] = (0.5, 0.5),
    ):
        """Initialize vignette.

        Args:
            intensity: Vignette strength (0 to 1)
            softness: Edge softness (0 = hard, 1 = soft)
            roundness: Shape (1 = circular, <1 = more rectangular)
            center: Vignette center as normalized coordinates
        """
        self.intensity = intensity
        self.softness = softness
        self.roundness = roundness
        self.center = center

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply vignette to image.

        Args:
            image: Input image (H, W, 3)

        Returns:
            Image with vignette applied
        """
        h, w = image.shape[:2]

        # Create coordinate grid centered at vignette center
        y_coords = np.linspace(0, 1, h) - self.center[1]
        x_coords = np.linspace(0, 1, w) - self.center[0]
        xx, yy = np.meshgrid(x_coords, y_coords)

        # Aspect ratio correction
        aspect = w / h
        xx *= aspect

        # Calculate distance with roundness
        if self.roundness < 1.0:
            # More rectangular
            distance = np.maximum(np.abs(xx), np.abs(yy))
        else:
            # Circular
            distance = np.sqrt(xx ** 2 + yy ** 2)

        # Normalize distance
        max_dist = math.sqrt(0.5 ** 2 + (0.5 * aspect) ** 2)
        distance = distance / max_dist

        # Apply softness using smoothstep
        inner = self.softness
        outer = 1.0
        t = np.clip((distance - inner) / (outer - inner + 1e-8), 0.0, 1.0)
        vignette_mask = 1.0 - (t * t * (3.0 - 2.0 * t))

        # Apply intensity
        vignette_mask = 1.0 - (1.0 - vignette_mask) * self.intensity

        # Apply to image
        result = image.copy()
        for i in range(3):
            result[:, :, i] *= vignette_mask

        return result


def apply_color_correction(
    image: np.ndarray,
    exposure: float = 0.0,
    contrast: float = 1.0,
    saturation: float = 1.0,
    temperature: float = 0.0,
    tint: float = 0.0,
) -> ColorCorrectionResult:
    """Convenience function for basic color correction.

    Args:
        image: Input image (H, W, 3)
        exposure: Exposure adjustment in EV
        contrast: Contrast multiplier
        saturation: Saturation multiplier
        temperature: Color temperature offset
        tint: Green-magenta tint offset

    Returns:
        ColorCorrectionResult
    """
    settings = ColorCorrectionSettings(
        exposure=exposure,
        contrast=contrast,
        saturation=saturation,
        temperature=temperature,
        tint=tint,
    )
    corrector = ColorCorrector(settings)
    return corrector.apply(image)
