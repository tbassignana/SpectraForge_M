"""
Tone mapping operators for HDR to LDR conversion.

Implements various tone mapping algorithms:
- Reinhard (global and local)
- ACES Filmic (Academy Color Encoding System)
- Uncharted 2 Filmic (Hable)
- Exposure-based
- Gamma correction
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import math

import numpy as np


class ToneMappingOperator(Enum):
    """Available tone mapping operators."""
    LINEAR = "linear"
    REINHARD = "reinhard"
    REINHARD_EXTENDED = "reinhard_extended"
    ACES_FILMIC = "aces_filmic"
    UNCHARTED2 = "uncharted2"
    EXPOSURE = "exposure"


@dataclass
class ToneMappingResult:
    """Result of tone mapping operation."""

    image: np.ndarray  # LDR image (H, W, 3), values in [0, 1]
    operator: str  # Name of operator used
    parameters: dict  # Parameters used for the operation


class ToneMapper(ABC):
    """Abstract base class for tone mapping operators."""

    @abstractmethod
    def apply(self, hdr_image: np.ndarray) -> np.ndarray:
        """Apply tone mapping to HDR image.

        Args:
            hdr_image: HDR image (H, W, 3), linear float values

        Returns:
            LDR image (H, W, 3), values in [0, 1]
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the operator."""
        pass

    def get_parameters(self) -> dict:
        """Get operator parameters for result metadata."""
        return {}


class LinearToneMapper(ToneMapper):
    """Simple linear clamping (no tone mapping).

    Just clamps values to [0, 1] range. Only useful for
    already normalized images or testing.
    """

    @property
    def name(self) -> str:
        return "Linear"

    def apply(self, hdr_image: np.ndarray) -> np.ndarray:
        """Clamp HDR values to [0, 1]."""
        return np.clip(hdr_image, 0.0, 1.0)


class ReinhardToneMapper(ToneMapper):
    """Reinhard global tone mapping operator.

    A simple but effective operator that compresses highlights
    while preserving shadow detail. Based on the paper:
    "Photographic Tone Reproduction for Digital Images" (2002)

    The formula is: L_out = L_in / (1 + L_in)
    """

    def __init__(self, key: float = 0.18, white_point: Optional[float] = None):
        """Initialize Reinhard tone mapper.

        Args:
            key: Scene key value (default 0.18 for typical scenes)
            white_point: If set, enables extended Reinhard with white burning
        """
        self.key = key
        self.white_point = white_point

    @property
    def name(self) -> str:
        return "Reinhard" if self.white_point is None else "Reinhard Extended"

    def get_parameters(self) -> dict:
        return {"key": self.key, "white_point": self.white_point}

    def apply(self, hdr_image: np.ndarray) -> np.ndarray:
        """Apply Reinhard tone mapping.

        Args:
            hdr_image: HDR image (H, W, 3)

        Returns:
            Tone-mapped LDR image (H, W, 3)
        """
        # Calculate luminance
        luminance = self._rgb_to_luminance(hdr_image)

        # Calculate average log luminance (avoiding log(0))
        delta = 1e-6
        avg_luminance = np.exp(np.mean(np.log(luminance + delta)))

        # Scale by key value
        scaled_luminance = (self.key / avg_luminance) * luminance

        if self.white_point is not None:
            # Extended Reinhard with white burning
            # L_out = L * (1 + L/Lw^2) / (1 + L)
            white_sq = self.white_point ** 2
            mapped_luminance = (
                scaled_luminance * (1.0 + scaled_luminance / white_sq) /
                (1.0 + scaled_luminance)
            )
        else:
            # Simple Reinhard: L_out = L / (1 + L)
            mapped_luminance = scaled_luminance / (1.0 + scaled_luminance)

        # Apply to color channels while preserving hue
        result = self._apply_luminance_mapping(hdr_image, luminance, mapped_luminance)

        return np.clip(result, 0.0, 1.0)

    def _rgb_to_luminance(self, rgb: np.ndarray) -> np.ndarray:
        """Convert RGB to luminance using Rec. 709 coefficients."""
        return 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]

    def _apply_luminance_mapping(
        self,
        rgb: np.ndarray,
        old_luminance: np.ndarray,
        new_luminance: np.ndarray
    ) -> np.ndarray:
        """Apply luminance change while preserving color ratios."""
        # Avoid division by zero
        safe_old = np.maximum(old_luminance, 1e-8)
        scale = new_luminance / safe_old

        # Apply scale to each channel
        result = np.zeros_like(rgb)
        for i in range(3):
            result[:, :, i] = rgb[:, :, i] * scale

        return result


class ACESFilmicToneMapper(ToneMapper):
    """ACES Filmic tone mapping (approximation).

    Academy Color Encoding System filmic tone curve. Provides
    cinematic look with good highlight roll-off and shadow detail.

    This is the approximation by Stephen Hill commonly used in games.
    """

    def __init__(self, exposure_bias: float = 0.0):
        """Initialize ACES filmic tone mapper.

        Args:
            exposure_bias: EV adjustment before tone mapping
        """
        self.exposure_bias = exposure_bias

    @property
    def name(self) -> str:
        return "ACES Filmic"

    def get_parameters(self) -> dict:
        return {"exposure_bias": self.exposure_bias}

    def apply(self, hdr_image: np.ndarray) -> np.ndarray:
        """Apply ACES filmic tone mapping.

        Uses the fitted curve approximation:
        (x * (a*x + b)) / (x * (c*x + d) + e)
        where x is input color and a,b,c,d,e are fitted constants.
        """
        # Apply exposure bias
        exposure_scale = 2.0 ** self.exposure_bias
        x = hdr_image * exposure_scale

        # ACES approximation constants (by Stephen Hill)
        a = 2.51
        b = 0.03
        c = 2.43
        d = 0.59
        e = 0.14

        # Apply the fitted curve
        result = (x * (a * x + b)) / (x * (c * x + d) + e)

        return np.clip(result, 0.0, 1.0)


class Uncharted2ToneMapper(ToneMapper):
    """Uncharted 2 filmic tone mapping (Hable).

    The tone curve used in Uncharted 2, designed by John Hable.
    Provides good highlight compression with adjustable shoulder.
    """

    def __init__(
        self,
        shoulder_strength: float = 0.22,
        linear_strength: float = 0.30,
        linear_angle: float = 0.10,
        toe_strength: float = 0.20,
        toe_numerator: float = 0.01,
        toe_denominator: float = 0.30,
        white_point: float = 11.2,
        exposure_bias: float = 2.0,
    ):
        """Initialize Uncharted 2 tone mapper.

        Args:
            shoulder_strength: A coefficient
            linear_strength: B coefficient
            linear_angle: C coefficient
            toe_strength: D coefficient
            toe_numerator: E coefficient
            toe_denominator: F coefficient
            white_point: Reference white point
            exposure_bias: Exposure multiplier
        """
        self.A = shoulder_strength
        self.B = linear_strength
        self.C = linear_angle
        self.D = toe_strength
        self.E = toe_numerator
        self.F = toe_denominator
        self.white_point = white_point
        self.exposure_bias = exposure_bias

    @property
    def name(self) -> str:
        return "Uncharted 2 Filmic"

    def get_parameters(self) -> dict:
        return {
            "A": self.A, "B": self.B, "C": self.C,
            "D": self.D, "E": self.E, "F": self.F,
            "white_point": self.white_point,
            "exposure_bias": self.exposure_bias
        }

    def _uncharted_curve(self, x: np.ndarray) -> np.ndarray:
        """Apply the Uncharted 2 curve function.

        f(x) = ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F)) - E/F
        """
        return (
            ((x * (self.A * x + self.C * self.B) + self.D * self.E) /
             (x * (self.A * x + self.B) + self.D * self.F))
            - self.E / self.F
        )

    def apply(self, hdr_image: np.ndarray) -> np.ndarray:
        """Apply Uncharted 2 filmic tone mapping."""
        # Apply exposure bias
        exposed = hdr_image * self.exposure_bias

        # Apply curve
        mapped = self._uncharted_curve(exposed)

        # Normalize by white point
        white_scale = 1.0 / self._uncharted_curve(np.array([self.white_point]))[0]
        result = mapped * white_scale

        return np.clip(result, 0.0, 1.0)


class ExposureToneMapper(ToneMapper):
    """Simple exposure-based tone mapping.

    Applies exposure adjustment and gamma correction.
    Simpler than filmic curves but gives more control.
    """

    def __init__(self, exposure: float = 0.0, gamma: float = 2.2):
        """Initialize exposure tone mapper.

        Args:
            exposure: Exposure value in EV (f-stops)
            gamma: Gamma value for correction (2.2 for sRGB)
        """
        self.exposure = exposure
        self.gamma = gamma

    @property
    def name(self) -> str:
        return "Exposure"

    def get_parameters(self) -> dict:
        return {"exposure": self.exposure, "gamma": self.gamma}

    def apply(self, hdr_image: np.ndarray) -> np.ndarray:
        """Apply exposure and gamma correction."""
        # Exposure adjustment: multiply by 2^EV
        exposure_scale = 2.0 ** self.exposure
        exposed = hdr_image * exposure_scale

        # Gamma correction: x^(1/gamma)
        # Handle negative values (shouldn't happen but be safe)
        result = np.power(np.maximum(exposed, 0.0), 1.0 / self.gamma)

        return np.clip(result, 0.0, 1.0)


def apply_gamma(image: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    """Apply gamma correction to an image.

    Args:
        image: Input image (H, W, 3), values in [0, 1]
        gamma: Gamma value (2.2 for sRGB)

    Returns:
        Gamma-corrected image
    """
    return np.power(np.clip(image, 0.0, 1.0), 1.0 / gamma)


def linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    """Convert linear RGB to sRGB with proper gamma curve.

    Uses the actual sRGB transfer function, not just gamma 2.2.

    Args:
        linear: Linear RGB image (H, W, 3), values >= 0

    Returns:
        sRGB image (H, W, 3), values in [0, 1]
    """
    # sRGB transfer function:
    # - Linear for small values: 12.92 * x
    # - Power curve for larger: 1.055 * x^(1/2.4) - 0.055

    linear = np.clip(linear, 0.0, None)

    # Threshold
    threshold = 0.0031308

    # Create output array
    srgb = np.zeros_like(linear)

    # Linear part
    linear_mask = linear <= threshold
    srgb[linear_mask] = linear[linear_mask] * 12.92

    # Power curve part
    power_mask = ~linear_mask
    srgb[power_mask] = 1.055 * np.power(linear[power_mask], 1.0 / 2.4) - 0.055

    return np.clip(srgb, 0.0, 1.0)


def srgb_to_linear(srgb: np.ndarray) -> np.ndarray:
    """Convert sRGB to linear RGB.

    Args:
        srgb: sRGB image (H, W, 3), values in [0, 1]

    Returns:
        Linear RGB image (H, W, 3)
    """
    srgb = np.clip(srgb, 0.0, 1.0)

    threshold = 0.04045
    linear = np.zeros_like(srgb)

    # Linear part
    linear_mask = srgb <= threshold
    linear[linear_mask] = srgb[linear_mask] / 12.92

    # Power curve part
    power_mask = ~linear_mask
    linear[power_mask] = np.power((srgb[power_mask] + 0.055) / 1.055, 2.4)

    return linear


def create_tone_mapper(
    operator: ToneMappingOperator,
    **kwargs
) -> ToneMapper:
    """Create a tone mapper by operator type.

    Args:
        operator: Type of tone mapping operator
        **kwargs: Additional arguments for the specific operator

    Returns:
        ToneMapper instance
    """
    if operator == ToneMappingOperator.LINEAR:
        return LinearToneMapper()
    elif operator == ToneMappingOperator.REINHARD:
        return ReinhardToneMapper(**kwargs)
    elif operator == ToneMappingOperator.REINHARD_EXTENDED:
        kwargs.setdefault("white_point", 4.0)
        return ReinhardToneMapper(**kwargs)
    elif operator == ToneMappingOperator.ACES_FILMIC:
        return ACESFilmicToneMapper(**kwargs)
    elif operator == ToneMappingOperator.UNCHARTED2:
        return Uncharted2ToneMapper(**kwargs)
    elif operator == ToneMappingOperator.EXPOSURE:
        return ExposureToneMapper(**kwargs)
    else:
        raise ValueError(f"Unknown tone mapping operator: {operator}")


def tone_map(
    hdr_image: np.ndarray,
    operator: ToneMappingOperator = ToneMappingOperator.ACES_FILMIC,
    convert_to_srgb: bool = True,
    **kwargs
) -> ToneMappingResult:
    """Convenience function to tone map an HDR image.

    Args:
        hdr_image: HDR image (H, W, 3), linear float values
        operator: Tone mapping operator to use
        convert_to_srgb: Whether to apply sRGB conversion after tone mapping
        **kwargs: Additional arguments for the operator

    Returns:
        ToneMappingResult with tone-mapped image
    """
    mapper = create_tone_mapper(operator, **kwargs)
    result = mapper.apply(hdr_image)

    if convert_to_srgb:
        result = linear_to_srgb(result)

    return ToneMappingResult(
        image=result,
        operator=mapper.name,
        parameters=mapper.get_parameters()
    )
