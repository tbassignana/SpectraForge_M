"""
Post-processing pipeline orchestrator.

Provides a unified interface for chaining multiple post-processing
effects in the correct order with proper color space handling.

Also includes additional post-processing effects:
- Chromatic aberration
- Sharpen filter
- Film grain
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Callable, Any
from enum import Enum
import math

import numpy as np

from .tonemapping import (
    ToneMapper, ToneMappingOperator, create_tone_mapper,
    linear_to_srgb, srgb_to_linear
)
from .bloom import BloomEffect, BloomQuality
from .color_correction import ColorCorrector, ColorCorrectionSettings
from .denoiser import Denoiser, create_denoiser


class PostProcessStage(Enum):
    """Post-processing pipeline stages in order."""
    DENOISE = "denoise"
    BLOOM = "bloom"
    COLOR_CORRECTION = "color_correction"
    TONE_MAP = "tone_map"
    CHROMATIC_ABERRATION = "chromatic_aberration"
    SHARPEN = "sharpen"
    FILM_GRAIN = "film_grain"
    SRGB_CONVERT = "srgb_convert"


@dataclass
class PipelineResult:
    """Result from post-processing pipeline."""

    image: np.ndarray  # Final processed image
    intermediate: dict  # Intermediate results by stage name
    enabled_stages: List[str]  # List of enabled stages that were run


@dataclass
class PostProcessingPipeline:
    """Unified post-processing pipeline.

    Chains multiple effects in the optimal order:
    1. Denoising (on raw HDR)
    2. Bloom (on clean HDR)
    3. Color correction (HDR adjustments)
    4. Tone mapping (HDR -> LDR)
    5. Chromatic aberration (after tone map, before final)
    6. Sharpen (restore detail)
    7. Film grain (artistic effect)
    8. sRGB conversion (final output)

    All effects are optional and can be individually enabled/disabled.
    """

    # Denoising
    enable_denoise: bool = False
    denoiser: Optional[Denoiser] = None

    # Bloom
    enable_bloom: bool = False
    bloom_threshold: float = 1.0
    bloom_intensity: float = 0.5
    bloom_radius: float = 1.0
    bloom_quality: BloomQuality = BloomQuality.MEDIUM

    # Color correction
    enable_color_correction: bool = False
    color_settings: Optional[ColorCorrectionSettings] = None

    # Tone mapping
    enable_tone_map: bool = True
    tone_map_operator: ToneMappingOperator = ToneMappingOperator.ACES_FILMIC
    tone_map_kwargs: dict = field(default_factory=dict)

    # Chromatic aberration
    enable_chromatic_aberration: bool = False
    chromatic_intensity: float = 0.005
    chromatic_samples: int = 3

    # Sharpen
    enable_sharpen: bool = False
    sharpen_amount: float = 0.5
    sharpen_radius: float = 1.0

    # Film grain
    enable_film_grain: bool = False
    grain_intensity: float = 0.05
    grain_size: float = 1.0

    # sRGB conversion
    enable_srgb: bool = True

    # Keep intermediate results
    store_intermediate: bool = False

    def process(
        self,
        image: np.ndarray,
        albedo: Optional[np.ndarray] = None,
        normal: Optional[np.ndarray] = None,
    ) -> PipelineResult:
        """Run the complete post-processing pipeline.

        Args:
            image: Input HDR image (H, W, 3), linear space
            albedo: Optional albedo buffer for denoising
            normal: Optional normal buffer for denoising

        Returns:
            PipelineResult with final image and metadata
        """
        result = image.copy()
        intermediate = {}
        enabled_stages = []

        # Stage 1: Denoising
        if self.enable_denoise:
            denoiser = self.denoiser or create_denoiser()
            denoise_result = denoiser.denoise(result, albedo, normal)
            result = denoise_result.image
            enabled_stages.append(PostProcessStage.DENOISE.value)
            if self.store_intermediate:
                intermediate["denoise"] = result.copy()

        # Stage 2: Bloom
        if self.enable_bloom:
            bloom = BloomEffect(
                threshold=self.bloom_threshold,
                intensity=self.bloom_intensity,
                radius=self.bloom_radius,
                quality=self.bloom_quality,
            )
            bloom_result = bloom.apply(result)
            result = bloom_result.image
            enabled_stages.append(PostProcessStage.BLOOM.value)
            if self.store_intermediate:
                intermediate["bloom"] = result.copy()

        # Stage 3: Color correction (still in HDR)
        if self.enable_color_correction:
            settings = self.color_settings or ColorCorrectionSettings()
            corrector = ColorCorrector(settings)
            correction_result = corrector.apply(result)
            result = correction_result.image
            enabled_stages.append(PostProcessStage.COLOR_CORRECTION.value)
            if self.store_intermediate:
                intermediate["color_correction"] = result.copy()

        # Stage 4: Tone mapping (HDR -> LDR)
        if self.enable_tone_map:
            mapper = create_tone_mapper(self.tone_map_operator, **self.tone_map_kwargs)
            result = mapper.apply(result)
            enabled_stages.append(PostProcessStage.TONE_MAP.value)
            if self.store_intermediate:
                intermediate["tone_map"] = result.copy()

        # Stage 5: Chromatic aberration (after tone map, on LDR)
        if self.enable_chromatic_aberration:
            result = apply_chromatic_aberration(
                result,
                intensity=self.chromatic_intensity,
                samples=self.chromatic_samples
            )
            enabled_stages.append(PostProcessStage.CHROMATIC_ABERRATION.value)
            if self.store_intermediate:
                intermediate["chromatic_aberration"] = result.copy()

        # Stage 6: Sharpen
        if self.enable_sharpen:
            result = apply_sharpen(
                result,
                amount=self.sharpen_amount,
                radius=self.sharpen_radius
            )
            enabled_stages.append(PostProcessStage.SHARPEN.value)
            if self.store_intermediate:
                intermediate["sharpen"] = result.copy()

        # Stage 7: Film grain
        if self.enable_film_grain:
            result = apply_film_grain(
                result,
                intensity=self.grain_intensity,
                size=self.grain_size
            )
            enabled_stages.append(PostProcessStage.FILM_GRAIN.value)
            if self.store_intermediate:
                intermediate["film_grain"] = result.copy()

        # Stage 8: sRGB conversion
        if self.enable_srgb:
            result = linear_to_srgb(result)
            enabled_stages.append(PostProcessStage.SRGB_CONVERT.value)
            if self.store_intermediate:
                intermediate["srgb"] = result.copy()

        return PipelineResult(
            image=result,
            intermediate=intermediate,
            enabled_stages=enabled_stages
        )


class ChromaticAberration:
    """Chromatic aberration effect.

    Simulates color fringing caused by lens dispersion, where
    different wavelengths of light focus at different distances.
    """

    def __init__(
        self,
        intensity: float = 0.005,
        samples: int = 3,
        falloff: float = 2.0,
    ):
        """Initialize chromatic aberration.

        Args:
            intensity: Base displacement amount (0-0.1, relative to image size)
            samples: Number of samples per channel for smooth blur (1-5)
            falloff: Radial falloff power (higher = more edge-focused)
        """
        self.intensity = intensity
        self.samples = max(1, min(samples, 5))
        self.falloff = falloff

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply chromatic aberration.

        Shifts red channel outward and blue channel inward from center,
        simulating typical lens dispersion.

        Args:
            image: Input image (H, W, 3), values in [0, 1]

        Returns:
            Image with chromatic aberration (H, W, 3)
        """
        h, w = image.shape[:2]
        result = np.zeros_like(image)

        # Center coordinates
        cy, cx = h / 2, w / 2

        # Create coordinate grids
        y_coords = np.arange(h) - cy
        x_coords = np.arange(w) - cx
        xx, yy = np.meshgrid(x_coords, y_coords)

        # Distance from center (normalized)
        distance = np.sqrt(xx ** 2 + yy ** 2)
        max_dist = np.sqrt(cx ** 2 + cy ** 2)
        norm_dist = distance / max_dist

        # Apply falloff to make effect stronger at edges
        displacement_factor = np.power(norm_dist, self.falloff)

        # Displacement directions (radial)
        angle = np.arctan2(yy, xx)

        # Channel shifts: Red outward (+), Green none, Blue inward (-)
        shifts = [1.0, 0.0, -1.0]  # R, G, B

        for c, shift in enumerate(shifts):
            if shift == 0 and self.samples == 1:
                # Green channel unchanged
                result[:, :, c] = image[:, :, c]
                continue

            # Accumulate samples
            channel_result = np.zeros((h, w))

            for s in range(self.samples):
                # Sample offset for anti-aliasing
                sample_offset = (s - (self.samples - 1) / 2) / max(self.samples - 1, 1)
                sample_intensity = self.intensity * (1.0 + sample_offset * 0.2)

                # Calculate displacement
                disp = displacement_factor * shift * sample_intensity

                # Displaced coordinates
                src_x = xx + np.cos(angle) * disp * w
                src_y = yy + np.sin(angle) * disp * h

                # Convert back to image coordinates
                src_x = src_x + cx
                src_y = src_y + cy

                # Sample with bilinear interpolation
                sampled = self._bilinear_sample(image[:, :, c], src_x, src_y)
                channel_result += sampled

            result[:, :, c] = channel_result / self.samples

        return np.clip(result, 0.0, 1.0)

    def _bilinear_sample(
        self,
        channel: np.ndarray,
        x: np.ndarray,
        y: np.ndarray
    ) -> np.ndarray:
        """Bilinear interpolation sampling.

        Args:
            channel: Single channel image (H, W)
            x: X coordinates to sample
            y: Y coordinates to sample

        Returns:
            Sampled values
        """
        h, w = channel.shape

        # Clamp coordinates
        x0 = np.clip(np.floor(x).astype(int), 0, w - 1)
        x1 = np.clip(x0 + 1, 0, w - 1)
        y0 = np.clip(np.floor(y).astype(int), 0, h - 1)
        y1 = np.clip(y0 + 1, 0, h - 1)

        # Fractional parts
        fx = x - np.floor(x)
        fy = y - np.floor(y)

        # Bilinear interpolation
        result = (
            channel[y0, x0] * (1 - fx) * (1 - fy) +
            channel[y0, x1] * fx * (1 - fy) +
            channel[y1, x0] * (1 - fx) * fy +
            channel[y1, x1] * fx * fy
        )

        return result


class SharpenFilter:
    """Unsharp mask sharpening filter.

    Enhances edges and fine detail by subtracting a blurred
    version of the image from the original.
    """

    def __init__(
        self,
        amount: float = 0.5,
        radius: float = 1.0,
        threshold: float = 0.0,
    ):
        """Initialize sharpen filter.

        Args:
            amount: Sharpening strength (0-2, 0 = none, 1 = strong)
            radius: Blur radius for unsharp mask
            threshold: Minimum difference to sharpen (reduces noise amplification)
        """
        self.amount = amount
        self.radius = radius
        self.threshold = threshold

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply unsharp mask sharpening.

        Args:
            image: Input image (H, W, 3)

        Returns:
            Sharpened image (H, W, 3)
        """
        if self.amount == 0:
            return image

        # Create blurred version
        blurred = self._gaussian_blur(image, self.radius)

        # Calculate difference (detail layer)
        detail = image - blurred

        # Apply threshold
        if self.threshold > 0:
            mask = np.abs(detail) > self.threshold
            detail = detail * mask

        # Add scaled detail back
        result = image + detail * self.amount

        return np.clip(result, 0.0, 1.0)

    def _gaussian_blur(self, image: np.ndarray, sigma: float) -> np.ndarray:
        """Apply Gaussian blur."""
        kernel_size = int(math.ceil(sigma * 6)) | 1
        kernel_size = max(3, min(kernel_size, 15))

        half_k = kernel_size // 2
        x = np.arange(-half_k, half_k + 1)
        kernel = np.exp(-x ** 2 / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()

        # Separable convolution
        h, w = image.shape[:2]
        padded = np.pad(
            image,
            ((half_k, half_k), (half_k, half_k), (0, 0)),
            mode='reflect'
        )

        # Horizontal pass
        temp = np.zeros_like(image)
        for i, k in enumerate(kernel):
            temp += padded[:h, i:i + w, :] * k

        # Vertical pass
        padded = np.pad(temp, ((half_k, half_k), (0, 0), (0, 0)), mode='reflect')
        result = np.zeros_like(image)
        for i, k in enumerate(kernel):
            result += padded[i:i + h, :, :] * k

        return result


class FilmGrain:
    """Film grain effect.

    Adds noise that simulates photographic film grain for
    artistic/cinematic look.
    """

    def __init__(
        self,
        intensity: float = 0.05,
        size: float = 1.0,
        colored: bool = False,
        seed: Optional[int] = None,
    ):
        """Initialize film grain.

        Args:
            intensity: Grain visibility (0-0.2)
            size: Grain size multiplier (0.5 = fine, 2 = coarse)
            colored: Whether to add color variation to grain
            seed: Random seed for reproducibility
        """
        self.intensity = intensity
        self.size = size
        self.colored = colored
        self.seed = seed

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply film grain effect.

        Args:
            image: Input image (H, W, 3)

        Returns:
            Image with film grain (H, W, 3)
        """
        if self.intensity == 0:
            return image

        rng = np.random.default_rng(self.seed)
        h, w = image.shape[:2]

        # Generate grain at potentially lower resolution
        grain_h = max(1, int(h / self.size))
        grain_w = max(1, int(w / self.size))

        if self.colored:
            # Colored grain (different noise per channel)
            grain = rng.standard_normal((grain_h, grain_w, 3))
        else:
            # Monochrome grain
            grain = rng.standard_normal((grain_h, grain_w))
            grain = np.stack([grain, grain, grain], axis=-1)

        # Upscale grain to image size
        if self.size != 1.0:
            grain = self._upscale(grain, (h, w))

        # Scale grain intensity based on image luminance
        # Grain is less visible in bright/dark areas
        luminance = 0.2126 * image[:, :, 0] + 0.7152 * image[:, :, 1] + 0.0722 * image[:, :, 2]
        luminance_factor = 4 * luminance * (1 - luminance)  # Peaks at 0.5
        luminance_factor = luminance_factor[:, :, np.newaxis]

        # Apply grain
        result = image + grain * self.intensity * luminance_factor

        return np.clip(result, 0.0, 1.0)

    def _upscale(self, grain: np.ndarray, target_shape: tuple) -> np.ndarray:
        """Upscale grain using nearest neighbor for crisp grain."""
        h, w = grain.shape[:2]
        target_h, target_w = target_shape

        result = np.zeros((target_h, target_w, 3))

        for i in range(target_h):
            for j in range(target_w):
                src_i = min(int(i * h / target_h), h - 1)
                src_j = min(int(j * w / target_w), w - 1)
                result[i, j] = grain[src_i, src_j]

        return result


def apply_chromatic_aberration(
    image: np.ndarray,
    intensity: float = 0.005,
    samples: int = 3,
) -> np.ndarray:
    """Apply chromatic aberration effect.

    Args:
        image: Input image (H, W, 3)
        intensity: Effect intensity
        samples: Quality samples

    Returns:
        Image with chromatic aberration
    """
    ca = ChromaticAberration(intensity=intensity, samples=samples)
    return ca.apply(image)


def apply_sharpen(
    image: np.ndarray,
    amount: float = 0.5,
    radius: float = 1.0,
) -> np.ndarray:
    """Apply sharpening filter.

    Args:
        image: Input image (H, W, 3)
        amount: Sharpen strength
        radius: Blur radius

    Returns:
        Sharpened image
    """
    sharpen = SharpenFilter(amount=amount, radius=radius)
    return sharpen.apply(image)


def apply_film_grain(
    image: np.ndarray,
    intensity: float = 0.05,
    size: float = 1.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Apply film grain effect.

    Args:
        image: Input image (H, W, 3)
        intensity: Grain intensity
        size: Grain size
        seed: Random seed

    Returns:
        Image with film grain
    """
    grain = FilmGrain(intensity=intensity, size=size, seed=seed)
    return grain.apply(image)


def create_pipeline(
    denoise: bool = False,
    bloom: bool = False,
    color_correct: bool = False,
    tone_map: bool = True,
    chromatic_aberration: bool = False,
    sharpen: bool = False,
    film_grain: bool = False,
    **kwargs
) -> PostProcessingPipeline:
    """Create a post-processing pipeline with common presets.

    Args:
        denoise: Enable denoising
        bloom: Enable bloom effect
        color_correct: Enable color correction
        tone_map: Enable tone mapping (default True)
        chromatic_aberration: Enable chromatic aberration
        sharpen: Enable sharpening
        film_grain: Enable film grain
        **kwargs: Additional parameters passed to pipeline

    Returns:
        Configured PostProcessingPipeline
    """
    return PostProcessingPipeline(
        enable_denoise=denoise,
        enable_bloom=bloom,
        enable_color_correction=color_correct,
        enable_tone_map=tone_map,
        enable_chromatic_aberration=chromatic_aberration,
        enable_sharpen=sharpen,
        enable_film_grain=film_grain,
        **kwargs
    )
