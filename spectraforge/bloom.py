"""
Bloom/glow post-processing effect for HDR renders.

Implements:
- Threshold-based bright pixel extraction
- Multi-scale Gaussian blur (progressive downsampling)
- Bloom intensity and color controls
- Lens flare/star patterns (optional)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List
from enum import Enum
import math

import numpy as np


class BloomQuality(Enum):
    """Bloom quality presets."""
    LOW = "low"       # 3 blur passes
    MEDIUM = "medium"  # 5 blur passes
    HIGH = "high"     # 7 blur passes
    ULTRA = "ultra"   # 9 blur passes


@dataclass
class BloomResult:
    """Result of bloom operation."""

    image: np.ndarray  # Image with bloom applied (H, W, 3)
    bloom_only: np.ndarray  # Just the bloom contribution
    parameters: dict  # Bloom parameters used


class BloomEffect:
    """Bloom/glow post-processing effect.

    Creates a glow around bright areas by:
    1. Extracting pixels above a luminance threshold
    2. Applying multi-scale Gaussian blur (kawase or gaussian)
    3. Blending the blurred result back with the original
    """

    def __init__(
        self,
        threshold: float = 1.0,
        intensity: float = 0.5,
        radius: float = 1.0,
        quality: BloomQuality = BloomQuality.MEDIUM,
        soft_threshold: float = 0.5,
        tint: Optional[Tuple[float, float, float]] = None,
    ):
        """Initialize bloom effect.

        Args:
            threshold: Luminance threshold for bloom extraction (HDR value)
            intensity: Bloom intensity multiplier
            radius: Bloom radius multiplier (affects blur size)
            quality: Quality preset (affects number of blur passes)
            soft_threshold: Soft threshold knee (0 = hard, 1 = very soft)
            tint: Optional color tint for bloom (R, G, B)
        """
        self.threshold = threshold
        self.intensity = intensity
        self.radius = radius
        self.quality = quality
        self.soft_threshold = soft_threshold
        self.tint = tint or (1.0, 1.0, 1.0)

    def _get_num_passes(self) -> int:
        """Get number of blur passes based on quality."""
        return {
            BloomQuality.LOW: 3,
            BloomQuality.MEDIUM: 5,
            BloomQuality.HIGH: 7,
            BloomQuality.ULTRA: 9,
        }[self.quality]

    def _rgb_to_luminance(self, rgb: np.ndarray) -> np.ndarray:
        """Convert RGB to luminance."""
        return 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]

    def _extract_bright_pixels(self, image: np.ndarray) -> np.ndarray:
        """Extract pixels above threshold with soft knee.

        Args:
            image: HDR image (H, W, 3)

        Returns:
            Bright pixel contribution (H, W, 3)
        """
        luminance = self._rgb_to_luminance(image)

        # Soft threshold using smooth curve
        # knee = threshold * soft_threshold
        knee = self.threshold * self.soft_threshold

        # For soft threshold, use smooth hermite interpolation
        if self.soft_threshold > 0:
            # Compute soft knee blend
            knee_start = self.threshold - knee
            knee_end = self.threshold + knee

            # Create mask for different regions
            below_knee = luminance < knee_start
            in_knee = (luminance >= knee_start) & (luminance < knee_end)
            above_threshold = luminance >= knee_end

            # Compute contribution
            contribution = np.zeros_like(luminance)

            # Smooth blend in knee region
            t = (luminance[in_knee] - knee_start) / (2 * knee + 1e-8)
            contribution[in_knee] = t * t * (3 - 2 * t)  # Smoothstep

            # Full contribution above threshold
            contribution[above_threshold] = 1.0

            # Apply contribution to color
            result = np.zeros_like(image)
            for i in range(3):
                # Scale by how much pixel exceeds threshold
                excess = np.maximum(luminance - knee_start, 0)
                result[:, :, i] = image[:, :, i] * contribution * (excess / (luminance + 1e-8))

        else:
            # Hard threshold
            mask = luminance > self.threshold
            result = np.zeros_like(image)
            for i in range(3):
                result[:, :, i] = np.where(mask, image[:, :, i], 0)

        return result

    def _gaussian_blur(
        self,
        image: np.ndarray,
        sigma: float
    ) -> np.ndarray:
        """Apply Gaussian blur.

        Args:
            image: Input image (H, W, 3)
            sigma: Standard deviation of Gaussian

        Returns:
            Blurred image (H, W, 3)
        """
        # Kernel size based on sigma (3-sigma rule)
        kernel_size = int(math.ceil(sigma * 6)) | 1  # Ensure odd
        kernel_size = max(3, min(kernel_size, 31))  # Clamp to reasonable size

        # Create 1D Gaussian kernel
        half_k = kernel_size // 2
        x = np.arange(-half_k, half_k + 1)
        kernel = np.exp(-x ** 2 / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()

        # Separable convolution (horizontal then vertical)
        # Pad image
        padded = np.pad(
            image,
            ((half_k, half_k), (half_k, half_k), (0, 0)),
            mode='reflect'
        )

        # Horizontal pass
        temp = np.zeros_like(image)
        for i, k in enumerate(kernel):
            temp += padded[:image.shape[0], i:i + image.shape[1], :] * k

        # Vertical pass
        padded = np.pad(
            temp,
            ((half_k, half_k), (0, 0), (0, 0)),
            mode='reflect'
        )
        result = np.zeros_like(image)
        for i, k in enumerate(kernel):
            result += padded[i:i + image.shape[0], :, :] * k

        return result

    def _downsample(self, image: np.ndarray) -> np.ndarray:
        """Downsample image by 2x using bilinear filtering."""
        h, w = image.shape[:2]
        new_h, new_w = h // 2, w // 2

        if new_h < 1 or new_w < 1:
            return image

        # Simple 2x2 box filter downsample
        result = np.zeros((new_h, new_w, 3), dtype=image.dtype)
        for i in range(new_h):
            for j in range(new_w):
                result[i, j] = (
                    image[2*i, 2*j] +
                    image[2*i + 1, 2*j] +
                    image[2*i, 2*j + 1] +
                    image[2*i + 1, 2*j + 1]
                ) / 4.0

        return result

    def _upsample(self, image: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """Upsample image to target shape using bilinear interpolation."""
        h, w = image.shape[:2]
        target_h, target_w = target_shape

        if h == target_h and w == target_w:
            return image

        result = np.zeros((target_h, target_w, 3), dtype=image.dtype)

        # Bilinear interpolation
        for i in range(target_h):
            for j in range(target_w):
                # Map to source coordinates
                src_i = i * (h - 1) / max(target_h - 1, 1)
                src_j = j * (w - 1) / max(target_w - 1, 1)

                # Get integer and fractional parts
                i0 = int(src_i)
                j0 = int(src_j)
                i1 = min(i0 + 1, h - 1)
                j1 = min(j0 + 1, w - 1)

                fi = src_i - i0
                fj = src_j - j0

                # Bilinear interpolation
                result[i, j] = (
                    image[i0, j0] * (1 - fi) * (1 - fj) +
                    image[i1, j0] * fi * (1 - fj) +
                    image[i0, j1] * (1 - fi) * fj +
                    image[i1, j1] * fi * fj
                )

        return result

    def _progressive_blur(self, image: np.ndarray) -> np.ndarray:
        """Apply multi-scale progressive blur.

        Uses progressive downsampling and upsampling for efficient
        large-radius blur (similar to Kawase bloom).
        """
        num_passes = self._get_num_passes()
        base_sigma = 1.5 * self.radius

        # Build mipmap pyramid
        mips: List[np.ndarray] = [image]
        current = image

        for i in range(num_passes - 1):
            # Blur then downsample
            blurred = self._gaussian_blur(current, base_sigma)
            downsampled = self._downsample(blurred)
            mips.append(downsampled)
            current = downsampled

        # Upsample and accumulate (from smallest to largest)
        result = mips[-1]
        original_shape = image.shape[:2]

        for i in range(len(mips) - 2, -1, -1):
            # Upsample to match current level
            target_shape = mips[i].shape[:2]
            upsampled = self._upsample(result, target_shape)

            # Blur the upsampled result
            blurred = self._gaussian_blur(upsampled, base_sigma)

            # Blend with this level's image
            result = (blurred + mips[i]) * 0.5

        # Final upsample to original size if needed
        if result.shape[:2] != original_shape:
            result = self._upsample(result, original_shape)

        return result

    def apply(self, image: np.ndarray) -> BloomResult:
        """Apply bloom effect to image.

        Args:
            image: HDR image (H, W, 3)

        Returns:
            BloomResult with bloom-applied image
        """
        # Extract bright pixels
        bright = self._extract_bright_pixels(image)

        # Apply progressive blur
        bloom = self._progressive_blur(bright)

        # Apply intensity and tint
        bloom = bloom * self.intensity
        for i, t in enumerate(self.tint):
            bloom[:, :, i] *= t

        # Combine with original (additive blending)
        result = image + bloom

        return BloomResult(
            image=result,
            bloom_only=bloom,
            parameters={
                "threshold": self.threshold,
                "intensity": self.intensity,
                "radius": self.radius,
                "quality": self.quality.value,
                "soft_threshold": self.soft_threshold,
                "tint": self.tint,
            }
        )


class LensFlare:
    """Lens flare/star pattern effect.

    Creates star/streak patterns around very bright light sources,
    simulating diffraction from camera aperture blades.
    """

    def __init__(
        self,
        threshold: float = 2.0,
        intensity: float = 0.3,
        num_blades: int = 6,
        blade_angle: float = 0.0,
        streak_length: float = 0.1,
    ):
        """Initialize lens flare effect.

        Args:
            threshold: Luminance threshold for flare generation
            intensity: Flare intensity
            num_blades: Number of aperture blades (affects star points)
            blade_angle: Rotation angle of flare pattern (radians)
            streak_length: Length of streaks (0-1, relative to image size)
        """
        self.threshold = threshold
        self.intensity = intensity
        self.num_blades = num_blades
        self.blade_angle = blade_angle
        self.streak_length = streak_length

    def _rgb_to_luminance(self, rgb: np.ndarray) -> np.ndarray:
        """Convert RGB to luminance."""
        return 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]

    def _create_star_kernel(self, size: int) -> np.ndarray:
        """Create star/flare convolution kernel.

        Args:
            size: Kernel size (should be odd)

        Returns:
            Kernel array (size, size)
        """
        center = size // 2
        kernel = np.zeros((size, size))

        # Create star pattern with num_blades rays
        for i in range(self.num_blades):
            angle = self.blade_angle + (i * math.pi / self.num_blades)

            # Draw line from center
            for r in range(center):
                x = int(center + r * math.cos(angle))
                y = int(center + r * math.sin(angle))

                if 0 <= x < size and 0 <= y < size:
                    # Falloff with distance
                    falloff = 1.0 - (r / center)
                    kernel[y, x] = max(kernel[y, x], falloff ** 2)

                # Opposite direction
                x = int(center - r * math.cos(angle))
                y = int(center - r * math.sin(angle))

                if 0 <= x < size and 0 <= y < size:
                    falloff = 1.0 - (r / center)
                    kernel[y, x] = max(kernel[y, x], falloff ** 2)

        # Normalize
        kernel_sum = kernel.sum()
        if kernel_sum > 0:
            kernel /= kernel_sum

        return kernel

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply lens flare effect.

        Args:
            image: HDR image (H, W, 3)

        Returns:
            Image with lens flare (H, W, 3)
        """
        h, w = image.shape[:2]

        # Calculate kernel size based on streak length
        kernel_size = int(min(h, w) * self.streak_length)
        kernel_size = kernel_size | 1  # Ensure odd
        kernel_size = max(3, min(kernel_size, 101))  # Clamp

        # Create star kernel
        kernel = self._create_star_kernel(kernel_size)

        # Extract very bright pixels
        luminance = self._rgb_to_luminance(image)
        mask = luminance > self.threshold
        bright = np.zeros_like(image)
        for i in range(3):
            bright[:, :, i] = np.where(mask, image[:, :, i], 0)

        # Convolve with star kernel (for each channel)
        half_k = kernel_size // 2
        padded = np.pad(
            bright,
            ((half_k, half_k), (half_k, half_k), (0, 0)),
            mode='constant'
        )

        flare = np.zeros_like(image)
        for y in range(h):
            for x in range(w):
                for c in range(3):
                    flare[y, x, c] = np.sum(
                        padded[y:y + kernel_size, x:x + kernel_size, c] * kernel
                    )

        # Add flare to original
        result = image + flare * self.intensity

        return result


def apply_bloom(
    image: np.ndarray,
    threshold: float = 1.0,
    intensity: float = 0.5,
    radius: float = 1.0,
    quality: BloomQuality = BloomQuality.MEDIUM,
) -> BloomResult:
    """Convenience function to apply bloom effect.

    Args:
        image: HDR image (H, W, 3)
        threshold: Luminance threshold for bloom
        intensity: Bloom intensity
        radius: Bloom radius multiplier
        quality: Quality preset

    Returns:
        BloomResult with bloom-applied image
    """
    bloom = BloomEffect(
        threshold=threshold,
        intensity=intensity,
        radius=radius,
        quality=quality
    )
    return bloom.apply(image)
