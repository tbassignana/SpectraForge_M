"""
Denoising integration for path-traced images.

Implements:
- Intel Open Image Denoise (OIDN) integration (primary)
- Auxiliary buffer (albedo, normal) support for better quality
- Fallback bilateral filter for environments without OIDN
- HDR-aware processing
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple
import math

import numpy as np

from .vec3 import Color


@dataclass
class DenoiseResult:
    """Result of denoising operation."""

    image: np.ndarray  # Denoised HDR image (H, W, 3)
    quality: float  # Estimated quality improvement (0-1)
    method: str  # Name of denoising method used


class Denoiser(ABC):
    """Abstract base class for denoisers."""

    @abstractmethod
    def denoise(
        self,
        image: np.ndarray,
        albedo: Optional[np.ndarray] = None,
        normal: Optional[np.ndarray] = None,
    ) -> DenoiseResult:
        """Denoise a rendered image.

        Args:
            image: HDR color image (H, W, 3), float32, linear space
            albedo: Optional albedo buffer (H, W, 3) for better edge preservation
            normal: Optional world-space normal buffer (H, W, 3) for geometry guidance

        Returns:
            DenoiseResult with denoised image and metadata
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the denoiser."""
        pass

    @property
    @abstractmethod
    def supports_auxiliary(self) -> bool:
        """Whether this denoiser can use albedo/normal buffers."""
        pass


class OIDNDenoiser(Denoiser):
    """Intel Open Image Denoise integration.

    OIDN uses deep learning models trained specifically for path-traced
    images. It's the industry standard for high-quality denoising.
    """

    def __init__(self, hdr: bool = True, srgb: bool = False):
        """Initialize OIDN denoiser.

        Args:
            hdr: Enable HDR mode for high dynamic range images
            srgb: Input/output in sRGB (False = linear)
        """
        self.hdr = hdr
        self.srgb = srgb
        self._device = None
        self._filter = None
        self._available = None

    def _check_available(self) -> bool:
        """Check if OIDN is available."""
        if self._available is None:
            try:
                import oidn
                self._available = True
            except ImportError:
                self._available = False
        return self._available

    def _ensure_initialized(self) -> None:
        """Initialize OIDN device and filter on first use."""
        if self._device is not None:
            return

        if not self._check_available():
            raise RuntimeError(
                "Intel Open Image Denoise (oidn) is not installed. "
                "Install it with: pip install oidn"
            )

        import oidn

        self._device = oidn.NewDevice()
        oidn.CommitDevice(self._device)

    @property
    def name(self) -> str:
        return "Intel Open Image Denoise"

    @property
    def supports_auxiliary(self) -> bool:
        return True

    def is_available(self) -> bool:
        """Check if OIDN is available on this system."""
        return self._check_available()

    def denoise(
        self,
        image: np.ndarray,
        albedo: Optional[np.ndarray] = None,
        normal: Optional[np.ndarray] = None,
    ) -> DenoiseResult:
        """Denoise using OIDN.

        Args:
            image: HDR color image (H, W, 3), float32
            albedo: Optional albedo buffer (H, W, 3)
            normal: Optional world-space normal buffer (H, W, 3)

        Returns:
            DenoiseResult with denoised image
        """
        self._ensure_initialized()
        import oidn

        # Ensure proper format
        image = np.ascontiguousarray(image.astype(np.float32))
        height, width = image.shape[:2]

        # Create output buffer
        output = np.zeros_like(image)

        # Create filter
        filter_handle = oidn.NewFilter(self._device, "RT")

        # Set color image
        oidn.SetFilterImage(
            filter_handle, "color", image, oidn.FORMAT_FLOAT3,
            width, height, 0, 0, 0
        )

        # Set albedo if provided
        if albedo is not None:
            albedo = np.ascontiguousarray(albedo.astype(np.float32))
            oidn.SetFilterImage(
                filter_handle, "albedo", albedo, oidn.FORMAT_FLOAT3,
                width, height, 0, 0, 0
            )

        # Set normal if provided
        if normal is not None:
            normal = np.ascontiguousarray(normal.astype(np.float32))
            oidn.SetFilterImage(
                filter_handle, "normal", normal, oidn.FORMAT_FLOAT3,
                width, height, 0, 0, 0
            )

        # Set output
        oidn.SetFilterImage(
            filter_handle, "output", output, oidn.FORMAT_FLOAT3,
            width, height, 0, 0, 0
        )

        # Configure filter
        oidn.SetFilter1b(filter_handle, "hdr", self.hdr)
        oidn.SetFilter1b(filter_handle, "srgb", self.srgb)

        # Execute
        oidn.CommitFilter(filter_handle)
        oidn.ExecuteFilter(filter_handle)

        # Check for errors
        error = oidn.GetDeviceError(self._device)
        if error[0] != oidn.ERROR_NONE:
            raise RuntimeError(f"OIDN error: {error[1]}")

        # Release filter
        oidn.ReleaseFilter(filter_handle)

        # Estimate quality improvement (rough heuristic based on variance reduction)
        original_variance = np.var(image)
        denoised_variance = np.var(output)
        quality = 1.0 - min(1.0, denoised_variance / (original_variance + 1e-8))

        return DenoiseResult(
            image=output,
            quality=quality,
            method=self.name
        )

    def __del__(self):
        """Release OIDN resources."""
        if self._device is not None:
            try:
                import oidn
                oidn.ReleaseDevice(self._device)
            except Exception:
                pass


class BilateralDenoiser(Denoiser):
    """Simple bilateral filter denoiser as fallback.

    A bilateral filter preserves edges while smoothing noise. It's not
    as effective as ML-based denoisers but works everywhere.
    """

    def __init__(
        self,
        sigma_spatial: float = 3.0,
        sigma_range: float = 0.1,
        kernel_size: int = 7,
    ):
        """Initialize bilateral denoiser.

        Args:
            sigma_spatial: Spatial standard deviation (larger = more blur)
            sigma_range: Range standard deviation (larger = less edge preservation)
            kernel_size: Size of the filter kernel (must be odd)
        """
        self.sigma_spatial = sigma_spatial
        self.sigma_range = sigma_range
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1

    @property
    def name(self) -> str:
        return "Bilateral Filter"

    @property
    def supports_auxiliary(self) -> bool:
        return False

    def denoise(
        self,
        image: np.ndarray,
        albedo: Optional[np.ndarray] = None,
        normal: Optional[np.ndarray] = None,
    ) -> DenoiseResult:
        """Apply bilateral filtering.

        Args:
            image: HDR color image (H, W, 3)
            albedo: Ignored (not supported)
            normal: Ignored (not supported)

        Returns:
            DenoiseResult with filtered image
        """
        # Bilateral filter implementation
        output = self._bilateral_filter(image)

        # Estimate quality improvement
        original_variance = np.var(image)
        denoised_variance = np.var(output)
        quality = 1.0 - min(1.0, denoised_variance / (original_variance + 1e-8))

        return DenoiseResult(
            image=output,
            quality=quality * 0.5,  # Lower quality estimate for simple filter
            method=self.name
        )

    def _bilateral_filter(self, image: np.ndarray) -> np.ndarray:
        """Apply bilateral filter to image.

        Args:
            image: Input image (H, W, 3)

        Returns:
            Filtered image (H, W, 3)
        """
        height, width, channels = image.shape
        output = np.zeros_like(image)

        # Precompute spatial weights
        half_k = self.kernel_size // 2
        spatial_weights = np.zeros((self.kernel_size, self.kernel_size))
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                di = i - half_k
                dj = j - half_k
                spatial_weights[i, j] = math.exp(
                    -(di * di + dj * dj) / (2 * self.sigma_spatial ** 2)
                )

        # Pad image
        padded = np.pad(
            image,
            ((half_k, half_k), (half_k, half_k), (0, 0)),
            mode='reflect'
        )

        # Apply filter
        for y in range(height):
            for x in range(width):
                center_pixel = image[y, x]
                weighted_sum = np.zeros(channels)
                weight_sum = 0.0

                # Process neighborhood
                for i in range(self.kernel_size):
                    for j in range(self.kernel_size):
                        neighbor = padded[y + i, x + j]

                        # Range weight based on color difference
                        color_diff = np.sum((neighbor - center_pixel) ** 2)
                        range_weight = math.exp(
                            -color_diff / (2 * self.sigma_range ** 2)
                        )

                        # Combined weight
                        weight = spatial_weights[i, j] * range_weight

                        weighted_sum += weight * neighbor
                        weight_sum += weight

                output[y, x] = weighted_sum / (weight_sum + 1e-8)

        return output


class JointBilateralDenoiser(Denoiser):
    """Joint bilateral filter using auxiliary buffers.

    Uses albedo and normal buffers to guide edge preservation,
    providing better results than simple bilateral filtering.
    """

    def __init__(
        self,
        sigma_spatial: float = 3.0,
        sigma_color: float = 0.1,
        sigma_albedo: float = 0.1,
        sigma_normal: float = 0.5,
        kernel_size: int = 7,
    ):
        """Initialize joint bilateral denoiser.

        Args:
            sigma_spatial: Spatial standard deviation
            sigma_color: Color range standard deviation
            sigma_albedo: Albedo range standard deviation
            sigma_normal: Normal range standard deviation
            kernel_size: Filter kernel size (must be odd)
        """
        self.sigma_spatial = sigma_spatial
        self.sigma_color = sigma_color
        self.sigma_albedo = sigma_albedo
        self.sigma_normal = sigma_normal
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1

    @property
    def name(self) -> str:
        return "Joint Bilateral Filter"

    @property
    def supports_auxiliary(self) -> bool:
        return True

    def denoise(
        self,
        image: np.ndarray,
        albedo: Optional[np.ndarray] = None,
        normal: Optional[np.ndarray] = None,
    ) -> DenoiseResult:
        """Apply joint bilateral filtering.

        Args:
            image: HDR color image (H, W, 3)
            albedo: Optional albedo buffer (H, W, 3)
            normal: Optional normal buffer (H, W, 3)

        Returns:
            DenoiseResult with filtered image
        """
        output = self._joint_bilateral_filter(image, albedo, normal)

        # Estimate quality improvement
        original_variance = np.var(image)
        denoised_variance = np.var(output)
        quality = 1.0 - min(1.0, denoised_variance / (original_variance + 1e-8))

        # Adjust quality based on auxiliary buffer usage
        quality_factor = 0.6
        if albedo is not None:
            quality_factor += 0.1
        if normal is not None:
            quality_factor += 0.1

        return DenoiseResult(
            image=output,
            quality=quality * quality_factor,
            method=self.name
        )

    def _joint_bilateral_filter(
        self,
        image: np.ndarray,
        albedo: Optional[np.ndarray],
        normal: Optional[np.ndarray],
    ) -> np.ndarray:
        """Apply joint bilateral filter.

        Args:
            image: Input color image (H, W, 3)
            albedo: Albedo buffer (H, W, 3) or None
            normal: Normal buffer (H, W, 3) or None

        Returns:
            Filtered image (H, W, 3)
        """
        height, width, channels = image.shape
        output = np.zeros_like(image)

        # Precompute spatial weights
        half_k = self.kernel_size // 2
        spatial_weights = np.zeros((self.kernel_size, self.kernel_size))
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                di = i - half_k
                dj = j - half_k
                spatial_weights[i, j] = math.exp(
                    -(di * di + dj * dj) / (2 * self.sigma_spatial ** 2)
                )

        # Pad arrays
        padded_img = np.pad(
            image,
            ((half_k, half_k), (half_k, half_k), (0, 0)),
            mode='reflect'
        )
        padded_albedo = None
        padded_normal = None

        if albedo is not None:
            padded_albedo = np.pad(
                albedo,
                ((half_k, half_k), (half_k, half_k), (0, 0)),
                mode='reflect'
            )
        if normal is not None:
            padded_normal = np.pad(
                normal,
                ((half_k, half_k), (half_k, half_k), (0, 0)),
                mode='reflect'
            )

        # Apply filter
        for y in range(height):
            for x in range(width):
                center_color = image[y, x]
                center_albedo = albedo[y, x] if albedo is not None else None
                center_normal = normal[y, x] if normal is not None else None

                weighted_sum = np.zeros(channels)
                weight_sum = 0.0

                for i in range(self.kernel_size):
                    for j in range(self.kernel_size):
                        neighbor_color = padded_img[y + i, x + j]

                        # Color weight
                        color_diff = np.sum((neighbor_color - center_color) ** 2)
                        color_weight = math.exp(
                            -color_diff / (2 * self.sigma_color ** 2)
                        )

                        # Start with spatial * color weight
                        weight = spatial_weights[i, j] * color_weight

                        # Albedo weight
                        if padded_albedo is not None and center_albedo is not None:
                            neighbor_albedo = padded_albedo[y + i, x + j]
                            albedo_diff = np.sum((neighbor_albedo - center_albedo) ** 2)
                            albedo_weight = math.exp(
                                -albedo_diff / (2 * self.sigma_albedo ** 2)
                            )
                            weight *= albedo_weight

                        # Normal weight
                        if padded_normal is not None and center_normal is not None:
                            neighbor_normal = padded_normal[y + i, x + j]
                            # Use dot product for normal comparison
                            dot = np.dot(neighbor_normal, center_normal)
                            normal_diff = 1.0 - max(-1.0, min(1.0, dot))
                            normal_weight = math.exp(
                                -normal_diff / (2 * self.sigma_normal ** 2)
                            )
                            weight *= normal_weight

                        weighted_sum += weight * neighbor_color
                        weight_sum += weight

                output[y, x] = weighted_sum / (weight_sum + 1e-8)

        return output


def create_denoiser(prefer_oidn: bool = True) -> Denoiser:
    """Create the best available denoiser.

    Args:
        prefer_oidn: Prefer Intel OIDN if available

    Returns:
        Denoiser instance
    """
    if prefer_oidn:
        oidn_denoiser = OIDNDenoiser()
        if oidn_denoiser.is_available():
            return oidn_denoiser

    # Fall back to joint bilateral filter
    return JointBilateralDenoiser()


def denoise_image(
    image: np.ndarray,
    albedo: Optional[np.ndarray] = None,
    normal: Optional[np.ndarray] = None,
    method: str = "auto",
) -> DenoiseResult:
    """Convenience function to denoise an image.

    Args:
        image: HDR color image (H, W, 3), float32
        albedo: Optional albedo buffer (H, W, 3)
        normal: Optional normal buffer (H, W, 3)
        method: Denoising method - "auto", "oidn", "bilateral", or "joint"

    Returns:
        DenoiseResult with denoised image
    """
    if method == "auto":
        denoiser = create_denoiser(prefer_oidn=True)
    elif method == "oidn":
        denoiser = OIDNDenoiser()
    elif method == "bilateral":
        denoiser = BilateralDenoiser()
    elif method == "joint":
        denoiser = JointBilateralDenoiser()
    else:
        raise ValueError(f"Unknown denoising method: {method}")

    return denoiser.denoise(image, albedo, normal)


class AuxiliaryBufferRenderer:
    """Helper class to render auxiliary buffers for denoising.

    Renders albedo and normal buffers alongside the color buffer
    to improve denoising quality.
    """

    def __init__(self, width: int, height: int):
        """Initialize auxiliary buffer storage.

        Args:
            width: Image width
            height: Image height
        """
        self.width = width
        self.height = height
        self.albedo_buffer = np.zeros((height, width, 3), dtype=np.float32)
        self.normal_buffer = np.zeros((height, width, 3), dtype=np.float32)
        self.sample_count = np.zeros((height, width), dtype=np.int32)

    def add_sample(
        self,
        x: int,
        y: int,
        albedo: Tuple[float, float, float],
        normal: Tuple[float, float, float],
    ) -> None:
        """Add a sample to the auxiliary buffers.

        Args:
            x: Pixel x coordinate
            y: Pixel y coordinate
            albedo: Albedo color (R, G, B)
            normal: World-space normal (X, Y, Z)
        """
        count = self.sample_count[y, x]
        new_count = count + 1

        # Running average
        self.albedo_buffer[y, x] = (
            self.albedo_buffer[y, x] * count + np.array(albedo)
        ) / new_count

        self.normal_buffer[y, x] = (
            self.normal_buffer[y, x] * count + np.array(normal)
        ) / new_count

        self.sample_count[y, x] = new_count

    def get_albedo(self) -> np.ndarray:
        """Get the albedo buffer.

        Returns:
            Albedo buffer (H, W, 3)
        """
        return self.albedo_buffer

    def get_normal(self) -> np.ndarray:
        """Get the normal buffer (normalized).

        Returns:
            Normal buffer (H, W, 3), each normal is unit length
        """
        # Normalize normals
        norms = np.linalg.norm(self.normal_buffer, axis=2, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        return self.normal_buffer / norms

    def clear(self) -> None:
        """Clear all buffers."""
        self.albedo_buffer.fill(0)
        self.normal_buffer.fill(0)
        self.sample_count.fill(0)
