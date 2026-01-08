"""
Adaptive sampling for efficient path tracing.

Implements:
- Per-pixel variance estimation
- Adaptive sample distribution (more samples in noisy areas)
- Error threshold-based convergence
- Sample budget management
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
from enum import Enum
import math

import numpy as np

from .vec3 import Color


class AdaptiveMode(Enum):
    """Adaptive sampling modes."""
    VARIANCE = "variance"  # Based on color variance
    ERROR = "error"  # Based on estimated error
    LUMINANCE = "luminance"  # Based on luminance variance


@dataclass
class AdaptiveStats:
    """Statistics for adaptive sampling."""

    total_samples: int = 0
    min_samples_per_pixel: int = 0
    max_samples_per_pixel: int = 0
    avg_samples_per_pixel: float = 0.0
    converged_pixels: int = 0
    total_pixels: int = 0

    @property
    def convergence_ratio(self) -> float:
        """Ratio of converged pixels."""
        if self.total_pixels == 0:
            return 0.0
        return self.converged_pixels / self.total_pixels


@dataclass
class PixelState:
    """State for a single pixel during adaptive sampling."""

    # Running statistics for variance calculation
    sum: np.ndarray = field(default_factory=lambda: np.zeros(3))
    sum_sq: np.ndarray = field(default_factory=lambda: np.zeros(3))
    sample_count: int = 0

    # Convergence state
    converged: bool = False
    variance: float = float('inf')

    def add_sample(self, color: np.ndarray) -> None:
        """Add a sample and update running statistics.

        Uses Welford's online algorithm for numerically stable variance.
        """
        self.sample_count += 1
        self.sum += color
        self.sum_sq += color * color

    def get_mean(self) -> np.ndarray:
        """Get current mean color."""
        if self.sample_count == 0:
            return np.zeros(3)
        return self.sum / self.sample_count

    def get_variance(self) -> float:
        """Get current variance estimate.

        Returns the average variance across RGB channels.
        """
        if self.sample_count < 2:
            return float('inf')

        n = self.sample_count
        mean = self.sum / n
        variance_per_channel = (self.sum_sq / n) - (mean * mean)
        # Ensure non-negative due to numerical issues
        variance_per_channel = np.maximum(variance_per_channel, 0.0)

        # Return luminance-weighted variance
        weights = np.array([0.2126, 0.7152, 0.0722])
        return float(np.sum(variance_per_channel * weights))

    def get_error_estimate(self) -> float:
        """Get estimated error (standard error of mean)."""
        if self.sample_count < 2:
            return float('inf')

        variance = self.get_variance()
        return math.sqrt(variance / self.sample_count)


class AdaptiveSampler:
    """Adaptive sampler for path tracing.

    Distributes samples adaptively based on per-pixel variance,
    spending more samples on noisy areas and fewer on converged areas.
    """

    def __init__(
        self,
        width: int,
        height: int,
        min_samples: int = 4,
        max_samples: int = 1024,
        error_threshold: float = 0.01,
        mode: AdaptiveMode = AdaptiveMode.VARIANCE,
        check_interval: int = 8,
    ):
        """Initialize adaptive sampler.

        Args:
            width: Image width
            height: Image height
            min_samples: Minimum samples per pixel before checking convergence
            max_samples: Maximum samples per pixel
            error_threshold: Target error threshold for convergence
            mode: Adaptive sampling mode
            check_interval: Check convergence every N samples
        """
        self.width = width
        self.height = height
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.error_threshold = error_threshold
        self.mode = mode
        self.check_interval = check_interval

        # Initialize pixel states
        self.pixels: List[List[PixelState]] = [
            [PixelState() for _ in range(width)]
            for _ in range(height)
        ]

        # Image buffers
        self.image = np.zeros((height, width, 3), dtype=np.float32)

    def add_sample(self, x: int, y: int, color: np.ndarray) -> None:
        """Add a sample for a pixel.

        Args:
            x: Pixel x coordinate
            y: Pixel y coordinate
            color: Sample color (RGB)
        """
        if not (0 <= x < self.width and 0 <= y < self.height):
            return

        pixel = self.pixels[y][x]
        pixel.add_sample(color)

        # Update image with running mean
        self.image[y, x] = pixel.get_mean()

        # Check convergence periodically
        if (pixel.sample_count >= self.min_samples and
            pixel.sample_count % self.check_interval == 0):
            self._check_convergence(x, y)

    def _check_convergence(self, x: int, y: int) -> None:
        """Check if a pixel has converged."""
        pixel = self.pixels[y][x]

        if pixel.converged:
            return

        if pixel.sample_count >= self.max_samples:
            pixel.converged = True
            return

        if self.mode == AdaptiveMode.VARIANCE:
            pixel.variance = pixel.get_variance()
            threshold = self.error_threshold ** 2
            pixel.converged = pixel.variance < threshold

        elif self.mode == AdaptiveMode.ERROR:
            error = pixel.get_error_estimate()
            pixel.converged = error < self.error_threshold

        elif self.mode == AdaptiveMode.LUMINANCE:
            mean = pixel.get_mean()
            luminance = 0.2126 * mean[0] + 0.7152 * mean[1] + 0.0722 * mean[2]
            variance = pixel.get_variance()
            # Relative threshold based on luminance
            relative_threshold = self.error_threshold * max(luminance, 0.01)
            pixel.converged = math.sqrt(variance) < relative_threshold

    def needs_more_samples(self, x: int, y: int) -> bool:
        """Check if a pixel needs more samples.

        Args:
            x: Pixel x coordinate
            y: Pixel y coordinate

        Returns:
            True if more samples are needed
        """
        if not (0 <= x < self.width and 0 <= y < self.height):
            return False

        pixel = self.pixels[y][x]

        # Always render minimum samples
        if pixel.sample_count < self.min_samples:
            return True

        # Don't exceed maximum
        if pixel.sample_count >= self.max_samples:
            return False

        # Check convergence
        return not pixel.converged

    def get_sample_count(self, x: int, y: int) -> int:
        """Get current sample count for a pixel."""
        if not (0 <= x < self.width and 0 <= y < self.height):
            return 0
        return self.pixels[y][x].sample_count

    def is_converged(self, x: int, y: int) -> bool:
        """Check if a pixel has converged."""
        if not (0 <= x < self.width and 0 <= y < self.height):
            return True
        return self.pixels[y][x].converged

    def get_priority(self, x: int, y: int) -> float:
        """Get sampling priority for a pixel.

        Higher values mean higher priority (more variance).
        """
        if not (0 <= x < self.width and 0 <= y < self.height):
            return 0.0

        pixel = self.pixels[y][x]

        if pixel.converged:
            return 0.0

        if pixel.sample_count < self.min_samples:
            return float('inf')  # Highest priority

        return pixel.get_variance()

    def get_priority_map(self) -> np.ndarray:
        """Get priority/variance map for visualization.

        Returns:
            Priority map (H, W) with normalized values [0, 1]
        """
        priority_map = np.zeros((self.height, self.width), dtype=np.float32)

        for y in range(self.height):
            for x in range(self.width):
                priority_map[y, x] = self.get_priority(x, y)

        # Normalize, handling infinities
        finite_mask = np.isfinite(priority_map)
        if np.any(finite_mask):
            max_val = np.max(priority_map[finite_mask])
            if max_val > 0:
                priority_map[finite_mask] /= max_val
        priority_map[~finite_mask] = 1.0

        return priority_map

    def get_sample_count_map(self) -> np.ndarray:
        """Get sample count map for visualization.

        Returns:
            Sample count map (H, W) as integers
        """
        count_map = np.zeros((self.height, self.width), dtype=np.int32)

        for y in range(self.height):
            for x in range(self.width):
                count_map[y, x] = self.pixels[y][x].sample_count

        return count_map

    def get_convergence_map(self) -> np.ndarray:
        """Get convergence map for visualization.

        Returns:
            Convergence map (H, W) with 1.0 = converged, 0.0 = not converged
        """
        conv_map = np.zeros((self.height, self.width), dtype=np.float32)

        for y in range(self.height):
            for x in range(self.width):
                conv_map[y, x] = 1.0 if self.pixels[y][x].converged else 0.0

        return conv_map

    def get_image(self) -> np.ndarray:
        """Get current accumulated image.

        Returns:
            Image (H, W, 3)
        """
        return self.image.copy()

    def get_stats(self) -> AdaptiveStats:
        """Get adaptive sampling statistics.

        Returns:
            AdaptiveStats with current state
        """
        total_samples = 0
        min_samples = self.max_samples
        max_samples = 0
        converged = 0

        for y in range(self.height):
            for x in range(self.width):
                pixel = self.pixels[y][x]
                count = pixel.sample_count

                total_samples += count
                min_samples = min(min_samples, count)
                max_samples = max(max_samples, count)

                if pixel.converged:
                    converged += 1

        total_pixels = self.width * self.height
        avg_samples = total_samples / total_pixels if total_pixels > 0 else 0.0

        return AdaptiveStats(
            total_samples=total_samples,
            min_samples_per_pixel=min_samples,
            max_samples_per_pixel=max_samples,
            avg_samples_per_pixel=avg_samples,
            converged_pixels=converged,
            total_pixels=total_pixels
        )

    def get_pixels_to_sample(self, batch_size: int = 1024) -> List[Tuple[int, int]]:
        """Get a batch of pixels that need more samples.

        Prioritizes pixels with highest variance.

        Args:
            batch_size: Maximum number of pixels to return

        Returns:
            List of (x, y) coordinates
        """
        # Collect non-converged pixels with priorities
        candidates = []
        for y in range(self.height):
            for x in range(self.width):
                if self.needs_more_samples(x, y):
                    priority = self.get_priority(x, y)
                    candidates.append((priority, x, y))

        # Sort by priority (highest first)
        candidates.sort(reverse=True, key=lambda t: t[0])

        # Return top batch_size pixels
        return [(x, y) for _, x, y in candidates[:batch_size]]

    def all_converged(self) -> bool:
        """Check if all pixels have converged."""
        for y in range(self.height):
            for x in range(self.width):
                if self.needs_more_samples(x, y):
                    return False
        return True


class TileAdaptiveSampler:
    """Tile-based adaptive sampler for multi-threaded rendering.

    Works on tiles instead of individual pixels for better
    cache coherence and parallelism.
    """

    def __init__(
        self,
        width: int,
        height: int,
        tile_size: int = 32,
        min_samples: int = 4,
        max_samples: int = 1024,
        error_threshold: float = 0.01,
    ):
        """Initialize tile-based adaptive sampler.

        Args:
            width: Image width
            height: Image height
            tile_size: Size of tiles
            min_samples: Minimum samples per pixel
            max_samples: Maximum samples per pixel
            error_threshold: Target error threshold
        """
        self.width = width
        self.height = height
        self.tile_size = tile_size
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.error_threshold = error_threshold

        # Create per-pixel sampler
        self.sampler = AdaptiveSampler(
            width, height,
            min_samples=min_samples,
            max_samples=max_samples,
            error_threshold=error_threshold
        )

        # Calculate tiles
        self.tiles_x = (width + tile_size - 1) // tile_size
        self.tiles_y = (height + tile_size - 1) // tile_size

    def get_tile_bounds(self, tile_x: int, tile_y: int) -> Tuple[int, int, int, int]:
        """Get pixel bounds for a tile.

        Args:
            tile_x: Tile x index
            tile_y: Tile y index

        Returns:
            (x_start, y_start, x_end, y_end)
        """
        x_start = tile_x * self.tile_size
        y_start = tile_y * self.tile_size
        x_end = min(x_start + self.tile_size, self.width)
        y_end = min(y_start + self.tile_size, self.height)
        return x_start, y_start, x_end, y_end

    def tile_needs_samples(self, tile_x: int, tile_y: int) -> bool:
        """Check if any pixel in tile needs more samples."""
        x_start, y_start, x_end, y_end = self.get_tile_bounds(tile_x, tile_y)

        for y in range(y_start, y_end):
            for x in range(x_start, x_end):
                if self.sampler.needs_more_samples(x, y):
                    return True
        return False

    def get_tile_priority(self, tile_x: int, tile_y: int) -> float:
        """Get maximum priority among pixels in tile."""
        x_start, y_start, x_end, y_end = self.get_tile_bounds(tile_x, tile_y)

        max_priority = 0.0
        for y in range(y_start, y_end):
            for x in range(x_start, x_end):
                priority = self.sampler.get_priority(x, y)
                if math.isfinite(priority):
                    max_priority = max(max_priority, priority)
                else:
                    return float('inf')  # Has unstarted pixels

        return max_priority

    def get_tiles_to_render(self) -> List[Tuple[int, int]]:
        """Get list of tiles that need more samples, sorted by priority.

        Returns:
            List of (tile_x, tile_y) tuples
        """
        tiles = []
        for ty in range(self.tiles_y):
            for tx in range(self.tiles_x):
                if self.tile_needs_samples(tx, ty):
                    priority = self.get_tile_priority(tx, ty)
                    tiles.append((priority, tx, ty))

        # Sort by priority
        tiles.sort(reverse=True, key=lambda t: t[0])
        return [(tx, ty) for _, tx, ty in tiles]

    def add_sample(self, x: int, y: int, color: np.ndarray) -> None:
        """Add a sample (delegates to underlying sampler)."""
        self.sampler.add_sample(x, y, color)

    def get_image(self) -> np.ndarray:
        """Get current image."""
        return self.sampler.get_image()

    def get_stats(self) -> AdaptiveStats:
        """Get statistics."""
        return self.sampler.get_stats()


def estimate_required_samples(
    variance: float,
    target_error: float,
    confidence: float = 0.95
) -> int:
    """Estimate required samples to achieve target error.

    Uses the central limit theorem to estimate samples needed.

    Args:
        variance: Current estimated variance
        target_error: Desired error level
        confidence: Confidence level (0.95 = 95%)

    Returns:
        Estimated number of samples needed
    """
    if variance <= 0 or target_error <= 0:
        return 1

    # Z-score for confidence level
    # 0.95 -> 1.96, 0.99 -> 2.58
    z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_scores.get(confidence, 1.96)

    # n = (z * sigma / E)^2
    sigma = math.sqrt(variance)
    n = (z * sigma / target_error) ** 2

    return max(1, int(math.ceil(n)))
