"""Tests for bloom/glow post-processing effect."""

import pytest
import numpy as np

from spectraforge.bloom import (
    BloomEffect, BloomQuality, BloomResult, LensFlare, apply_bloom
)


class TestBloomQuality:
    """Tests for BloomQuality enum."""

    def test_quality_values(self):
        assert BloomQuality.LOW.value == "low"
        assert BloomQuality.MEDIUM.value == "medium"
        assert BloomQuality.HIGH.value == "high"
        assert BloomQuality.ULTRA.value == "ultra"


class TestBloomEffect:
    """Tests for BloomEffect class."""

    def test_default_creation(self):
        bloom = BloomEffect()
        assert bloom.threshold == 1.0
        assert bloom.intensity == 0.5
        assert bloom.radius == 1.0
        assert bloom.quality == BloomQuality.MEDIUM

    def test_custom_creation(self):
        bloom = BloomEffect(
            threshold=0.5,
            intensity=0.8,
            radius=2.0,
            quality=BloomQuality.HIGH,
            soft_threshold=0.3,
            tint=(1.0, 0.9, 0.8),
        )
        assert bloom.threshold == 0.5
        assert bloom.intensity == 0.8
        assert bloom.radius == 2.0
        assert bloom.quality == BloomQuality.HIGH
        assert bloom.soft_threshold == 0.3
        assert bloom.tint == (1.0, 0.9, 0.8)

    def test_num_passes_by_quality(self):
        for quality, expected in [
            (BloomQuality.LOW, 3),
            (BloomQuality.MEDIUM, 5),
            (BloomQuality.HIGH, 7),
            (BloomQuality.ULTRA, 9),
        ]:
            bloom = BloomEffect(quality=quality)
            assert bloom._get_num_passes() == expected

    def test_extract_bright_pixels_above_threshold(self):
        bloom = BloomEffect(threshold=0.5, soft_threshold=0.0)
        # Image with one bright pixel
        image = np.array([[[0.3, 0.3, 0.3], [1.0, 1.0, 1.0]]])
        bright = bloom._extract_bright_pixels(image)

        # First pixel should be black (below threshold)
        assert np.allclose(bright[0, 0], [0, 0, 0])
        # Second pixel should have some value
        assert np.sum(bright[0, 1]) > 0

    def test_apply_returns_result(self):
        bloom = BloomEffect()
        image = np.random.rand(20, 20, 3).astype(np.float32)
        result = bloom.apply(image)

        assert isinstance(result, BloomResult)
        assert result.image.shape == image.shape
        assert result.bloom_only.shape == image.shape
        assert "threshold" in result.parameters
        assert "intensity" in result.parameters

    def test_bloom_increases_brightness(self):
        # Create image with bright spot
        image = np.zeros((20, 20, 3), dtype=np.float32)
        image[10, 10] = [5.0, 5.0, 5.0]  # Very bright pixel

        bloom = BloomEffect(threshold=1.0, intensity=1.0)
        result = bloom.apply(image)

        # Total brightness should increase due to bloom spreading
        assert np.sum(result.image) > np.sum(image)

    def test_bloom_spreads_from_bright_areas(self):
        # Create image with bright spot
        image = np.zeros((30, 30, 3), dtype=np.float32)
        image[15, 15] = [3.0, 3.0, 3.0]

        bloom = BloomEffect(threshold=1.0, intensity=0.5, radius=1.0)
        result = bloom.apply(image)

        # Neighboring pixels should have some glow
        assert result.image[14, 15, 0] > 0
        assert result.image[16, 15, 0] > 0
        assert result.image[15, 14, 0] > 0
        assert result.image[15, 16, 0] > 0

    def test_no_bloom_below_threshold(self):
        # Image with all values below threshold
        image = np.full((10, 10, 3), 0.3, dtype=np.float32)

        bloom = BloomEffect(threshold=1.0, intensity=1.0)
        result = bloom.apply(image)

        # Result should be nearly identical to input (no bloom)
        np.testing.assert_array_almost_equal(result.image, image, decimal=2)

    def test_tint_affects_bloom_color(self):
        # Create bright white spot
        image = np.zeros((20, 20, 3), dtype=np.float32)
        image[10, 10] = [5.0, 5.0, 5.0]

        # Apply red tint
        bloom = BloomEffect(threshold=1.0, intensity=1.0, tint=(1.0, 0.0, 0.0))
        result = bloom.apply(image)

        # Bloom should have red tint (more red than green/blue in bloom_only)
        assert result.bloom_only[10, 10, 0] > result.bloom_only[10, 10, 1]
        assert result.bloom_only[10, 10, 0] > result.bloom_only[10, 10, 2]

    def test_soft_threshold(self):
        bloom_hard = BloomEffect(threshold=1.0, soft_threshold=0.0)
        bloom_soft = BloomEffect(threshold=1.0, soft_threshold=0.5)

        # Image with gradient near threshold
        image = np.zeros((1, 10, 3), dtype=np.float32)
        image[0, :, :] = np.linspace(0.5, 1.5, 10)[:, np.newaxis]

        bright_hard = bloom_hard._extract_bright_pixels(image)
        bright_soft = bloom_soft._extract_bright_pixels(image)

        # Soft threshold should have smoother transition
        # (more non-zero values in the transition region)
        hard_nonzero = np.sum(bright_hard > 0)
        soft_nonzero = np.sum(bright_soft > 0)
        assert soft_nonzero >= hard_nonzero


class TestBloomDownsampleUpsample:
    """Tests for bloom downsample/upsample operations."""

    def test_downsample_halves_size(self):
        bloom = BloomEffect()
        image = np.random.rand(20, 20, 3).astype(np.float32)
        downsampled = bloom._downsample(image)
        assert downsampled.shape == (10, 10, 3)

    def test_downsample_preserves_average(self):
        bloom = BloomEffect()
        image = np.full((20, 20, 3), 0.5, dtype=np.float32)
        downsampled = bloom._downsample(image)
        np.testing.assert_array_almost_equal(
            np.mean(downsampled), 0.5, decimal=5
        )

    def test_upsample_doubles_size(self):
        bloom = BloomEffect()
        image = np.random.rand(10, 10, 3).astype(np.float32)
        upsampled = bloom._upsample(image, (20, 20))
        assert upsampled.shape == (20, 20, 3)

    def test_upsample_same_size_returns_input(self):
        bloom = BloomEffect()
        image = np.random.rand(10, 10, 3).astype(np.float32)
        upsampled = bloom._upsample(image, (10, 10))
        np.testing.assert_array_equal(upsampled, image)


class TestGaussianBlur:
    """Tests for Gaussian blur in bloom."""

    def test_blur_preserves_brightness(self):
        bloom = BloomEffect()
        image = np.random.rand(20, 20, 3).astype(np.float32)
        blurred = bloom._gaussian_blur(image, sigma=2.0)

        # Total brightness should be approximately preserved
        # Allow some tolerance due to edge effects with reflect padding
        relative_diff = abs(np.sum(image) - np.sum(blurred)) / np.sum(image)
        assert relative_diff < 0.1  # Within 10%

    def test_blur_uniform_image_unchanged(self):
        bloom = BloomEffect()
        image = np.full((20, 20, 3), 0.5, dtype=np.float32)
        blurred = bloom._gaussian_blur(image, sigma=2.0)

        # Uniform image should remain uniform
        np.testing.assert_array_almost_equal(blurred, image, decimal=2)


class TestLensFlare:
    """Tests for LensFlare class."""

    def test_default_creation(self):
        flare = LensFlare()
        assert flare.threshold == 2.0
        assert flare.intensity == 0.3
        assert flare.num_blades == 6

    def test_custom_creation(self):
        flare = LensFlare(
            threshold=1.5,
            intensity=0.5,
            num_blades=4,
            blade_angle=0.5,
            streak_length=0.2,
        )
        assert flare.threshold == 1.5
        assert flare.intensity == 0.5
        assert flare.num_blades == 4

    def test_star_kernel_creation(self):
        flare = LensFlare(num_blades=6)
        kernel = flare._create_star_kernel(21)

        assert kernel.shape == (21, 21)
        # Kernel should be normalized (sums to ~1)
        np.testing.assert_almost_equal(np.sum(kernel), 1.0, decimal=2)

    def test_apply_preserves_shape(self):
        flare = LensFlare()
        image = np.random.rand(30, 30, 3).astype(np.float32)
        result = flare.apply(image)
        assert result.shape == image.shape

    def test_flare_adds_to_bright_spots(self):
        # Create image with very bright spot
        image = np.zeros((30, 30, 3), dtype=np.float32)
        image[15, 15] = [5.0, 5.0, 5.0]

        flare = LensFlare(threshold=2.0, intensity=0.5, streak_length=0.2)
        result = flare.apply(image)

        # Result should have at least as many bright pixels
        # and total brightness should increase
        original_brightness = np.sum(image)
        result_brightness = np.sum(result)
        assert result_brightness >= original_brightness


class TestApplyBloomFunction:
    """Tests for apply_bloom convenience function."""

    def test_returns_bloom_result(self):
        image = np.random.rand(20, 20, 3).astype(np.float32)
        result = apply_bloom(image)
        assert isinstance(result, BloomResult)

    def test_passes_parameters(self):
        image = np.random.rand(20, 20, 3).astype(np.float32)
        result = apply_bloom(
            image,
            threshold=0.5,
            intensity=0.8,
            radius=1.5,
            quality=BloomQuality.HIGH,
        )
        assert result.parameters["threshold"] == 0.5
        assert result.parameters["intensity"] == 0.8
        assert result.parameters["radius"] == 1.5
        assert result.parameters["quality"] == "high"


class TestBloomIntegration:
    """Integration tests for bloom effect."""

    def test_bloom_pipeline(self):
        # Create test image with varying brightness
        image = np.zeros((50, 50, 3), dtype=np.float32)
        # Add some bright spots
        image[10, 10] = [2.0, 2.0, 2.0]
        image[25, 25] = [3.0, 2.0, 1.0]  # Colored bright spot
        image[40, 40] = [5.0, 5.0, 5.0]

        bloom = BloomEffect(
            threshold=1.0,
            intensity=0.5,
            radius=1.0,
            quality=BloomQuality.MEDIUM,
        )
        result = bloom.apply(image)

        # Output should be valid
        assert not np.any(np.isnan(result.image))
        assert not np.any(np.isinf(result.image))

        # Bloom should add brightness
        assert np.sum(result.image) > np.sum(image)

    def test_bloom_large_image(self):
        # Test with larger image
        image = np.random.rand(100, 100, 3).astype(np.float32) * 2.0
        bloom = BloomEffect(quality=BloomQuality.LOW)
        result = bloom.apply(image)

        assert result.image.shape == image.shape
        assert not np.any(np.isnan(result.image))
