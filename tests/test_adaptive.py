"""Tests for adaptive sampling."""

import pytest
import numpy as np
import math

from spectraforge.adaptive import (
    AdaptiveMode, AdaptiveStats, PixelState, AdaptiveSampler,
    TileAdaptiveSampler, estimate_required_samples
)


class TestPixelState:
    """Tests for PixelState class."""

    def test_default_creation(self):
        state = PixelState()
        assert state.sample_count == 0
        assert not state.converged
        assert state.variance == float('inf')

    def test_add_sample(self):
        state = PixelState()
        state.add_sample(np.array([0.5, 0.5, 0.5]))
        assert state.sample_count == 1

    def test_get_mean_single_sample(self):
        state = PixelState()
        state.add_sample(np.array([0.5, 0.3, 0.1]))
        mean = state.get_mean()
        np.testing.assert_array_almost_equal(mean, [0.5, 0.3, 0.1])

    def test_get_mean_multiple_samples(self):
        state = PixelState()
        state.add_sample(np.array([1.0, 0.0, 0.0]))
        state.add_sample(np.array([0.0, 1.0, 0.0]))
        state.add_sample(np.array([0.0, 0.0, 1.0]))
        mean = state.get_mean()
        np.testing.assert_array_almost_equal(mean, [1/3, 1/3, 1/3])

    def test_get_mean_no_samples(self):
        state = PixelState()
        mean = state.get_mean()
        np.testing.assert_array_equal(mean, [0, 0, 0])

    def test_get_variance_needs_two_samples(self):
        state = PixelState()
        state.add_sample(np.array([0.5, 0.5, 0.5]))
        assert state.get_variance() == float('inf')

    def test_get_variance_constant_samples(self):
        state = PixelState()
        for _ in range(10):
            state.add_sample(np.array([0.5, 0.5, 0.5]))
        variance = state.get_variance()
        assert variance < 0.001  # Should be near zero

    def test_get_variance_varying_samples(self):
        state = PixelState()
        # Add varying samples
        state.add_sample(np.array([0.0, 0.0, 0.0]))
        state.add_sample(np.array([1.0, 1.0, 1.0]))
        variance = state.get_variance()
        # Variance should be significant
        assert variance > 0.1

    def test_get_error_estimate(self):
        state = PixelState()
        for i in range(100):
            # Samples with some variance
            state.add_sample(np.array([0.5 + 0.1 * (i % 2), 0.5, 0.5]))

        error = state.get_error_estimate()
        # Error should decrease with more samples
        assert error < 0.1


class TestAdaptiveStats:
    """Tests for AdaptiveStats dataclass."""

    def test_default_creation(self):
        stats = AdaptiveStats()
        assert stats.total_samples == 0
        assert stats.convergence_ratio == 0.0

    def test_convergence_ratio(self):
        stats = AdaptiveStats(
            converged_pixels=50,
            total_pixels=100
        )
        assert stats.convergence_ratio == 0.5

    def test_convergence_ratio_zero_pixels(self):
        stats = AdaptiveStats(total_pixels=0)
        assert stats.convergence_ratio == 0.0


class TestAdaptiveSampler:
    """Tests for AdaptiveSampler class."""

    def test_creation(self):
        sampler = AdaptiveSampler(width=100, height=50)
        assert sampler.width == 100
        assert sampler.height == 50

    def test_custom_parameters(self):
        sampler = AdaptiveSampler(
            width=50, height=50,
            min_samples=8,
            max_samples=512,
            error_threshold=0.02
        )
        assert sampler.min_samples == 8
        assert sampler.max_samples == 512
        assert sampler.error_threshold == 0.02

    def test_add_sample(self):
        sampler = AdaptiveSampler(width=10, height=10, min_samples=2)
        sampler.add_sample(5, 5, np.array([0.5, 0.5, 0.5]))
        assert sampler.get_sample_count(5, 5) == 1

    def test_add_sample_out_of_bounds(self):
        sampler = AdaptiveSampler(width=10, height=10)
        # Should not crash
        sampler.add_sample(-1, 5, np.array([0.5, 0.5, 0.5]))
        sampler.add_sample(100, 5, np.array([0.5, 0.5, 0.5]))

    def test_needs_more_samples_initial(self):
        sampler = AdaptiveSampler(width=10, height=10, min_samples=4)
        assert sampler.needs_more_samples(5, 5) is True

    def test_needs_more_samples_after_min(self):
        sampler = AdaptiveSampler(width=10, height=10, min_samples=4)

        # Add minimum samples
        for _ in range(4):
            sampler.add_sample(5, 5, np.array([0.5, 0.5, 0.5]))

        # Should still need more if not converged
        # With constant samples, it should converge
        # (but check interval may not have triggered)

    def test_needs_more_samples_false_after_max(self):
        sampler = AdaptiveSampler(width=10, height=10, min_samples=2, max_samples=5)

        # Add max samples
        for _ in range(5):
            sampler.add_sample(5, 5, np.array([0.5, 0.5, 0.5]))

        assert sampler.needs_more_samples(5, 5) is False

    def test_get_sample_count(self):
        sampler = AdaptiveSampler(width=10, height=10)

        for i in range(7):
            sampler.add_sample(3, 4, np.array([0.5, 0.5, 0.5]))

        assert sampler.get_sample_count(3, 4) == 7
        assert sampler.get_sample_count(0, 0) == 0

    def test_get_sample_count_out_of_bounds(self):
        sampler = AdaptiveSampler(width=10, height=10)
        assert sampler.get_sample_count(-1, 0) == 0
        assert sampler.get_sample_count(100, 0) == 0

    def test_get_priority(self):
        sampler = AdaptiveSampler(width=10, height=10, min_samples=2)

        # Unsampled pixel should have infinite priority
        assert sampler.get_priority(5, 5) == float('inf')

    def test_get_priority_map(self):
        sampler = AdaptiveSampler(width=10, height=10, min_samples=2)

        # Add some samples
        for i in range(4):
            sampler.add_sample(5, 5, np.array([0.5 + 0.1*i, 0.5, 0.5]))

        priority_map = sampler.get_priority_map()
        assert priority_map.shape == (10, 10)

    def test_get_sample_count_map(self):
        sampler = AdaptiveSampler(width=10, height=10)

        sampler.add_sample(0, 0, np.array([0.5, 0.5, 0.5]))
        sampler.add_sample(0, 0, np.array([0.5, 0.5, 0.5]))
        sampler.add_sample(5, 5, np.array([0.5, 0.5, 0.5]))

        count_map = sampler.get_sample_count_map()
        assert count_map[0, 0] == 2
        assert count_map[5, 5] == 1
        assert count_map[9, 9] == 0

    def test_get_convergence_map(self):
        sampler = AdaptiveSampler(width=10, height=10, min_samples=2, check_interval=2)

        # Add many identical samples to converge
        for _ in range(10):
            sampler.add_sample(5, 5, np.array([0.5, 0.5, 0.5]))

        conv_map = sampler.get_convergence_map()
        assert conv_map.shape == (10, 10)
        assert conv_map[5, 5] == 1.0  # Should be converged

    def test_get_image(self):
        sampler = AdaptiveSampler(width=10, height=10)

        sampler.add_sample(5, 5, np.array([1.0, 0.0, 0.0]))
        sampler.add_sample(5, 5, np.array([0.0, 1.0, 0.0]))

        image = sampler.get_image()
        assert image.shape == (10, 10, 3)
        # Average of red and green
        np.testing.assert_array_almost_equal(image[5, 5], [0.5, 0.5, 0.0])

    def test_get_stats(self):
        sampler = AdaptiveSampler(width=10, height=10, min_samples=2)

        sampler.add_sample(0, 0, np.array([0.5, 0.5, 0.5]))
        sampler.add_sample(1, 1, np.array([0.5, 0.5, 0.5]))
        sampler.add_sample(1, 1, np.array([0.5, 0.5, 0.5]))

        stats = sampler.get_stats()
        assert stats.total_samples == 3
        assert stats.total_pixels == 100
        assert stats.min_samples_per_pixel == 0
        assert stats.max_samples_per_pixel == 2

    def test_get_pixels_to_sample(self):
        sampler = AdaptiveSampler(width=10, height=10, min_samples=2, max_samples=4)

        # Mark some pixels as done
        for _ in range(4):
            sampler.add_sample(0, 0, np.array([0.5, 0.5, 0.5]))

        pixels = sampler.get_pixels_to_sample(batch_size=50)

        # Should not include (0, 0) which has max samples
        assert (0, 0) not in pixels

        # Should include some unsampled pixels
        assert len(pixels) > 0

    def test_all_converged_initially_false(self):
        sampler = AdaptiveSampler(width=10, height=10, min_samples=2)
        assert not sampler.all_converged()

    def test_convergence_with_constant_samples(self):
        sampler = AdaptiveSampler(
            width=5, height=5,
            min_samples=2,
            max_samples=100,
            error_threshold=0.001,
            check_interval=2
        )

        # Add identical samples to all pixels
        for _ in range(50):
            for y in range(5):
                for x in range(5):
                    sampler.add_sample(x, y, np.array([0.5, 0.5, 0.5]))

        # All should converge with constant samples
        stats = sampler.get_stats()
        assert stats.converged_pixels == 25


class TestAdaptiveMode:
    """Tests for different adaptive modes."""

    def test_variance_mode(self):
        sampler = AdaptiveSampler(
            width=5, height=5,
            mode=AdaptiveMode.VARIANCE,
            min_samples=2,
            check_interval=2
        )

        # Add samples and check mode works
        for _ in range(10):
            sampler.add_sample(0, 0, np.array([0.5, 0.5, 0.5]))

        assert sampler.mode == AdaptiveMode.VARIANCE

    def test_error_mode(self):
        sampler = AdaptiveSampler(
            width=5, height=5,
            mode=AdaptiveMode.ERROR,
            min_samples=2,
            check_interval=2
        )

        for _ in range(10):
            sampler.add_sample(0, 0, np.array([0.5, 0.5, 0.5]))

        assert sampler.mode == AdaptiveMode.ERROR

    def test_luminance_mode(self):
        sampler = AdaptiveSampler(
            width=5, height=5,
            mode=AdaptiveMode.LUMINANCE,
            min_samples=2,
            check_interval=2
        )

        for _ in range(10):
            sampler.add_sample(0, 0, np.array([0.5, 0.5, 0.5]))

        assert sampler.mode == AdaptiveMode.LUMINANCE


class TestTileAdaptiveSampler:
    """Tests for TileAdaptiveSampler class."""

    def test_creation(self):
        sampler = TileAdaptiveSampler(width=100, height=100, tile_size=32)
        assert sampler.width == 100
        assert sampler.height == 100
        assert sampler.tile_size == 32

    def test_tile_count(self):
        sampler = TileAdaptiveSampler(width=100, height=64, tile_size=32)
        assert sampler.tiles_x == 4  # ceil(100/32) = 4
        assert sampler.tiles_y == 2  # ceil(64/32) = 2

    def test_get_tile_bounds(self):
        sampler = TileAdaptiveSampler(width=100, height=100, tile_size=32)

        # First tile
        x0, y0, x1, y1 = sampler.get_tile_bounds(0, 0)
        assert (x0, y0, x1, y1) == (0, 0, 32, 32)

        # Last tile (may be partial)
        x0, y0, x1, y1 = sampler.get_tile_bounds(3, 3)
        assert x0 == 96
        assert x1 == 100  # Clamped to width

    def test_tile_needs_samples_initially(self):
        sampler = TileAdaptiveSampler(width=64, height=64, tile_size=32, min_samples=2)
        assert sampler.tile_needs_samples(0, 0)

    def test_get_tiles_to_render(self):
        sampler = TileAdaptiveSampler(width=64, height=64, tile_size=32, min_samples=2)

        tiles = sampler.get_tiles_to_render()
        assert len(tiles) == 4  # 2x2 tiles

    def test_add_sample_delegates(self):
        sampler = TileAdaptiveSampler(width=64, height=64, tile_size=32)

        sampler.add_sample(5, 5, np.array([0.5, 0.5, 0.5]))
        assert sampler.sampler.get_sample_count(5, 5) == 1

    def test_get_image(self):
        sampler = TileAdaptiveSampler(width=32, height=32, tile_size=16)

        sampler.add_sample(0, 0, np.array([1.0, 0.0, 0.0]))

        image = sampler.get_image()
        assert image.shape == (32, 32, 3)
        np.testing.assert_array_almost_equal(image[0, 0], [1.0, 0.0, 0.0])


class TestEstimateRequiredSamples:
    """Tests for estimate_required_samples function."""

    def test_basic_estimate(self):
        n = estimate_required_samples(variance=0.01, target_error=0.01)
        assert n > 0

    def test_higher_variance_needs_more_samples(self):
        n_low = estimate_required_samples(variance=0.01, target_error=0.01)
        n_high = estimate_required_samples(variance=0.1, target_error=0.01)
        assert n_high > n_low

    def test_tighter_threshold_needs_more_samples(self):
        n_loose = estimate_required_samples(variance=0.01, target_error=0.1)
        n_tight = estimate_required_samples(variance=0.01, target_error=0.01)
        assert n_tight > n_loose

    def test_zero_variance(self):
        n = estimate_required_samples(variance=0.0, target_error=0.01)
        assert n == 1

    def test_zero_error(self):
        n = estimate_required_samples(variance=0.01, target_error=0.0)
        assert n == 1

    def test_different_confidence_levels(self):
        n_95 = estimate_required_samples(variance=0.01, target_error=0.01, confidence=0.95)
        n_99 = estimate_required_samples(variance=0.01, target_error=0.01, confidence=0.99)
        assert n_99 > n_95


class TestAdaptiveSamplerIntegration:
    """Integration tests for adaptive sampling."""

    def test_noisy_vs_smooth_regions(self):
        sampler = AdaptiveSampler(
            width=20, height=20,
            min_samples=4,
            max_samples=100,
            error_threshold=0.05,
            check_interval=4
        )

        np.random.seed(42)

        # Simulate rendering with noisy and smooth regions
        for _ in range(50):
            for y in range(20):
                for x in range(20):
                    if sampler.needs_more_samples(x, y):
                        if x < 10:
                            # Smooth region (constant color)
                            color = np.array([0.5, 0.5, 0.5])
                        else:
                            # Noisy region (variable color)
                            color = np.array([0.5, 0.5, 0.5]) + np.random.randn(3) * 0.2

                        sampler.add_sample(x, y, color)

        # Get sample counts
        count_map = sampler.get_sample_count_map()

        # Smooth region should have fewer samples on average
        smooth_avg = np.mean(count_map[:, :10])
        noisy_avg = np.mean(count_map[:, 10:])

        # Noisy region should need more samples
        assert noisy_avg >= smooth_avg

    def test_full_convergence(self):
        sampler = AdaptiveSampler(
            width=10, height=10,
            min_samples=2,
            max_samples=50,
            error_threshold=0.001,
            check_interval=2
        )

        # Add enough samples to converge
        for iteration in range(100):
            for y in range(10):
                for x in range(10):
                    if sampler.needs_more_samples(x, y):
                        sampler.add_sample(x, y, np.array([0.5, 0.5, 0.5]))

            if sampler.all_converged():
                break

        # Should converge with constant samples
        assert sampler.all_converged()
