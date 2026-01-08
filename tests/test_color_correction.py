"""Tests for color correction and grading."""

import pytest
import numpy as np
import tempfile
import os

from spectraforge.color_correction import (
    ColorCorrector, ColorCorrectionSettings, ColorCorrectionResult,
    LUT, Vignette, apply_color_correction
)


class TestColorCorrectionSettings:
    """Tests for ColorCorrectionSettings dataclass."""

    def test_default_values(self):
        settings = ColorCorrectionSettings()
        assert settings.exposure == 0.0
        assert settings.contrast == 1.0
        assert settings.saturation == 1.0
        assert settings.temperature == 0.0
        assert settings.tint == 0.0
        assert settings.shadows == 0.0
        assert settings.midtones == 0.0
        assert settings.highlights == 0.0
        assert settings.red_gain == 1.0
        assert settings.green_gain == 1.0
        assert settings.blue_gain == 1.0
        assert settings.vignette_intensity == 0.0

    def test_custom_values(self):
        settings = ColorCorrectionSettings(
            exposure=1.5,
            contrast=1.2,
            saturation=0.8,
        )
        assert settings.exposure == 1.5
        assert settings.contrast == 1.2
        assert settings.saturation == 0.8


class TestColorCorrector:
    """Tests for ColorCorrector class."""

    def test_default_creation(self):
        corrector = ColorCorrector()
        assert corrector.settings is not None
        assert corrector.settings.exposure == 0.0

    def test_custom_settings(self):
        settings = ColorCorrectionSettings(exposure=2.0)
        corrector = ColorCorrector(settings)
        assert corrector.settings.exposure == 2.0

    def test_apply_returns_result(self):
        corrector = ColorCorrector()
        image = np.random.rand(10, 10, 3).astype(np.float32)
        result = corrector.apply(image)

        assert isinstance(result, ColorCorrectionResult)
        assert result.image.shape == image.shape
        assert result.settings is not None

    def test_no_change_with_defaults(self):
        corrector = ColorCorrector()
        image = np.random.rand(10, 10, 3).astype(np.float32)
        result = corrector.apply(image)

        # With default settings, output should be similar to input
        np.testing.assert_array_almost_equal(result.image, image, decimal=4)


class TestExposure:
    """Tests for exposure adjustment."""

    def test_positive_exposure_brightens(self):
        settings = ColorCorrectionSettings(exposure=1.0)
        corrector = ColorCorrector(settings)
        image = np.array([[[0.25, 0.25, 0.25]]])
        result = corrector.apply(image)

        # +1 EV should double brightness
        expected = np.array([[[0.5, 0.5, 0.5]]])
        np.testing.assert_array_almost_equal(result.image, expected)

    def test_negative_exposure_darkens(self):
        settings = ColorCorrectionSettings(exposure=-1.0)
        corrector = ColorCorrector(settings)
        image = np.array([[[0.5, 0.5, 0.5]]])
        result = corrector.apply(image)

        # -1 EV should halve brightness
        expected = np.array([[[0.25, 0.25, 0.25]]])
        np.testing.assert_array_almost_equal(result.image, expected)

    def test_zero_exposure_no_change(self):
        settings = ColorCorrectionSettings(exposure=0.0)
        corrector = ColorCorrector(settings)
        image = np.array([[[0.5, 0.5, 0.5]]])
        result = corrector.apply(image)

        np.testing.assert_array_almost_equal(result.image, image)


class TestContrast:
    """Tests for contrast adjustment."""

    def test_increased_contrast(self):
        settings = ColorCorrectionSettings(contrast=2.0)
        corrector = ColorCorrector(settings)

        # Middle gray should be unaffected (pivot point)
        middle_gray = np.array([[[0.18, 0.18, 0.18]]])
        result = corrector.apply(middle_gray)
        np.testing.assert_array_almost_equal(result.image, middle_gray, decimal=3)

    def test_contrast_expands_range(self):
        settings = ColorCorrectionSettings(contrast=1.5)
        corrector = ColorCorrector(settings)

        # Values above pivot should increase more
        bright = np.array([[[0.5, 0.5, 0.5]]])
        dark = np.array([[[0.05, 0.05, 0.05]]])

        bright_result = corrector.apply(bright)
        dark_result = corrector.apply(dark)

        # Bright should be brighter, dark should be darker
        assert bright_result.image[0, 0, 0] > 0.5
        assert dark_result.image[0, 0, 0] < 0.05

    def test_unity_contrast_no_change(self):
        settings = ColorCorrectionSettings(contrast=1.0)
        corrector = ColorCorrector(settings)
        image = np.random.rand(5, 5, 3).astype(np.float32)
        result = corrector.apply(image)

        np.testing.assert_array_almost_equal(result.image, image)


class TestSaturation:
    """Tests for saturation adjustment."""

    def test_zero_saturation_grayscale(self):
        settings = ColorCorrectionSettings(saturation=0.0)
        corrector = ColorCorrector(settings)

        # Colored image
        image = np.array([[[1.0, 0.5, 0.2]]])
        result = corrector.apply(image)

        # Result should be grayscale (all channels equal)
        assert result.image[0, 0, 0] == result.image[0, 0, 1]
        assert result.image[0, 0, 1] == result.image[0, 0, 2]

    def test_increased_saturation(self):
        settings = ColorCorrectionSettings(saturation=1.5)
        corrector = ColorCorrector(settings)

        # Image with some color
        image = np.array([[[0.8, 0.5, 0.3]]])
        result = corrector.apply(image)

        # Color difference should be larger
        original_range = image[0, 0, 0] - image[0, 0, 2]
        result_range = result.image[0, 0, 0] - result.image[0, 0, 2]
        assert result_range > original_range

    def test_unity_saturation_no_change(self):
        settings = ColorCorrectionSettings(saturation=1.0)
        corrector = ColorCorrector(settings)
        image = np.random.rand(5, 5, 3).astype(np.float32)
        result = corrector.apply(image)

        np.testing.assert_array_almost_equal(result.image, image)


class TestTemperatureTint:
    """Tests for color temperature and tint."""

    def test_warm_temperature(self):
        settings = ColorCorrectionSettings(temperature=50.0)
        corrector = ColorCorrector(settings)

        # Neutral gray
        image = np.array([[[0.5, 0.5, 0.5]]])
        result = corrector.apply(image)

        # Warmer = more red, less blue
        assert result.image[0, 0, 0] > result.image[0, 0, 2]

    def test_cool_temperature(self):
        settings = ColorCorrectionSettings(temperature=-50.0)
        corrector = ColorCorrector(settings)

        image = np.array([[[0.5, 0.5, 0.5]]])
        result = corrector.apply(image)

        # Cooler = more blue, less red
        assert result.image[0, 0, 2] > result.image[0, 0, 0]

    def test_positive_tint(self):
        settings = ColorCorrectionSettings(tint=50.0)
        corrector = ColorCorrector(settings)

        image = np.array([[[0.5, 0.5, 0.5]]])
        result = corrector.apply(image)

        # Positive tint = less green
        assert result.image[0, 0, 1] < 0.5

    def test_zero_temperature_tint_no_change(self):
        settings = ColorCorrectionSettings(temperature=0.0, tint=0.0)
        corrector = ColorCorrector(settings)
        image = np.random.rand(5, 5, 3).astype(np.float32)
        result = corrector.apply(image)

        np.testing.assert_array_almost_equal(result.image, image)


class TestShadowsMidtonesHighlights:
    """Tests for shadow/midtone/highlight controls."""

    def test_lift_shadows(self):
        settings = ColorCorrectionSettings(shadows=0.5)
        corrector = ColorCorrector(settings)

        # Dark pixel
        dark = np.array([[[0.05, 0.05, 0.05]]])
        result = corrector.apply(dark)

        # Shadow should be lifted (brighter)
        assert result.image[0, 0, 0] > 0.05

    def test_lower_shadows(self):
        settings = ColorCorrectionSettings(shadows=-0.5)
        corrector = ColorCorrector(settings)

        dark = np.array([[[0.1, 0.1, 0.1]]])
        result = corrector.apply(dark)

        # Shadow should be darker (may clamp to 0)
        assert result.image[0, 0, 0] <= 0.1

    def test_boost_highlights(self):
        settings = ColorCorrectionSettings(highlights=0.5)
        corrector = ColorCorrector(settings)

        bright = np.array([[[0.8, 0.8, 0.8]]])
        result = corrector.apply(bright)

        # Highlights should be brighter
        assert result.image[0, 0, 0] > 0.8

    def test_no_adjustment_no_change(self):
        settings = ColorCorrectionSettings(shadows=0.0, midtones=0.0, highlights=0.0)
        corrector = ColorCorrector(settings)
        image = np.random.rand(5, 5, 3).astype(np.float32)
        result = corrector.apply(image)

        np.testing.assert_array_almost_equal(result.image, image)


class TestChannelGains:
    """Tests for per-channel gain adjustments."""

    def test_red_gain(self):
        settings = ColorCorrectionSettings(red_gain=1.5)
        corrector = ColorCorrector(settings)

        image = np.array([[[0.5, 0.5, 0.5]]])
        result = corrector.apply(image)

        # Red should be higher
        assert result.image[0, 0, 0] > result.image[0, 0, 1]
        assert result.image[0, 0, 0] > result.image[0, 0, 2]

    def test_unity_gains_no_change(self):
        settings = ColorCorrectionSettings(red_gain=1.0, green_gain=1.0, blue_gain=1.0)
        corrector = ColorCorrector(settings)
        image = np.random.rand(5, 5, 3).astype(np.float32)
        result = corrector.apply(image)

        np.testing.assert_array_almost_equal(result.image, image)


class TestVignetteInCorrector:
    """Tests for vignette in ColorCorrector."""

    def test_vignette_darkens_corners(self):
        settings = ColorCorrectionSettings(vignette_intensity=0.5)
        corrector = ColorCorrector(settings)

        # Uniform image
        image = np.full((20, 20, 3), 0.5, dtype=np.float32)
        result = corrector.apply(image)

        # Center should be brighter than corners
        center_val = result.image[10, 10, 0]
        corner_val = result.image[0, 0, 0]
        assert center_val > corner_val

    def test_no_vignette_no_change(self):
        settings = ColorCorrectionSettings(vignette_intensity=0.0)
        corrector = ColorCorrector(settings)

        image = np.full((20, 20, 3), 0.5, dtype=np.float32)
        result = corrector.apply(image)

        np.testing.assert_array_almost_equal(result.image, image)


class TestLUT:
    """Tests for LUT class."""

    def test_identity_lut_creation(self):
        lut = LUT.identity(size=17)
        assert lut.size == 17
        assert lut.data.shape == (17, 17, 17, 3)

    def test_identity_lut_no_change(self):
        lut = LUT.identity(size=17)
        image = np.random.rand(10, 10, 3).astype(np.float32)
        result = lut.apply(image)

        # Identity LUT should not change image significantly
        np.testing.assert_array_almost_equal(result, image, decimal=2)

    def test_lut_clamps_input(self):
        lut = LUT.identity(size=17)
        # Image with out-of-range values
        image = np.array([[[-0.5, 1.5, 0.5]]])
        result = lut.apply(image)

        # Should handle out-of-range gracefully
        assert not np.any(np.isnan(result))

    def test_lut_preserves_shape(self):
        lut = LUT.identity(size=17)
        image = np.random.rand(25, 30, 3).astype(np.float32)
        result = lut.apply(image)

        assert result.shape == image.shape

    def test_cube_file_parsing(self):
        # Create temporary .cube file
        cube_content = """# Test LUT
TITLE "Test"
LUT_3D_SIZE 2
0.0 0.0 0.0
1.0 0.0 0.0
0.0 1.0 0.0
1.0 1.0 0.0
0.0 0.0 1.0
1.0 0.0 1.0
0.0 1.0 1.0
1.0 1.0 1.0
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cube', delete=False) as f:
            f.write(cube_content)
            temp_path = f.name

        try:
            lut = LUT.from_cube_file(temp_path)
            assert lut.size == 2
            assert lut.data.shape == (2, 2, 2, 3)
        finally:
            os.unlink(temp_path)


class TestVignette:
    """Tests for standalone Vignette class."""

    def test_default_creation(self):
        vignette = Vignette()
        assert vignette.intensity == 0.5
        assert vignette.softness == 0.5
        assert vignette.roundness == 1.0
        assert vignette.center == (0.5, 0.5)

    def test_custom_creation(self):
        vignette = Vignette(
            intensity=0.8,
            softness=0.3,
            roundness=0.5,
            center=(0.4, 0.6),
        )
        assert vignette.intensity == 0.8
        assert vignette.softness == 0.3

    def test_apply_preserves_shape(self):
        vignette = Vignette()
        image = np.random.rand(30, 40, 3).astype(np.float32)
        result = vignette.apply(image)

        assert result.shape == image.shape

    def test_darkens_edges(self):
        vignette = Vignette(intensity=0.8)
        image = np.full((50, 50, 3), 1.0, dtype=np.float32)
        result = vignette.apply(image)

        # Center should be brightest
        center_val = result[25, 25, 0]
        corner_val = result[0, 0, 0]
        edge_val = result[0, 25, 0]

        assert center_val > edge_val
        assert center_val > corner_val

    def test_zero_intensity_no_change(self):
        vignette = Vignette(intensity=0.0)
        image = np.random.rand(20, 20, 3).astype(np.float32)
        result = vignette.apply(image)

        np.testing.assert_array_almost_equal(result, image)

    def test_off_center_vignette(self):
        vignette = Vignette(intensity=0.5, center=(0.25, 0.25))
        image = np.full((40, 40, 3), 1.0, dtype=np.float32)
        result = vignette.apply(image)

        # Brightest point should be at the offset center
        brightest_y = np.argmax(np.mean(result, axis=(1, 2)))
        brightest_x = np.argmax(np.mean(result, axis=(0, 2)))

        # Should be closer to (10, 10) than to (20, 20)
        assert abs(brightest_y - 10) < 15
        assert abs(brightest_x - 10) < 15


class TestApplyColorCorrection:
    """Tests for apply_color_correction convenience function."""

    def test_returns_result(self):
        image = np.random.rand(10, 10, 3).astype(np.float32)
        result = apply_color_correction(image)

        assert isinstance(result, ColorCorrectionResult)

    def test_passes_parameters(self):
        image = np.random.rand(10, 10, 3).astype(np.float32)
        result = apply_color_correction(
            image,
            exposure=1.0,
            contrast=1.5,
            saturation=0.8,
        )

        assert result.settings.exposure == 1.0
        assert result.settings.contrast == 1.5
        assert result.settings.saturation == 0.8


class TestColorCorrectionIntegration:
    """Integration tests for color correction pipeline."""

    def test_full_pipeline(self):
        settings = ColorCorrectionSettings(
            exposure=0.5,
            contrast=1.1,
            saturation=1.2,
            temperature=10.0,
            shadows=0.1,
            highlights=-0.1,
            vignette_intensity=0.2,
        )
        corrector = ColorCorrector(settings)

        image = np.random.rand(50, 50, 3).astype(np.float32)
        result = corrector.apply(image)

        # Output should be valid
        assert not np.any(np.isnan(result.image))
        assert not np.any(np.isinf(result.image))
        assert result.image.shape == image.shape

    def test_extreme_settings(self):
        settings = ColorCorrectionSettings(
            exposure=3.0,
            contrast=2.0,
            saturation=2.0,
            temperature=100.0,
            tint=-100.0,
        )
        corrector = ColorCorrector(settings)

        image = np.random.rand(20, 20, 3).astype(np.float32)
        result = corrector.apply(image)

        # Should not produce NaN or Inf
        assert not np.any(np.isnan(result.image))
        assert not np.any(np.isinf(result.image))

    def test_order_matters(self):
        # Exposure before contrast should give different results
        # than contrast before exposure (pipeline order is fixed)
        image = np.random.rand(10, 10, 3).astype(np.float32)

        settings = ColorCorrectionSettings(exposure=1.0, contrast=1.5)
        corrector = ColorCorrector(settings)
        result = corrector.apply(image)

        # Just verify it completes without error
        assert result.image.shape == image.shape
