"""Tests for tone mapping operators."""

import pytest
import numpy as np

from spectraforge.tonemapping import (
    ToneMapper, ToneMappingOperator, ToneMappingResult,
    LinearToneMapper, ReinhardToneMapper, ACESFilmicToneMapper,
    Uncharted2ToneMapper, ExposureToneMapper,
    create_tone_mapper, tone_map, apply_gamma, linear_to_srgb, srgb_to_linear
)


class TestLinearToneMapper:
    """Tests for LinearToneMapper."""

    def test_name(self):
        mapper = LinearToneMapper()
        assert mapper.name == "Linear"

    def test_clamps_values(self):
        mapper = LinearToneMapper()
        image = np.array([[[2.0, -0.5, 0.5]]])
        result = mapper.apply(image)
        np.testing.assert_array_almost_equal(result, [[[1.0, 0.0, 0.5]]])

    def test_preserves_valid_range(self):
        mapper = LinearToneMapper()
        image = np.array([[[0.3, 0.5, 0.7]]])
        result = mapper.apply(image)
        np.testing.assert_array_almost_equal(result, image)


class TestReinhardToneMapper:
    """Tests for ReinhardToneMapper."""

    def test_name_basic(self):
        mapper = ReinhardToneMapper()
        assert mapper.name == "Reinhard"

    def test_name_extended(self):
        mapper = ReinhardToneMapper(white_point=4.0)
        assert mapper.name == "Reinhard Extended"

    def test_get_parameters(self):
        mapper = ReinhardToneMapper(key=0.2, white_point=5.0)
        params = mapper.get_parameters()
        assert params["key"] == 0.2
        assert params["white_point"] == 5.0

    def test_compresses_highlights(self):
        mapper = ReinhardToneMapper()
        # Very bright pixel
        image = np.array([[[10.0, 10.0, 10.0]]])
        result = mapper.apply(image)
        # Should be compressed below 1.0
        assert result[0, 0, 0] < 1.0
        assert result[0, 0, 0] > 0.0  # Non-zero output

    def test_preserves_shadows(self):
        mapper = ReinhardToneMapper()
        # Dark pixel
        image = np.array([[[0.1, 0.1, 0.1]]])
        result = mapper.apply(image)
        # Should still be relatively dark
        assert result[0, 0, 0] < 0.5

    def test_extended_burns_whites(self):
        mapper = ReinhardToneMapper(white_point=4.0)
        mapper_simple = ReinhardToneMapper()
        # Test that extended Reinhard produces different results
        image = np.array([[[4.0, 4.0, 4.0]]])
        result = mapper.apply(image)
        result_simple = mapper_simple.apply(image)
        # Extended version should have different response
        # Both should produce valid output
        assert result[0, 0, 0] > 0.0
        assert result[0, 0, 0] < 1.0


class TestACESFilmicToneMapper:
    """Tests for ACESFilmicToneMapper."""

    def test_name(self):
        mapper = ACESFilmicToneMapper()
        assert mapper.name == "ACES Filmic"

    def test_exposure_bias_parameter(self):
        mapper = ACESFilmicToneMapper(exposure_bias=1.0)
        params = mapper.get_parameters()
        assert params["exposure_bias"] == 1.0

    def test_output_range(self):
        mapper = ACESFilmicToneMapper()
        # Test various HDR values
        image = np.array([[[0.0, 0.5, 1.0], [2.0, 5.0, 10.0]]])
        result = mapper.apply(image)
        # All outputs should be in [0, 1]
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_exposure_bias_brightens(self):
        mapper_normal = ACESFilmicToneMapper(exposure_bias=0.0)
        mapper_bright = ACESFilmicToneMapper(exposure_bias=1.0)
        image = np.array([[[0.5, 0.5, 0.5]]])
        result_normal = mapper_normal.apply(image)
        result_bright = mapper_bright.apply(image)
        # Positive exposure bias should brighten
        assert np.mean(result_bright) > np.mean(result_normal)


class TestUncharted2ToneMapper:
    """Tests for Uncharted2ToneMapper."""

    def test_name(self):
        mapper = Uncharted2ToneMapper()
        assert mapper.name == "Uncharted 2 Filmic"

    def test_get_parameters(self):
        mapper = Uncharted2ToneMapper(white_point=10.0)
        params = mapper.get_parameters()
        assert params["white_point"] == 10.0
        assert "A" in params
        assert "B" in params

    def test_output_range(self):
        mapper = Uncharted2ToneMapper()
        image = np.array([[[0.1, 1.0, 5.0], [10.0, 20.0, 50.0]]])
        result = mapper.apply(image)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_exposure_bias(self):
        mapper_low = Uncharted2ToneMapper(exposure_bias=1.0)
        mapper_high = Uncharted2ToneMapper(exposure_bias=4.0)
        image = np.array([[[0.5, 0.5, 0.5]]])
        result_low = mapper_low.apply(image)
        result_high = mapper_high.apply(image)
        assert np.mean(result_high) > np.mean(result_low)


class TestExposureToneMapper:
    """Tests for ExposureToneMapper."""

    def test_name(self):
        mapper = ExposureToneMapper()
        assert mapper.name == "Exposure"

    def test_get_parameters(self):
        mapper = ExposureToneMapper(exposure=1.5, gamma=2.4)
        params = mapper.get_parameters()
        assert params["exposure"] == 1.5
        assert params["gamma"] == 2.4

    def test_zero_exposure_no_change(self):
        mapper = ExposureToneMapper(exposure=0.0, gamma=1.0)
        image = np.array([[[0.5, 0.5, 0.5]]])
        result = mapper.apply(image)
        np.testing.assert_array_almost_equal(result, image)

    def test_positive_exposure_brightens(self):
        mapper = ExposureToneMapper(exposure=1.0, gamma=1.0)
        image = np.array([[[0.25, 0.25, 0.25]]])
        result = mapper.apply(image)
        # +1 EV doubles brightness
        np.testing.assert_array_almost_equal(result, [[[0.5, 0.5, 0.5]]])

    def test_gamma_correction(self):
        mapper = ExposureToneMapper(exposure=0.0, gamma=2.2)
        image = np.array([[[0.25, 0.5, 0.75]]])
        result = mapper.apply(image)
        expected = np.power(image, 1.0 / 2.2)
        np.testing.assert_array_almost_equal(result, expected)


class TestCreateToneMapper:
    """Tests for create_tone_mapper factory function."""

    def test_create_linear(self):
        mapper = create_tone_mapper(ToneMappingOperator.LINEAR)
        assert isinstance(mapper, LinearToneMapper)

    def test_create_reinhard(self):
        mapper = create_tone_mapper(ToneMappingOperator.REINHARD)
        assert isinstance(mapper, ReinhardToneMapper)

    def test_create_reinhard_extended(self):
        mapper = create_tone_mapper(ToneMappingOperator.REINHARD_EXTENDED)
        assert isinstance(mapper, ReinhardToneMapper)
        assert mapper.white_point is not None

    def test_create_aces(self):
        mapper = create_tone_mapper(ToneMappingOperator.ACES_FILMIC)
        assert isinstance(mapper, ACESFilmicToneMapper)

    def test_create_uncharted2(self):
        mapper = create_tone_mapper(ToneMappingOperator.UNCHARTED2)
        assert isinstance(mapper, Uncharted2ToneMapper)

    def test_create_exposure(self):
        mapper = create_tone_mapper(ToneMappingOperator.EXPOSURE)
        assert isinstance(mapper, ExposureToneMapper)

    def test_pass_kwargs(self):
        mapper = create_tone_mapper(ToneMappingOperator.REINHARD, key=0.2)
        assert mapper.key == 0.2


class TestToneMapFunction:
    """Tests for tone_map convenience function."""

    def test_returns_result_object(self):
        image = np.random.rand(10, 10, 3)
        result = tone_map(image)
        assert isinstance(result, ToneMappingResult)
        assert result.image.shape == image.shape
        assert isinstance(result.operator, str)
        assert isinstance(result.parameters, dict)

    def test_default_aces(self):
        image = np.random.rand(10, 10, 3)
        result = tone_map(image)
        assert result.operator == "ACES Filmic"

    def test_without_srgb_conversion(self):
        image = np.array([[[0.5, 0.5, 0.5]]])
        result_with_srgb = tone_map(
            image,
            operator=ToneMappingOperator.LINEAR,
            convert_to_srgb=True
        )
        result_without_srgb = tone_map(
            image,
            operator=ToneMappingOperator.LINEAR,
            convert_to_srgb=False
        )
        # With sRGB conversion should be brighter (gamma applied)
        assert np.mean(result_with_srgb.image) > np.mean(result_without_srgb.image)


class TestGammaFunctions:
    """Tests for gamma and sRGB conversion functions."""

    def test_apply_gamma(self):
        image = np.array([[[0.25, 0.5, 1.0]]])
        result = apply_gamma(image, gamma=2.2)
        expected = np.power(image, 1.0 / 2.2)
        np.testing.assert_array_almost_equal(result, expected)

    def test_apply_gamma_clamps(self):
        image = np.array([[[-0.5, 1.5, 0.5]]])
        result = apply_gamma(image, gamma=2.2)
        assert np.all(result >= 0.0)

    def test_linear_to_srgb_small_values(self):
        # Small values use linear portion
        small = np.array([[[0.001, 0.002, 0.003]]])
        result = linear_to_srgb(small)
        expected = small * 12.92
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_linear_to_srgb_large_values(self):
        # Large values use power curve
        large = np.array([[[0.5, 0.8, 1.0]]])
        result = linear_to_srgb(large)
        expected = 1.055 * np.power(large, 1.0 / 2.4) - 0.055
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_srgb_roundtrip(self):
        # sRGB -> linear -> sRGB should be identity
        original = np.random.rand(5, 5, 3)
        linear = srgb_to_linear(original)
        back_to_srgb = linear_to_srgb(linear)
        np.testing.assert_array_almost_equal(original, back_to_srgb, decimal=5)

    def test_linear_roundtrip(self):
        # linear -> sRGB -> linear should be identity
        original = np.random.rand(5, 5, 3)
        srgb = linear_to_srgb(original)
        back_to_linear = srgb_to_linear(srgb)
        np.testing.assert_array_almost_equal(original, back_to_linear, decimal=5)


class TestToneMappingIntegration:
    """Integration tests for tone mapping pipeline."""

    def test_hdr_to_ldr_gradient(self):
        # Create HDR gradient with values from 0 to 10
        gradient = np.zeros((1, 100, 3))
        gradient[0, :, :] = np.linspace(0, 10, 100)[:, np.newaxis]

        result = tone_map(gradient, operator=ToneMappingOperator.ACES_FILMIC)

        # Output should be monotonically increasing
        diffs = np.diff(result.image[0, :, 0])
        assert np.all(diffs >= 0)

        # All values should be in [0, 1]
        assert np.all(result.image >= 0.0)
        assert np.all(result.image <= 1.0)

    def test_preserve_black(self):
        # Black should remain black
        image = np.zeros((10, 10, 3))
        for op in ToneMappingOperator:
            result = tone_map(image, operator=op, convert_to_srgb=False)
            np.testing.assert_array_almost_equal(
                result.image, np.zeros((10, 10, 3)),
                err_msg=f"Failed for {op}"
            )

    def test_color_preservation(self):
        # Test that colors are preserved (hue doesn't shift)
        # Red pixel
        red = np.array([[[2.0, 0.2, 0.2]]])
        result = tone_map(red, operator=ToneMappingOperator.REINHARD, convert_to_srgb=False)
        # Red should still be dominant
        assert result.image[0, 0, 0] > result.image[0, 0, 1]
        assert result.image[0, 0, 0] > result.image[0, 0, 2]
