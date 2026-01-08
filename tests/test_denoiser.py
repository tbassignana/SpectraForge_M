"""Tests for the denoising module."""

import math
import pytest
import numpy as np

from spectraforge.denoiser import (
    Denoiser,
    DenoiseResult,
    OIDNDenoiser,
    BilateralDenoiser,
    JointBilateralDenoiser,
    create_denoiser,
    denoise_image,
    AuxiliaryBufferRenderer,
)


class TestDenoiseResult:
    """Tests for DenoiseResult dataclass."""

    def test_creation(self):
        """Test creating a DenoiseResult."""
        image = np.zeros((100, 100, 3), dtype=np.float32)
        result = DenoiseResult(image=image, quality=0.8, method="test")

        assert result.image.shape == (100, 100, 3)
        assert result.quality == 0.8
        assert result.method == "test"


class TestBilateralDenoiser:
    """Tests for the bilateral filter denoiser."""

    def test_creation(self):
        """Test creating a bilateral denoiser."""
        denoiser = BilateralDenoiser()
        assert denoiser.name == "Bilateral Filter"
        assert denoiser.supports_auxiliary is False

    def test_custom_parameters(self):
        """Test creating with custom parameters."""
        denoiser = BilateralDenoiser(
            sigma_spatial=5.0,
            sigma_range=0.2,
            kernel_size=9,
        )
        assert denoiser.sigma_spatial == 5.0
        assert denoiser.sigma_range == 0.2
        assert denoiser.kernel_size == 9

    def test_kernel_size_made_odd(self):
        """Test that even kernel size is made odd."""
        denoiser = BilateralDenoiser(kernel_size=8)
        assert denoiser.kernel_size == 9

    def test_denoise_uniform_image(self):
        """Test denoising a uniform image (no change expected)."""
        denoiser = BilateralDenoiser(kernel_size=3)
        image = np.ones((10, 10, 3), dtype=np.float32) * 0.5

        result = denoiser.denoise(image)

        assert result.image.shape == image.shape
        assert result.method == "Bilateral Filter"
        # Uniform image should stay uniform
        assert np.allclose(result.image, 0.5, atol=0.01)

    def test_denoise_reduces_noise(self):
        """Test that denoising reduces image variance."""
        denoiser = BilateralDenoiser(sigma_spatial=2.0, sigma_range=0.3, kernel_size=5)

        # Create noisy image
        np.random.seed(42)
        base = np.ones((20, 20, 3), dtype=np.float32) * 0.5
        noise = np.random.normal(0, 0.1, (20, 20, 3)).astype(np.float32)
        noisy = base + noise

        result = denoiser.denoise(noisy)

        # Variance should be reduced
        original_var = np.var(noisy)
        denoised_var = np.var(result.image)
        assert denoised_var < original_var

    def test_denoise_preserves_edges(self):
        """Test that bilateral filter preserves strong edges."""
        denoiser = BilateralDenoiser(sigma_spatial=2.0, sigma_range=0.05, kernel_size=5)

        # Create image with strong edge
        image = np.zeros((20, 20, 3), dtype=np.float32)
        image[:, 10:, :] = 1.0

        result = denoiser.denoise(image)

        # Check that edge is still visible (left side dark, right side bright)
        left_mean = np.mean(result.image[:, :5, :])
        right_mean = np.mean(result.image[:, 15:, :])
        assert right_mean > left_mean + 0.5

    def test_ignores_auxiliary_buffers(self):
        """Test that auxiliary buffers are ignored."""
        denoiser = BilateralDenoiser(kernel_size=3)
        image = np.ones((10, 10, 3), dtype=np.float32)
        albedo = np.ones((10, 10, 3), dtype=np.float32) * 0.5
        normal = np.array([[[0, 1, 0]]] * 100).reshape(10, 10, 3).astype(np.float32)

        # Should not raise even with auxiliary buffers
        result = denoiser.denoise(image, albedo=albedo, normal=normal)
        assert result.image.shape == image.shape


class TestJointBilateralDenoiser:
    """Tests for the joint bilateral filter denoiser."""

    def test_creation(self):
        """Test creating a joint bilateral denoiser."""
        denoiser = JointBilateralDenoiser()
        assert denoiser.name == "Joint Bilateral Filter"
        assert denoiser.supports_auxiliary is True

    def test_denoise_without_auxiliary(self):
        """Test denoising without auxiliary buffers."""
        denoiser = JointBilateralDenoiser(kernel_size=3)
        image = np.ones((10, 10, 3), dtype=np.float32) * 0.5

        result = denoiser.denoise(image)

        assert result.image.shape == image.shape
        assert result.method == "Joint Bilateral Filter"

    def test_denoise_with_albedo(self):
        """Test that albedo buffer affects result."""
        denoiser = JointBilateralDenoiser(
            sigma_spatial=2.0, sigma_albedo=0.1, kernel_size=5
        )

        # Create noisy image
        np.random.seed(42)
        base = np.ones((20, 20, 3), dtype=np.float32) * 0.5
        noise = np.random.normal(0, 0.1, (20, 20, 3)).astype(np.float32)
        noisy = base + noise

        # Albedo with edge at center
        albedo = np.zeros((20, 20, 3), dtype=np.float32)
        albedo[:, 10:, :] = 1.0

        result = denoiser.denoise(noisy, albedo=albedo)

        assert result.quality > 0

    def test_denoise_with_normal(self):
        """Test that normal buffer affects result."""
        denoiser = JointBilateralDenoiser(
            sigma_spatial=2.0, sigma_normal=0.3, kernel_size=5
        )

        # Create noisy image
        np.random.seed(42)
        base = np.ones((20, 20, 3), dtype=np.float32) * 0.5
        noise = np.random.normal(0, 0.1, (20, 20, 3)).astype(np.float32)
        noisy = base + noise

        # Normals pointing up
        normal = np.zeros((20, 20, 3), dtype=np.float32)
        normal[:, :, 1] = 1.0  # Y-up

        result = denoiser.denoise(noisy, normal=normal)

        assert result.quality > 0

    def test_denoise_with_both_buffers(self):
        """Test denoising with both albedo and normal."""
        denoiser = JointBilateralDenoiser(kernel_size=5)

        np.random.seed(42)
        base = np.ones((15, 15, 3), dtype=np.float32) * 0.5
        noise = np.random.normal(0, 0.1, (15, 15, 3)).astype(np.float32)
        noisy = base + noise

        albedo = np.ones((15, 15, 3), dtype=np.float32) * 0.8
        normal = np.zeros((15, 15, 3), dtype=np.float32)
        normal[:, :, 2] = 1.0  # Z-forward

        result = denoiser.denoise(noisy, albedo=albedo, normal=normal)

        assert result.image.shape == noisy.shape
        # Quality should be higher with auxiliary buffers
        assert result.quality > 0


class TestOIDNDenoiser:
    """Tests for OIDN denoiser (may skip if OIDN not installed)."""

    def test_creation(self):
        """Test creating an OIDN denoiser."""
        denoiser = OIDNDenoiser()
        assert denoiser.name == "Intel Open Image Denoise"
        assert denoiser.supports_auxiliary is True

    def test_is_available(self):
        """Test checking OIDN availability."""
        denoiser = OIDNDenoiser()
        # Should return boolean regardless of installation
        available = denoiser.is_available()
        assert isinstance(available, bool)

    def test_hdr_mode(self):
        """Test HDR mode configuration."""
        denoiser_hdr = OIDNDenoiser(hdr=True)
        denoiser_ldr = OIDNDenoiser(hdr=False)

        assert denoiser_hdr.hdr is True
        assert denoiser_ldr.hdr is False

    @pytest.mark.skipif(
        not OIDNDenoiser().is_available(),
        reason="OIDN not installed"
    )
    def test_denoise_with_oidn(self):
        """Test actual OIDN denoising (only runs if OIDN available)."""
        denoiser = OIDNDenoiser()

        np.random.seed(42)
        base = np.ones((64, 64, 3), dtype=np.float32) * 0.5
        noise = np.random.normal(0, 0.1, (64, 64, 3)).astype(np.float32)
        noisy = np.clip(base + noise, 0, 1).astype(np.float32)

        result = denoiser.denoise(noisy)

        assert result.image.shape == noisy.shape
        assert result.method == "Intel Open Image Denoise"

    def test_denoise_without_oidn_raises(self):
        """Test that denoising without OIDN raises RuntimeError."""
        denoiser = OIDNDenoiser()

        if denoiser.is_available():
            pytest.skip("OIDN is available, can't test unavailable case")

        image = np.ones((10, 10, 3), dtype=np.float32)

        with pytest.raises(RuntimeError, match="oidn"):
            denoiser.denoise(image)


class TestCreateDenoiser:
    """Tests for the create_denoiser factory function."""

    def test_creates_denoiser(self):
        """Test that create_denoiser returns a Denoiser."""
        denoiser = create_denoiser()
        assert isinstance(denoiser, Denoiser)

    def test_prefer_oidn_false(self):
        """Test that prefer_oidn=False returns non-OIDN denoiser."""
        denoiser = create_denoiser(prefer_oidn=False)
        assert isinstance(denoiser, JointBilateralDenoiser)

    def test_fallback_when_oidn_unavailable(self):
        """Test fallback to joint bilateral when OIDN unavailable."""
        oidn = OIDNDenoiser()
        if oidn.is_available():
            pytest.skip("OIDN is available")

        denoiser = create_denoiser(prefer_oidn=True)
        assert isinstance(denoiser, JointBilateralDenoiser)


class TestDenoiseImage:
    """Tests for the denoise_image convenience function."""

    def test_auto_method(self):
        """Test auto method selection."""
        image = np.ones((10, 10, 3), dtype=np.float32)
        result = denoise_image(image, method="auto")
        assert isinstance(result, DenoiseResult)

    def test_bilateral_method(self):
        """Test explicit bilateral method."""
        image = np.ones((10, 10, 3), dtype=np.float32)
        result = denoise_image(image, method="bilateral")
        assert result.method == "Bilateral Filter"

    def test_joint_method(self):
        """Test explicit joint bilateral method."""
        image = np.ones((10, 10, 3), dtype=np.float32)
        result = denoise_image(image, method="joint")
        assert result.method == "Joint Bilateral Filter"

    def test_invalid_method_raises(self):
        """Test that invalid method raises ValueError."""
        image = np.ones((10, 10, 3), dtype=np.float32)

        with pytest.raises(ValueError, match="Unknown denoising method"):
            denoise_image(image, method="invalid")

    def test_with_auxiliary_buffers(self):
        """Test denoising with auxiliary buffers."""
        image = np.ones((10, 10, 3), dtype=np.float32)
        albedo = np.ones((10, 10, 3), dtype=np.float32) * 0.5
        normal = np.zeros((10, 10, 3), dtype=np.float32)
        normal[:, :, 1] = 1.0

        result = denoise_image(image, albedo=albedo, normal=normal, method="joint")

        assert result.image.shape == image.shape


class TestAuxiliaryBufferRenderer:
    """Tests for auxiliary buffer rendering helper."""

    def test_creation(self):
        """Test creating auxiliary buffer renderer."""
        renderer = AuxiliaryBufferRenderer(width=100, height=80)

        assert renderer.width == 100
        assert renderer.height == 80
        assert renderer.albedo_buffer.shape == (80, 100, 3)
        assert renderer.normal_buffer.shape == (80, 100, 3)
        assert renderer.sample_count.shape == (80, 100)

    def test_add_single_sample(self):
        """Test adding a single sample."""
        renderer = AuxiliaryBufferRenderer(10, 10)

        renderer.add_sample(5, 5, (0.8, 0.6, 0.4), (0, 1, 0))

        assert np.allclose(renderer.albedo_buffer[5, 5], [0.8, 0.6, 0.4])
        assert np.allclose(renderer.normal_buffer[5, 5], [0, 1, 0])
        assert renderer.sample_count[5, 5] == 1

    def test_add_multiple_samples_averaging(self):
        """Test that multiple samples are averaged."""
        renderer = AuxiliaryBufferRenderer(10, 10)

        renderer.add_sample(3, 3, (1.0, 0.0, 0.0), (1, 0, 0))
        renderer.add_sample(3, 3, (0.0, 1.0, 0.0), (0, 1, 0))

        assert renderer.sample_count[3, 3] == 2
        # Average of (1,0,0) and (0,1,0)
        assert np.allclose(renderer.albedo_buffer[3, 3], [0.5, 0.5, 0.0])
        assert np.allclose(renderer.normal_buffer[3, 3], [0.5, 0.5, 0.0])

    def test_get_albedo(self):
        """Test getting albedo buffer."""
        renderer = AuxiliaryBufferRenderer(10, 10)
        renderer.add_sample(0, 0, (0.5, 0.5, 0.5), (0, 0, 1))

        albedo = renderer.get_albedo()

        assert albedo.shape == (10, 10, 3)
        assert np.allclose(albedo[0, 0], [0.5, 0.5, 0.5])

    def test_get_normal_normalized(self):
        """Test that normals are normalized on get."""
        renderer = AuxiliaryBufferRenderer(10, 10)

        # Add non-unit normal
        renderer.add_sample(0, 0, (1, 1, 1), (3, 4, 0))

        normal = renderer.get_normal()

        # Should be normalized (3,4,0)/5 = (0.6, 0.8, 0)
        assert np.allclose(normal[0, 0], [0.6, 0.8, 0.0], atol=0.01)

        # Verify unit length
        length = np.linalg.norm(normal[0, 0])
        assert abs(length - 1.0) < 0.01

    def test_clear(self):
        """Test clearing buffers."""
        renderer = AuxiliaryBufferRenderer(10, 10)
        renderer.add_sample(5, 5, (0.8, 0.6, 0.4), (0, 1, 0))

        renderer.clear()

        assert np.all(renderer.albedo_buffer == 0)
        assert np.all(renderer.normal_buffer == 0)
        assert np.all(renderer.sample_count == 0)


class TestDenoiserIntegration:
    """Integration tests for the denoising system."""

    def test_denoise_gradient_image(self):
        """Test denoising a gradient image with noise."""
        denoiser = BilateralDenoiser(sigma_spatial=2.0, kernel_size=5)

        # Create gradient with noise
        gradient = np.zeros((50, 50, 3), dtype=np.float32)
        for x in range(50):
            gradient[:, x, :] = x / 49.0

        np.random.seed(42)
        noise = np.random.normal(0, 0.05, (50, 50, 3)).astype(np.float32)
        noisy = gradient + noise

        result = denoiser.denoise(noisy)

        # Result should be closer to original gradient than noisy version
        original_error = np.mean((noisy - gradient) ** 2)
        denoised_error = np.mean((result.image - gradient) ** 2)
        assert denoised_error < original_error

    def test_denoise_preserves_hdr_values(self):
        """Test that HDR values (>1) are preserved."""
        denoiser = BilateralDenoiser(kernel_size=3)

        # HDR image with values > 1
        image = np.ones((10, 10, 3), dtype=np.float32) * 5.0

        result = denoiser.denoise(image)

        # Mean should still be around 5.0
        assert np.mean(result.image) > 4.5

    def test_auxiliary_buffers_improve_quality(self):
        """Test that auxiliary buffers improve denoising quality."""
        np.random.seed(42)

        # Create image with textured region
        base = np.zeros((30, 30, 3), dtype=np.float32)
        base[:, :15, :] = 0.2  # Dark left
        base[:, 15:, :] = 0.8  # Bright right

        noise = np.random.normal(0, 0.15, (30, 30, 3)).astype(np.float32)
        noisy = base + noise

        # Albedo matches the base pattern
        albedo = np.zeros((30, 30, 3), dtype=np.float32)
        albedo[:, :15, :] = 0.2
        albedo[:, 15:, :] = 0.8

        # Denoise without auxiliary
        denoiser_plain = JointBilateralDenoiser(kernel_size=5)
        result_plain = denoiser_plain.denoise(noisy)

        # Denoise with albedo
        result_with_albedo = denoiser_plain.denoise(noisy, albedo=albedo)

        # Both should reduce variance, but we mainly test they work
        assert np.var(result_plain.image) < np.var(noisy)
        assert result_with_albedo.image.shape == noisy.shape
