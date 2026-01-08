"""Tests for Arbitrary Output Variables (AOV) / Render Passes."""

import pytest
import numpy as np

from spectraforge.aov import (
    AOVType, AOVSample, AOVBuffer, AOVManager,
    normalize_depth, pack_normal, unpack_normal,
    create_object_id_colormap, combine_direct_indirect,
    depth_to_world_position, RenderPassCompositor
)
from spectraforge.vec3 import Vec3, Color, Point3


class TestAOVType:
    """Tests for AOVType enum."""

    def test_common_aov_types(self):
        assert AOVType.BEAUTY.value == "beauty"
        assert AOVType.DEPTH.value == "depth"
        assert AOVType.NORMAL.value == "normal"
        assert AOVType.ALBEDO.value == "albedo"
        assert AOVType.OBJECT_ID.value == "object_id"

    def test_lighting_aov_types(self):
        assert AOVType.DIRECT.value == "direct"
        assert AOVType.INDIRECT.value == "indirect"
        assert AOVType.DIFFUSE.value == "diffuse"
        assert AOVType.SPECULAR.value == "specular"


class TestAOVSample:
    """Tests for AOVSample dataclass."""

    def test_default_creation(self):
        sample = AOVSample()
        assert sample.depth == float('inf')
        assert sample.normal is None
        assert sample.albedo is None
        assert sample.object_id == 0

    def test_custom_creation(self):
        sample = AOVSample(
            depth=5.0,
            normal=Vec3(0, 1, 0),
            albedo=Color(0.8, 0.2, 0.1),
            object_id=42,
            material_id=7
        )
        assert sample.depth == 5.0
        assert sample.normal == Vec3(0, 1, 0)
        assert sample.albedo == Color(0.8, 0.2, 0.1)
        assert sample.object_id == 42
        assert sample.material_id == 7


class TestAOVBuffer:
    """Tests for AOVBuffer class."""

    def test_creation(self):
        buffer = AOVBuffer(width=100, height=50, aov_type=AOVType.DEPTH)
        assert buffer.width == 100
        assert buffer.height == 50
        assert buffer.aov_type == AOVType.DEPTH

    def test_depth_buffer_single_channel(self):
        buffer = AOVBuffer(width=10, height=10, aov_type=AOVType.DEPTH)
        assert buffer.data.shape == (10, 10, 1)

    def test_normal_buffer_three_channels(self):
        buffer = AOVBuffer(width=10, height=10, aov_type=AOVType.NORMAL)
        assert buffer.data.shape == (10, 10, 3)

    def test_uv_buffer_two_channels(self):
        buffer = AOVBuffer(width=10, height=10, aov_type=AOVType.UV)
        assert buffer.data.shape == (10, 10, 2)

    def test_add_sample_scalar(self):
        buffer = AOVBuffer(width=10, height=10, aov_type=AOVType.DEPTH)
        buffer.add_sample(5, 5, 3.5)
        assert buffer.data[5, 5, 0] == 3.5

    def test_add_sample_vec3(self):
        buffer = AOVBuffer(width=10, height=10, aov_type=AOVType.NORMAL)
        buffer.add_sample(5, 5, Vec3(0.5, 0.8, -0.3))
        np.testing.assert_array_almost_equal(
            buffer.data[5, 5], [0.5, 0.8, -0.3]
        )

    def test_add_sample_tuple(self):
        buffer = AOVBuffer(width=10, height=10, aov_type=AOVType.UV)
        buffer.add_sample(5, 5, (0.25, 0.75))
        np.testing.assert_array_almost_equal(
            buffer.data[5, 5], [0.25, 0.75]
        )

    def test_sample_averaging(self):
        buffer = AOVBuffer(width=10, height=10, aov_type=AOVType.DEPTH)

        # Add multiple samples to same pixel
        buffer.add_sample(5, 5, 1.0)
        buffer.add_sample(5, 5, 3.0)
        buffer.add_sample(5, 5, 2.0)

        # Should be average
        assert buffer.data[5, 5, 0] == 2.0

    def test_out_of_bounds_ignored(self):
        buffer = AOVBuffer(width=10, height=10, aov_type=AOVType.DEPTH)

        # Should not crash
        buffer.add_sample(-1, 5, 1.0)
        buffer.add_sample(5, -1, 1.0)
        buffer.add_sample(100, 5, 1.0)
        buffer.add_sample(5, 100, 1.0)

    def test_get_image(self):
        buffer = AOVBuffer(width=10, height=10, aov_type=AOVType.DEPTH)
        buffer.add_sample(5, 5, 2.5)

        image = buffer.get_image()
        assert image.shape == (10, 10, 1)
        assert image[5, 5, 0] == 2.5

    def test_clear(self):
        buffer = AOVBuffer(width=10, height=10, aov_type=AOVType.DEPTH)
        buffer.add_sample(5, 5, 2.5)
        buffer.clear()

        assert buffer.data[5, 5, 0] == 0.0
        assert buffer.sample_count[5, 5] == 0


class TestAOVManager:
    """Tests for AOVManager class."""

    def test_creation(self):
        manager = AOVManager(width=100, height=50)
        assert manager.width == 100
        assert manager.height == 50
        assert len(manager.buffers) == 0

    def test_enable_aov(self):
        manager = AOVManager(width=50, height=50)
        manager.enable_aov(AOVType.DEPTH)

        assert manager.is_enabled(AOVType.DEPTH)
        assert not manager.is_enabled(AOVType.NORMAL)

    def test_enable_multiple_aovs(self):
        manager = AOVManager(width=50, height=50)
        manager.enable_aovs([AOVType.DEPTH, AOVType.NORMAL, AOVType.ALBEDO])

        assert manager.is_enabled(AOVType.DEPTH)
        assert manager.is_enabled(AOVType.NORMAL)
        assert manager.is_enabled(AOVType.ALBEDO)

    def test_disable_aov(self):
        manager = AOVManager(width=50, height=50)
        manager.enable_aov(AOVType.DEPTH)
        manager.disable_aov(AOVType.DEPTH)

        assert not manager.is_enabled(AOVType.DEPTH)

    def test_add_sample_distributes_to_buffers(self):
        manager = AOVManager(width=10, height=10)
        manager.enable_aovs([AOVType.DEPTH, AOVType.NORMAL])

        sample = AOVSample(
            depth=5.0,
            normal=Vec3(0, 1, 0)
        )
        manager.add_sample(5, 5, sample)

        depth_img = manager.get_aov(AOVType.DEPTH)
        normal_img = manager.get_aov(AOVType.NORMAL)

        assert depth_img[5, 5, 0] == 5.0
        np.testing.assert_array_almost_equal(
            normal_img[5, 5], [0, 1, 0]
        )

    def test_get_aov_returns_none_if_disabled(self):
        manager = AOVManager(width=10, height=10)
        assert manager.get_aov(AOVType.DEPTH) is None

    def test_get_all_aovs(self):
        manager = AOVManager(width=10, height=10)
        manager.enable_aovs([AOVType.DEPTH, AOVType.NORMAL])

        all_aovs = manager.get_all_aovs()
        assert "depth" in all_aovs
        assert "normal" in all_aovs
        assert len(all_aovs) == 2

    def test_clear(self):
        manager = AOVManager(width=10, height=10)
        manager.enable_aov(AOVType.DEPTH)

        sample = AOVSample(depth=5.0)
        manager.add_sample(5, 5, sample)
        manager.clear()

        depth_img = manager.get_aov(AOVType.DEPTH)
        assert depth_img[5, 5, 0] == 0.0


class TestNormalizeDepth:
    """Tests for normalize_depth function."""

    def test_basic_normalization(self):
        depth = np.array([[1.0, 5.0], [10.0, 20.0]])
        normalized = normalize_depth(depth, near=0.0, far=20.0)

        expected = np.array([[0.05, 0.25], [0.5, 1.0]])
        np.testing.assert_array_almost_equal(normalized, expected)

    def test_auto_far_plane(self):
        depth = np.array([[1.0, 5.0], [10.0, 20.0]])
        normalized = normalize_depth(depth)

        # Max value should be approximately 1.0
        np.testing.assert_almost_equal(normalized.max(), 1.0, decimal=5)
        assert normalized.min() >= 0.0

    def test_invert(self):
        depth = np.array([[1.0, 5.0], [10.0, 20.0]])
        normalized = normalize_depth(depth, far=20.0, invert=True)

        # Closest should be brightest (near 1.0)
        assert normalized[0, 0] > normalized[1, 1]

    def test_handles_infinity(self):
        depth = np.array([[1.0, float('inf')], [5.0, 10.0]])
        normalized = normalize_depth(depth, far=10.0)

        # Infinite values should become 1.0 (far plane)
        assert normalized[0, 1] == 1.0


class TestNormalPacking:
    """Tests for normal packing/unpacking functions."""

    def test_pack_normal(self):
        normals = np.array([[[0, 0, 1], [-1, 0, 0], [0, 1, 0]]], dtype=np.float32)
        packed = pack_normal(normals)

        expected = np.array([[[0.5, 0.5, 1.0], [0.0, 0.5, 0.5], [0.5, 1.0, 0.5]]])
        np.testing.assert_array_almost_equal(packed, expected)

    def test_unpack_normal(self):
        packed = np.array([[[0.5, 0.5, 1.0], [0.0, 0.5, 0.5]]], dtype=np.float32)
        unpacked = unpack_normal(packed)

        expected = np.array([[[0, 0, 1], [-1, 0, 0]]], dtype=np.float32)
        np.testing.assert_array_almost_equal(unpacked, expected)

    def test_roundtrip(self):
        original = np.random.rand(10, 10, 3) * 2 - 1  # Random normals in [-1, 1]
        packed = pack_normal(original)
        unpacked = unpack_normal(packed)

        np.testing.assert_array_almost_equal(original, unpacked)


class TestCreateObjectIDColormap:
    """Tests for create_object_id_colormap function."""

    def test_basic_colormap(self):
        ids = np.array([[0, 1], [2, 1]])
        colored = create_object_id_colormap(ids)

        assert colored.shape == (2, 2, 3)

    def test_background_is_black(self):
        ids = np.array([[0, 1], [0, 2]])
        colored = create_object_id_colormap(ids)

        # Background (ID 0) should be black
        np.testing.assert_array_equal(colored[0, 0], [0, 0, 0])
        np.testing.assert_array_equal(colored[1, 0], [0, 0, 0])

    def test_same_id_same_color(self):
        ids = np.array([[1, 2], [1, 2]])
        colored = create_object_id_colormap(ids, seed=42)

        # Same ID should have same color
        np.testing.assert_array_equal(colored[0, 0], colored[1, 0])
        np.testing.assert_array_equal(colored[0, 1], colored[1, 1])

    def test_different_ids_different_colors(self):
        ids = np.array([[1, 2], [3, 4]])
        colored = create_object_id_colormap(ids, seed=42)

        # Different IDs should have different colors
        assert not np.array_equal(colored[0, 0], colored[0, 1])
        assert not np.array_equal(colored[0, 0], colored[1, 0])

    def test_seed_reproducibility(self):
        ids = np.array([[1, 2], [3, 4]])
        colored1 = create_object_id_colormap(ids, seed=42)
        colored2 = create_object_id_colormap(ids, seed=42)

        np.testing.assert_array_equal(colored1, colored2)


class TestCombineDirectIndirect:
    """Tests for combine_direct_indirect function."""

    def test_basic_combination(self):
        direct = np.full((10, 10, 3), 0.3, dtype=np.float32)
        indirect = np.full((10, 10, 3), 0.2, dtype=np.float32)

        combined = combine_direct_indirect(direct, indirect)
        np.testing.assert_array_almost_equal(
            combined, np.full((10, 10, 3), 0.5)
        )

    def test_with_albedo(self):
        direct = np.full((10, 10, 3), 1.0, dtype=np.float32)
        indirect = np.full((10, 10, 3), 1.0, dtype=np.float32)
        albedo = np.full((10, 10, 3), 0.5, dtype=np.float32)

        combined = combine_direct_indirect(direct, indirect, albedo)
        # (1.0 + 1.0) * 0.5 = 1.0
        np.testing.assert_array_almost_equal(
            combined, np.full((10, 10, 3), 1.0)
        )


class TestDepthToWorldPosition:
    """Tests for depth_to_world_position function."""

    def test_basic_reconstruction(self):
        depth = np.full((10, 10), 5.0, dtype=np.float32)
        origin = np.array([0, 0, 0])
        forward = np.array([0, 0, -1])
        right = np.array([1, 0, 0])
        up = np.array([0, 1, 0])

        positions = depth_to_world_position(
            depth, origin, forward, right, up,
            fov=90.0, aspect_ratio=1.0
        )

        assert positions.shape == (10, 10, 3)
        # Center should be roughly at (0, 0, -5)
        center = positions[5, 5]
        assert center[2] < 0  # Negative Z (forward)


class TestRenderPassCompositor:
    """Tests for RenderPassCompositor class."""

    def test_creation(self):
        manager = AOVManager(width=10, height=10)
        compositor = RenderPassCompositor(manager)
        assert compositor.aov_manager is manager

    def test_get_beauty(self):
        manager = AOVManager(width=10, height=10)
        manager.enable_aov(AOVType.BEAUTY)

        sample = AOVSample()
        # Note: beauty is not directly set in sample, would come from render
        manager.add_sample(5, 5, sample)

        compositor = RenderPassCompositor(manager)
        beauty = compositor.get_beauty()
        assert beauty is not None
        assert beauty.shape == (10, 10, 3)

    def test_get_depth_visualization(self):
        manager = AOVManager(width=10, height=10)
        manager.enable_aov(AOVType.DEPTH)

        sample = AOVSample(depth=5.0)
        manager.add_sample(5, 5, sample)

        compositor = RenderPassCompositor(manager)
        depth_vis = compositor.get_depth_visualization()

        assert depth_vis is not None
        assert depth_vis.shape == (10, 10, 3)

    def test_get_normal_visualization(self):
        manager = AOVManager(width=10, height=10)
        manager.enable_aov(AOVType.NORMAL)

        sample = AOVSample(normal=Vec3(0, 1, 0))
        manager.add_sample(5, 5, sample)

        compositor = RenderPassCompositor(manager)
        normal_vis = compositor.get_normal_visualization()

        assert normal_vis is not None
        assert normal_vis.shape == (10, 10, 3)
        # Packed normal (0, 1, 0) -> (0.5, 1.0, 0.5)
        np.testing.assert_array_almost_equal(
            normal_vis[5, 5], [0.5, 1.0, 0.5]
        )

    def test_get_object_id_visualization(self):
        manager = AOVManager(width=10, height=10)
        manager.enable_aov(AOVType.OBJECT_ID)

        for i in range(3):
            sample = AOVSample(object_id=i + 1)
            manager.add_sample(i, 0, sample)

        compositor = RenderPassCompositor(manager)
        obj_vis = compositor.get_object_id_visualization()

        assert obj_vis is not None
        assert obj_vis.shape == (10, 10, 3)

    def test_reconstruct_beauty(self):
        manager = AOVManager(width=10, height=10)
        manager.enable_aovs([AOVType.DIRECT, AOVType.INDIRECT, AOVType.ALBEDO])

        # Add sample data
        for y in range(10):
            for x in range(10):
                sample = AOVSample(
                    direct=Color(0.5, 0.5, 0.5),
                    indirect=Color(0.3, 0.3, 0.3),
                    albedo=Color(1.0, 0.5, 0.0)  # Orange
                )
                manager.add_sample(x, y, sample)

        compositor = RenderPassCompositor(manager)
        reconstructed = compositor.reconstruct_beauty()

        assert reconstructed is not None
        assert reconstructed.shape == (10, 10, 3)

    def test_returns_none_if_not_available(self):
        manager = AOVManager(width=10, height=10)
        compositor = RenderPassCompositor(manager)

        assert compositor.get_beauty() is None
        assert compositor.get_depth_visualization() is None
        assert compositor.reconstruct_beauty() is None


class TestAOVIntegration:
    """Integration tests for AOV system."""

    def test_full_render_simulation(self):
        manager = AOVManager(width=20, height=20)
        manager.enable_aovs([
            AOVType.BEAUTY,
            AOVType.DEPTH,
            AOVType.NORMAL,
            AOVType.ALBEDO,
            AOVType.OBJECT_ID
        ])

        # Simulate rendering multiple samples per pixel
        for y in range(20):
            for x in range(20):
                # Multiple samples per pixel
                for _ in range(4):
                    sample = AOVSample(
                        depth=5.0 + np.random.rand() * 0.1,
                        normal=Vec3(0, 1, 0),
                        albedo=Color(0.8, 0.2, 0.1),
                        object_id=1 + (x // 10)  # Two objects
                    )
                    manager.add_sample(x, y, sample)

        # Get all AOVs
        all_aovs = manager.get_all_aovs()

        assert len(all_aovs) == 5
        for name, data in all_aovs.items():
            assert data.shape[0] == 20
            assert data.shape[1] == 20
            assert not np.any(np.isnan(data))
