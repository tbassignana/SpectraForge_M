"""Tests for OBJ file loader."""

import pytest
import tempfile
import os
from spectraforge.vec3 import Vec3, Point3, Color
from spectraforge.ray import Ray
from spectraforge.obj_loader import (
    OBJLoader, SmoothTriangle, load_obj, get_mesh_bounds, get_mesh_stats
)
from spectraforge.shapes import Triangle
from spectraforge.materials import Lambertian


class TestSmoothTriangle:
    """Test SmoothTriangle with interpolated normals."""

    def test_creation(self):
        v0, v1, v2 = Point3(0, 0, 0), Point3(1, 0, 0), Point3(0, 1, 0)
        n0, n1, n2 = Vec3(0, 0, 1), Vec3(0, 0, 1), Vec3(0, 0, 1)
        tri = SmoothTriangle(v0, v1, v2, n0, n1, n2)

        assert tri.v0 == v0
        assert tri.n0 == n0

    def test_hit_returns_interpolated_normal(self):
        # Triangle in XY plane
        v0, v1, v2 = Point3(0, 0, 0), Point3(1, 0, 0), Point3(0, 1, 0)
        # Normals tilted differently at each vertex
        n0 = Vec3(0, 0, 1)
        n1 = Vec3(0.5, 0, 0.866).normalize()  # Tilted toward +x
        n2 = Vec3(0, 0.5, 0.866).normalize()  # Tilted toward +y

        tri = SmoothTriangle(v0, v1, v2, n0, n1, n2)

        # Hit near v0 - normal should be close to n0
        ray = Ray(Point3(0.1, 0.1, -1), Vec3(0, 0, 1))
        hit = tri.hit(ray, 0.001, float('inf'))

        assert hit is not None
        # Normal should be interpolated and facing against ray (z < 0 after flip)
        # The magnitude should still be close to 1 (normalized)
        assert abs(hit.normal.z) > 0.8  # Mostly pointing along z

    def test_uv_interpolation(self):
        v0, v1, v2 = Point3(0, 0, 0), Point3(1, 0, 0), Point3(0, 1, 0)
        n = Vec3(0, 0, 1)
        uv0, uv1, uv2 = (0, 0), (1, 0), (0, 1)

        tri = SmoothTriangle(v0, v1, v2, n, n, n, uv0, uv1, uv2)

        # Hit at center of triangle
        ray = Ray(Point3(0.33, 0.33, -1), Vec3(0, 0, 1))
        hit = tri.hit(ray, 0.001, float('inf'))

        assert hit is not None
        # UV should be near center of UV space
        assert 0.2 < hit.u < 0.5
        assert 0.2 < hit.v < 0.5

    def test_bounding_box(self):
        v0, v1, v2 = Point3(0, 0, 0), Point3(1, 0, 0), Point3(0, 1, 0)
        tri = SmoothTriangle(v0, v1, v2)

        bbox = tri.bounding_box()
        assert bbox is not None


class TestOBJLoader:
    """Test OBJ file loading."""

    @pytest.fixture
    def simple_cube_obj(self):
        """Create a simple cube OBJ file."""
        content = """# Simple cube
v -1 -1 -1
v  1 -1 -1
v  1  1 -1
v -1  1 -1
v -1 -1  1
v  1 -1  1
v  1  1  1
v -1  1  1

vn 0 0 -1
vn 0 0  1
vn 0 -1 0
vn 0  1 0
vn -1 0 0
vn  1 0 0

# Front face
f 1//1 2//1 3//1
f 1//1 3//1 4//1
# Back face
f 5//2 7//2 6//2
f 5//2 8//2 7//2
# Bottom face
f 1//3 6//3 2//3
f 1//3 5//3 6//3
# Top face
f 4//4 3//4 7//4
f 4//4 7//4 8//4
# Left face
f 1//5 4//5 8//5
f 1//5 8//5 5//5
# Right face
f 2//6 6//6 7//6
f 2//6 7//6 3//6
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.obj', delete=False) as f:
            f.write(content)
            f.flush()
            yield f.name
        os.unlink(f.name)

    @pytest.fixture
    def simple_triangle_obj(self):
        """Create a minimal triangle OBJ file."""
        content = """# Simple triangle
v 0 0 0
v 1 0 0
v 0 1 0
f 1 2 3
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.obj', delete=False) as f:
            f.write(content)
            f.flush()
            yield f.name
        os.unlink(f.name)

    @pytest.fixture
    def quad_obj(self):
        """Create a quad that needs triangulation."""
        content = """# Quad (will be triangulated)
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
f 1 2 3 4
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.obj', delete=False) as f:
            f.write(content)
            f.flush()
            yield f.name
        os.unlink(f.name)

    @pytest.fixture
    def obj_with_uvs(self):
        """Create OBJ with texture coordinates."""
        content = """# Triangle with UVs
v 0 0 0
v 1 0 0
v 0 1 0
vt 0 0
vt 1 0
vt 0 1
f 1/1 2/2 3/3
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.obj', delete=False) as f:
            f.write(content)
            f.flush()
            yield f.name
        os.unlink(f.name)

    def test_load_simple_triangle(self, simple_triangle_obj):
        mesh = load_obj(simple_triangle_obj)

        assert len(mesh) == 1

    def test_load_cube(self, simple_cube_obj):
        mesh = load_obj(simple_cube_obj)

        # Cube has 6 faces, each made of 2 triangles
        assert len(mesh) == 12

    def test_triangulate_quad(self, quad_obj):
        mesh = load_obj(quad_obj)

        # Quad should be split into 2 triangles
        assert len(mesh) == 2

    def test_hit_loaded_mesh(self, simple_triangle_obj):
        mesh = load_obj(simple_triangle_obj)

        ray = Ray(Point3(0.25, 0.25, -1), Vec3(0, 0, 1))
        hit = mesh.hit(ray, 0.001, float('inf'))

        assert hit is not None
        assert abs(hit.point.z) < 1e-6

    def test_miss_loaded_mesh(self, simple_triangle_obj):
        mesh = load_obj(simple_triangle_obj)

        ray = Ray(Point3(5, 5, -1), Vec3(0, 0, 1))
        hit = mesh.hit(ray, 0.001, float('inf'))

        assert hit is None

    def test_scale_parameter(self, simple_triangle_obj):
        mesh = load_obj(simple_triangle_obj, scale=2.0)

        # Check that mesh is scaled
        bbox = mesh.bounding_box()
        assert bbox is not None
        # Original triangle was at (0,0,0)-(1,1,0), scaled should be 0-2
        assert bbox.maximum.x > 1.5

    def test_center_parameter(self, simple_triangle_obj):
        mesh = load_obj(simple_triangle_obj, center=True)

        # Check that mesh is centered
        bbox = mesh.bounding_box()
        assert bbox is not None
        # Center should be at origin
        center_x = (bbox.minimum.x + bbox.maximum.x) / 2
        center_y = (bbox.minimum.y + bbox.maximum.y) / 2
        assert abs(center_x) < 0.01
        assert abs(center_y) < 0.01

    def test_custom_material(self, simple_triangle_obj):
        material = Lambertian(Color(1, 0, 0))
        mesh = load_obj(simple_triangle_obj, material=material)

        ray = Ray(Point3(0.25, 0.25, -1), Vec3(0, 0, 1))
        hit = mesh.hit(ray, 0.001, float('inf'))

        assert hit is not None
        assert hit.material is material

    def test_smooth_shading(self, simple_cube_obj):
        mesh = load_obj(simple_cube_obj, smooth_shading=True)

        # With normals in file, should create SmoothTriangles
        for obj in mesh:
            assert isinstance(obj, SmoothTriangle)

    def test_flat_shading(self, simple_cube_obj):
        mesh = load_obj(simple_cube_obj, smooth_shading=False)

        # Should create regular Triangles
        for obj in mesh:
            assert isinstance(obj, Triangle)

    def test_bounding_box(self, simple_cube_obj):
        mesh = load_obj(simple_cube_obj)

        bbox = mesh.bounding_box()
        assert bbox is not None
        # Cube is from -1 to 1 on all axes
        assert abs(bbox.minimum.x - (-1)) < 0.01
        assert abs(bbox.maximum.x - 1) < 0.01

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_obj("/nonexistent/path/model.obj")

    def test_uvs_loaded(self, obj_with_uvs):
        mesh = load_obj(obj_with_uvs)

        ray = Ray(Point3(0.25, 0.25, -1), Vec3(0, 0, 1))
        hit = mesh.hit(ray, 0.001, float('inf'))

        assert hit is not None
        # Should have interpolated UV coordinates
        assert 0 <= hit.u <= 1
        assert 0 <= hit.v <= 1

    def test_negative_indices(self):
        """Test that negative indices work (count from end)."""
        content = """v 0 0 0
v 1 0 0
v 0 1 0
f -3 -2 -1
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.obj', delete=False) as f:
            f.write(content)
            f.flush()
            mesh = load_obj(f.name)
            os.unlink(f.name)

        assert len(mesh) == 1


class TestMeshUtilities:
    """Test mesh utility functions."""

    @pytest.fixture
    def temp_obj(self):
        content = """v 0 0 0
v 2 0 0
v 0 2 0
f 1 2 3
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.obj', delete=False) as f:
            f.write(content)
            f.flush()
            yield f.name
        os.unlink(f.name)

    def test_get_mesh_bounds(self, temp_obj):
        mesh = load_obj(temp_obj)
        min_pt, max_pt = get_mesh_bounds(mesh)

        assert abs(min_pt.x) < 0.01
        assert abs(max_pt.x - 2) < 0.01

    def test_get_mesh_stats(self, temp_obj):
        mesh = load_obj(temp_obj)
        stats = get_mesh_stats(mesh)

        assert stats['triangle_count'] == 1
        assert stats['size'][0] > 1.9  # Width ~2


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_file(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.obj', delete=False) as f:
            f.write("# Empty file\n")
            f.flush()
            mesh = load_obj(f.name)
            os.unlink(f.name)

        assert len(mesh) == 0

    def test_comments_ignored(self):
        content = """# This is a comment
v 0 0 0
# Another comment
v 1 0 0
v 0 1 0
f 1 2 3
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.obj', delete=False) as f:
            f.write(content)
            f.flush()
            mesh = load_obj(f.name)
            os.unlink(f.name)

        assert len(mesh) == 1

    def test_malformed_lines_skipped(self):
        content = """v 0 0 0
v 1 0 0
v 0 1 0
v bad line
f 1 2 3
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.obj', delete=False) as f:
            f.write(content)
            f.flush()
            mesh = load_obj(f.name)
            os.unlink(f.name)

        # Should still load the valid triangle
        assert len(mesh) == 1
