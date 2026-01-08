"""
OBJ file loader for importing 3D meshes.

Supports:
- Vertices (v)
- Texture coordinates (vt)
- Normals (vn)
- Faces (f) with triangulation
- Material references (usemtl)
- Object groups (o, g)
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass

from .vec3 import Vec3, Point3
from .shapes import Triangle, HittableList
from .materials import Material, Lambertian, Color


@dataclass
class OBJVertex:
    """A vertex with optional texture coords and normal indices."""
    position_idx: int
    texcoord_idx: Optional[int] = None
    normal_idx: Optional[int] = None


class SmoothTriangle(Triangle):
    """A triangle with per-vertex normals for smooth shading."""

    def __init__(
        self,
        v0: Point3, v1: Point3, v2: Point3,
        n0: Vec3 = None, n1: Vec3 = None, n2: Vec3 = None,
        uv0: Tuple[float, float] = None,
        uv1: Tuple[float, float] = None,
        uv2: Tuple[float, float] = None,
        material: Optional[Material] = None
    ):
        """Create a smooth-shaded triangle.

        Args:
            v0, v1, v2: Vertex positions
            n0, n1, n2: Per-vertex normals (optional, uses face normal if None)
            uv0, uv1, uv2: Per-vertex UV coordinates (optional)
            material: Material for shading
        """
        super().__init__(v0, v1, v2, material)

        # Store vertex normals (use face normal if not provided)
        self.n0 = n0.normalize() if n0 else self.normal
        self.n1 = n1.normalize() if n1 else self.normal
        self.n2 = n2.normalize() if n2 else self.normal

        # Store UV coordinates
        self.uv0 = uv0 if uv0 else (0.0, 0.0)
        self.uv1 = uv1 if uv1 else (1.0, 0.0)
        self.uv2 = uv2 if uv2 else (0.0, 1.0)

    def hit(self, ray, t_min: float, t_max: float):
        """Hit test with interpolated normals and UVs."""
        from .ray import Ray

        h = ray.direction.cross(self.e2)
        a = self.e1.dot(h)

        if abs(a) < 1e-8:
            return None

        f = 1.0 / a
        s = ray.origin - self.v0
        u_bary = f * s.dot(h)

        if u_bary < 0.0 or u_bary > 1.0:
            return None

        q = s.cross(self.e1)
        v_bary = f * ray.direction.dot(q)

        if v_bary < 0.0 or u_bary + v_bary > 1.0:
            return None

        t = f * self.e2.dot(q)

        if t < t_min or t > t_max:
            return None

        point = ray.at(t)

        # Interpolate normal using barycentric coordinates
        w = 1.0 - u_bary - v_bary
        interpolated_normal = (
            self.n0 * w + self.n1 * u_bary + self.n2 * v_bary
        ).normalize()

        # Interpolate UV coordinates
        u = self.uv0[0] * w + self.uv1[0] * u_bary + self.uv2[0] * v_bary
        v = self.uv0[1] * w + self.uv1[1] * u_bary + self.uv2[1] * v_bary

        from .shapes import HitRecord
        hit_record = HitRecord(
            point=point,
            normal=interpolated_normal,
            t=t,
            front_face=True,
            material=self.material,
            u=u,
            v=v
        )
        hit_record.set_face_normal(ray, interpolated_normal)

        return hit_record


class OBJLoader:
    """Loader for Wavefront OBJ files."""

    def __init__(self):
        self.vertices: List[Point3] = []
        self.texcoords: List[Tuple[float, float]] = []
        self.normals: List[Vec3] = []
        self.current_material: Optional[Material] = None
        self.default_material = Lambertian(Color(0.7, 0.7, 0.7))

    def load(
        self,
        filename: str,
        material: Material = None,
        scale: float = 1.0,
        center: bool = False,
        smooth_shading: bool = True
    ) -> HittableList:
        """Load an OBJ file and return a list of triangles.

        Args:
            filename: Path to the OBJ file
            material: Default material (overrides file materials if set)
            scale: Scale factor to apply to the mesh
            center: If True, center the mesh at origin
            smooth_shading: If True, use per-vertex normals for smooth shading

        Returns:
            HittableList containing all triangles
        """
        path = Path(filename)
        if not path.exists():
            raise FileNotFoundError(f"OBJ file not found: {filename}")

        self.vertices = []
        self.texcoords = []
        self.normals = []
        self.current_material = material if material else self.default_material

        triangles = HittableList()

        with open(filename, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()

                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue

                parts = line.split()
                if not parts:
                    continue

                cmd = parts[0]

                try:
                    if cmd == 'v':
                        # Vertex position
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                        self.vertices.append(Point3(x * scale, y * scale, z * scale))

                    elif cmd == 'vt':
                        # Texture coordinate
                        u = float(parts[1])
                        v = float(parts[2]) if len(parts) > 2 else 0.0
                        self.texcoords.append((u, v))

                    elif cmd == 'vn':
                        # Vertex normal
                        nx, ny, nz = float(parts[1]), float(parts[2]), float(parts[3])
                        self.normals.append(Vec3(nx, ny, nz).normalize())

                    elif cmd == 'f':
                        # Face (may be polygon, triangulate)
                        face_verts = self._parse_face(parts[1:])
                        tris = self._triangulate_face(face_verts, smooth_shading)
                        for tri in tris:
                            triangles.add(tri)

                    elif cmd == 'usemtl':
                        # Material reference (not loading MTL files for now)
                        pass

                    elif cmd in ('o', 'g'):
                        # Object/group name (ignored for now)
                        pass

                    elif cmd == 'mtllib':
                        # Material library (not loading for now)
                        pass

                except (ValueError, IndexError) as e:
                    # Skip malformed lines with a warning
                    print(f"Warning: Skipping malformed line {line_num}: {line}")
                    continue

        # Center the mesh if requested
        if center and len(triangles) > 0:
            triangles = self._center_mesh(triangles)

        return triangles

    def _parse_face(self, face_parts: List[str]) -> List[OBJVertex]:
        """Parse face vertex indices (handles v, v/vt, v/vt/vn, v//vn formats)."""
        vertices = []

        for part in face_parts:
            indices = part.split('/')

            # Position index (required, 1-indexed)
            pos_idx = int(indices[0])
            if pos_idx < 0:
                pos_idx = len(self.vertices) + pos_idx + 1
            pos_idx -= 1  # Convert to 0-indexed

            # Texture coordinate index (optional)
            tex_idx = None
            if len(indices) > 1 and indices[1]:
                tex_idx = int(indices[1])
                if tex_idx < 0:
                    tex_idx = len(self.texcoords) + tex_idx + 1
                tex_idx -= 1

            # Normal index (optional)
            norm_idx = None
            if len(indices) > 2 and indices[2]:
                norm_idx = int(indices[2])
                if norm_idx < 0:
                    norm_idx = len(self.normals) + norm_idx + 1
                norm_idx -= 1

            vertices.append(OBJVertex(pos_idx, tex_idx, norm_idx))

        return vertices

    def _triangulate_face(self, face_verts: List[OBJVertex], smooth: bool) -> List[Triangle]:
        """Triangulate a face (fan triangulation for convex polygons)."""
        triangles = []

        if len(face_verts) < 3:
            return triangles

        # Fan triangulation: v0, v1, v2 then v0, v2, v3 etc.
        v0 = face_verts[0]

        for i in range(1, len(face_verts) - 1):
            v1 = face_verts[i]
            v2 = face_verts[i + 1]

            # Get positions
            p0 = self.vertices[v0.position_idx]
            p1 = self.vertices[v1.position_idx]
            p2 = self.vertices[v2.position_idx]

            if smooth and self.normals:
                # Get normals (if available)
                n0 = self.normals[v0.normal_idx] if v0.normal_idx is not None else None
                n1 = self.normals[v1.normal_idx] if v1.normal_idx is not None else None
                n2 = self.normals[v2.normal_idx] if v2.normal_idx is not None else None

                # Get UVs (if available)
                uv0 = self.texcoords[v0.texcoord_idx] if v0.texcoord_idx is not None else None
                uv1 = self.texcoords[v1.texcoord_idx] if v1.texcoord_idx is not None else None
                uv2 = self.texcoords[v2.texcoord_idx] if v2.texcoord_idx is not None else None

                tri = SmoothTriangle(
                    p0, p1, p2,
                    n0, n1, n2,
                    uv0, uv1, uv2,
                    self.current_material
                )
            else:
                tri = Triangle(p0, p1, p2, self.current_material)

                # Set UV coordinates on basic triangle too
                if v0.texcoord_idx is not None:
                    pass  # Basic Triangle doesn't store per-vertex UVs

            triangles.append(tri)

        return triangles

    def _center_mesh(self, triangles: HittableList) -> HittableList:
        """Center the mesh at the origin."""
        # Calculate bounding box
        bbox = triangles.bounding_box()
        if bbox is None:
            return triangles

        center = (bbox.minimum + bbox.maximum) * 0.5

        # Create new triangles offset by the center
        centered = HittableList()
        for obj in triangles:
            if isinstance(obj, SmoothTriangle):
                centered.add(SmoothTriangle(
                    obj.v0 - center, obj.v1 - center, obj.v2 - center,
                    obj.n0, obj.n1, obj.n2,
                    obj.uv0, obj.uv1, obj.uv2,
                    obj.material
                ))
            elif isinstance(obj, Triangle):
                centered.add(Triangle(
                    obj.v0 - center, obj.v1 - center, obj.v2 - center,
                    obj.material
                ))

        return centered


def load_obj(
    filename: str,
    material: Material = None,
    scale: float = 1.0,
    center: bool = False,
    smooth_shading: bool = True
) -> HittableList:
    """Convenience function to load an OBJ file.

    Args:
        filename: Path to the OBJ file
        material: Material to apply (uses default gray if None)
        scale: Scale factor for the mesh
        center: Center the mesh at origin
        smooth_shading: Use interpolated normals for smooth shading

    Returns:
        HittableList containing triangles
    """
    loader = OBJLoader()
    return loader.load(filename, material, scale, center, smooth_shading)


def get_mesh_bounds(mesh: HittableList) -> Tuple[Point3, Point3]:
    """Get the bounding box corners of a mesh.

    Returns:
        Tuple of (min_point, max_point)
    """
    bbox = mesh.bounding_box()
    if bbox is None:
        return Point3(0, 0, 0), Point3(0, 0, 0)
    return bbox.minimum, bbox.maximum


def get_mesh_stats(mesh: HittableList) -> Dict[str, any]:
    """Get statistics about a loaded mesh.

    Returns:
        Dictionary with mesh statistics
    """
    bbox = mesh.bounding_box()
    min_pt, max_pt = (bbox.minimum, bbox.maximum) if bbox else (Point3(0, 0, 0), Point3(0, 0, 0))
    size = max_pt - min_pt

    return {
        'triangle_count': len(mesh),
        'bounds_min': (min_pt.x, min_pt.y, min_pt.z),
        'bounds_max': (max_pt.x, max_pt.y, max_pt.z),
        'size': (size.x, size.y, size.z),
    }
