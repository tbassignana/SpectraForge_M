"""
Scene description language parser.

Supports a YAML-based scene description format with:
- Camera configuration
- Render settings
- Materials library
- Objects (shapes with materials)
- Lights

Example scene file:
```yaml
camera:
  look_from: [13, 2, 3]
  look_at: [0, 0, 0]
  vfov: 20
  aperture: 0.1
  focus_dist: 10

render:
  width: 800
  height: 600
  samples: 100
  max_depth: 50

materials:
  ground:
    type: lambertian
    albedo: [0.5, 0.5, 0.5]

  glass:
    type: dielectric
    ior: 1.5

objects:
  - type: sphere
    center: [0, -1000, 0]
    radius: 1000
    material: ground

  - type: sphere
    center: [0, 1, 0]
    radius: 1
    material: glass

lights:
  - type: point
    position: [0, 10, 0]
    color: [1, 1, 1]
    intensity: 10
```
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import json

from .vec3 import Vec3, Point3, Color
from .camera import Camera
from .shapes import Sphere, Plane, Triangle, HittableList
from .materials import Material, Lambertian, Metal, Dielectric, Emissive, PBRMaterial
from .lights import Light, PointLight, DirectionalLight, AreaLight, SphereLight, LightList
from .renderer import RenderSettings
from .bvh import BVH


class SceneParseError(Exception):
    """Error during scene parsing."""
    pass


class SceneParser:
    """Parser for scene description files."""

    def __init__(self):
        self.materials: Dict[str, Material] = {}
        self.objects: HittableList = HittableList()
        self.lights: LightList = LightList()
        self.camera: Optional[Camera] = None
        self.settings: Optional[RenderSettings] = None

    def parse_file(self, filepath: str) -> Tuple[HittableList, Camera, RenderSettings, LightList]:
        """Parse a scene file.

        Args:
            filepath: Path to the scene file (YAML or JSON)

        Returns:
            Tuple of (scene, camera, settings, lights)
        """
        path = Path(filepath)
        if not path.exists():
            raise SceneParseError(f"Scene file not found: {filepath}")

        content = path.read_text()

        if path.suffix in ('.yaml', '.yml'):
            try:
                import yaml
                data = yaml.safe_load(content)
            except ImportError:
                raise SceneParseError("PyYAML not installed. Install with: pip install pyyaml")
        elif path.suffix == '.json':
            data = json.loads(content)
        else:
            # Try YAML first, then JSON
            try:
                import yaml
                data = yaml.safe_load(content)
            except:
                data = json.loads(content)

        return self.parse_dict(data)

    def parse_dict(self, data: Dict[str, Any]) -> Tuple[HittableList, Camera, RenderSettings, LightList]:
        """Parse a scene from a dictionary.

        Args:
            data: Scene description dictionary

        Returns:
            Tuple of (scene, camera, settings, lights)
        """
        # Parse materials first (objects reference them)
        if 'materials' in data:
            self._parse_materials(data['materials'])

        # Parse objects
        if 'objects' in data:
            self._parse_objects(data['objects'])

        # Parse lights
        if 'lights' in data:
            self._parse_lights(data['lights'])

        # Parse camera
        if 'camera' in data:
            self._parse_camera(data['camera'])
        else:
            # Default camera
            self.camera = Camera(
                look_from=Point3(0, 0, 5),
                look_at=Point3(0, 0, 0),
                vfov=60
            )

        # Parse render settings
        if 'render' in data:
            self._parse_settings(data['render'])
        else:
            self.settings = RenderSettings()

        return self.objects, self.camera, self.settings, self.lights

    def _parse_vec3(self, data: Any) -> Vec3:
        """Parse a Vec3 from various formats."""
        if isinstance(data, (list, tuple)):
            if len(data) != 3:
                raise SceneParseError(f"Vec3 must have 3 components, got {len(data)}")
            return Vec3(float(data[0]), float(data[1]), float(data[2]))
        elif isinstance(data, dict):
            return Vec3(
                float(data.get('x', 0)),
                float(data.get('y', 0)),
                float(data.get('z', 0))
            )
        else:
            raise SceneParseError(f"Cannot parse Vec3 from: {data}")

    def _parse_color(self, data: Any) -> Color:
        """Parse a Color from various formats."""
        if isinstance(data, (list, tuple)):
            if len(data) != 3:
                raise SceneParseError(f"Color must have 3 components, got {len(data)}")
            return Color(float(data[0]), float(data[1]), float(data[2]))
        elif isinstance(data, dict):
            return Color(
                float(data.get('r', 0)),
                float(data.get('g', 0)),
                float(data.get('b', 0))
            )
        elif isinstance(data, str):
            # Handle hex colors
            if data.startswith('#'):
                hex_color = data[1:]
                if len(hex_color) == 6:
                    r = int(hex_color[0:2], 16) / 255.0
                    g = int(hex_color[2:4], 16) / 255.0
                    b = int(hex_color[4:6], 16) / 255.0
                    return Color(r, g, b)
            raise SceneParseError(f"Cannot parse color from string: {data}")
        else:
            raise SceneParseError(f"Cannot parse Color from: {data}")

    def _parse_materials(self, materials_data: Dict[str, Any]) -> None:
        """Parse materials section."""
        for name, mat_data in materials_data.items():
            mat_type = mat_data.get('type', 'lambertian').lower()

            if mat_type == 'lambertian':
                albedo = self._parse_color(mat_data.get('albedo', [0.5, 0.5, 0.5]))
                self.materials[name] = Lambertian(albedo)

            elif mat_type == 'metal':
                albedo = self._parse_color(mat_data.get('albedo', [0.8, 0.8, 0.8]))
                roughness = float(mat_data.get('roughness', 0.0))
                self.materials[name] = Metal(albedo, roughness)

            elif mat_type == 'dielectric':
                ior = float(mat_data.get('ior', 1.5))
                tint = None
                if 'tint' in mat_data:
                    tint = self._parse_color(mat_data['tint'])
                self.materials[name] = Dielectric(ior, tint)

            elif mat_type == 'emissive':
                color = self._parse_color(mat_data.get('color', [1, 1, 1]))
                intensity = float(mat_data.get('intensity', 1.0))
                self.materials[name] = Emissive(color, intensity)

            elif mat_type == 'pbr':
                albedo = self._parse_color(mat_data.get('albedo', [0.8, 0.8, 0.8]))
                metallic = float(mat_data.get('metallic', 0.0))
                roughness = float(mat_data.get('roughness', 0.5))
                ior = float(mat_data.get('ior', 1.5))
                emission = None
                if 'emission' in mat_data:
                    emission = self._parse_color(mat_data['emission'])
                emission_strength = float(mat_data.get('emission_strength', 0.0))
                self.materials[name] = PBRMaterial(
                    albedo, metallic, roughness, ior, emission, emission_strength
                )

            else:
                raise SceneParseError(f"Unknown material type: {mat_type}")

    def _get_material(self, mat_ref: Any) -> Optional[Material]:
        """Get a material by name or inline definition."""
        if mat_ref is None:
            return None
        if isinstance(mat_ref, str):
            if mat_ref not in self.materials:
                raise SceneParseError(f"Unknown material: {mat_ref}")
            return self.materials[mat_ref]
        elif isinstance(mat_ref, dict):
            # Inline material definition
            temp_materials = {'_inline': mat_ref}
            self._parse_materials(temp_materials)
            return self.materials['_inline']
        else:
            raise SceneParseError(f"Invalid material reference: {mat_ref}")

    def _parse_objects(self, objects_data: list) -> None:
        """Parse objects section."""
        for obj_data in objects_data:
            obj_type = obj_data.get('type', 'sphere').lower()
            material = self._get_material(obj_data.get('material'))

            if obj_type == 'sphere':
                center = self._parse_vec3(obj_data.get('center', [0, 0, 0]))
                radius = float(obj_data.get('radius', 1.0))
                self.objects.add(Sphere(center, radius, material))

            elif obj_type == 'plane':
                point = self._parse_vec3(obj_data.get('point', [0, 0, 0]))
                normal = self._parse_vec3(obj_data.get('normal', [0, 1, 0]))
                self.objects.add(Plane(point, normal, material))

            elif obj_type == 'triangle':
                v0 = self._parse_vec3(obj_data['v0'])
                v1 = self._parse_vec3(obj_data['v1'])
                v2 = self._parse_vec3(obj_data['v2'])
                self.objects.add(Triangle(v0, v1, v2, material))

            else:
                raise SceneParseError(f"Unknown object type: {obj_type}")

    def _parse_lights(self, lights_data: list) -> None:
        """Parse lights section."""
        for light_data in lights_data:
            light_type = light_data.get('type', 'point').lower()

            if light_type == 'point':
                position = self._parse_vec3(light_data.get('position', [0, 5, 0]))
                color = self._parse_color(light_data.get('color', [1, 1, 1]))
                intensity = float(light_data.get('intensity', 1.0))
                self.lights.add(PointLight(position, color, intensity))

            elif light_type == 'directional':
                direction = self._parse_vec3(light_data.get('direction', [0, -1, 0]))
                color = self._parse_color(light_data.get('color', [1, 1, 1]))
                intensity = float(light_data.get('intensity', 1.0))
                self.lights.add(DirectionalLight(direction, color, intensity))

            elif light_type == 'area':
                corner = self._parse_vec3(light_data['corner'])
                edge1 = self._parse_vec3(light_data['edge1'])
                edge2 = self._parse_vec3(light_data['edge2'])
                color = self._parse_color(light_data.get('color', [1, 1, 1]))
                intensity = float(light_data.get('intensity', 1.0))
                area_light = AreaLight(corner, edge1, edge2, color, intensity)
                self.lights.add(area_light)
                # Also add as geometry
                self.objects.add(area_light)

            elif light_type == 'sphere':
                center = self._parse_vec3(light_data.get('center', [0, 5, 0]))
                radius = float(light_data.get('radius', 0.5))
                color = self._parse_color(light_data.get('color', [1, 1, 1]))
                intensity = float(light_data.get('intensity', 1.0))
                sphere_light = SphereLight(center, radius, color, intensity)
                self.lights.add(sphere_light)
                # Also add as geometry
                self.objects.add(sphere_light)

            else:
                raise SceneParseError(f"Unknown light type: {light_type}")

    def _parse_camera(self, camera_data: Dict[str, Any]) -> None:
        """Parse camera section."""
        look_from = self._parse_vec3(camera_data.get('look_from', [0, 0, 5]))
        look_at = self._parse_vec3(camera_data.get('look_at', [0, 0, 0]))
        vup = self._parse_vec3(camera_data.get('vup', [0, 1, 0]))
        vfov = float(camera_data.get('vfov', 60))
        aspect_ratio = float(camera_data.get('aspect_ratio', 16/9))
        aperture = float(camera_data.get('aperture', 0.0))
        focus_dist = float(camera_data.get('focus_dist', 1.0))

        self.camera = Camera(
            look_from=look_from,
            look_at=look_at,
            vup=vup,
            vfov=vfov,
            aspect_ratio=aspect_ratio,
            aperture=aperture,
            focus_dist=focus_dist
        )

    def _parse_settings(self, settings_data: Dict[str, Any]) -> None:
        """Parse render settings section."""
        self.settings = RenderSettings(
            width=int(settings_data.get('width', 800)),
            height=int(settings_data.get('height', 600)),
            samples_per_pixel=int(settings_data.get('samples', 100)),
            max_depth=int(settings_data.get('max_depth', 50)),
            tile_size=int(settings_data.get('tile_size', 32)),
            num_threads=int(settings_data.get('threads', 0)),
            gamma=float(settings_data.get('gamma', 2.2))
        )


def load_scene(filepath: str) -> Tuple[HittableList, Camera, RenderSettings, LightList]:
    """Convenience function to load a scene file.

    Args:
        filepath: Path to the scene file

    Returns:
        Tuple of (scene, camera, settings, lights)
    """
    parser = SceneParser()
    return parser.parse_file(filepath)


def parse_scene(data: Dict[str, Any]) -> Tuple[HittableList, Camera, RenderSettings, LightList]:
    """Convenience function to parse a scene from a dictionary.

    Args:
        data: Scene description dictionary

    Returns:
        Tuple of (scene, camera, settings, lights)
    """
    parser = SceneParser()
    return parser.parse_dict(data)
