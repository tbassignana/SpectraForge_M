"""
Renderer module - the heart of the ray tracer.

Implements:
- Path tracing with Russian roulette termination
- Progressive rendering
- Multi-threaded tile-based rendering
- HDR output support
"""

from __future__ import annotations
import math
import sys
import platform
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass
from typing import Optional, Callable, Tuple
import numpy as np

from .vec3 import Vec3, Color
from .ray import Ray
from .camera import Camera
from .shapes import Hittable, HitRecord


@dataclass
class RenderSettings:
    """Configuration for the renderer."""
    width: int = 800
    height: int = 600
    samples_per_pixel: int = 100
    max_depth: int = 50
    tile_size: int = 32
    num_threads: int = 0  # 0 = auto-detect
    background_color: Color = None
    use_sky_gradient: bool = True
    gamma: float = 2.2

    def __post_init__(self):
        if self.background_color is None:
            self.background_color = Color(0.0, 0.0, 0.0)
        if self.num_threads == 0:
            import os
            self.num_threads = os.cpu_count() or 4


class Renderer:
    """Path tracing renderer with multi-threading support."""

    def __init__(self, settings: RenderSettings = None):
        """Create a renderer with the given settings.

        Args:
            settings: Render configuration (uses defaults if None)
        """
        self.settings = settings if settings else RenderSettings()
        self._progress_callback: Optional[Callable[[float], None]] = None

    def set_progress_callback(self, callback: Callable[[float], None]) -> None:
        """Set a callback function for progress updates.

        Args:
            callback: Function that takes progress as float (0.0 to 1.0)
        """
        self._progress_callback = callback

    def render(self, scene: Hittable, camera: Camera) -> np.ndarray:
        """Render the scene and return the image as a numpy array.

        Args:
            scene: The scene to render (any Hittable)
            camera: The camera to render from

        Returns:
            HDR image as numpy array of shape (height, width, 3)
        """
        width = self.settings.width
        height = self.settings.height
        samples = self.settings.samples_per_pixel
        max_depth = self.settings.max_depth

        # Initialize output image (HDR, no clamping during accumulation)
        image = np.zeros((height, width, 3), dtype=np.float64)

        # Generate tiles for parallel processing
        tiles = self._generate_tiles(width, height)
        total_tiles = len(tiles)
        completed_tiles = [0]  # Use list for mutable in closure

        def render_tile(tile: Tuple[int, int, int, int]) -> Tuple[Tuple, np.ndarray]:
            """Render a single tile."""
            x0, y0, x1, y1 = tile
            tile_height = y1 - y0
            tile_width = x1 - x0
            tile_image = np.zeros((tile_height, tile_width, 3), dtype=np.float64)

            for j in range(tile_height):
                for i in range(tile_width):
                    pixel_color = Color(0, 0, 0)

                    for _ in range(samples):
                        u = (x0 + i + np.random.random()) / (width - 1)
                        v = (height - 1 - (y0 + j) + np.random.random()) / (height - 1)

                        ray = camera.get_ray(u, v)
                        pixel_color = pixel_color + self._ray_color(ray, scene, max_depth)

                    tile_image[j, i] = pixel_color._data / samples

            completed_tiles[0] += 1
            if self._progress_callback:
                self._progress_callback(completed_tiles[0] / total_tiles)

            return tile, tile_image

        # Render tiles in parallel
        if self.settings.num_threads > 1:
            # Use ProcessPoolExecutor for true parallelism (bypassing GIL)
            # But for simplicity in initial version, use ThreadPool
            with ThreadPoolExecutor(max_workers=self.settings.num_threads) as executor:
                results = list(executor.map(render_tile, tiles))
        else:
            results = [render_tile(tile) for tile in tiles]

        # Combine tiles into final image
        for tile, tile_image in results:
            x0, y0, x1, y1 = tile
            image[y0:y1, x0:x1] = tile_image

        return image

    def _ray_color(self, ray: Ray, scene: Hittable, depth: int) -> Color:
        """Compute the color for a ray using path tracing.

        Args:
            ray: The ray to trace
            scene: The scene to trace against
            depth: Maximum recursion depth

        Returns:
            The computed color for this ray
        """
        # Russian roulette termination
        if depth <= 0:
            return Color(0, 0, 0)

        # Find closest hit
        hit_record = scene.hit(ray, 0.001, float('inf'))

        if hit_record is None:
            # No hit - return background color
            if self.settings.use_sky_gradient:
                return self._sky_color(ray)
            return self.settings.background_color

        # Get emission from material
        emitted = Color(0, 0, 0)
        if hit_record.material:
            emitted = hit_record.material.emitted(
                hit_record.u, hit_record.v, hit_record.point
            )

        # Scatter ray
        if hit_record.material:
            scatter_result = hit_record.material.scatter(
                ray, hit_record.point, hit_record.normal, hit_record.front_face
            )
            if scatter_result:
                return emitted + scatter_result.attenuation * self._ray_color(
                    scatter_result.scattered_ray, scene, depth - 1
                )
            return emitted

        # No material - return normal as color (for debugging)
        return (hit_record.normal + Color(1, 1, 1)) * 0.5

    def _sky_color(self, ray: Ray) -> Color:
        """Generate a sky gradient background.

        Args:
            ray: The ray direction to use for gradient

        Returns:
            Sky color at this direction
        """
        unit_direction = ray.direction.normalize()
        t = 0.5 * (unit_direction.y + 1.0)
        return Color(1.0, 1.0, 1.0) * (1.0 - t) + Color(0.5, 0.7, 1.0) * t

    def _generate_tiles(self, width: int, height: int) -> list[Tuple[int, int, int, int]]:
        """Generate tiles for parallel rendering.

        Args:
            width: Image width
            height: Image height

        Returns:
            List of tiles as (x0, y0, x1, y1) tuples
        """
        tile_size = self.settings.tile_size
        tiles = []

        for y in range(0, height, tile_size):
            for x in range(0, width, tile_size):
                x1 = min(x + tile_size, width)
                y1 = min(y + tile_size, height)
                tiles.append((x, y, x1, y1))

        return tiles

    def to_ldr(self, hdr_image: np.ndarray) -> np.ndarray:
        """Convert HDR image to 8-bit LDR with gamma correction.

        Args:
            hdr_image: HDR image array (float64)

        Returns:
            LDR image as uint8 array
        """
        # Apply gamma correction
        gamma = self.settings.gamma
        corrected = np.power(np.clip(hdr_image, 0, None), 1.0 / gamma)

        # Clamp and convert to 8-bit
        ldr = np.clip(corrected * 255, 0, 255).astype(np.uint8)
        return ldr

    def save_image(self, image: np.ndarray, filename: str) -> None:
        """Save image to file.

        Args:
            image: Image array (HDR or LDR)
            filename: Output filename (extension determines format)
        """
        from PIL import Image as PILImage

        if filename.endswith('.exr') or filename.endswith('.hdr'):
            self._save_hdr(image, filename)
        else:
            # Convert to LDR if needed
            if image.dtype == np.float64 or image.dtype == np.float32:
                image = self.to_ldr(image)

            pil_image = PILImage.fromarray(image, 'RGB')
            pil_image.save(filename)

    def _save_hdr(self, image: np.ndarray, filename: str) -> None:
        """Save image in HDR format.

        Args:
            image: HDR image array
            filename: Output filename
        """
        if filename.endswith('.exr'):
            try:
                import OpenEXR
                import Imath
                self._save_exr(image, filename)
            except ImportError:
                print("Warning: OpenEXR not available, saving as .hdr instead")
                filename = filename.replace('.exr', '.hdr')
                self._save_radiance_hdr(image, filename)
        else:
            self._save_radiance_hdr(image, filename)

    def _save_radiance_hdr(self, image: np.ndarray, filename: str) -> None:
        """Save image in Radiance HDR format."""
        import struct

        height, width = image.shape[:2]

        with open(filename, 'wb') as f:
            # Write header
            f.write(b'#?RADIANCE\n')
            f.write(b'FORMAT=32-bit_rle_rgbe\n')
            f.write(b'\n')
            f.write(f'-Y {height} +X {width}\n'.encode())

            # Convert to RGBE and write
            for y in range(height):
                scanline = []
                for x in range(width):
                    r, g, b = image[y, x]
                    rgbe = self._float_to_rgbe(r, g, b)
                    scanline.extend(rgbe)
                f.write(bytes(scanline))

    @staticmethod
    def _float_to_rgbe(r: float, g: float, b: float) -> tuple[int, int, int, int]:
        """Convert RGB float to RGBE format."""
        v = max(r, g, b)
        if v < 1e-32:
            return (0, 0, 0, 0)

        m, e = math.frexp(v)
        v = m * 256.0 / v

        return (
            int(r * v),
            int(g * v),
            int(b * v),
            int(e + 128)
        )

    def _save_exr(self, image: np.ndarray, filename: str) -> None:
        """Save image in OpenEXR format."""
        import OpenEXR
        import Imath

        height, width = image.shape[:2]

        header = OpenEXR.Header(width, height)
        header['channels'] = {
            'R': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
            'G': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
            'B': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
        }

        out = OpenEXR.OutputFile(filename, header)
        r = image[:, :, 0].astype(np.float32).tobytes()
        g = image[:, :, 1].astype(np.float32).tobytes()
        b = image[:, :, 2].astype(np.float32).tobytes()
        out.writePixels({'R': r, 'G': g, 'B': b})
        out.close()


def get_platform_info() -> dict:
    """Get information about the current platform for optimization decisions.

    Returns:
        Dictionary with platform details
    """
    import os

    info = {
        'system': platform.system(),
        'machine': platform.machine(),
        'processor': platform.processor(),
        'python_version': platform.python_version(),
        'cpu_count': os.cpu_count(),
        'is_arm': platform.machine().lower() in ('arm64', 'aarch64'),
        'is_x86': platform.machine().lower() in ('x86_64', 'amd64', 'x86'),
    }

    # Check for Apple Silicon
    if info['system'] == 'Darwin' and info['is_arm']:
        info['is_apple_silicon'] = True
    else:
        info['is_apple_silicon'] = False

    return info
