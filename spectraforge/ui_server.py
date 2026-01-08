"""
SpectraForge Web UI Server.

A platform-agnostic web-based UI for the ray tracer using only Python stdlib.
No external dependencies required - works on any platform with Python 3.8+.

Usage:
    python -m spectraforge.ui
    # Opens browser to http://localhost:8080
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import threading
import time
import webbrowser
from dataclasses import dataclass, asdict
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from urllib.parse import parse_qs, urlparse

# Import spectraforge components
from .vec3 import Vec3, Point3, Color
from .camera import Camera
from .shapes import Sphere, HittableList
from .materials import Lambertian, Metal, Dielectric, Emissive, PBRMaterial
from .renderer import Renderer, RenderSettings, get_platform_info
from .bvh import build_bvh

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


@dataclass
class RenderJob:
    """Represents an active render job."""
    id: str
    status: str  # 'pending', 'running', 'completed', 'cancelled'
    progress: float
    start_time: float
    end_time: Optional[float]
    settings: Dict[str, Any]
    image_data: Optional[str]  # Base64 encoded PNG
    error: Optional[str]


class RenderManager:
    """Manages render jobs and state."""

    def __init__(self):
        self.current_job: Optional[RenderJob] = None
        self.render_thread: Optional[threading.Thread] = None
        self.cancel_flag = threading.Event()
        self.lock = threading.Lock()
        self.job_counter = 0

    def create_scene(self, scene_type: str, custom_objects: Optional[list] = None) -> HittableList:
        """Create a scene based on type."""
        world = HittableList()

        if scene_type == 'demo':
            # Ground
            ground = Lambertian(Color(0.5, 0.5, 0.5))
            world.add(Sphere(Point3(0, -1000, 0), 1000, ground))

            # Glass sphere
            glass = Dielectric(1.5)
            world.add(Sphere(Point3(0, 1, 0), 1.0, glass))

            # Diffuse sphere
            diffuse = Lambertian(Color(0.4, 0.2, 0.1))
            world.add(Sphere(Point3(-4, 1, 0), 1.0, diffuse))

            # Metal sphere
            metal = Metal(Color(0.7, 0.6, 0.5), 0.0)
            world.add(Sphere(Point3(4, 1, 0), 1.0, metal))

            # PBR gold
            gold = PBRMaterial(Color(1.0, 0.766, 0.336), metallic=1.0, roughness=0.3)
            world.add(Sphere(Point3(1.5, 0.5, 2), 0.5, gold))

            # Light
            light = Emissive(Color(1, 1, 1), 5.0)
            world.add(Sphere(Point3(0, 5, 0), 1.0, light))

        elif scene_type == 'cornell':
            # Cornell box using spheres
            red = Lambertian(Color(0.65, 0.05, 0.05))
            white = Lambertian(Color(0.73, 0.73, 0.73))
            green = Lambertian(Color(0.12, 0.45, 0.15))
            light = Emissive(Color(1, 1, 1), 15.0)

            # Walls
            world.add(Sphere(Point3(-1005, 0, 0), 1000, red))
            world.add(Sphere(Point3(1005, 0, 0), 1000, green))
            world.add(Sphere(Point3(0, 0, -1005), 1000, white))
            world.add(Sphere(Point3(0, -1000, 0), 1000, white))
            world.add(Sphere(Point3(0, 1010, 0), 1000, white))

            # Light
            world.add(Sphere(Point3(0, 9.9, 0), 1.5, light))

            # Objects
            glass = Dielectric(1.5)
            world.add(Sphere(Point3(-2, 1.5, -2), 1.5, glass))
            metal = Metal(Color(0.8, 0.8, 0.8), 0.1)
            world.add(Sphere(Point3(2, 1, 1), 1.0, metal))

        elif scene_type == 'custom' and custom_objects:
            # Build scene from custom objects
            for obj in custom_objects:
                material = self._create_material(obj.get('material', {}))
                if obj.get('type') == 'sphere':
                    center = Point3(*obj.get('center', [0, 0, 0]))
                    radius = obj.get('radius', 1.0)
                    world.add(Sphere(center, radius, material))

        else:
            # Minimal scene
            ground = Lambertian(Color(0.5, 0.5, 0.5))
            world.add(Sphere(Point3(0, -1000, 0), 1000, ground))
            sphere = Lambertian(Color(0.8, 0.3, 0.3))
            world.add(Sphere(Point3(0, 1, 0), 1.0, sphere))

        return world

    def _create_material(self, mat_config: Dict[str, Any]):
        """Create a material from config."""
        mat_type = mat_config.get('type', 'lambertian')

        if mat_type == 'lambertian':
            color = Color(*mat_config.get('color', [0.5, 0.5, 0.5]))
            return Lambertian(color)
        elif mat_type == 'metal':
            color = Color(*mat_config.get('color', [0.8, 0.8, 0.8]))
            roughness = mat_config.get('roughness', 0.0)
            return Metal(color, roughness)
        elif mat_type == 'dielectric':
            ior = mat_config.get('ior', 1.5)
            return Dielectric(ior)
        elif mat_type == 'emissive':
            color = Color(*mat_config.get('color', [1, 1, 1]))
            intensity = mat_config.get('intensity', 5.0)
            return Emissive(color, intensity)
        elif mat_type == 'pbr':
            color = Color(*mat_config.get('color', [0.5, 0.5, 0.5]))
            metallic = mat_config.get('metallic', 0.0)
            roughness = mat_config.get('roughness', 0.5)
            return PBRMaterial(color, metallic=metallic, roughness=roughness)
        else:
            return Lambertian(Color(0.5, 0.5, 0.5))

    def start_render(self, config: Dict[str, Any]) -> RenderJob:
        """Start a new render job."""
        with self.lock:
            # Cancel any existing job
            if self.current_job and self.current_job.status == 'running':
                self.cancel_flag.set()
                if self.render_thread:
                    self.render_thread.join(timeout=2.0)

            self.cancel_flag.clear()
            self.job_counter += 1

            job = RenderJob(
                id=f"job_{self.job_counter}",
                status='pending',
                progress=0.0,
                start_time=time.time(),
                end_time=None,
                settings=config,
                image_data=None,
                error=None
            )
            self.current_job = job

            # Start render in background thread
            self.render_thread = threading.Thread(
                target=self._render_worker,
                args=(job, config),
                daemon=True
            )
            self.render_thread.start()

            return job

    def _render_worker(self, job: RenderJob, config: Dict[str, Any]):
        """Worker thread for rendering."""
        try:
            job.status = 'running'

            # Extract settings
            width = config.get('width', 400)
            height = config.get('height', 300)
            samples = config.get('samples', 50)
            max_depth = config.get('max_depth', 25)
            threads = config.get('threads', 0)
            scene_type = config.get('scene', 'demo')

            # Camera settings
            cam_config = config.get('camera', {})
            look_from = Point3(*cam_config.get('look_from', [13, 2, 3]))
            look_at = Point3(*cam_config.get('look_at', [0, 0, 0]))
            vfov = cam_config.get('vfov', 20)
            aperture = cam_config.get('aperture', 0.1)
            focus_dist = cam_config.get('focus_dist', 10.0)

            # Create scene
            world = self.create_scene(scene_type, config.get('objects'))

            # Use BVH for acceleration
            scene = build_bvh(world)

            # Create camera
            camera = Camera(
                look_from=look_from,
                look_at=look_at,
                vup=Vec3(0, 1, 0),
                vfov=vfov,
                aspect_ratio=width / height,
                aperture=aperture,
                focus_dist=focus_dist
            )

            # Create renderer
            settings = RenderSettings(
                width=width,
                height=height,
                samples_per_pixel=samples,
                max_depth=max_depth,
                num_threads=threads
            )
            renderer = Renderer(settings)

            # Progress callback
            def progress_callback(progress: float):
                if self.cancel_flag.is_set():
                    raise InterruptedError("Render cancelled")
                job.progress = progress

            renderer.set_progress_callback(progress_callback)

            # Render
            image = renderer.render(scene, camera)

            if self.cancel_flag.is_set():
                job.status = 'cancelled'
                return

            # Convert to base64 PNG
            if HAS_PIL:
                # Convert HDR to LDR
                ldr_image = renderer.to_ldr(image)

                # Create PIL image
                img_array = (ldr_image * 255).astype('uint8')
                pil_image = Image.fromarray(img_array)

                # Encode as base64 PNG
                buffer = io.BytesIO()
                pil_image.save(buffer, format='PNG')
                buffer.seek(0)
                job.image_data = base64.b64encode(buffer.read()).decode('utf-8')
            else:
                job.error = "PIL not available for image encoding"

            job.status = 'completed'
            job.progress = 1.0
            job.end_time = time.time()

        except InterruptedError:
            job.status = 'cancelled'
        except Exception as e:
            job.status = 'error'
            job.error = str(e)
            job.end_time = time.time()

    def cancel_render(self) -> bool:
        """Cancel the current render job."""
        with self.lock:
            if self.current_job and self.current_job.status == 'running':
                self.cancel_flag.set()
                return True
            return False

    def get_status(self) -> Dict[str, Any]:
        """Get current render status."""
        with self.lock:
            if self.current_job:
                return asdict(self.current_job)
            return {'status': 'idle'}


# Global render manager
render_manager = RenderManager()


class SpectraForgeHandler(SimpleHTTPRequestHandler):
    """HTTP request handler for SpectraForge UI."""

    # Directory for static files
    static_dir = Path(__file__).parent / 'static'

    def __init__(self, *args, **kwargs):
        # Set directory for static files
        super().__init__(*args, directory=str(self.static_dir), **kwargs)

    def do_GET(self):
        """Handle GET requests."""
        parsed = urlparse(self.path)
        path = parsed.path

        if path == '/api/status':
            self._send_json(render_manager.get_status())
        elif path == '/api/platform':
            self._send_json(get_platform_info())
        elif path == '/':
            # Serve index.html
            self.path = '/index.html'
            super().do_GET()
        else:
            # Serve static files
            super().do_GET()

    def do_POST(self):
        """Handle POST requests."""
        parsed = urlparse(self.path)
        path = parsed.path

        if path == '/api/render':
            # Start a new render
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            config = json.loads(body) if body else {}

            job = render_manager.start_render(config)
            self._send_json({'job_id': job.id, 'status': job.status})

        elif path == '/api/cancel':
            # Cancel current render
            cancelled = render_manager.cancel_render()
            self._send_json({'cancelled': cancelled})

        else:
            self.send_error(404)

    def _send_json(self, data: Any):
        """Send JSON response."""
        response = json.dumps(data).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(response))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, format, *args):
        """Suppress default logging."""
        pass


def create_static_files():
    """Create static files for the UI if they don't exist."""
    static_dir = Path(__file__).parent / 'static'
    static_dir.mkdir(exist_ok=True)

    # index.html will be created separately
    return static_dir


def run_server(host: str = 'localhost', port: int = 8080, open_browser: bool = True):
    """Run the SpectraForge UI server."""
    # Ensure static files exist
    create_static_files()

    # Create and start server
    server = HTTPServer((host, port), SpectraForgeHandler)

    url = f'http://{host}:{port}'
    print(f"SpectraForge UI running at {url}")
    print("Press Ctrl+C to stop")

    if open_browser:
        # Open browser after short delay
        def open_browser_delayed():
            time.sleep(0.5)
            webbrowser.open(url)

        threading.Thread(target=open_browser_delayed, daemon=True).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


def main():
    """Main entry point for UI."""
    import argparse

    parser = argparse.ArgumentParser(description='SpectraForge Web UI')
    parser.add_argument('--host', default='localhost', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8080, help='Port to listen on')
    parser.add_argument('--no-browser', action='store_true', help='Don\'t open browser automatically')

    args = parser.parse_args()
    run_server(args.host, args.port, not args.no_browser)


if __name__ == '__main__':
    main()
