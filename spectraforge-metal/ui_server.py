#!/usr/bin/env python3
"""
SpectraForge Metal - Web UI Server

Connects the web UI to the high-performance Metal GPU renderer.
Achieves 1000x+ speedup over the Python renderer.

Usage:
    python ui_server.py
    # Opens browser to http://localhost:8080
"""

import base64
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
from dataclasses import dataclass, asdict
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Optional, Dict, Any
from urllib.parse import urlparse
import platform


@dataclass
class RenderJob:
    """Represents an active render job."""
    id: str
    status: str  # 'pending', 'running', 'completed', 'cancelled', 'error'
    progress: float
    start_time: float
    end_time: Optional[float]
    settings: Dict[str, Any]
    image_data: Optional[str]  # Base64 encoded PNG
    error: Optional[str]


class MetalRenderManager:
    """Manages render jobs using the Metal GPU renderer."""

    def __init__(self):
        self.current_job: Optional[RenderJob] = None
        self.render_thread: Optional[threading.Thread] = None
        self.cancel_flag = threading.Event()
        self.lock = threading.Lock()
        self.job_counter = 0

        # Find the Metal renderer binary
        self.metal_binary = self._find_metal_binary()
        if not self.metal_binary:
            print("WARNING: Metal renderer not found. Run 'make' to build it.")

    def _find_metal_binary(self) -> Optional[Path]:
        """Find the Metal renderer binary."""
        # Look in common locations
        candidates = [
            Path(__file__).parent / 'build' / 'spectraforge',
            Path(__file__).parent.parent / 'spectraforge-metal' / 'build' / 'spectraforge',
            Path('build/spectraforge'),
        ]

        for path in candidates:
            if path.exists() and os.access(path, os.X_OK):
                return path.resolve()

        return None

    def _create_scene_json(self, config: Dict[str, Any], output_path: Path) -> Path:
        """Create a JSON scene file from the UI config."""
        scene_type = config.get('scene', 'demo')
        cam_config = config.get('camera', {})

        # Map scene types to built-in scenes or create custom JSON
        scene_data = {
            "materials": [],
            "spheres": [],
            "camera": {
                "position": cam_config.get('look_from', [13, 2, 3]),
                "look_at": cam_config.get('look_at', [0, 0, 0]),
                "up": [0, 1, 0],
                "fov": cam_config.get('vfov', 20),
                "aperture": cam_config.get('aperture', 0.1),
                "focus_distance": cam_config.get('focus_dist', 10.0)
            },
            "settings": {
                "samples": config.get('samples', 16),
                "max_depth": config.get('max_depth', 10),
                "sky_gradient": True
            }
        }

        # For built-in scenes, we'll use --scene flag instead
        # Only create JSON for custom scenes
        if scene_type == 'custom' and config.get('objects'):
            # Convert custom objects to JSON format
            mat_idx = 0
            for obj in config.get('objects', []):
                mat = obj.get('material', {})
                mat_type = mat.get('type', 'lambertian')

                material_data = {"type": mat_type}
                if 'color' in mat:
                    material_data['albedo'] = mat['color']
                if 'roughness' in mat:
                    material_data['roughness'] = mat['roughness']
                if 'ior' in mat:
                    material_data['ior'] = mat['ior']

                scene_data['materials'].append(material_data)

                if obj.get('type') == 'sphere':
                    scene_data['spheres'].append({
                        "center": obj.get('center', [0, 0, 0]),
                        "radius": obj.get('radius', 1.0),
                        "material": mat_idx
                    })
                    mat_idx += 1

            scene_file = output_path / 'scene.json'
            with open(scene_file, 'w') as f:
                json.dump(scene_data, f, indent=2)
            return scene_file

        return None

    def start_render(self, config: Dict[str, Any]) -> RenderJob:
        """Start a new render job using Metal renderer."""
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
        """Worker thread for Metal rendering."""
        try:
            job.status = 'running'

            if not self.metal_binary:
                raise RuntimeError("Metal renderer not found. Run 'make' to build.")

            # Extract settings
            width = config.get('width', 800)
            height = config.get('height', 600)
            samples = config.get('samples', 16)
            max_depth = config.get('max_depth', 10)
            scene_type = config.get('scene', 'demo')

            # Camera settings
            cam_config = config.get('camera', {})

            # Create temp directory for output
            with tempfile.TemporaryDirectory() as tmpdir:
                output_file = Path(tmpdir) / 'render.png'

                # Build command
                cmd = [
                    str(self.metal_binary),
                    '--width', str(width),
                    '--height', str(height),
                    '--samples', str(samples),
                    '--depth', str(max_depth),
                    '--output', str(output_file),
                ]

                # Use built-in scene or custom JSON
                scene_file = self._create_scene_json(config, Path(tmpdir))
                if scene_file:
                    cmd.extend(['--scene-file', str(scene_file)])
                else:
                    # Map UI scene names to Metal scene names
                    scene_map = {
                        'demo': 'demo',
                        'cornell': 'cornell',
                        'minimal': 'demo',
                        'pbr': 'pbr',
                        'dof': 'dof',
                        'motion': 'motion',
                    }
                    cmd.extend(['--scene', scene_map.get(scene_type, 'demo')])

                # Simulate progress (Metal doesn't report progress yet)
                job.progress = 0.1

                # Run Metal renderer
                start = time.time()
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=str(self.metal_binary.parent.parent)
                )
                elapsed = time.time() - start

                if self.cancel_flag.is_set():
                    job.status = 'cancelled'
                    return

                if result.returncode != 0:
                    raise RuntimeError(f"Render failed: {result.stderr}")

                job.progress = 0.9

                # Read output image and convert to base64
                if output_file.exists():
                    with open(output_file, 'rb') as f:
                        job.image_data = base64.b64encode(f.read()).decode('utf-8')
                else:
                    raise RuntimeError("Output image not created")

                job.status = 'completed'
                job.progress = 1.0
                job.end_time = time.time()

                # Log performance
                print(f"Render complete: {width}x{height} @ {samples}spp in {elapsed:.2f}s")

        except Exception as e:
            job.status = 'error'
            job.error = str(e)
            job.end_time = time.time()
            print(f"Render error: {e}")

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
render_manager = MetalRenderManager()


def get_platform_info() -> Dict[str, Any]:
    """Get platform information."""
    return {
        'system': platform.system(),
        'machine': platform.machine(),
        'cpu_count': os.cpu_count(),
        'renderer': 'Metal GPU' if render_manager.metal_binary else 'Not available',
        'binary': str(render_manager.metal_binary) if render_manager.metal_binary else None
    }


class SpectraForgeHandler(SimpleHTTPRequestHandler):
    """HTTP request handler for SpectraForge Metal UI."""

    # Try multiple locations for static files
    @property
    def static_dir(self):
        # First try the spectraforge Python package static dir
        python_static = Path(__file__).parent.parent / 'spectraforge' / 'static'
        if python_static.exists():
            return python_static

        # Fall back to local static dir
        local_static = Path(__file__).parent / 'static'
        if local_static.exists():
            return local_static

        return python_static  # Default

    def __init__(self, *args, **kwargs):
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
            self.path = '/index.html'
            super().do_GET()
        else:
            super().do_GET()

    def do_POST(self):
        """Handle POST requests."""
        parsed = urlparse(self.path)
        path = parsed.path

        if path == '/api/render':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            config = json.loads(body) if body else {}

            job = render_manager.start_render(config)
            self._send_json({'job_id': job.id, 'status': job.status})

        elif path == '/api/cancel':
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
        """Custom logging."""
        print(f"[{time.strftime('%H:%M:%S')}] {args[0]}")


def run_server(host: str = 'localhost', port: int = 8080, open_browser: bool = True):
    """Run the SpectraForge Metal UI server."""
    server = HTTPServer((host, port), SpectraForgeHandler)

    url = f'http://{host}:{port}'
    print("=" * 50)
    print("SpectraForge Metal - Web UI")
    print("=" * 50)
    print(f"URL: {url}")
    print(f"Renderer: {'Metal GPU' if render_manager.metal_binary else 'NOT FOUND'}")
    if render_manager.metal_binary:
        print(f"Binary: {render_manager.metal_binary}")
    else:
        print("\nWARNING: Run 'make' to build the Metal renderer")
    print("\nPress Ctrl+C to stop")
    print("=" * 50)

    if open_browser:
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
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='SpectraForge Metal Web UI')
    parser.add_argument('--host', default='localhost', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8080, help='Port to listen on')
    parser.add_argument('--no-browser', action='store_true', help="Don't open browser")

    args = parser.parse_args()
    run_server(args.host, args.port, not args.no_browser)


if __name__ == '__main__':
    main()
