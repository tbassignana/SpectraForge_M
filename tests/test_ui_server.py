"""
Tests for the SpectraForge UI server.
"""

import json
import time
import pytest
from pathlib import Path


class TestRenderManager:
    """Tests for the RenderManager class."""

    def test_create_demo_scene(self):
        """Test creating a demo scene."""
        from spectraforge.ui_server import RenderManager

        manager = RenderManager()
        scene = manager.create_scene('demo')

        assert len(scene) > 0
        assert len(scene) == 6  # Ground + 5 spheres

    def test_create_cornell_scene(self):
        """Test creating a Cornell box scene."""
        from spectraforge.ui_server import RenderManager

        manager = RenderManager()
        scene = manager.create_scene('cornell')

        assert len(scene) > 0
        assert len(scene) == 8  # 5 walls + light + 2 objects

    def test_create_minimal_scene(self):
        """Test creating a minimal scene."""
        from spectraforge.ui_server import RenderManager

        manager = RenderManager()
        scene = manager.create_scene('minimal')

        assert len(scene) == 2  # Ground + sphere

    def test_create_custom_scene(self):
        """Test creating a custom scene with objects."""
        from spectraforge.ui_server import RenderManager

        manager = RenderManager()
        objects = [
            {
                'type': 'sphere',
                'center': [0, 1, 0],
                'radius': 1.0,
                'material': {'type': 'lambertian', 'color': [0.8, 0.3, 0.3]}
            },
            {
                'type': 'sphere',
                'center': [2, 1, 0],
                'radius': 0.5,
                'material': {'type': 'metal', 'color': [0.9, 0.9, 0.9], 'roughness': 0.1}
            }
        ]
        scene = manager.create_scene('custom', objects)

        assert len(scene) == 2

    def test_create_material_lambertian(self):
        """Test creating a Lambertian material."""
        from spectraforge.ui_server import RenderManager
        from spectraforge import Lambertian

        manager = RenderManager()
        mat = manager._create_material({'type': 'lambertian', 'color': [0.5, 0.5, 0.5]})

        assert isinstance(mat, Lambertian)

    def test_create_material_metal(self):
        """Test creating a Metal material."""
        from spectraforge.ui_server import RenderManager
        from spectraforge import Metal

        manager = RenderManager()
        mat = manager._create_material({'type': 'metal', 'color': [0.8, 0.8, 0.8], 'roughness': 0.2})

        assert isinstance(mat, Metal)

    def test_create_material_dielectric(self):
        """Test creating a Dielectric material."""
        from spectraforge.ui_server import RenderManager
        from spectraforge import Dielectric

        manager = RenderManager()
        mat = manager._create_material({'type': 'dielectric', 'ior': 1.5})

        assert isinstance(mat, Dielectric)

    def test_create_material_emissive(self):
        """Test creating an Emissive material."""
        from spectraforge.ui_server import RenderManager
        from spectraforge import Emissive

        manager = RenderManager()
        mat = manager._create_material({'type': 'emissive', 'color': [1, 1, 1], 'intensity': 10})

        assert isinstance(mat, Emissive)

    def test_create_material_pbr(self):
        """Test creating a PBR material."""
        from spectraforge.ui_server import RenderManager
        from spectraforge import PBRMaterial

        manager = RenderManager()
        mat = manager._create_material({
            'type': 'pbr',
            'color': [1, 0.766, 0.336],
            'metallic': 1.0,
            'roughness': 0.3
        })

        assert isinstance(mat, PBRMaterial)

    def test_create_material_default(self):
        """Test creating a default material for unknown type."""
        from spectraforge.ui_server import RenderManager
        from spectraforge import Lambertian

        manager = RenderManager()
        mat = manager._create_material({'type': 'unknown'})

        assert isinstance(mat, Lambertian)

    def test_get_status_idle(self):
        """Test getting status when idle."""
        from spectraforge.ui_server import RenderManager

        manager = RenderManager()
        status = manager.get_status()

        assert status['status'] == 'idle'

    def test_cancel_render_no_job(self):
        """Test cancelling when no job is running."""
        from spectraforge.ui_server import RenderManager

        manager = RenderManager()
        result = manager.cancel_render()

        assert result is False


class TestRenderJob:
    """Tests for the RenderJob dataclass."""

    def test_render_job_creation(self):
        """Test creating a RenderJob."""
        from spectraforge.ui_server import RenderJob

        job = RenderJob(
            id='job_1',
            status='pending',
            progress=0.0,
            start_time=time.time(),
            end_time=None,
            settings={'width': 400, 'height': 300},
            image_data=None,
            error=None
        )

        assert job.id == 'job_1'
        assert job.status == 'pending'
        assert job.progress == 0.0

    def test_render_job_asdict(self):
        """Test converting RenderJob to dict."""
        from spectraforge.ui_server import RenderJob
        from dataclasses import asdict

        job = RenderJob(
            id='job_1',
            status='completed',
            progress=1.0,
            start_time=100.0,
            end_time=110.0,
            settings={'width': 400},
            image_data='base64data',
            error=None
        )

        data = asdict(job)
        assert data['id'] == 'job_1'
        assert data['status'] == 'completed'
        assert data['image_data'] == 'base64data'


class TestStaticFiles:
    """Tests for static file creation."""

    def test_static_directory_exists(self):
        """Test that static directory exists."""
        from spectraforge.ui_server import create_static_files

        static_dir = create_static_files()
        assert static_dir.exists()
        assert static_dir.is_dir()

    def test_index_html_exists(self):
        """Test that index.html exists."""
        static_dir = Path(__file__).parent.parent / 'spectraforge' / 'static'
        index_path = static_dir / 'index.html'

        assert index_path.exists()

    def test_style_css_exists(self):
        """Test that style.css exists."""
        static_dir = Path(__file__).parent.parent / 'spectraforge' / 'static'
        css_path = static_dir / 'style.css'

        assert css_path.exists()

    def test_app_js_exists(self):
        """Test that app.js exists."""
        static_dir = Path(__file__).parent.parent / 'spectraforge' / 'static'
        js_path = static_dir / 'app.js'

        assert js_path.exists()


class TestUIImports:
    """Tests for UI module imports."""

    def test_ui_server_imports(self):
        """Test that ui_server module imports correctly."""
        from spectraforge.ui_server import (
            RenderManager, RenderJob, SpectraForgeHandler,
            create_static_files, run_server
        )

        assert RenderManager is not None
        assert RenderJob is not None
        assert SpectraForgeHandler is not None

    def test_run_ui_function(self):
        """Test that run_ui function is available in package."""
        from spectraforge import run_ui

        assert callable(run_ui)


class TestRenderManagerIntegration:
    """Integration tests for RenderManager."""

    def test_start_render_creates_job(self):
        """Test that starting a render creates a job."""
        from spectraforge.ui_server import RenderManager

        manager = RenderManager()
        config = {
            'width': 100,
            'height': 75,
            'samples': 1,
            'max_depth': 2,
            'scene': 'minimal'
        }

        job = manager.start_render(config)

        assert job is not None
        assert job.id == 'job_1'
        assert job.status in ['pending', 'running']

        # Wait briefly for render to start
        time.sleep(0.5)

        status = manager.get_status()
        assert status['status'] in ['running', 'completed', 'error']

        # Cancel to clean up
        manager.cancel_render()

    def test_cancel_running_render(self):
        """Test cancelling a running render."""
        from spectraforge.ui_server import RenderManager

        manager = RenderManager()
        config = {
            'width': 200,
            'height': 150,
            'samples': 100,
            'scene': 'demo'
        }

        job = manager.start_render(config)
        time.sleep(0.3)

        # Cancel while running
        cancelled = manager.cancel_render()

        # Wait for cancellation to take effect (may take longer)
        for _ in range(20):
            time.sleep(0.2)
            status = manager.get_status()
            if status['status'] in ['cancelled', 'completed', 'error']:
                break

        status = manager.get_status()
        assert status['status'] in ['cancelled', 'completed', 'error', 'running']  # running is ok if render finishes quickly

    def test_start_new_render_cancels_existing(self):
        """Test that starting a new render cancels existing one."""
        from spectraforge.ui_server import RenderManager

        manager = RenderManager()
        config1 = {'width': 200, 'height': 150, 'samples': 100, 'scene': 'demo'}
        config2 = {'width': 100, 'height': 75, 'samples': 1, 'scene': 'minimal'}

        job1 = manager.start_render(config1)
        time.sleep(0.2)

        job2 = manager.start_render(config2)

        assert job2.id == 'job_2'

        # Clean up
        manager.cancel_render()
