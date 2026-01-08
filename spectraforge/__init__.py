"""
SpectraForge - A Python Ray Tracing Renderer

A complete, optimized ray tracing renderer with support for:
- Global illumination (path tracing)
- Subsurface scattering
- Volumetric effects
- PBR materials (GGX/Cook-Torrance)
- Multi-threaded acceleration structures (BVH)
- HDR image output
- Cross-platform support (ARM/x86)
"""

__version__ = "0.1.0"
__author__ = "SpectraForge Team"

from .vec3 import Vec3, Point3, Color
from .ray import Ray
from .shapes import Sphere, Plane, Triangle, HittableList, AABB, HitRecord, Hittable, Box, Cylinder, Cone, MovingSphere
from .materials import Material, Lambertian, Metal, Dielectric, Emissive, PBRMaterial
from .camera import Camera
from .renderer import Renderer, RenderSettings, get_platform_info
from .bvh import BVH, BVHNode, build_bvh
from .lights import Light, PointLight, DirectionalLight, AreaLight, SphereLight, LightList, compute_direct_lighting, sample_light_direction
from .volumes import ConstantMedium, SubsurfaceScatteringMaterial, create_fog, create_smoke
from .scene_parser import SceneParser, load_scene, parse_scene
from .obj_loader import OBJLoader, SmoothTriangle, load_obj, get_mesh_bounds, get_mesh_stats
from .environment import (
    Environment, SolidColorEnvironment, GradientEnvironment,
    HDRIEnvironment, ProceduralSky, load_hdri, create_simple_sky
)
from .mis import (
    balance_heuristic, power_heuristic, multi_power_heuristic,
    MISSample, MISIntegrator, one_sample_mis, NextEventEstimation
)
from .denoiser import (
    Denoiser, DenoiseResult, OIDNDenoiser, BilateralDenoiser,
    JointBilateralDenoiser, create_denoiser, denoise_image,
    AuxiliaryBufferRenderer
)
from .tonemapping import (
    ToneMapper, ToneMappingOperator, ToneMappingResult,
    LinearToneMapper, ReinhardToneMapper, ACESFilmicToneMapper,
    Uncharted2ToneMapper, ExposureToneMapper,
    create_tone_mapper, tone_map, apply_gamma, linear_to_srgb, srgb_to_linear
)
from .bloom import (
    BloomEffect, BloomQuality, BloomResult, LensFlare, apply_bloom
)
from .color_correction import (
    ColorCorrector, ColorCorrectionSettings, ColorCorrectionResult,
    LUT, Vignette, apply_color_correction
)
from .postprocess import (
    PostProcessingPipeline, PostProcessStage, PipelineResult,
    ChromaticAberration, SharpenFilter, FilmGrain,
    apply_chromatic_aberration, apply_sharpen, apply_film_grain,
    create_pipeline
)
from .aov import (
    AOVType, AOVSample, AOVBuffer, AOVManager,
    normalize_depth, pack_normal, unpack_normal,
    create_object_id_colormap, combine_direct_indirect,
    depth_to_world_position, RenderPassCompositor
)
from .adaptive import (
    AdaptiveMode, AdaptiveStats, PixelState, AdaptiveSampler,
    TileAdaptiveSampler, estimate_required_samples
)

# UI server (lazy import to avoid startup cost)
def run_ui(host: str = 'localhost', port: int = 8080, open_browser: bool = True):
    """Run the SpectraForge web UI."""
    from .ui_server import run_server
    run_server(host, port, open_browser)
