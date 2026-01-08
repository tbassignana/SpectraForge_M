# SpectraForge Quick Start Guide

A complete Python ray tracing renderer featuring path tracing, PBR materials, volumetrics, and post-processing.

## Installation

### Requirements
- Python 3.8+
- NumPy
- Pillow (PIL)

### Setup

```bash
# Clone/navigate to the project
cd SpectraForge

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Optional Dependencies

```bash
# For OpenEXR output
pip install OpenEXR

# For YAML scene files
pip install pyyaml

# For Intel Open Image Denoise (high-quality denoising)
pip install pyoidn
```

## Quick Render (Command Line)

Render a demo scene immediately:

```bash
# Basic render
python main.py

# High-quality render
python main.py --width 1920 --height 1080 --samples 500 --output hd_render.png

# Cornell box scene
python main.py --scene cornell --samples 1000 --output cornell.png

# Show platform info
python main.py --info
```

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--width` | 800 | Image width in pixels |
| `--height` | 600 | Image height in pixels |
| `--samples` | 100 | Samples per pixel (more = less noise) |
| `--depth` | 50 | Maximum ray bounce depth |
| `--threads` | 0 | Number of threads (0 = auto-detect) |
| `--output` | output/render.png | Output file path (.png, .hdr, .exr) |
| `--scene` | demo | Scene preset: `demo` or `cornell` |

## Web UI (Recommended for Beginners)

Launch the browser-based interface for an interactive rendering experience:

```bash
python -m spectraforge.ui
```

This opens http://localhost:8080 in your browser with:
- **Scene Presets**: Demo, Cornell Box, Minimal
- **Render Settings**: Resolution, samples, depth, threads
- **Camera Controls**: Position, look-at, FOV, aperture, focus distance
- **Post-Processing**: Denoise, tone mapping, bloom, exposure
- **Real-time Progress**: Progress bar with ETA
- **Download**: Save rendered images as PNG

No additional dependencies required - works on any platform with Python 3.8+.

## Python API Usage

### Basic Render

```python
from spectraforge import (
    Vec3, Point3, Color, Camera, Renderer, RenderSettings,
    Sphere, HittableList, Lambertian, Metal, Dielectric
)

# Create scene
world = HittableList()

# Ground
world.add(Sphere(Point3(0, -1000, 0), 1000, Lambertian(Color(0.5, 0.5, 0.5))))

# Spheres with different materials
world.add(Sphere(Point3(0, 1, 0), 1.0, Dielectric(1.5)))        # Glass
world.add(Sphere(Point3(-4, 1, 0), 1.0, Lambertian(Color(0.4, 0.2, 0.1))))  # Matte
world.add(Sphere(Point3(4, 1, 0), 1.0, Metal(Color(0.7, 0.6, 0.5), 0.0)))   # Mirror

# Camera
camera = Camera(
    look_from=Point3(13, 2, 3),
    look_at=Point3(0, 0, 0),
    vup=Vec3(0, 1, 0),
    vfov=20,
    aspect_ratio=800 / 600,
    aperture=0.1,
    focus_dist=10.0
)

# Render
settings = RenderSettings(width=800, height=600, samples_per_pixel=100)
renderer = Renderer(settings)
image = renderer.render(world, camera)

# Save
renderer.save_image(image, "output.png")
```

### Materials

```python
from spectraforge import Lambertian, Metal, Dielectric, Emissive, PBRMaterial, Color

# Diffuse (matte)
matte_red = Lambertian(Color(0.8, 0.1, 0.1))

# Metal (reflective)
chrome = Metal(Color(0.9, 0.9, 0.9), roughness=0.0)    # Perfect mirror
brushed = Metal(Color(0.8, 0.7, 0.6), roughness=0.3)   # Brushed metal

# Glass (transparent)
glass = Dielectric(ior=1.5)      # Standard glass
water = Dielectric(ior=1.33)     # Water
diamond = Dielectric(ior=2.42)   # Diamond

# Emissive (light source)
light = Emissive(Color(1, 1, 1), intensity=10.0)

# PBR (physically-based)
gold = PBRMaterial(
    albedo=Color(1.0, 0.766, 0.336),
    metallic=1.0,
    roughness=0.3
)
```

### Shapes

```python
from spectraforge import Sphere, Plane, Box, Cylinder, Cone, Triangle, Point3, Vec3

# Sphere
sphere = Sphere(center=Point3(0, 1, 0), radius=1.0, material=mat)

# Plane (infinite)
floor = Plane(point=Point3(0, 0, 0), normal=Vec3(0, 1, 0), material=mat)

# Box
box = Box(min_corner=Point3(-1, 0, -1), max_corner=Point3(1, 2, 1), material=mat)

# Cylinder
cyl = Cylinder(center=Point3(0, 1, 0), radius=0.5, height=2.0, material=mat)

# Cone
cone = Cone(apex=Point3(0, 2, 0), radius=1.0, height=2.0, material=mat)

# Triangle
tri = Triangle(v0=Point3(0, 0, 0), v1=Point3(1, 0, 0), v2=Point3(0.5, 1, 0), material=mat)
```

### Loading OBJ Meshes

```python
from spectraforge import load_obj

# Load mesh with options
mesh = load_obj(
    "model.obj",
    material=my_material,
    scale=1.0,
    center=Point3(0, 0, 0),
    smooth=True
)
world.add(mesh)
```

## Scene Files (JSON/YAML)

Create reusable scene definitions:

```json
{
  "render": {
    "width": 800,
    "height": 600,
    "samples": 100
  },
  "camera": {
    "look_from": [13, 2, 3],
    "look_at": [0, 0, 0],
    "vfov": 20,
    "aperture": 0.1,
    "focus_dist": 10
  },
  "materials": {
    "glass": { "type": "dielectric", "ior": 1.5 },
    "gold": { "type": "pbr", "albedo": [1.0, 0.766, 0.336], "metallic": 1.0, "roughness": 0.3 }
  },
  "objects": [
    { "type": "sphere", "center": [0, 1, 0], "radius": 1.0, "material": "glass" }
  ],
  "lights": [
    { "type": "point", "position": [0, 10, 0], "color": [1, 1, 1], "intensity": 10 }
  ]
}
```

Load and render:

```python
from spectraforge import load_scene, Renderer

world, camera, settings, lights = load_scene("scenes/my_scene.json")
renderer = Renderer(settings)
image = renderer.render(world, camera)
```

## Post-Processing Pipeline

Apply cinematic effects to your renders:

```python
from spectraforge import (
    create_pipeline, tone_map, apply_bloom,
    apply_color_correction, denoise_image
)

# Full pipeline
pipeline = create_pipeline(
    denoise=True,
    bloom=True,
    bloom_intensity=0.3,
    tone_mapping="aces",
    color_correction=True,
    exposure=0.5,
    contrast=1.1,
    saturation=1.2,
    chromatic_aberration=True,
    film_grain=True,
    grain_intensity=0.05
)

result = pipeline.process(hdr_image)
final_image = result.image

# Or apply effects individually
denoised = denoise_image(hdr_image).image
bloomed = apply_bloom(denoised, intensity=0.3, threshold=1.0).image
tonemapped = tone_map(bloomed, method="aces").image
```

### Available Post-Processing Effects

| Effect | Description |
|--------|-------------|
| **Denoise** | OIDN or bilateral filter noise reduction |
| **Bloom** | Glow effect from bright areas |
| **Tone Mapping** | HDR to LDR (reinhard, aces, uncharted2) |
| **Color Correction** | Exposure, contrast, saturation, temperature |
| **Vignette** | Darkened corners |
| **Chromatic Aberration** | Lens color fringing |
| **Sharpen** | Edge enhancement |
| **Film Grain** | Cinematic grain texture |

## Environment Lighting

```python
from spectraforge import HDRIEnvironment, GradientEnvironment, ProceduralSky

# HDRI environment map
env = HDRIEnvironment("studio.hdr", intensity=1.0, rotation=45)

# Gradient sky
sky = GradientEnvironment(
    zenith=Color(0.5, 0.7, 1.0),
    horizon=Color(1.0, 0.9, 0.8),
    ground=Color(0.4, 0.3, 0.2)
)

# Procedural sky with sun
sky = ProceduralSky(sun_direction=Vec3(0.5, 0.3, 0.8), turbidity=2.0)
```

## Render Passes (AOV)

Extract separate passes for compositing:

```python
from spectraforge import AOVManager, AOVType, RenderPassCompositor

# Enable passes
aov = AOVManager(width=800, height=600)
aov.enable(AOVType.DEPTH)
aov.enable(AOVType.NORMAL)
aov.enable(AOVType.ALBEDO)
aov.enable(AOVType.OBJECT_ID)

# After rendering, extract passes
compositor = RenderPassCompositor(aov)
depth_vis = compositor.get_depth_visualization()
normal_vis = compositor.get_normal_visualization()
```

## Adaptive Sampling

Spend more samples where needed:

```python
from spectraforge import AdaptiveSampler, TileAdaptiveSampler

sampler = AdaptiveSampler(
    width=800,
    height=600,
    min_samples=16,
    max_samples=1024,
    error_threshold=0.01
)

# Check which pixels need more samples
pixels_to_sample = sampler.get_pixels_to_sample(batch_size=1000)
```

## Performance Tips

1. **Use BVH acceleration** for scenes with many objects:
   ```python
   from spectraforge import build_bvh
   bvh = build_bvh(world)  # Much faster ray intersection
   image = renderer.render(bvh, camera)
   ```

2. **Adjust thread count** for your CPU:
   ```python
   settings = RenderSettings(num_threads=8)  # Or 0 for auto
   ```

3. **Lower samples for preview**, increase for final:
   - Preview: 16-64 samples
   - Draft: 100-256 samples
   - Final: 500-2000+ samples

4. **Use adaptive sampling** to focus computation on noisy areas

## Output Formats

| Extension | Format | Notes |
|-----------|--------|-------|
| `.png` | PNG | 8-bit LDR, good for sharing |
| `.hdr` | Radiance HDR | 32-bit HDR, for compositing |
| `.exr` | OpenEXR | 32-bit HDR, industry standard |

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_renderer.py -v

# With coverage
pytest tests/ --cov=spectraforge
```

## Example Workflow

```python
from spectraforge import *

# 1. Create materials
glass = Dielectric(1.5)
gold = PBRMaterial(Color(1, 0.766, 0.336), metallic=1.0, roughness=0.2)
light_mat = Emissive(Color(1, 1, 1), 15)

# 2. Build scene
world = HittableList()
world.add(Sphere(Point3(0, -1000, 0), 1000, Lambertian(Color(0.5, 0.5, 0.5))))
world.add(Sphere(Point3(0, 1, 0), 1.0, glass))
world.add(Sphere(Point3(-2, 0.5, 1), 0.5, gold))
world.add(Sphere(Point3(0, 5, 0), 0.5, light_mat))

# 3. Accelerate with BVH
scene = build_bvh(world)

# 4. Setup camera
camera = Camera(
    look_from=Point3(6, 2, 4),
    look_at=Point3(0, 0.5, 0),
    vfov=30,
    aperture=0.05,
    focus_dist=6.0
)

# 5. Render HDR
settings = RenderSettings(width=1280, height=720, samples_per_pixel=256)
renderer = Renderer(settings)
hdr_image = renderer.render(scene, camera)

# 6. Post-process
pipeline = create_pipeline(
    denoise=True,
    bloom=True,
    tone_mapping="aces",
    exposure=0.3
)
result = pipeline.process(hdr_image)

# 7. Save
renderer.save_image(result.image, "final_render.png")
```

## Troubleshooting

**Render is too dark**: Increase light intensity or add more lights

**Too much noise**: Increase samples per pixel or enable denoising

**Slow renders**: Use BVH, reduce samples, or decrease resolution

**Memory issues with large scenes**: Process in tiles with `TileAdaptiveSampler`

---

For more examples, see `main.py` and `scenes/demo.json`.
