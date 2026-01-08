# SpectraForge Metal

**High-performance GPU path tracer for Apple Silicon**

Achieves **3.9 billion rays/second** on M4 Pro with full path tracing, BVH acceleration, and real-time preview.

![Demo Render](output/render.png)

## Features

- **Path Tracing** - Global illumination with multiple bounces
- **BVH Acceleration** - O(log n) ray-scene intersection with SAH construction
- **PBR Materials** - Lambertian, metal, dielectric, emissive, Cook-Torrance
- **Depth of Field** - Physically-based lens simulation
- **Motion Blur** - Time-sampled sphere animation
- **Bloom & Tone Mapping** - ACES filmic with multi-pass Gaussian glow
- **Real-time Preview** - Interactive camera with progressive rendering
- **Web UI** - Browser-based interface with scene presets and camera controls
- **JSON Scenes** - Define scenes without recompiling
- **Procedural Textures** - Checker, noise, marble, wood patterns

## Quick Start

```bash
# Build (no dependencies required)
make

# Render demo scene
./build/spectraforge

# Interactive preview
./build/spectraforge --preview

# Web UI (browser-based)
python ui_server.py

# Load custom scene
./build/spectraforge --scene-file scenes/simple.json --samples 128
```

**See [QUICKSTART.md](QUICKSTART.md) for comprehensive usage guide.**

## Built-in Scenes

```bash
./build/spectraforge --scene demo      # Random spheres (default)
./build/spectraforge --scene cornell   # Classic Cornell box
./build/spectraforge --scene pbr       # PBR material showcase
./build/spectraforge --scene dof       # Depth of field demo
./build/spectraforge --scene motion    # Motion blur demo
./build/spectraforge --scene mesh      # Triangle mesh test
./build/spectraforge --scene stress    # 1600+ spheres benchmark
```

## Performance (Apple M4 Pro)

| Resolution | Samples | Render Time | Throughput |
|------------|---------|-------------|------------|
| 1280×720 | 16 spp | 46ms | 3.2 Grays/sec |
| 1920×1080 | 64 spp | 347ms | **3.9 Grays/sec** |
| 1920×1080 | 256 spp | ~1.4s | 3.9 Grays/sec |

BVH acceleration enables 17× more primitives with only 40% overhead.

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Xcode Command Line Tools (`xcode-select --install`)

No Metal toolchain required - shaders compile at runtime from source.

## Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--width N` | 800 | Image width in pixels |
| `--height N` | 600 | Image height in pixels |
| `--samples N` | 16 | Samples per pixel |
| `--depth N` | 10 | Maximum ray bounces |
| `--output FILE` | output/render.png | Output path (.png or .hdr) |
| `--scene NAME` | demo | Built-in scene preset |
| `--scene-file F` | - | Load JSON scene file |
| `--preview` | - | Open interactive preview window |
| `--benchmark` | - | Run performance benchmark |

## JSON Scene Format

```json
{
  "materials": [
    { "type": "lambertian", "albedo": [0.8, 0.3, 0.3] },
    { "type": "metal", "albedo": [0.9, 0.9, 0.9], "roughness": 0.1 },
    { "type": "dielectric", "ior": 1.5 }
  ],
  "spheres": [
    { "center": [0, -1000, 0], "radius": 1000, "material": 0 },
    { "center": [0, 1, 0], "radius": 1, "material": 2, "velocity": [1, 0, 0] }
  ],
  "camera": {
    "position": [5, 2, 5],
    "look_at": [0, 0, 0],
    "fov": 40,
    "aperture": 0.1,
    "focus_distance": 6,
    "shutter": [0, 1]
  }
}
```

## Project Structure

```
spectraforge-metal/
├── src/              # C/Objective-C source
│   ├── main.m        # CLI and scene setup
│   ├── renderer.m    # Metal renderer core
│   ├── preview.m     # Real-time preview window
│   ├── scene_loader.c # JSON scene parser
│   ├── bvh.c         # BVH construction
│   └── image.c       # PNG/HDR output
├── shaders/          # Metal compute shaders
│   ├── types.metal   # Shared types, RNG
│   ├── intersect.metal # Ray-primitive, BVH traversal
│   ├── materials.metal # BRDF implementations
│   ├── textures.metal  # Procedural textures
│   ├── pathtracer.metal # Path tracing kernels
│   └── tonemap.metal   # Bloom, tone mapping
├── include/
│   └── spectraforge.h  # Public C API
├── scenes/           # Example JSON scenes
├── lib/              # Third-party (cJSON)
├── QUICKSTART.md     # User guide
├── TODO.md           # Development progress
└── Makefile
```

## Architecture

- **Runtime Shader Compilation** - No Metal toolchain required
- **Unified Memory** - Shared CPU/GPU buffers for fast data transfer
- **Compute Shaders** - Path tracing, BVH traversal, post-processing
- **Progressive Rendering** - Accumulate samples across frames
- **Adaptive Quality** - Low samples while moving, high when still

## License

MIT License - see LICENSE file.

## Credits

Metal implementation following Apple's best practices for GPU compute.
Uses [cJSON](https://github.com/DaveGamble/cJSON) for scene file parsing.
