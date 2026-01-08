# SpectraForge Metal - Quick Start Guide

A high-performance GPU ray tracer for Apple Silicon, achieving **3.9 billion rays/second** on M4 Pro.

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Xcode Command Line Tools (`xcode-select --install`)

## Build

```bash
cd spectraforge-metal
make
```

That's it! No additional dependencies required - shaders compile at runtime.

## Your First Render

```bash
./build/spectraforge
```

This renders the demo scene at 800×600 with 16 samples per pixel. Output: `output/render.png`

## Quick Examples

### High Quality Render
```bash
./build/spectraforge --width 1920 --height 1080 --samples 256 --output my_render.png
```

### Interactive Preview
```bash
./build/spectraforge --preview
```
Controls:
- **Mouse drag**: Orbit camera
- **Scroll**: Zoom in/out
- **W/S**: Move forward/back
- **A/D**: Rotate left/right
- **Q/E**: Look up/down
- **Space**: Reset view
- **Escape**: Close

### Built-in Scenes
```bash
./build/spectraforge --scene demo      # Random spheres (default)
./build/spectraforge --scene cornell   # Cornell box
./build/spectraforge --scene pbr       # PBR material showcase
./build/spectraforge --scene dof       # Depth of field demo
./build/spectraforge --scene motion    # Motion blur demo
./build/spectraforge --scene stress    # 1600+ spheres benchmark
```

### Custom JSON Scene
```bash
./build/spectraforge --scene-file scenes/simple.json
```

## Create Your Own Scene

Create a JSON file (e.g., `scenes/myscene.json`):

```json
{
  "materials": [
    { "type": "lambertian", "albedo": [0.5, 0.5, 0.5] },
    { "type": "metal", "albedo": [0.8, 0.8, 0.8], "roughness": 0.1 },
    { "type": "dielectric", "ior": 1.5 },
    { "type": "emissive", "emission": { "color": [1, 1, 1], "intensity": 10 } }
  ],
  "spheres": [
    { "center": [0, -1000, 0], "radius": 1000, "material": 0 },
    { "center": [0, 1, 0], "radius": 1, "material": 2 },
    { "center": [-2, 1, 0], "radius": 1, "material": 1 }
  ],
  "camera": {
    "position": [5, 3, 5],
    "look_at": [0, 1, 0],
    "fov": 40,
    "aperture": 0.1,
    "focus_distance": 6
  }
}
```

Render it:
```bash
./build/spectraforge --scene-file scenes/myscene.json --samples 128
```

### Material Types

| Type | Properties |
|------|------------|
| `lambertian` | `albedo` - diffuse color [r,g,b] |
| `metal` | `albedo`, `roughness` (0=mirror, 1=rough) |
| `dielectric` | `ior` - index of refraction (glass=1.5) |
| `emissive` | `emission.color`, `emission.intensity` |
| `pbr` | `albedo`, `metallic`, `roughness` |

### Motion Blur

Add velocity to spheres:
```json
{
  "center": [0, 1, 0],
  "radius": 0.5,
  "material": 1,
  "velocity": [2, 0, 0]
}
```

Enable camera shutter:
```json
"camera": {
  "shutter": [0, 1]
}
```

## Performance Tips

1. **Start low, go high**: Begin with `--samples 4` to preview, then increase
2. **Use preview mode**: `--preview` for real-time camera adjustment
3. **BVH acceleration**: Automatically enabled for complex scenes
4. **Benchmark your GPU**:
   ```bash
   ./build/spectraforge --benchmark --width 1920 --height 1080 --samples 64
   ```

## Command Reference

| Option | Description | Default |
|--------|-------------|---------|
| `--width N` | Image width | 800 |
| `--height N` | Image height | 600 |
| `--samples N` | Samples per pixel | 16 |
| `--depth N` | Max ray bounces | 10 |
| `--output FILE` | Output path | output/render.png |
| `--scene NAME` | Built-in scene | demo |
| `--scene-file F` | JSON scene file | - |
| `--preview` | Interactive window | - |
| `--benchmark` | Performance test | - |

## Typical Performance (M4 Pro)

| Resolution | Samples | Time | Throughput |
|------------|---------|------|------------|
| 1280×720 | 16 | 46ms | 3.2 Grays/sec |
| 1920×1080 | 64 | 347ms | 3.9 Grays/sec |
| 1920×1080 | 256 | ~1.4s | 3.9 Grays/sec |

## Troubleshooting

**Black image?**
- Ensure camera is positioned outside objects
- Check material indices match your materials array

**Slow performance?**
- Reduce `--samples` for preview
- Use `--preview` mode for interactive adjustment

**Build errors?**
- Run `xcode-select --install` for command line tools
- Ensure you're on macOS with Apple Silicon

## Web UI

SpectraForge Metal includes a web-based interface for easy rendering without command-line knowledge.

### Start the Web UI

```bash
cd spectraforge-metal
python ui_server.py
```

This opens a browser to `http://localhost:8080` with:
- **Scene presets**: Demo, Cornell Box, PBR, DOF, Motion Blur
- **Quality presets**: Preview (800x600) to High (1080p @ 256spp)
- **Camera controls**: Position, look-at, FOV, aperture, focus distance
- **Post-processing**: Tone mapping, bloom

### Web UI Options

```bash
python ui_server.py --port 9000         # Use different port
python ui_server.py --no-browser        # Don't auto-open browser
python ui_server.py --host 0.0.0.0      # Allow external connections
```

## Next Steps

- Explore `scenes/` directory for example JSON files
- Try different material combinations
- Experiment with depth of field (`aperture` > 0)
- Add motion blur with sphere velocities
- Use the Web UI for quick iterations

Happy rendering!
