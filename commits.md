# SpectraForge Commit Log

## Commit History

### Phase 7: Post-Processing (Complete)

#### `1ecf80e` - Add quick start guide for users
- Comprehensive documentation for new users
- Installation, API usage, scene files, post-processing examples

#### `1b23731` - Add adaptive sampling for efficient path tracing
- Per-pixel variance estimation with Welford's algorithm
- Error threshold-based convergence detection
- Priority-based sample distribution
- Tile-based adaptive sampling for multi-threading
- 48 unit tests

#### `7786e03` - Add AOV (Arbitrary Output Variables) / render pass support
- Beauty, Depth, Normal, Albedo, Emission passes
- Direct/Indirect lighting separation
- Object ID and Material ID for compositing
- UV and Position passes
- RenderPassCompositor for visualization
- 47 unit tests

#### `4b066fd` - Add post-processing pipeline orchestrator with lens effects
- Unified pipeline with optimal stage ordering
- Chromatic aberration (lens color fringing)
- Sharpen filter (unsharp mask)
- Film grain effect
- Intermediate result storage
- 37 unit tests

#### `9c5006a` - Add comprehensive post-processing (tone mapping, bloom, color correction)
- Tone mapping: Reinhard, ACES Filmic, Uncharted 2, Exposure-based
- Bloom/glow with multi-scale blur and quality presets
- Color correction: exposure, contrast, saturation, temperature
- 3D LUT support, vignette effect
- 100 unit tests

#### `35a9e0b` - Add denoising integration with OIDN and bilateral filter fallback
- Intel Open Image Denoise (OIDN) integration
- Bilateral and joint bilateral filter fallback
- Auxiliary buffer support (albedo, normal)
- 27 unit tests

### Phase 6: Advanced Rendering (Complete)

#### `67b74c6` - Add Multiple Importance Sampling (MIS) for variance reduction
- Balance and power heuristics
- MIS integrator with next event estimation
- One-sample MIS implementation

#### `e92319f` - Enhance importance sampling for lights
- Power-weighted light sampling
- Per-light PDF computation
- Improved direct lighting estimation

#### `72a9a73` - Add motion blur support with temporal sampling
- Camera motion blur with configurable shutter
- Temporal ray distribution

#### `3d8777f` - Add environment map lighting (HDRI) system
- HDRIEnvironment with importance sampling
- GradientEnvironment and ProceduralSky
- Rotation and intensity controls

#### `7b28990` - Add OBJ mesh loader with smooth shading support
- OBJ file parsing with materials
- Smooth shading via vertex normals
- UV coordinate support

#### `41be5e3` - Add texture mapping system and new geometric shapes
- Image textures, procedural textures (checker, noise, marble)
- Normal mapping support
- Box, Cylinder, Cone primitives

### Phase 1-5: Core Implementation (Complete)

#### `f46fecc` - Complete ray tracing renderer with PBR, BVH, volumetrics
- Path tracing with Russian roulette
- PBR materials (Cook-Torrance BRDF with GGX)
- BVH acceleration with parallel build
- Volumetric effects (fog, smoke, SSS)
- Multiple light types
- Scene parser (YAML/JSON)
- HDR output, cross-platform support
- 164 initial unit tests

#### `e458360` - Initial commit

---

## Project Statistics

| Metric | Count |
|--------|-------|
| Total Commits | 14 |
| Unit Tests | 611 |
| Source Files | 22 modules |
| Test Files | 20 test modules |

## Module Summary

### Core (Phase 1-5)
- `vec3.py` - Vector3 math (Point3, Color)
- `ray.py` - Ray class
- `shapes.py` - Sphere, Plane, Triangle, Box, Cylinder, Cone, AABB
- `materials.py` - Lambertian, Metal, Dielectric, PBR, Emissive
- `camera.py` - Camera with DOF and motion blur
- `renderer.py` - Path tracer, HDR/EXR output
- `bvh.py` - BVH acceleration structure
- `lights.py` - Point, Directional, Area, Sphere lights
- `volumes.py` - Fog, smoke, SSS
- `scene_parser.py` - YAML/JSON scene loader

### Advanced Rendering (Phase 6)
- `textures.py` - Image, procedural, normal map textures
- `environment.py` - HDRI environment maps
- `obj_loader.py` - OBJ mesh loading
- `mis.py` - Multiple importance sampling

### Post-Processing (Phase 7)
- `denoiser.py` - OIDN + bilateral filter denoising
- `tonemapping.py` - HDR to LDR tone mapping
- `bloom.py` - Bloom/glow effects
- `color_correction.py` - Color grading
- `postprocess.py` - Pipeline orchestrator
- `aov.py` - Render passes (AOV)
- `adaptive.py` - Adaptive sampling
