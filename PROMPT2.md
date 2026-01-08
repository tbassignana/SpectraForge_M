# PROMPT2.md - Metal/C GPU Ray Tracer Rewrite

You are an expert systems programmer tasked with rewriting the SpectraForge ray tracing renderer from Python to **Metal** (Apple's GPU compute API) with **C++** for CPU-side code. The goal is to achieve real-time or near-real-time performance on Apple Silicon (M1/M2/M3/M4).  This new project is titled SpectraForge_M

## Reference Implementation

The existing Python implementation in `spectraforge/` contains all the algorithms you need:
- `vec3.py` - Vector math (translate to Metal's `float3`)
- `ray.py` - Ray structure
- `shapes.py` - Sphere, Plane, Triangle, Box, Cylinder, Cone intersection
- `materials.py` - Lambertian, Metal, Dielectric, PBR (GGX/Cook-Torrance)
- `camera.py` - Camera with DOF and motion blur
- `renderer.py` - Path tracing with Russian roulette
- `bvh.py` - BVH acceleration structure
- `lights.py` - Point, Directional, Area, Sphere lights
- `volumes.py` - Volumetric fog, SSS
- `tonemapping.py` - ACES, Reinhard, Uncharted2

Study these files to understand the algorithms, then implement them efficiently in Metal shaders.

## RALPH Loop Workflow

Use a RALPH loop: **R**ead PROMPT2.md, **A**nalyze current state, **L**ist to-do items in TODO.md, **P**lan next steps, **H**andle one chunk of work, then compact and loop back.

**CRITICAL**: At the start of each loop iteration:
1. Read PROMPT2.md (this file) to stay on track
2. Read TODO.md to see completed work and remaining tasks
3. Check git log for recent commits to understand progress
4. Update TODO.md with current status before proceeding

## Project Structure

Create a new directory structure:

```
spectraforge-metal/
├── src/
│   ├── main.c                 # Entry point, window management
│   ├── renderer.h/.c          # Metal renderer setup
│   ├── scene.h/.c             # Scene data structures
│   ├── bvh.h/.c               # BVH construction (CPU-side)
│   ├── camera.h/.c            # Camera math
│   └── image.h/.c             # Image I/O (PNG export)
├── shaders/
│   ├── types.metal            # Shared types (Ray, HitRecord, Material)
│   ├── intersect.metal        # Intersection functions
│   ├── materials.metal        # Material evaluation (BRDF)
│   ├── pathtracer.metal       # Main path tracing kernel
│   ├── tonemap.metal          # Post-processing
│   └── denoise.metal          # Simple denoiser
├── include/
│   └── spectraforge.h         # Public API
├── tests/
│   └── test_intersect.c       # Unit tests
├── Makefile
├── TODO.md                    # Track progress across loops
└── README.md
```

## Implementation Phases

### Phase 1: Foundation
- [ ] Project setup with Makefile for Apple Silicon
- [ ] Metal device/command queue initialization
- [ ] Basic compute shader pipeline
- [ ] Simple ray generation kernel
- [ ] Sphere intersection in Metal
- [ ] Output to PNG

### Phase 2: Core Path Tracing
- [ ] Full ray-scene intersection (BVH traversal in shader)
- [ ] Material system (Lambertian, Metal, Dielectric)
- [ ] Path tracing with configurable bounces
- [ ] Russian roulette termination
- [ ] Accumulation buffer for progressive rendering

### Phase 3: Advanced Materials
- [ ] PBR/GGX microfacet BRDF
- [ ] Importance sampling for materials
- [ ] Emissive materials
- [ ] Texture sampling (procedural + image)

### Phase 4: Acceleration
- [ ] BVH construction on CPU, traversal on GPU
- [ ] Stackless BVH traversal for GPU efficiency
- [ ] Tiled rendering for better cache utilization
- [ ] Wavefront path tracing (optional)

### Phase 5: Advanced Features
- [ ] Depth of field
- [ ] Motion blur
- [ ] Volumetric rendering (fog, participating media)
- [ ] Environment maps (HDRI)

### Phase 6: Post-Processing
- [ ] Tone mapping (ACES Filmic)
- [ ] Bloom/glow
- [ ] Simple temporal denoising
- [ ] sRGB conversion

### Phase 7: Integration
- [ ] Real-time preview window (Metal + CAMetalLayer)
- [ ] Interactive camera controls
- [ ] Scene file loading (JSON format from Python version)
- [ ] Benchmarking vs Python implementation

## Technical Requirements

### Metal Specifics
- Use `MTLComputePipelineState` for compute shaders
- Use `MTLBuffer` with `storageModeShared` for CPU-GPU data sharing
- Implement proper synchronization with `MTLCommandBuffer`
- Use `threadgroup` memory for local BVH stack
- Target Metal 3.0 features (Apple Silicon)

### Performance Targets
- **Preview**: 1080p @ 1 spp in <100ms (10+ FPS interactive)
- **Quality**: 1080p @ 64 spp in <5 seconds
- **Final**: 4K @ 256 spp in <60 seconds

### Memory Layout
```c
// GPU-friendly structures (aligned, packed)
typedef struct {
    float3 origin;
    float tmin;
    float3 direction;
    float tmax;
} Ray;

typedef struct {
    float3 albedo;
    float metallic;
    float roughness;
    float ior;
    uint32_t type;  // 0=Lambertian, 1=Metal, 2=Dielectric, 3=Emissive
    float emission_intensity;
} Material;

typedef struct {
    float3 center;
    float radius;
    uint32_t material_id;
} Sphere;
```

### BVH Structure
```c
// Linear BVH node for GPU traversal
typedef struct {
    float3 bbox_min;
    uint32_t left_or_first;  // Left child index or first primitive
    float3 bbox_max;
    uint32_t count;          // 0 = internal node, >0 = leaf with count primitives
} BVHNode;
```

## Commit Strategy

Commit after completing each major feature:
```bash
git add .
git commit -m "feat(metal): <description>"
git push origin main
```

Example commits:
- "feat(metal): Initialize Metal compute pipeline"
- "feat(metal): Implement sphere intersection kernel"
- "feat(metal): Add path tracing with 3 bounces"
- "feat(metal): BVH traversal on GPU"

## Testing Strategy

1. **Visual comparison**: Render same scene in Python and Metal, compare output
2. **Unit tests**: Test intersection math against Python reference
3. **Performance benchmarks**: Log rays/second, compare to Python (~4,000/sec baseline)

## Context Preservation

**IMPORTANT**: When the context window compacts:
1. All progress is tracked in TODO.md - read it first
2. Git history shows completed work - check recent commits
3. This file (PROMPT2.md) contains the full specification
4. The Python implementation is the reference for algorithms

Always update TODO.md before ending a session or when completing significant work.

## Getting Started

First loop iteration should:
1. Create the `spectraforge-metal/` directory structure
2. Set up a basic Makefile for macOS/Metal
3. Write minimal Metal initialization code
4. Create a simple test that clears a texture to a color
5. Update TODO.md with Phase 1 tasks

## Performance Notes

Key optimizations for Metal:
- Minimize CPU-GPU synchronization
- Use persistent mapped buffers
- Batch work into large dispatches (e.g., 1920x1080 = ~2M threads)
- Use SIMD group functions for warp-level operations
- Prefer `half` precision where acceptable (faster on Apple Silicon)

## Reference Resources

- Metal Shading Language Specification
- Metal Best Practices Guide
- "Ray Tracing in One Weekend" (algorithm reference)
- PBRT Book (advanced techniques)

---

When compacting, ALWAYS reread PROMPT2.md and TODO.md to maintain context. The goal is a production-quality GPU ray tracer that achieves 1000-10000x speedup over the Python version.
