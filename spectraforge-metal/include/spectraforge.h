/**
 * SpectraForge Metal - GPU Ray Tracer for Apple Silicon
 *
 * Public API header defining all types and functions for the ray tracer.
 * This header is shared between CPU (C) and GPU (Metal) code.
 */

#ifndef SPECTRAFORGE_H
#define SPECTRAFORGE_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __METAL_VERSION__
    // Metal shading language
    #include <metal_stdlib>
    using namespace metal;
    #define SF_CONSTANT constant
    #define SF_DEVICE device
    #define SF_THREAD thread
#else
    // CPU-side C code
    #define SF_CONSTANT const
    #define SF_DEVICE
    #define SF_THREAD

    // Define float3 for CPU (packed)
    typedef struct __attribute__((packed)) {
        float x, y, z;
    } float3;

    // Helper to create float3
    static inline float3 make_float3(float x, float y, float z) {
        return (float3){x, y, z};
    }
#endif

// ============================================================================
// CORE TYPES - GPU-friendly, 16-byte aligned structures
// ============================================================================

/**
 * Ray structure for GPU traversal.
 * Packed to 32 bytes for efficient memory access.
 */
typedef struct {
    float3 origin;      // 12 bytes
    float tmin;         // 4 bytes  (16 total)
    float3 direction;   // 12 bytes
    float tmax;         // 4 bytes  (32 total)
} Ray;

/**
 * Material types enum.
 */
typedef enum {
    MATERIAL_LAMBERTIAN = 0,
    MATERIAL_METAL = 1,
    MATERIAL_DIELECTRIC = 2,
    MATERIAL_EMISSIVE = 3,
    MATERIAL_PBR = 4
} MaterialType;

/**
 * Material structure for GPU.
 * Contains all properties needed for any material type.
 */
typedef struct {
    float3 albedo;              // 12 bytes - base color
    float metallic;             // 4 bytes  (16 total)

    float roughness;            // 4 bytes
    float ior;                  // 4 bytes - index of refraction
    uint32_t type;              // 4 bytes - MaterialType
    float emission_intensity;   // 4 bytes  (32 total)

    float3 emission_color;      // 12 bytes
    float _padding;             // 4 bytes  (48 total)
} Material;

/**
 * Hit record storing intersection information.
 */
typedef struct {
    float3 point;           // 12 bytes - intersection point
    float t;                // 4 bytes  (16 total)

    float3 normal;          // 12 bytes - surface normal
    uint32_t material_id;   // 4 bytes  (32 total)

    float u, v;             // 8 bytes - texture coordinates
    uint32_t front_face;    // 4 bytes - hit from outside?
    uint32_t hit;           // 4 bytes - did we hit anything? (48 total)
} HitRecord;

/**
 * Sphere primitive for GPU.
 * NOTE: Using individual floats for proper C/Metal struct alignment.
 * Includes velocity for motion blur support.
 */
typedef struct {
    float center_x, center_y, center_z;  // 12 bytes
    float radius;           // 4 bytes  (16 total)
    uint32_t material_id;   // 4 bytes
    float velocity_x, velocity_y, velocity_z;  // 12 bytes - motion blur velocity (32 total)
} Sphere;

/**
 * Triangle primitive for GPU.
 * NOTE: Using individual floats for proper C/Metal struct alignment.
 */
typedef struct {
    float v0_x, v0_y, v0_z;        // 12 bytes
    float _pad0;                    // 4 bytes  (16 total)
    float v1_x, v1_y, v1_z;        // 12 bytes
    float _pad1;                    // 4 bytes  (32 total)
    float v2_x, v2_y, v2_z;        // 12 bytes
    float _pad2;                    // 4 bytes  (48 total)
    float normal_x, normal_y, normal_z;  // 12 bytes - precomputed normal
    uint32_t material_id;          // 4 bytes  (64 total)
} Triangle;

/**
 * Axis-Aligned Bounding Box for BVH.
 */
typedef struct {
    float3 min;             // 12 bytes
    float _pad0;            // 4 bytes  (16 total)
    float3 max;             // 12 bytes
    float _pad1;            // 4 bytes  (32 total)
} AABB;

/**
 * BVH Node for GPU traversal.
 * Uses a linear/flat BVH layout for efficient GPU access.
 * NOTE: Using float4 for proper Metal alignment (float3 is 16-byte aligned in Metal structs).
 */
typedef struct {
    float bbox_min_x, bbox_min_y, bbox_min_z;  // 12 bytes
    uint32_t left_or_first;     // 4 bytes - left child index OR first primitive (16 total)

    float bbox_max_x, bbox_max_y, bbox_max_z;  // 12 bytes
    uint32_t count;             // 4 bytes - 0 = internal, >0 = leaf with count prims (32 total)
} BVHNode;

/**
 * Camera parameters for GPU.
 * NOTE: Using individual floats for proper C/Metal struct alignment.
 */
typedef struct {
    float origin_x, origin_y, origin_z;           // 12 bytes
    float lens_radius;                             // 4 bytes  (16 total)

    float lower_left_x, lower_left_y, lower_left_z;  // 12 bytes
    float focus_dist;                              // 4 bytes  (32 total)

    float horizontal_x, horizontal_y, horizontal_z;  // 12 bytes
    float _pad0;                                   // 4 bytes  (48 total)

    float vertical_x, vertical_y, vertical_z;     // 12 bytes
    float _pad1;                                   // 4 bytes  (64 total)

    float u_x, u_y, u_z;                          // 12 bytes
    float _pad2;                                   // 4 bytes  (80 total)

    float v_x, v_y, v_z;                          // 12 bytes
    float _pad3;                                   // 4 bytes  (96 total)

    float w_x, w_y, w_z;                          // 12 bytes
    float _pad4;                                   // 4 bytes  (112 total)

    float time0, time1;                           // 8 bytes
    float _pad5[2];                               // 8 bytes  (128 total)
} Camera;

/**
 * Render settings passed to GPU.
 * NOTE: Using individual floats for proper C/Metal struct alignment.
 */
typedef struct {
    uint32_t width;             // 4 bytes
    uint32_t height;            // 4 bytes
    uint32_t samples_per_pixel; // 4 bytes
    uint32_t max_depth;         // 4 bytes  (16 total)

    uint32_t frame_number;      // For accumulation
    uint32_t num_spheres;       // Number of spheres in scene
    uint32_t num_triangles;     // Number of triangles
    uint32_t num_bvh_nodes;     // Number of BVH nodes (32 total)

    float background_r, background_g, background_b;  // 12 bytes
    uint32_t use_sky_gradient;  // Use gradient sky? (48 total)

    uint32_t num_primitive_indices;  // For combined BVH (>0 = use combined)
    uint32_t _pad[3];           // Pad to 64 bytes
} RenderSettings;

// ============================================================================
// CPU-SIDE API (only available in C, not Metal)
// ============================================================================

#ifndef __METAL_VERSION__

/**
 * Opaque handle to the Metal renderer.
 */
typedef struct MetalRenderer MetalRenderer;

/**
 * Scene data structure for CPU-side management.
 */
typedef struct {
    Sphere* spheres;
    uint32_t num_spheres;
    uint32_t capacity_spheres;

    Triangle* triangles;
    uint32_t num_triangles;
    uint32_t capacity_triangles;

    Material* materials;
    uint32_t num_materials;
    uint32_t capacity_materials;

    BVHNode* bvh_nodes;
    uint32_t num_bvh_nodes;

    // Primitive index buffer for combined BVH (encoded: high bit = triangle)
    uint32_t* primitive_indices;
    uint32_t num_primitive_indices;
} Scene;

/**
 * Initialize the Metal renderer.
 * @return Renderer handle, or NULL on failure.
 */
MetalRenderer* sf_renderer_create(void);

/**
 * Destroy the Metal renderer and free resources.
 */
void sf_renderer_destroy(MetalRenderer* renderer);

/**
 * Set render settings.
 */
void sf_renderer_set_settings(MetalRenderer* renderer, const RenderSettings* settings);

/**
 * Set camera parameters.
 */
void sf_renderer_set_camera(MetalRenderer* renderer, const Camera* camera);

/**
 * Upload scene data to GPU.
 */
void sf_renderer_upload_scene(MetalRenderer* renderer, const Scene* scene);

/**
 * Render a frame and store result in output buffer.
 * @param output RGB float buffer of size width*height*3
 */
void sf_renderer_render(MetalRenderer* renderer, float* output);

/**
 * Get the accumulated image (for progressive rendering).
 */
void sf_renderer_get_accumulation(MetalRenderer* renderer, float* output);

/**
 * Reset accumulation buffer (call when camera/scene changes).
 */
void sf_renderer_reset_accumulation(MetalRenderer* renderer);

/**
 * Get last render time in milliseconds.
 */
double sf_renderer_get_render_time(MetalRenderer* renderer);

/**
 * Get total rays traced across all frames.
 */
uint64_t sf_renderer_get_total_rays(MetalRenderer* renderer);

// Scene management helpers
Scene* sf_scene_create(void);
void sf_scene_destroy(Scene* scene);
uint32_t sf_scene_add_sphere(Scene* scene, float3 center, float radius, uint32_t material_id);
uint32_t sf_scene_add_sphere_moving(Scene* scene, float3 center, float radius, float3 velocity, uint32_t material_id);
uint32_t sf_scene_add_triangle(Scene* scene, float3 v0, float3 v1, float3 v2, uint32_t material_id);
uint32_t sf_scene_add_material(Scene* scene, const Material* material);
void sf_scene_build_bvh(Scene* scene);
void sf_scene_build_bvh_full(Scene* scene);  // Combined BVH for spheres + triangles

// Camera helpers
Camera sf_camera_create(
    float3 look_from,
    float3 look_at,
    float3 vup,
    float vfov,         // Vertical field of view in degrees
    float aspect_ratio,
    float aperture,
    float focus_dist
);

// Image I/O
int sf_save_png(const char* filename, const float* rgb_data, uint32_t width, uint32_t height);
int sf_save_hdr(const char* filename, const float* rgb_data, uint32_t width, uint32_t height);

// Scene loading
int sf_scene_load_json(Scene* scene, const char* filename,
                       Camera* out_camera, RenderSettings* out_settings);

// Real-time preview
int sf_preview_run(MetalRenderer* renderer, Scene* scene, uint32_t width, uint32_t height);

#endif // __METAL_VERSION__

#endif // SPECTRAFORGE_H
