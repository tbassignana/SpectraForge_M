/**
 * SpectraForge Metal - Shared Types for GPU Shaders
 *
 * This file defines all structures used by the path tracing kernels.
 * Structures are designed for optimal GPU memory access patterns.
 */

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// CONSTANTS
// ============================================================================

constant float PI = 3.14159265358979323846f;
constant float TWO_PI = 6.28318530717958647693f;
constant float INV_PI = 0.31830988618379067154f;
constant float EPSILON = 1e-6f;
constant float INFINITY_F = 1e30f;

// ============================================================================
// MATERIAL TYPES
// ============================================================================

enum MaterialType : uint32_t {
    MATERIAL_LAMBERTIAN = 0,
    MATERIAL_METAL = 1,
    MATERIAL_DIELECTRIC = 2,
    MATERIAL_EMISSIVE = 3,
    MATERIAL_PBR = 4
};

// ============================================================================
// CORE STRUCTURES
// ============================================================================

/**
 * Ray structure - 32 bytes, cache-friendly layout.
 */
struct Ray {
    float3 origin;
    float tmin;
    float3 direction;
    float tmax;
};

/**
 * Material properties - 48 bytes.
 */
struct Material {
    float3 albedo;
    float metallic;

    float roughness;
    float ior;
    uint32_t type;
    float emission_intensity;

    float3 emission_color;
    float _padding;
};

/**
 * Hit record for intersection results - 48 bytes.
 */
struct HitRecord {
    float3 point;
    float t;

    float3 normal;
    uint32_t material_id;

    float u;
    float v;
    uint32_t front_face;
    uint32_t hit;
};

/**
 * Sphere primitive - 32 bytes.
 * Includes velocity for motion blur.
 */
struct Sphere {
    float center_x, center_y, center_z;  // 12 bytes
    float radius;                         // 4 bytes (16 total)
    uint32_t material_id;                 // 4 bytes
    float velocity_x, velocity_y, velocity_z;  // 12 bytes - motion blur (32 total)
};

/**
 * Triangle primitive - 64 bytes.
 */
struct Triangle {
    float v0_x, v0_y, v0_z;        // 12 bytes
    float _pad0;                    // 4 bytes (16 total)
    float v1_x, v1_y, v1_z;        // 12 bytes
    float _pad1;                    // 4 bytes (32 total)
    float v2_x, v2_y, v2_z;        // 12 bytes
    float _pad2;                    // 4 bytes (48 total)
    float normal_x, normal_y, normal_z;  // 12 bytes
    uint32_t material_id;          // 4 bytes (64 total)
};

/**
 * AABB for BVH - 32 bytes.
 */
struct AABB {
    float3 min_bound;
    float _pad0;
    float3 max_bound;
    float _pad1;
};

/**
 * BVH Node - 32 bytes.
 * Uses compact encoding: high bit of count indicates interior vs leaf.
 * NOTE: Using individual floats for proper C/Metal struct alignment.
 */
struct BVHNode {
    float bbox_min_x, bbox_min_y, bbox_min_z;  // 12 bytes
    uint32_t left_or_first;  // Left child index or first primitive index (16 total)

    float bbox_max_x, bbox_max_y, bbox_max_z;  // 12 bytes
    uint32_t count;          // high bit = interior, low bits = right child or prim count (32 total)
};

/**
 * Camera parameters - 128 bytes.
 */
struct Camera {
    float origin_x, origin_y, origin_z;        // 12 bytes
    float lens_radius;                          // 4 bytes (16 total)

    float lower_left_x, lower_left_y, lower_left_z;  // 12 bytes
    float focus_dist;                           // 4 bytes (32 total)

    float horizontal_x, horizontal_y, horizontal_z;  // 12 bytes
    float _pad0;                                // 4 bytes (48 total)

    float vertical_x, vertical_y, vertical_z;  // 12 bytes
    float _pad1;                                // 4 bytes (64 total)

    float u_x, u_y, u_z;                       // 12 bytes
    float _pad2;                                // 4 bytes (80 total)

    float v_x, v_y, v_z;                       // 12 bytes
    float _pad3;                                // 4 bytes (96 total)

    float w_x, w_y, w_z;                       // 12 bytes
    float _pad4;                                // 4 bytes (112 total)

    float time0;
    float time1;
    float2 _pad5;
};

/**
 * Render settings - 64 bytes.
 */
struct RenderSettings {
    uint32_t width;
    uint32_t height;
    uint32_t samples_per_pixel;
    uint32_t max_depth;

    uint32_t frame_number;
    uint32_t num_spheres;
    uint32_t num_triangles;
    uint32_t num_bvh_nodes;

    float background_r, background_g, background_b;  // 12 bytes
    uint32_t use_sky_gradient;  // 4 bytes (48 total)

    uint32_t num_primitive_indices;  // For combined BVH (>0 = use combined)
    uint32_t _pad[3];           // Pad to 64 bytes
};

// ============================================================================
// RANDOM NUMBER GENERATION
// ============================================================================

/**
 * PCG random number generator state.
 * High-quality, fast PRNG suitable for GPU use.
 */
struct RNG {
    uint64_t state;
    uint64_t inc;
};

/**
 * Initialize RNG with pixel coordinates and frame number for unique seeds.
 */
inline RNG rng_init(uint2 pixel, uint32_t frame, uint32_t sample_idx) {
    RNG rng;
    uint64_t seed = (uint64_t(pixel.x) * 1973u + uint64_t(pixel.y) * 9277u +
                     uint64_t(frame) * 26699u + uint64_t(sample_idx) * 13u);
    rng.state = seed;
    rng.inc = (seed << 1u) | 1u;
    return rng;
}

/**
 * Generate next random uint32.
 */
inline uint32_t rng_next_uint(thread RNG& rng) {
    uint64_t oldstate = rng.state;
    rng.state = oldstate * 6364136223846793005ULL + rng.inc;
    uint32_t xorshifted = uint32_t(((oldstate >> 18u) ^ oldstate) >> 27u);
    uint32_t rot = uint32_t(oldstate >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31u));
}

/**
 * Generate random float in [0, 1).
 */
inline float rng_next_float(thread RNG& rng) {
    return float(rng_next_uint(rng)) / float(0xFFFFFFFFu);
}

/**
 * Generate random float in [min, max).
 */
inline float rng_next_float_range(thread RNG& rng, float min_val, float max_val) {
    return min_val + (max_val - min_val) * rng_next_float(rng);
}

/**
 * Generate random point in unit sphere (rejection sampling).
 */
inline float3 random_in_unit_sphere(thread RNG& rng) {
    float3 p;
    do {
        p = float3(
            rng_next_float_range(rng, -1.0f, 1.0f),
            rng_next_float_range(rng, -1.0f, 1.0f),
            rng_next_float_range(rng, -1.0f, 1.0f)
        );
    } while (length_squared(p) >= 1.0f);
    return p;
}

/**
 * Generate random unit vector (uniform on sphere surface).
 */
inline float3 random_unit_vector(thread RNG& rng) {
    return normalize(random_in_unit_sphere(rng));
}

/**
 * Generate random point in unit disk.
 */
inline float2 random_in_unit_disk(thread RNG& rng) {
    float2 p;
    do {
        p = float2(
            rng_next_float_range(rng, -1.0f, 1.0f),
            rng_next_float_range(rng, -1.0f, 1.0f)
        );
    } while (length_squared(p) >= 1.0f);
    return p;
}

/**
 * Cosine-weighted hemisphere sampling (importance sampling for diffuse).
 */
inline float3 random_cosine_direction(thread RNG& rng) {
    float r1 = rng_next_float(rng);
    float r2 = rng_next_float(rng);

    float phi = TWO_PI * r1;
    float sqrt_r2 = sqrt(r2);

    float x = cos(phi) * sqrt_r2;
    float y = sin(phi) * sqrt_r2;
    float z = sqrt(1.0f - r2);

    return float3(x, y, z);
}

// ============================================================================
// VECTOR UTILITIES
// ============================================================================

/**
 * Reflect vector v around normal n.
 * Note: Using sf_ prefix to avoid conflict with Metal's built-in reflect()
 */
inline float3 sf_reflect(float3 v, float3 n) {
    return v - 2.0f * dot(v, n) * n;
}

/**
 * Refract vector v through surface with normal n and ratio eta.
 * Returns zero vector if total internal reflection.
 * Note: Using sf_ prefix to avoid conflict with Metal's built-in refract()
 */
inline float3 sf_refract(float3 v, float3 n, float eta) {
    float cos_theta = min(dot(-v, n), 1.0f);
    float3 r_out_perp = eta * (v + cos_theta * n);
    float perp_len_sq = length_squared(r_out_perp);

    if (perp_len_sq > 1.0f) {
        return float3(0.0f);  // Total internal reflection
    }

    float3 r_out_parallel = -sqrt(abs(1.0f - perp_len_sq)) * n;
    return r_out_perp + r_out_parallel;
}

/**
 * Schlick's approximation for Fresnel reflectance.
 */
inline float schlick_reflectance(float cosine, float ref_idx) {
    float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * pow(1.0f - cosine, 5.0f);
}

/**
 * Check if vector is near zero.
 */
inline bool near_zero(float3 v) {
    return (abs(v.x) < EPSILON) && (abs(v.y) < EPSILON) && (abs(v.z) < EPSILON);
}

/**
 * Build orthonormal basis from normal.
 * Returns tangent and bitangent vectors.
 */
inline void build_onb(float3 n, thread float3& tangent, thread float3& bitangent) {
    float3 up = abs(n.y) < 0.999f ? float3(0.0f, 1.0f, 0.0f) : float3(1.0f, 0.0f, 0.0f);
    tangent = normalize(cross(up, n));
    bitangent = cross(n, tangent);
}

/**
 * Transform direction from local (tangent space) to world space.
 */
inline float3 local_to_world(float3 local_dir, float3 normal, float3 tangent, float3 bitangent) {
    return tangent * local_dir.x + bitangent * local_dir.y + normal * local_dir.z;
}
