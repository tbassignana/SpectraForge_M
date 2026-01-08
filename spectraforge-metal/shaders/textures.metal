/**
 * SpectraForge Metal - Procedural Textures
 *
 * Implements procedural texture patterns for materials:
 * - Checker pattern
 * - Gradient noise (Perlin-like)
 * - Marble/turbulence
 */

#include <metal_stdlib>
#include "types.metal"

using namespace metal;

// ============================================================================
// NOISE UTILITIES
// ============================================================================

/**
 * Hash function for noise generation.
 * Returns pseudo-random float in [0,1] from integer input.
 */
inline float hash_float(int3 p) {
    int n = p.x * 73856093 ^ p.y * 19349663 ^ p.z * 83492791;
    n = (n << 13) ^ n;
    return float((n * (n * n * 15731 + 789221) + 1376312589) & 0x7FFFFFFF) / float(0x7FFFFFFF);
}

/**
 * Smooth interpolation (quintic).
 */
inline float3 smooth_step_vec(float3 t) {
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

/**
 * 3D gradient noise (Perlin-like).
 * Returns value in approximately [-1, 1].
 */
inline float gradient_noise(float3 p) {
    int3 i = int3(floor(p));
    float3 f = p - float3(i);

    float3 u = smooth_step_vec(f);

    // Hash corners
    float n000 = hash_float(i + int3(0, 0, 0));
    float n100 = hash_float(i + int3(1, 0, 0));
    float n010 = hash_float(i + int3(0, 1, 0));
    float n110 = hash_float(i + int3(1, 1, 0));
    float n001 = hash_float(i + int3(0, 0, 1));
    float n101 = hash_float(i + int3(1, 0, 1));
    float n011 = hash_float(i + int3(0, 1, 1));
    float n111 = hash_float(i + int3(1, 1, 1));

    // Trilinear interpolation
    float nx00 = mix(n000, n100, u.x);
    float nx10 = mix(n010, n110, u.x);
    float nx01 = mix(n001, n101, u.x);
    float nx11 = mix(n011, n111, u.x);

    float nxy0 = mix(nx00, nx10, u.y);
    float nxy1 = mix(nx01, nx11, u.y);

    return mix(nxy0, nxy1, u.z) * 2.0f - 1.0f;
}

/**
 * Fractal Brownian Motion (fBm) - layered noise.
 */
inline float fbm_noise(float3 p, int octaves, float persistence) {
    float value = 0.0f;
    float amplitude = 1.0f;
    float frequency = 1.0f;
    float max_value = 0.0f;

    for (int i = 0; i < octaves; i++) {
        value += amplitude * gradient_noise(p * frequency);
        max_value += amplitude;
        amplitude *= persistence;
        frequency *= 2.0f;
    }

    return value / max_value;
}

// ============================================================================
// PROCEDURAL PATTERNS
// ============================================================================

/**
 * 3D checker pattern.
 * Returns 0 or 1 based on position.
 */
inline float checker_pattern(float3 p, float scale) {
    float3 scaled = p * scale;
    int3 i = int3(floor(scaled));
    return float((i.x + i.y + i.z) & 1);
}

/**
 * Checker texture returning color.
 */
inline float3 checker_texture(float3 p, float scale, float3 color1, float3 color2) {
    float t = checker_pattern(p, scale);
    return mix(color1, color2, t);
}

/**
 * Gradient/ramp based on height.
 */
inline float3 height_gradient(float y, float3 bottom_color, float3 top_color, float bottom_y, float top_y) {
    float t = clamp((y - bottom_y) / (top_y - bottom_y), 0.0f, 1.0f);
    return mix(bottom_color, top_color, t);
}

/**
 * Marble-like pattern using turbulent noise.
 */
inline float3 marble_texture(float3 p, float scale, float3 color1, float3 color2, int octaves) {
    float noise = fbm_noise(p * scale, octaves, 0.5f);
    float t = sin(p.x * scale + noise * 5.0f) * 0.5f + 0.5f;
    return mix(color1, color2, t);
}

/**
 * Wood grain pattern.
 */
inline float3 wood_texture(float3 p, float scale, float3 light_wood, float3 dark_wood) {
    float dist = sqrt(p.x * p.x + p.z * p.z) * scale;
    float noise = gradient_noise(p * 0.5f) * 0.2f;
    float rings = sin(dist + noise) * 0.5f + 0.5f;
    return mix(light_wood, dark_wood, rings);
}

/**
 * Simple stripe pattern.
 */
inline float3 stripe_texture(float3 p, float scale, float3 color1, float3 color2, int axis) {
    float coord = (axis == 0) ? p.x : ((axis == 1) ? p.y : p.z);
    float t = sin(coord * scale * TWO_PI) * 0.5f + 0.5f;
    return mix(color1, color2, t);
}

/**
 * Voronoi/cellular noise for organic patterns.
 */
inline float voronoi_noise(float3 p) {
    int3 i = int3(floor(p));
    float3 f = p - float3(i);

    float min_dist = 10.0f;

    // Check neighboring cells
    for (int z = -1; z <= 1; z++) {
        for (int y = -1; y <= 1; y++) {
            for (int x = -1; x <= 1; x++) {
                int3 neighbor = int3(x, y, z);
                // Random point in neighboring cell
                float3 point = float3(neighbor) + float3(
                    hash_float(i + neighbor),
                    hash_float(i + neighbor + int3(57, 0, 0)),
                    hash_float(i + neighbor + int3(0, 57, 0))
                );
                float3 diff = point - f;
                float dist = dot(diff, diff);
                min_dist = min(min_dist, dist);
            }
        }
    }

    return sqrt(min_dist);
}

// ============================================================================
// TEXTURE TYPE ENUM AND EVALUATION
// ============================================================================

enum TextureType : uint32_t {
    TEXTURE_SOLID = 0,
    TEXTURE_CHECKER = 1,
    TEXTURE_NOISE = 2,
    TEXTURE_MARBLE = 3,
    TEXTURE_WOOD = 4,
    TEXTURE_STRIPE = 5
};

/**
 * Texture parameters structure.
 * Can be embedded in material or passed separately.
 */
struct TextureParams {
    uint32_t type;
    float scale;
    float3 color1;
    float3 color2;
    int octaves;
    int axis;
};

/**
 * Evaluate procedural texture at point.
 */
inline float3 evaluate_texture(TextureParams tex, float3 point, float2 uv) {
    switch (tex.type) {
        case TEXTURE_CHECKER:
            return checker_texture(point, tex.scale, tex.color1, tex.color2);

        case TEXTURE_NOISE: {
            float n = fbm_noise(point * tex.scale, tex.octaves, 0.5f) * 0.5f + 0.5f;
            return mix(tex.color1, tex.color2, n);
        }

        case TEXTURE_MARBLE:
            return marble_texture(point, tex.scale, tex.color1, tex.color2, tex.octaves);

        case TEXTURE_WOOD:
            return wood_texture(point, tex.scale, tex.color1, tex.color2);

        case TEXTURE_STRIPE:
            return stripe_texture(point, tex.scale, tex.color1, tex.color2, tex.axis);

        case TEXTURE_SOLID:
        default:
            return tex.color1;
    }
}

