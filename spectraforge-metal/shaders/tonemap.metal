/**
 * SpectraForge Metal - Tone Mapping and Post-Processing
 *
 * Implements HDR to LDR conversion with various tone mapping operators:
 * - ACES Filmic
 * - Reinhard
 * - Uncharted 2
 * - Simple exposure/gamma
 */

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// CONSTANTS
// ============================================================================

constant float GAMMA = 2.2f;
constant float INV_GAMMA = 1.0f / 2.2f;

// ============================================================================
// TONE MAPPING OPERATORS
// ============================================================================

/**
 * Simple Reinhard tone mapping.
 * L_out = L_in / (1 + L_in)
 */
inline float3 tonemap_reinhard(float3 color) {
    return color / (color + float3(1.0f));
}

/**
 * Extended Reinhard with white point.
 */
inline float3 tonemap_reinhard_extended(float3 color, float white_point) {
    float3 numerator = color * (1.0f + color / (white_point * white_point));
    return numerator / (1.0f + color);
}

/**
 * ACES Filmic tone mapping (approximate).
 * Industry standard, gives nice S-curve with pleasant roll-off.
 */
inline float3 tonemap_aces(float3 color) {
    // Fitted curve by Krzysztof Narkowicz
    float a = 2.51f;
    float b = 0.03f;
    float c = 2.43f;
    float d = 0.59f;
    float e = 0.14f;

    return clamp((color * (a * color + b)) / (color * (c * color + d) + e), 0.0f, 1.0f);
}

/**
 * Full ACES RRT+ODT approximation.
 * More accurate but more expensive.
 */
inline float3 tonemap_aces_full(float3 color) {
    // Input transform (convert from sRGB to ACES AP1)
    float3x3 input_mat = float3x3(
        float3(0.59719f, 0.07600f, 0.02840f),
        float3(0.35458f, 0.90834f, 0.13383f),
        float3(0.04823f, 0.01566f, 0.83777f)
    );

    // Output transform (ACES AP1 to sRGB)
    float3x3 output_mat = float3x3(
        float3(1.60475f, -0.10208f, -0.00327f),
        float3(-0.53108f, 1.10813f, -0.07276f),
        float3(-0.07367f, -0.00605f, 1.07602f)
    );

    color = input_mat * color;

    // RRT and ODT fit
    float3 a = color * (color + 0.0245786f) - 0.000090537f;
    float3 b = color * (0.983729f * color + 0.4329510f) + 0.238081f;
    color = a / b;

    color = output_mat * color;

    return clamp(color, 0.0f, 1.0f);
}

/**
 * Uncharted 2 filmic tone mapping.
 * Used in many games, good highlight compression.
 */
inline float3 uncharted2_partial(float3 x) {
    float A = 0.15f;
    float B = 0.50f;
    float C = 0.10f;
    float D = 0.20f;
    float E = 0.02f;
    float F = 0.30f;
    return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}

inline float3 tonemap_uncharted2(float3 color) {
    float exposure_bias = 2.0f;
    float3 curr = uncharted2_partial(color * exposure_bias);

    float W = 11.2f;  // White point
    float3 white_scale = float3(1.0f) / uncharted2_partial(float3(W));

    return curr * white_scale;
}

/**
 * Simple exposure adjustment.
 */
inline float3 apply_exposure(float3 color, float exposure) {
    return color * pow(2.0f, exposure);
}

/**
 * Gamma correction (linear to sRGB).
 */
inline float3 linear_to_srgb(float3 linear) {
    // Simple gamma
    return pow(max(linear, float3(0.0f)), float3(INV_GAMMA));
}

/**
 * Inverse gamma correction (sRGB to linear).
 */
inline float3 srgb_to_linear(float3 srgb) {
    return pow(max(srgb, float3(0.0f)), float3(GAMMA));
}

/**
 * Accurate sRGB conversion with linear segment.
 */
inline float3 linear_to_srgb_accurate(float3 linear) {
    float3 result;
    result.x = linear.x <= 0.0031308f ? linear.x * 12.92f : 1.055f * pow(linear.x, 1.0f/2.4f) - 0.055f;
    result.y = linear.y <= 0.0031308f ? linear.y * 12.92f : 1.055f * pow(linear.y, 1.0f/2.4f) - 0.055f;
    result.z = linear.z <= 0.0031308f ? linear.z * 12.92f : 1.055f * pow(linear.z, 1.0f/2.4f) - 0.055f;
    return result;
}

// ============================================================================
// POST-PROCESSING EFFECTS
// ============================================================================

/**
 * Simple vignette effect.
 */
inline float3 apply_vignette(float3 color, float2 uv, float strength) {
    float2 center = float2(0.5f);
    float dist = length(uv - center) * 1.414f;  // Normalize to [0,1] at corners
    float vignette = 1.0f - strength * dist * dist;
    return color * vignette;
}

/**
 * Contrast adjustment.
 */
inline float3 apply_contrast(float3 color, float contrast) {
    return (color - 0.5f) * contrast + 0.5f;
}

/**
 * Saturation adjustment.
 */
inline float3 apply_saturation(float3 color, float saturation) {
    float gray = dot(color, float3(0.2126f, 0.7152f, 0.0722f));
    return mix(float3(gray), color, saturation);
}

// ============================================================================
// TONE MAPPING KERNELS
// ============================================================================

/**
 * Tone mapping compute kernel.
 * Converts HDR input to LDR output with selected operator.
 */
kernel void tonemap_kernel(
    device float4* output                  [[buffer(0)]],
    device const float4* input             [[buffer(1)]],
    constant uint32_t& width               [[buffer(2)]],
    constant uint32_t& height              [[buffer(3)]],
    constant uint32_t& tonemap_operator    [[buffer(4)]],   // 0=ACES, 1=Reinhard, 2=Uncharted2, 3=None
    constant float& exposure               [[buffer(5)]],
    constant float& vignette_strength      [[buffer(6)]],
    constant float& contrast               [[buffer(7)]],
    constant float& saturation             [[buffer(8)]],
    uint2 gid                              [[thread_position_in_grid]]
) {
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    uint32_t idx = gid.y * width + gid.x;
    float3 color = input[idx].xyz;

    // Apply exposure
    color = apply_exposure(color, exposure);

    // Tone mapping
    switch (tonemap_operator) {
        case 0: color = tonemap_aces(color); break;
        case 1: color = tonemap_reinhard(color); break;
        case 2: color = tonemap_uncharted2(color); break;
        case 3: color = clamp(color, 0.0f, 1.0f); break;  // No tone mapping
    }

    // Post-processing
    float2 uv = float2(float(gid.x) / float(width - 1),
                       float(gid.y) / float(height - 1));

    if (vignette_strength > 0.0f) {
        color = apply_vignette(color, uv, vignette_strength);
    }

    if (abs(contrast - 1.0f) > 0.001f) {
        color = apply_contrast(color, contrast);
    }

    if (abs(saturation - 1.0f) > 0.001f) {
        color = apply_saturation(color, saturation);
    }

    // Gamma correction (linear to sRGB)
    color = linear_to_srgb(color);

    // Clamp final output
    color = clamp(color, 0.0f, 1.0f);

    output[idx] = float4(color, 1.0f);
}

/**
 * Simple tone mapping kernel with just ACES and gamma.
 */
kernel void tonemap_simple_kernel(
    device float4* output                  [[buffer(0)]],
    device const float4* input             [[buffer(1)]],
    constant uint32_t& width               [[buffer(2)]],
    constant uint32_t& height              [[buffer(3)]],
    constant float& exposure               [[buffer(4)]],
    uint2 gid                              [[thread_position_in_grid]]
) {
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    uint32_t idx = gid.y * width + gid.x;
    float3 color = input[idx].xyz;

    // Exposure
    color = apply_exposure(color, exposure);

    // ACES tone mapping
    color = tonemap_aces(color);

    // Gamma correction
    color = linear_to_srgb(color);

    output[idx] = float4(color, 1.0f);
}

/**
 * Copy kernel - no processing, just copy HDR values.
 * Useful for outputting to HDR formats.
 */
kernel void copy_kernel(
    device float4* output                  [[buffer(0)]],
    device const float4* input             [[buffer(1)]],
    constant uint32_t& width               [[buffer(2)]],
    constant uint32_t& height              [[buffer(3)]],
    uint2 gid                              [[thread_position_in_grid]]
) {
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    uint32_t idx = gid.y * width + gid.x;
    output[idx] = input[idx];
}

// ============================================================================
// BLOOM / GLOW KERNELS
// ============================================================================

// Gaussian weights for 13-tap blur (sigma ~= 4)
constant float BLUR_WEIGHTS[7] = {
    0.1964825501511404f,   // center
    0.1748840605498416f,
    0.1209853622595717f,
    0.0649730039917015f,
    0.0270518468663093f,
    0.0087266459239964f,
    0.0021874584242281f
};

/**
 * Extract bright pixels above threshold for bloom.
 * Outputs only the bright parts of the image.
 */
kernel void bloom_threshold_kernel(
    device float4* output                  [[buffer(0)]],
    device const float4* input             [[buffer(1)]],
    constant uint32_t& width               [[buffer(2)]],
    constant uint32_t& height              [[buffer(3)]],
    constant float& threshold              [[buffer(4)]],
    constant float& soft_threshold         [[buffer(5)]],  // Soft knee width
    uint2 gid                              [[thread_position_in_grid]]
) {
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    uint32_t idx = gid.y * width + gid.x;
    float3 color = input[idx].xyz;

    // Compute luminance
    float luma = dot(color, float3(0.2126f, 0.7152f, 0.0722f));

    // Soft threshold (smooth knee)
    float knee = threshold * soft_threshold;
    float soft = luma - threshold + knee;
    soft = clamp(soft, 0.0f, 2.0f * knee);
    soft = soft * soft / (4.0f * knee + 0.00001f);

    // Contribution factor
    float contribution = max(soft, luma - threshold) / max(luma, 0.00001f);
    contribution = clamp(contribution, 0.0f, 1.0f);

    output[idx] = float4(color * contribution, 1.0f);
}

/**
 * Horizontal Gaussian blur pass.
 * Uses separable 13-tap filter.
 */
kernel void bloom_blur_h_kernel(
    device float4* output                  [[buffer(0)]],
    device const float4* input             [[buffer(1)]],
    constant uint32_t& width               [[buffer(2)]],
    constant uint32_t& height              [[buffer(3)]],
    uint2 gid                              [[thread_position_in_grid]]
) {
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    float3 result = float3(0.0f);
    int y = int(gid.y);

    // Center sample
    result += input[y * width + gid.x].xyz * BLUR_WEIGHTS[0];

    // Symmetric samples
    for (int i = 1; i < 7; i++) {
        int x_left = max(0, int(gid.x) - i);
        int x_right = min(int(width) - 1, int(gid.x) + i);

        result += input[y * width + x_left].xyz * BLUR_WEIGHTS[i];
        result += input[y * width + x_right].xyz * BLUR_WEIGHTS[i];
    }

    output[gid.y * width + gid.x] = float4(result, 1.0f);
}

/**
 * Vertical Gaussian blur pass.
 * Uses separable 13-tap filter.
 */
kernel void bloom_blur_v_kernel(
    device float4* output                  [[buffer(0)]],
    device const float4* input             [[buffer(1)]],
    constant uint32_t& width               [[buffer(2)]],
    constant uint32_t& height              [[buffer(3)]],
    uint2 gid                              [[thread_position_in_grid]]
) {
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    float3 result = float3(0.0f);
    int x = int(gid.x);

    // Center sample
    result += input[gid.y * width + x].xyz * BLUR_WEIGHTS[0];

    // Symmetric samples
    for (int i = 1; i < 7; i++) {
        int y_up = max(0, int(gid.y) - i);
        int y_down = min(int(height) - 1, int(gid.y) + i);

        result += input[y_up * width + x].xyz * BLUR_WEIGHTS[i];
        result += input[y_down * width + x].xyz * BLUR_WEIGHTS[i];
    }

    output[gid.y * width + gid.x] = float4(result, 1.0f);
}

/**
 * Combine original image with bloom.
 * Applies tone mapping and gamma correction.
 */
kernel void bloom_combine_kernel(
    device float4* output                  [[buffer(0)]],
    device const float4* original          [[buffer(1)]],
    device const float4* bloom             [[buffer(2)]],
    constant uint32_t& width               [[buffer(3)]],
    constant uint32_t& height              [[buffer(4)]],
    constant float& bloom_intensity        [[buffer(5)]],
    constant float& exposure               [[buffer(6)]],
    uint2 gid                              [[thread_position_in_grid]]
) {
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    uint32_t idx = gid.y * width + gid.x;

    float3 color = original[idx].xyz;
    float3 bloom_color = bloom[idx].xyz;

    // Add bloom
    color += bloom_color * bloom_intensity;

    // Apply exposure
    color = apply_exposure(color, exposure);

    // Tone mapping
    color = tonemap_aces(color);

    // Gamma correction
    color = linear_to_srgb(color);

    output[idx] = float4(color, 1.0f);
}
