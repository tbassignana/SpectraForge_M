/**
 * SpectraForge Metal - Path Tracing Kernel
 *
 * Main compute shader that performs path tracing with:
 * - Progressive accumulation
 * - Russian roulette termination
 * - Multiple importance sampling ready
 */

#include <metal_stdlib>
#include "types.metal"
#include "intersect.metal"
#include "materials.metal"

using namespace metal;

// ============================================================================
// RAY GENERATION
// ============================================================================

/**
 * Generate a camera ray for the given pixel coordinates.
 * Includes depth of field support via lens sampling.
 */
inline Ray generate_camera_ray(
    float2 uv,              // Normalized [0,1] pixel coordinates
    Camera camera,
    thread RNG& rng
) {
    Ray ray;

    // Construct float3s from individual components
    float3 origin = float3(camera.origin_x, camera.origin_y, camera.origin_z);
    float3 lower_left = float3(camera.lower_left_x, camera.lower_left_y, camera.lower_left_z);
    float3 horizontal = float3(camera.horizontal_x, camera.horizontal_y, camera.horizontal_z);
    float3 vertical = float3(camera.vertical_x, camera.vertical_y, camera.vertical_z);
    float3 u = float3(camera.u_x, camera.u_y, camera.u_z);
    float3 v = float3(camera.v_x, camera.v_y, camera.v_z);

    // Depth of field - sample point on lens
    float2 lens_sample = camera.lens_radius > 0.0f
        ? camera.lens_radius * random_in_unit_disk(rng)
        : float2(0.0f);

    float3 offset = u * lens_sample.x + v * lens_sample.y;

    float3 pixel_target = lower_left + uv.x * horizontal + uv.y * vertical;

    ray.origin = origin + offset;
    ray.direction = normalize(pixel_target - origin - offset);
    ray.tmin = EPSILON;
    ray.tmax = INFINITY_F;

    return ray;
}

// ============================================================================
// BACKGROUND / SKY COLOR
// ============================================================================

// Sun direction (normalized) - can be made configurable later
constant float3 SUN_DIRECTION = float3(0.5f, 0.6f, -0.3f);
constant float3 SUN_COLOR = float3(1.0f, 0.95f, 0.85f);
constant float SUN_INTENSITY = 20.0f;
constant float SUN_SIZE = 0.01f;  // Angular radius (cosine)

/**
 * Simple sky gradient color based on ray direction.
 * Used when use_sky_gradient is enabled.
 */
inline float3 sky_color_simple(float3 direction) {
    float3 unit_dir = normalize(direction);
    float t = 0.5f * (unit_dir.y + 1.0f);
    float3 white = float3(1.0f, 1.0f, 1.0f);
    float3 blue = float3(0.5f, 0.7f, 1.0f);
    return mix(white, blue, t);
}

/**
 * Atmospheric sky with procedural sun.
 * Simulates basic Rayleigh scattering for blue sky and red sunset.
 */
inline float3 sky_color_atmospheric(float3 direction) {
    float3 unit_dir = normalize(direction);
    float3 sun_dir = normalize(SUN_DIRECTION);

    // Base sky color with altitude-based gradient
    float altitude = unit_dir.y;

    // Horizon color (warm white/orange near horizon)
    float3 horizon_color = float3(0.9f, 0.85f, 0.75f);

    // Zenith color (deep blue at top)
    float3 zenith_color = float3(0.3f, 0.5f, 0.9f);

    // Ground color (when looking below horizon)
    float3 ground_color = float3(0.3f, 0.25f, 0.2f);

    float3 sky;
    if (altitude > 0.0f) {
        // Above horizon: blend from horizon to zenith
        float t = pow(altitude, 0.4f);  // Non-linear for more realistic look
        sky = mix(horizon_color, zenith_color, t);
    } else {
        // Below horizon: dark ground
        sky = ground_color * (1.0f + altitude * 2.0f);  // Darken toward nadir
    }

    // Sun disc
    float sun_dot = dot(unit_dir, sun_dir);
    if (sun_dot > 1.0f - SUN_SIZE) {
        // Inside sun disc - bright sun color
        float sun_t = (sun_dot - (1.0f - SUN_SIZE)) / SUN_SIZE;
        sun_t = clamp(sun_t, 0.0f, 1.0f);
        sky = mix(sky, SUN_COLOR * SUN_INTENSITY, sun_t);
    } else if (sun_dot > 1.0f - SUN_SIZE * 4.0f) {
        // Sun glow/corona
        float glow_t = (sun_dot - (1.0f - SUN_SIZE * 4.0f)) / (SUN_SIZE * 3.0f);
        glow_t = clamp(glow_t * glow_t, 0.0f, 1.0f);
        sky += SUN_COLOR * glow_t * 2.0f;
    }

    return sky;
}

/**
 * Get sky color based on ray direction.
 * Wrapper that can switch between simple and atmospheric sky.
 */
inline float3 sky_color(float3 direction) {
    // Use atmospheric sky for better lighting
    return sky_color_atmospheric(direction);
}

// ============================================================================
// PATH TRACING CORE
// ============================================================================

/**
 * Trace a single ray through the scene, accumulating color.
 *
 * Uses iterative loop instead of recursion (more GPU-friendly).
 * Implements Russian roulette for unbiased early termination.
 */
inline float3 trace_ray(
    Ray ray,
    device const Sphere* spheres,
    uint32_t num_spheres,
    device const Triangle* triangles,
    uint32_t num_triangles,
    device const Material* materials,
    device const BVHNode* bvh_nodes,
    uint32_t num_bvh_nodes,
    device const uint32_t* primitive_indices,
    uint32_t num_primitive_indices,
    RenderSettings settings,
    float time,  // Motion blur time sample
    thread RNG& rng
) {
    float3 accumulated_color = float3(0.0f);
    float3 throughput = float3(1.0f);

    Ray current_ray = ray;

    for (uint32_t depth = 0; depth < settings.max_depth; depth++) {
        HitRecord rec;
        rec.hit = 0;

        // Find intersection
        // Note: BVH paths currently don't support motion blur (requires expanded bounds)
        // Motion blur works with brute force or simple_render_kernel
        bool hit;
        if (num_primitive_indices > 0) {
            // Combined BVH with primitive indices (spheres + triangles)
            hit = traverse_bvh_combined(current_ray, bvh_nodes, spheres, triangles,
                                       primitive_indices, num_bvh_nodes,
                                       current_ray.tmin, current_ray.tmax, rec);
        } else if (num_bvh_nodes > 0) {
            // Sphere-only BVH with brute force triangles
            hit = traverse_bvh(current_ray, bvh_nodes, spheres, triangles,
                             num_bvh_nodes, num_triangles,
                             current_ray.tmin, current_ray.tmax, rec);
        } else {
            // No BVH - brute force with motion blur support
            hit = hit_scene_motion(current_ray, spheres, num_spheres, triangles, num_triangles,
                                  time, current_ray.tmin, current_ray.tmax, rec);
        }

        if (!hit) {
            // No hit - add background color
            float3 bg_color;
            if (settings.use_sky_gradient) {
                bg_color = sky_color(current_ray.direction);
            } else {
                bg_color = float3(settings.background_r, settings.background_g, settings.background_b);
            }
            accumulated_color += throughput * bg_color;
            break;
        }

        // Get material and apply procedural textures
        Material mat = materials[rec.material_id];
        mat = apply_procedural_texture(mat, rec);

        // Add emission
        float3 emission = get_emission(mat);
        accumulated_color += throughput * emission;

        // Scatter ray
        ScatterResult scatter = scatter_material(current_ray, rec, mat, rng);

        if (!scatter.did_scatter) {
            break;  // Ray absorbed
        }

        // Update throughput
        throughput *= scatter.attenuation;

        // Russian roulette (start after a few bounces)
        if (depth > 3) {
            float p = max(max(throughput.x, throughput.y), throughput.z);
            if (rng_next_float(rng) > p) {
                break;  // Terminate path
            }
            throughput /= p;  // Compensate for termination probability
        }

        // Check for very low throughput
        if (max(max(throughput.x, throughput.y), throughput.z) < 0.001f) {
            break;
        }

        current_ray = scatter.scattered_ray;
    }

    return accumulated_color;
}

// ============================================================================
// MAIN COMPUTE KERNEL
// ============================================================================

/**
 * Main path tracing compute kernel.
 *
 * Each thread processes one pixel, tracing multiple samples and accumulating.
 */
kernel void path_trace_kernel(
    device float4* output_buffer           [[buffer(0)]],   // RGBA output
    device float4* accumulation_buffer     [[buffer(1)]],   // Progressive accumulation
    device const Sphere* spheres           [[buffer(2)]],
    device const Triangle* triangles       [[buffer(3)]],
    device const Material* materials       [[buffer(4)]],
    device const BVHNode* bvh_nodes        [[buffer(5)]],
    constant Camera& camera                [[buffer(6)]],
    constant RenderSettings& settings      [[buffer(7)]],
    device const uint32_t* primitive_indices [[buffer(8)]], // For combined BVH
    uint2 gid                              [[thread_position_in_grid]]
) {
    // Bounds check
    if (gid.x >= settings.width || gid.y >= settings.height) {
        return;
    }

    uint32_t pixel_idx = gid.y * settings.width + gid.x;

    // Initialize RNG with unique seed per pixel/frame
    RNG rng = rng_init(gid, settings.frame_number, 0);

    float3 pixel_color = float3(0.0f);

    // Trace multiple samples per pixel
    for (uint32_t s = 0; s < settings.samples_per_pixel; s++) {
        // Add jitter for anti-aliasing
        float u = (float(gid.x) + rng_next_float(rng)) / float(settings.width - 1);
        float v = (float(settings.height - 1 - gid.y) + rng_next_float(rng)) / float(settings.height - 1);

        // Sample time for motion blur (random within shutter interval)
        float time = camera.time0 + rng_next_float(rng) * (camera.time1 - camera.time0);

        // Generate camera ray
        Ray ray = generate_camera_ray(float2(u, v), camera, rng);

        // Trace ray
        float3 sample_color = trace_ray(
            ray,
            spheres, settings.num_spheres,
            triangles, settings.num_triangles,
            materials,
            bvh_nodes, settings.num_bvh_nodes,
            primitive_indices, settings.num_primitive_indices,
            settings,
            time,  // Motion blur time
            rng
        );

        pixel_color += sample_color;
    }

    // Average samples
    pixel_color /= float(settings.samples_per_pixel);

    // Progressive accumulation
    if (settings.frame_number > 0) {
        float4 prev_accum = accumulation_buffer[pixel_idx];
        float weight = float(settings.frame_number);
        pixel_color = (prev_accum.xyz * weight + pixel_color) / (weight + 1.0f);
    }

    // Store in accumulation buffer
    accumulation_buffer[pixel_idx] = float4(pixel_color, 1.0f);

    // Output (no tone mapping here - done in post-process)
    output_buffer[pixel_idx] = float4(pixel_color, 1.0f);
}

// ============================================================================
// SIMPLE RENDER KERNEL (No accumulation, for testing)
// ============================================================================

/**
 * Simple single-pass render kernel for testing.
 * No accumulation, outputs directly.
 */
kernel void simple_render_kernel(
    device float4* output_buffer           [[buffer(0)]],
    device float4* accumulation_buffer     [[buffer(1)]],  // For bloom post-processing
    device const Sphere* spheres           [[buffer(2)]],
    device const Material* materials       [[buffer(3)]],
    constant Camera& camera                [[buffer(4)]],
    constant RenderSettings& settings      [[buffer(5)]],
    uint2 gid                              [[thread_position_in_grid]]
) {
    if (gid.x >= settings.width || gid.y >= settings.height) {
        return;
    }

    uint32_t pixel_idx = gid.y * settings.width + gid.x;

    RNG rng = rng_init(gid, settings.frame_number, 0);

    float3 pixel_color = float3(0.0f);

    for (uint32_t s = 0; s < settings.samples_per_pixel; s++) {
        float u = (float(gid.x) + rng_next_float(rng)) / float(settings.width - 1);
        float v = (float(settings.height - 1 - gid.y) + rng_next_float(rng)) / float(settings.height - 1);

        // Sample time for motion blur (random within shutter interval)
        float time = camera.time0 + rng_next_float(rng) * (camera.time1 - camera.time0);

        Ray ray = generate_camera_ray(float2(u, v), camera, rng);

        // Simple trace - spheres only, no BVH
        float3 throughput = float3(1.0f);
        float3 color = float3(0.0f);
        Ray current_ray = ray;

        for (uint32_t d = 0; d < settings.max_depth; d++) {
            HitRecord rec;
            rec.hit = 0;

            // Use motion blur intersection with sampled time
            bool hit = hit_scene_spheres_motion(current_ray, spheres, settings.num_spheres,
                                               time, current_ray.tmin, current_ray.tmax, rec);

            if (!hit) {
                color += throughput * sky_color(current_ray.direction);
                break;
            }

            Material mat = materials[rec.material_id];
            color += throughput * get_emission(mat);

            ScatterResult scatter = scatter_material(current_ray, rec, mat, rng);
            if (!scatter.did_scatter) break;

            throughput *= scatter.attenuation;
            current_ray = scatter.scattered_ray;

            // Russian roulette
            if (d > 2) {
                float p = max(max(throughput.x, throughput.y), throughput.z);
                if (rng_next_float(rng) > p) break;
                throughput /= p;
            }
        }

        pixel_color += color;
    }

    pixel_color /= float(settings.samples_per_pixel);

    // Write to both buffers (bloom post-processing reads from accumulation)
    float4 result = float4(pixel_color, 1.0f);
    output_buffer[pixel_idx] = result;
    accumulation_buffer[pixel_idx] = result;
}

// ============================================================================
// DEBUG KERNELS
// ============================================================================

/**
 * Debug kernel - outputs normals as colors.
 */
kernel void debug_normals_kernel(
    device float4* output_buffer           [[buffer(0)]],
    device const Sphere* spheres           [[buffer(1)]],
    constant Camera& camera                [[buffer(2)]],
    constant RenderSettings& settings      [[buffer(3)]],
    uint2 gid                              [[thread_position_in_grid]]
) {
    if (gid.x >= settings.width || gid.y >= settings.height) {
        return;
    }

    uint32_t pixel_idx = gid.y * settings.width + gid.x;

    float u = float(gid.x) / float(settings.width - 1);
    float v = float(settings.height - 1 - gid.y) / float(settings.height - 1);

    RNG rng = rng_init(gid, 0, 0);
    Ray ray = generate_camera_ray(float2(u, v), camera, rng);

    HitRecord rec;
    rec.hit = 0;

    bool hit = hit_scene_spheres(ray, spheres, settings.num_spheres,
                                ray.tmin, ray.tmax, rec);

    float3 color;
    if (hit) {
        // Normal to color: [-1,1] -> [0,1]
        color = (rec.normal + 1.0f) * 0.5f;
    } else {
        color = sky_color(ray.direction);
    }

    output_buffer[pixel_idx] = float4(color, 1.0f);
}

/**
 * Debug kernel - outputs depth as grayscale.
 */
kernel void debug_depth_kernel(
    device float4* output_buffer           [[buffer(0)]],
    device const Sphere* spheres           [[buffer(1)]],
    constant Camera& camera                [[buffer(2)]],
    constant RenderSettings& settings      [[buffer(3)]],
    uint2 gid                              [[thread_position_in_grid]]
) {
    if (gid.x >= settings.width || gid.y >= settings.height) {
        return;
    }

    uint32_t pixel_idx = gid.y * settings.width + gid.x;

    float u = float(gid.x) / float(settings.width - 1);
    float v = float(settings.height - 1 - gid.y) / float(settings.height - 1);

    RNG rng = rng_init(gid, 0, 0);
    Ray ray = generate_camera_ray(float2(u, v), camera, rng);

    HitRecord rec;
    rec.hit = 0;

    bool hit = hit_scene_spheres(ray, spheres, settings.num_spheres,
                                ray.tmin, ray.tmax, rec);

    float depth;
    if (hit) {
        // Normalize depth to [0,1] range (assuming max distance ~100)
        depth = 1.0f - clamp(rec.t / 100.0f, 0.0f, 1.0f);
    } else {
        depth = 0.0f;
    }

    output_buffer[pixel_idx] = float4(depth, depth, depth, 1.0f);
}
