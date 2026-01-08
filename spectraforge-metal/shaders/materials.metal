/**
 * SpectraForge Metal - Material Evaluation
 *
 * Implements material scattering functions (BRDFs) for path tracing:
 * - Lambertian (diffuse)
 * - Metal (specular with roughness)
 * - Dielectric (glass with refraction)
 * - Emissive (light sources)
 * - PBR (Cook-Torrance microfacet BRDF)
 */

#include <metal_stdlib>
#include "types.metal"
#include "textures.metal"

using namespace metal;

// ============================================================================
// SCATTER RESULT
// ============================================================================

/**
 * Result of material scattering operation.
 */
struct ScatterResult {
    Ray scattered_ray;
    float3 attenuation;
    bool did_scatter;
    bool is_specular;
};

// ============================================================================
// LAMBERTIAN (DIFFUSE) MATERIAL
// ============================================================================

/**
 * Lambertian diffuse scattering.
 * Scatters ray in random direction in hemisphere, weighted by cosine.
 */
inline ScatterResult scatter_lambertian(
    Ray ray_in,
    HitRecord rec,
    Material mat,
    thread RNG& rng
) {
    ScatterResult result;

    // Cosine-weighted hemisphere sampling for better convergence
    float3 tangent, bitangent;
    build_onb(rec.normal, tangent, bitangent);

    float3 local_dir = random_cosine_direction(rng);
    float3 scatter_direction = local_to_world(local_dir, rec.normal, tangent, bitangent);

    // Handle degenerate case
    if (near_zero(scatter_direction)) {
        scatter_direction = rec.normal;
    }

    result.scattered_ray.origin = rec.point;
    result.scattered_ray.direction = normalize(scatter_direction);
    result.scattered_ray.tmin = EPSILON;
    result.scattered_ray.tmax = INFINITY_F;

    result.attenuation = mat.albedo;
    result.did_scatter = true;
    result.is_specular = false;

    return result;
}

// ============================================================================
// METAL (SPECULAR) MATERIAL
// ============================================================================

/**
 * Metal specular reflection with optional roughness.
 */
inline ScatterResult scatter_metal(
    Ray ray_in,
    HitRecord rec,
    Material mat,
    thread RNG& rng
) {
    ScatterResult result;

    float3 reflected = sf_reflect(normalize(ray_in.direction), rec.normal);

    // Add roughness as random perturbation
    if (mat.roughness > 0.0f) {
        float3 fuzz = random_in_unit_sphere(rng) * mat.roughness;
        reflected = normalize(reflected + fuzz);
    }

    result.scattered_ray.origin = rec.point;
    result.scattered_ray.direction = reflected;
    result.scattered_ray.tmin = EPSILON;
    result.scattered_ray.tmax = INFINITY_F;

    result.attenuation = mat.albedo;

    // Only scatter if reflection is in correct hemisphere
    result.did_scatter = dot(reflected, rec.normal) > 0.0f;
    result.is_specular = true;

    return result;
}

// ============================================================================
// DIELECTRIC (GLASS) MATERIAL
// ============================================================================

/**
 * Dielectric material with refraction (glass, water, diamond).
 * Uses Schlick's approximation for Fresnel reflectance.
 */
inline ScatterResult scatter_dielectric(
    Ray ray_in,
    HitRecord rec,
    Material mat,
    thread RNG& rng
) {
    ScatterResult result;
    result.attenuation = mat.albedo;  // Glass tint

    // Determine refraction ratio (entering vs exiting)
    float refraction_ratio = rec.front_face ? (1.0f / mat.ior) : mat.ior;

    float3 unit_direction = normalize(ray_in.direction);
    float cos_theta = min(dot(-unit_direction, rec.normal), 1.0f);
    float sin_theta = sqrt(1.0f - cos_theta * cos_theta);

    bool cannot_refract = refraction_ratio * sin_theta > 1.0f;

    float3 direction;
    if (cannot_refract || schlick_reflectance(cos_theta, refraction_ratio) > rng_next_float(rng)) {
        // Must reflect (total internal reflection or Fresnel)
        direction = sf_reflect(unit_direction, rec.normal);
    } else {
        // Can refract
        direction = sf_refract(unit_direction, rec.normal, refraction_ratio);
    }

    result.scattered_ray.origin = rec.point;
    result.scattered_ray.direction = normalize(direction);
    result.scattered_ray.tmin = EPSILON;
    result.scattered_ray.tmax = INFINITY_F;

    result.did_scatter = true;
    result.is_specular = true;

    return result;
}

// ============================================================================
// PBR (COOK-TORRANCE) MATERIAL
// ============================================================================

/**
 * GGX/Trowbridge-Reitz normal distribution function.
 */
inline float distribution_ggx(float3 N, float3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0f);
    float NdotH2 = NdotH * NdotH;

    float denom = NdotH2 * (a2 - 1.0f) + 1.0f;
    denom = PI * denom * denom;

    return a2 / max(denom, EPSILON);
}

/**
 * Smith's geometry function with GGX.
 */
inline float geometry_schlick_ggx(float NdotV, float roughness) {
    float r = roughness + 1.0f;
    float k = (r * r) / 8.0f;
    return NdotV / (NdotV * (1.0f - k) + k);
}

inline float geometry_smith(float3 N, float3 V, float3 L, float roughness) {
    float NdotV = max(dot(N, V), 0.0f);
    float NdotL = max(dot(N, L), 0.0f);
    return geometry_schlick_ggx(NdotV, roughness) * geometry_schlick_ggx(NdotL, roughness);
}

/**
 * Fresnel-Schlick approximation with roughness.
 */
inline float3 fresnel_schlick(float cos_theta, float3 F0) {
    return F0 + (1.0f - F0) * pow(1.0f - cos_theta, 5.0f);
}

/**
 * Sample GGX distribution (importance sampling).
 */
inline float3 sample_ggx(float3 N, float3 V, float roughness, thread RNG& rng) {
    float alpha = roughness * roughness;
    float alpha2 = alpha * alpha;

    float u1 = rng_next_float(rng);
    float u2 = rng_next_float(rng);

    float phi = TWO_PI * u1;
    float cos_theta = sqrt((1.0f - u2) / (1.0f + (alpha2 - 1.0f) * u2));
    float sin_theta = sqrt(1.0f - cos_theta * cos_theta);

    // Spherical to Cartesian in tangent space
    float3 H_tangent = float3(
        sin_theta * cos(phi),
        sin_theta * sin(phi),
        cos_theta
    );

    // Transform to world space
    float3 tangent, bitangent;
    build_onb(N, tangent, bitangent);
    float3 H = local_to_world(H_tangent, N, tangent, bitangent);

    // Reflect view direction around half vector
    float3 L = sf_reflect(-V, H);
    return L;
}

/**
 * PBR material scattering using Cook-Torrance BRDF.
 */
inline ScatterResult scatter_pbr(
    Ray ray_in,
    HitRecord rec,
    Material mat,
    thread RNG& rng
) {
    ScatterResult result;

    float3 V = -normalize(ray_in.direction);
    float3 N = rec.normal;

    // F0 calculation: blend between dielectric and metallic
    float3 F0_dielectric = float3(0.04f);  // Typical for plastics
    float3 F0 = mix(F0_dielectric, mat.albedo, mat.metallic);

    // Fresnel at view angle
    float cos_theta = max(dot(N, V), 0.0f);
    float3 F = fresnel_schlick(cos_theta, F0);

    // Probability of specular vs diffuse
    float spec_prob = (F.x + F.y + F.z) / 3.0f;
    spec_prob = spec_prob * (1.0f - mat.metallic) + mat.metallic;

    if (rng_next_float(rng) < spec_prob) {
        // Specular (GGX importance sampling)
        float3 L = sample_ggx(N, V, mat.roughness, rng);

        if (dot(N, L) <= 0.0f) {
            result.did_scatter = false;
            return result;
        }

        result.scattered_ray.origin = rec.point;
        result.scattered_ray.direction = normalize(L);
        result.scattered_ray.tmin = EPSILON;
        result.scattered_ray.tmax = INFINITY_F;

        // Specular attenuation - metals tint reflections with albedo
        float3 attenuation = mix(F, mat.albedo * F, mat.metallic);
        result.attenuation = attenuation;
        result.did_scatter = true;
        result.is_specular = true;
    } else {
        // Diffuse (cosine-weighted sampling)
        float3 tangent, bitangent;
        build_onb(N, tangent, bitangent);

        float3 local_dir = random_cosine_direction(rng);
        float3 L = local_to_world(local_dir, N, tangent, bitangent);

        result.scattered_ray.origin = rec.point;
        result.scattered_ray.direction = normalize(L);
        result.scattered_ray.tmin = EPSILON;
        result.scattered_ray.tmax = INFINITY_F;

        // Diffuse attenuation weighted by (1-F) and (1-metallic)
        result.attenuation = mat.albedo * (1.0f - F.x) * (1.0f - mat.metallic);
        result.did_scatter = true;
        result.is_specular = false;
    }

    return result;
}

// ============================================================================
// PROCEDURAL TEXTURE APPLICATION
// ============================================================================

/**
 * Apply procedural texture to material based on hit record.
 * This is a simple demonstration - real implementation would use texture IDs.
 *
 * For now, applies checker pattern to material ID 0 (typically ground).
 */
inline Material apply_procedural_texture(Material mat, HitRecord rec) {
    Material result = mat;

    // Apply checker to ground (material_id 0) using world position
    if (rec.material_id == 0 && mat.type == MATERIAL_LAMBERTIAN) {
        float3 checker_color = checker_texture(
            rec.point,
            0.5f,  // scale
            mat.albedo * 0.3f,   // darker squares
            mat.albedo           // original color squares
        );
        result.albedo = checker_color;
    }

    return result;
}

// ============================================================================
// MATERIAL DISPATCHER
// ============================================================================

/**
 * Get emission from material.
 */
inline float3 get_emission(Material mat) {
    if (mat.type == MATERIAL_EMISSIVE || mat.emission_intensity > 0.0f) {
        return mat.emission_color * mat.emission_intensity;
    }
    return float3(0.0f);
}

/**
 * Scatter ray based on material type.
 */
inline ScatterResult scatter_material(
    Ray ray_in,
    HitRecord rec,
    Material mat,
    thread RNG& rng
) {
    switch (mat.type) {
        case MATERIAL_LAMBERTIAN:
            return scatter_lambertian(ray_in, rec, mat, rng);

        case MATERIAL_METAL:
            return scatter_metal(ray_in, rec, mat, rng);

        case MATERIAL_DIELECTRIC:
            return scatter_dielectric(ray_in, rec, mat, rng);

        case MATERIAL_PBR:
            return scatter_pbr(ray_in, rec, mat, rng);

        case MATERIAL_EMISSIVE:
        default: {
            // Emissive materials don't scatter
            ScatterResult result;
            result.did_scatter = false;
            return result;
        }
    }
}
