/**
 * SpectraForge Metal - Intersection Functions
 *
 * Implements ray-primitive intersection tests optimized for GPU execution.
 */

#include <metal_stdlib>
#include "types.metal"

using namespace metal;

// ============================================================================
// RAY-SPHERE INTERSECTION
// ============================================================================

/**
 * Get sphere center at a given time (for motion blur).
 * center_at_time = center + velocity * time
 */
inline float3 sphere_center_at_time(Sphere sphere, float time) {
    return float3(
        sphere.center_x + sphere.velocity_x * time,
        sphere.center_y + sphere.velocity_y * time,
        sphere.center_z + sphere.velocity_z * time
    );
}

/**
 * Test ray-sphere intersection with motion blur support.
 *
 * The sphere equation: |P - C(t)|^2 = r^2
 * Where C(t) = C0 + V * time (center moves with velocity)
 *
 * Using half-b optimization: b' = (O-C)·D, discriminant = b'^2 - ac
 */
inline bool hit_sphere_motion(
    Ray ray,
    Sphere sphere,
    float time,       // Time in [0, 1] for motion blur
    float t_min,
    float t_max,
    thread HitRecord& rec
) {
    // Get sphere center at this time
    float3 center = sphere_center_at_time(sphere, time);

    float3 oc = ray.origin - center;
    float a = length_squared(ray.direction);
    float half_b = dot(oc, ray.direction);
    float c = length_squared(oc) - sphere.radius * sphere.radius;

    float discriminant = half_b * half_b - a * c;
    if (discriminant < 0.0f) {
        return false;
    }

    float sqrtd = sqrt(discriminant);

    // Find nearest root in acceptable range
    float root = (-half_b - sqrtd) / a;
    if (root < t_min || root > t_max) {
        root = (-half_b + sqrtd) / a;
        if (root < t_min || root > t_max) {
            return false;
        }
    }

    // Fill hit record
    rec.t = root;
    rec.point = ray.origin + root * ray.direction;

    // Outward normal
    float3 outward_normal = (rec.point - center) / sphere.radius;

    // Determine if we hit from outside
    rec.front_face = dot(ray.direction, outward_normal) < 0.0f ? 1 : 0;
    rec.normal = rec.front_face ? outward_normal : -outward_normal;

    rec.material_id = sphere.material_id;
    rec.hit = 1;

    // Spherical UV coordinates
    // u: angle around Y axis from X=-1 (phi)
    // v: angle from Y=-1 to Y=+1 (theta)
    float theta = acos(-outward_normal.y);
    float phi = atan2(-outward_normal.z, outward_normal.x) + PI;
    rec.u = phi * INV_PI * 0.5f;
    rec.v = theta * INV_PI;

    return true;
}

/**
 * Test ray-sphere intersection without motion blur (backwards compatible).
 * Calls hit_sphere_motion with time=0.
 */
inline bool hit_sphere(
    Ray ray,
    Sphere sphere,
    float t_min,
    float t_max,
    thread HitRecord& rec
) {
    return hit_sphere_motion(ray, sphere, 0.0f, t_min, t_max, rec);
}

// ============================================================================
// RAY-TRIANGLE INTERSECTION (Möller-Trumbore)
// ============================================================================

/**
 * Test ray-triangle intersection using Möller-Trumbore algorithm.
 *
 * This is an efficient algorithm that computes barycentric coordinates
 * as a byproduct, which are useful for interpolation.
 */
inline bool hit_triangle(
    Ray ray,
    Triangle tri,
    float t_min,
    float t_max,
    thread HitRecord& rec
) {
    // Construct float3 vertices from individual components
    float3 v0 = float3(tri.v0_x, tri.v0_y, tri.v0_z);
    float3 v1 = float3(tri.v1_x, tri.v1_y, tri.v1_z);
    float3 v2 = float3(tri.v2_x, tri.v2_y, tri.v2_z);

    float3 e1 = v1 - v0;
    float3 e2 = v2 - v0;

    float3 h = cross(ray.direction, e2);
    float a = dot(e1, h);

    // Ray parallel to triangle
    if (abs(a) < EPSILON) {
        return false;
    }

    float f = 1.0f / a;
    float3 s = ray.origin - v0;
    float u = f * dot(s, h);

    if (u < 0.0f || u > 1.0f) {
        return false;
    }

    float3 q = cross(s, e1);
    float v = f * dot(ray.direction, q);

    if (v < 0.0f || u + v > 1.0f) {
        return false;
    }

    float t = f * dot(e2, q);

    if (t < t_min || t > t_max) {
        return false;
    }

    // Fill hit record
    rec.t = t;
    rec.point = ray.origin + t * ray.direction;
    rec.u = u;
    rec.v = v;

    // Use precomputed normal or compute from edges
    float3 outward_normal = float3(tri.normal_x, tri.normal_y, tri.normal_z);
    if (length_squared(outward_normal) < 0.5f) {
        outward_normal = normalize(cross(e1, e2));
    }

    rec.front_face = dot(ray.direction, outward_normal) < 0.0f ? 1 : 0;
    rec.normal = rec.front_face ? outward_normal : -outward_normal;

    rec.material_id = tri.material_id;
    rec.hit = 1;

    return true;
}

// ============================================================================
// RAY-AABB INTERSECTION (Slab Method)
// ============================================================================

/**
 * Test ray-AABB intersection using optimized slab method.
 *
 * Returns true if ray intersects the box within [t_min, t_max].
 * This is crucial for BVH traversal efficiency.
 */
inline bool hit_aabb(Ray ray, float3 box_min, float3 box_max, float t_min, float t_max) {
    // Precompute inverse direction (handle div by zero)
    float3 inv_d = 1.0f / ray.direction;

    float3 t0 = (box_min - ray.origin) * inv_d;
    float3 t1 = (box_max - ray.origin) * inv_d;

    // Handle negative direction
    float3 t_smaller = min(t0, t1);
    float3 t_larger = max(t0, t1);

    float t_near = max(max(t_smaller.x, t_smaller.y), max(t_smaller.z, t_min));
    float t_far = min(min(t_larger.x, t_larger.y), min(t_larger.z, t_max));

    return t_near <= t_far;
}

// ============================================================================
// RAY-PLANE INTERSECTION
// ============================================================================

/**
 * Test ray-plane intersection.
 * Plane defined by point and normal.
 */
inline bool hit_plane(
    Ray ray,
    float3 plane_point,
    float3 plane_normal,
    float t_min,
    float t_max,
    thread HitRecord& rec,
    uint32_t material_id
) {
    float denom = dot(plane_normal, ray.direction);

    // Ray parallel to plane
    if (abs(denom) < EPSILON) {
        return false;
    }

    float t = dot(plane_point - ray.origin, plane_normal) / denom;

    if (t < t_min || t > t_max) {
        return false;
    }

    rec.t = t;
    rec.point = ray.origin + t * ray.direction;

    rec.front_face = denom < 0.0f ? 1 : 0;
    rec.normal = rec.front_face ? plane_normal : -plane_normal;

    // Planar UV mapping
    rec.u = rec.point.x - floor(rec.point.x);
    rec.v = rec.point.z - floor(rec.point.z);

    rec.material_id = material_id;
    rec.hit = 1;

    return true;
}

// ============================================================================
// BVH TRAVERSAL
// ============================================================================

/**
 * Traverse BVH and find closest intersection.
 *
 * Uses iterative traversal with a local stack to avoid recursion.
 * This is more GPU-friendly than recursive traversal.
 *
 * Current approach: BVH for spheres, brute-force for triangles.
 * TODO: Combined BVH for both primitive types.
 */
inline bool traverse_bvh(
    Ray ray,
    device const BVHNode* bvh_nodes,
    device const Sphere* spheres,
    device const Triangle* triangles,
    uint32_t num_bvh_nodes,
    uint32_t num_triangles,
    float t_min,
    float t_max,
    thread HitRecord& rec
) {
    bool hit_anything = false;
    float closest_t = t_max;
    HitRecord temp_rec;

    // BVH traversal for spheres
    if (num_bvh_nodes > 0) {
        // Local stack for traversal (max depth ~32 should be plenty)
        uint32_t stack[32];
        int stack_ptr = 0;
        stack[stack_ptr++] = 0;  // Start at root

        while (stack_ptr > 0) {
            uint32_t node_idx = stack[--stack_ptr];
            BVHNode node = bvh_nodes[node_idx];

            // Construct float3 from individual components
            float3 bbox_min = float3(node.bbox_min_x, node.bbox_min_y, node.bbox_min_z);
            float3 bbox_max = float3(node.bbox_max_x, node.bbox_max_y, node.bbox_max_z);

            // Test AABB intersection
            if (!hit_aabb(ray, bbox_min, bbox_max, t_min, closest_t)) {
                continue;
            }

            // Check if interior node (high bit set) or leaf node
            bool is_interior = (node.count & 0x80000000u) != 0;

            if (!is_interior) {
                // Leaf node - test primitives (spheres)
                uint32_t prim_count = node.count;
                for (uint32_t i = 0; i < prim_count; i++) {
                    uint32_t prim_idx = node.left_or_first + i;

                    if (hit_sphere(ray, spheres[prim_idx], t_min, closest_t, temp_rec)) {
                        hit_anything = true;
                        closest_t = temp_rec.t;
                        rec = temp_rec;
                    }
                }
            } else {
                // Interior node - push children
                // Left child is in left_or_first, right child is in low bits of count
                uint32_t left = node.left_or_first;
                uint32_t right = node.count & 0x7FFFFFFFu;

                // Simple heuristic: check which child is closer along ray direction
                BVHNode left_node = bvh_nodes[left];
                BVHNode right_node = bvh_nodes[right];
                float left_dist = dot(float3(left_node.bbox_min_x, left_node.bbox_min_y, left_node.bbox_min_z), ray.direction);
                float right_dist = dot(float3(right_node.bbox_min_x, right_node.bbox_min_y, right_node.bbox_min_z), ray.direction);

                if (left_dist > right_dist) {
                    stack[stack_ptr++] = left;
                    stack[stack_ptr++] = right;
                } else {
                    stack[stack_ptr++] = right;
                    stack[stack_ptr++] = left;
                }
            }
        }
    }

    // Brute-force triangle testing (use closest_t from sphere hits)
    for (uint32_t i = 0; i < num_triangles; i++) {
        if (hit_triangle(ray, triangles[i], t_min, closest_t, temp_rec)) {
            hit_anything = true;
            closest_t = temp_rec.t;
            rec = temp_rec;
        }
    }

    return hit_anything;
}

// ============================================================================
// COMBINED BVH TRAVERSAL (Spheres + Triangles)
// ============================================================================

// Primitive type encoding constants (must match bvh.c)
constant uint32_t PRIM_TYPE_TRIANGLE = 0x80000000u;
constant uint32_t PRIM_INDEX_MASK = 0x7FFFFFFFu;

/**
 * Traverse combined BVH with both spheres and triangles.
 *
 * Uses primitive_indices buffer for indirect lookup.
 * High bit of index indicates triangle vs sphere.
 */
inline bool traverse_bvh_combined(
    Ray ray,
    device const BVHNode* bvh_nodes,
    device const Sphere* spheres,
    device const Triangle* triangles,
    device const uint32_t* primitive_indices,
    uint32_t num_bvh_nodes,
    float t_min,
    float t_max,
    thread HitRecord& rec
) {
    if (num_bvh_nodes == 0) {
        return false;
    }

    bool hit_anything = false;
    float closest_t = t_max;
    HitRecord temp_rec;

    // Local stack for traversal
    uint32_t stack[32];
    int stack_ptr = 0;
    stack[stack_ptr++] = 0;  // Start at root

    while (stack_ptr > 0) {
        uint32_t node_idx = stack[--stack_ptr];
        BVHNode node = bvh_nodes[node_idx];

        // Construct float3 from individual components
        float3 bbox_min = float3(node.bbox_min_x, node.bbox_min_y, node.bbox_min_z);
        float3 bbox_max = float3(node.bbox_max_x, node.bbox_max_y, node.bbox_max_z);

        // Test AABB intersection
        if (!hit_aabb(ray, bbox_min, bbox_max, t_min, closest_t)) {
            continue;
        }

        // Check if interior node (high bit set) or leaf node
        bool is_interior = (node.count & 0x80000000u) != 0;

        if (!is_interior) {
            // Leaf node - test primitives via indirect lookup
            uint32_t prim_count = node.count;
            uint32_t first_idx = node.left_or_first;

            for (uint32_t i = 0; i < prim_count; i++) {
                uint32_t encoded_idx = primitive_indices[first_idx + i];
                bool is_triangle = (encoded_idx & PRIM_TYPE_TRIANGLE) != 0;
                uint32_t prim_idx = encoded_idx & PRIM_INDEX_MASK;

                if (is_triangle) {
                    if (hit_triangle(ray, triangles[prim_idx], t_min, closest_t, temp_rec)) {
                        hit_anything = true;
                        closest_t = temp_rec.t;
                        rec = temp_rec;
                    }
                } else {
                    if (hit_sphere(ray, spheres[prim_idx], t_min, closest_t, temp_rec)) {
                        hit_anything = true;
                        closest_t = temp_rec.t;
                        rec = temp_rec;
                    }
                }
            }
        } else {
            // Interior node - push children
            uint32_t left = node.left_or_first;
            uint32_t right = node.count & 0x7FFFFFFFu;

            // Simple heuristic: check which child is closer along ray direction
            BVHNode left_node = bvh_nodes[left];
            BVHNode right_node = bvh_nodes[right];
            float left_dist = dot(float3(left_node.bbox_min_x, left_node.bbox_min_y, left_node.bbox_min_z), ray.direction);
            float right_dist = dot(float3(right_node.bbox_min_x, right_node.bbox_min_y, right_node.bbox_min_z), ray.direction);

            if (left_dist > right_dist) {
                stack[stack_ptr++] = left;
                stack[stack_ptr++] = right;
            } else {
                stack[stack_ptr++] = right;
                stack[stack_ptr++] = left;
            }
        }
    }

    return hit_anything;
}

// ============================================================================
// SIMPLE SCENE INTERSECTION (No BVH)
// ============================================================================

/**
 * Test ray against all spheres with motion blur (brute force).
 */
inline bool hit_scene_spheres_motion(
    Ray ray,
    device const Sphere* spheres,
    uint32_t num_spheres,
    float time,  // Motion blur time [0, 1]
    float t_min,
    float t_max,
    thread HitRecord& rec
) {
    bool hit_anything = false;
    float closest_t = t_max;
    HitRecord temp_rec;

    for (uint32_t i = 0; i < num_spheres; i++) {
        if (hit_sphere_motion(ray, spheres[i], time, t_min, closest_t, temp_rec)) {
            hit_anything = true;
            closest_t = temp_rec.t;
            rec = temp_rec;
        }
    }

    return hit_anything;
}

/**
 * Test ray against all spheres (brute force, no motion blur).
 * Used when BVH is not available or for small scenes.
 */
inline bool hit_scene_spheres(
    Ray ray,
    device const Sphere* spheres,
    uint32_t num_spheres,
    float t_min,
    float t_max,
    thread HitRecord& rec
) {
    return hit_scene_spheres_motion(ray, spheres, num_spheres, 0.0f, t_min, t_max, rec);
}

/**
 * Test ray against all triangles (brute force).
 */
inline bool hit_scene_triangles(
    Ray ray,
    device const Triangle* triangles,
    uint32_t num_triangles,
    float t_min,
    float t_max,
    thread HitRecord& rec
) {
    bool hit_anything = false;
    float closest_t = t_max;
    HitRecord temp_rec;

    for (uint32_t i = 0; i < num_triangles; i++) {
        if (hit_triangle(ray, triangles[i], t_min, closest_t, temp_rec)) {
            hit_anything = true;
            closest_t = temp_rec.t;
            rec = temp_rec;
        }
    }

    return hit_anything;
}

/**
 * Combined scene intersection with motion blur - checks all primitives.
 * Spheres support motion blur; triangles are static.
 */
inline bool hit_scene_motion(
    Ray ray,
    device const Sphere* spheres,
    uint32_t num_spheres,
    device const Triangle* triangles,
    uint32_t num_triangles,
    float time,  // Motion blur time [0, 1]
    float t_min,
    float t_max,
    thread HitRecord& rec
) {
    bool hit_anything = false;
    float closest_t = t_max;
    HitRecord temp_rec;

    // Check spheres with motion blur
    if (hit_scene_spheres_motion(ray, spheres, num_spheres, time, t_min, closest_t, temp_rec)) {
        hit_anything = true;
        closest_t = temp_rec.t;
        rec = temp_rec;
    }

    // Check triangles (static, no motion blur)
    if (hit_scene_triangles(ray, triangles, num_triangles, t_min, closest_t, temp_rec)) {
        hit_anything = true;
        closest_t = temp_rec.t;
        rec = temp_rec;
    }

    return hit_anything;
}

/**
 * Combined scene intersection - checks all primitives (no motion blur).
 */
inline bool hit_scene(
    Ray ray,
    device const Sphere* spheres,
    uint32_t num_spheres,
    device const Triangle* triangles,
    uint32_t num_triangles,
    float t_min,
    float t_max,
    thread HitRecord& rec
) {
    return hit_scene_motion(ray, spheres, num_spheres, triangles, num_triangles,
                           0.0f, t_min, t_max, rec);
}
