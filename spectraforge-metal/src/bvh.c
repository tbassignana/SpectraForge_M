/**
 * SpectraForge Metal - BVH Construction
 *
 * Builds a linear BVH (Bounding Volume Hierarchy) on the CPU for GPU traversal.
 * Uses a flattened array layout optimized for GPU memory access patterns.
 */

#include "../include/spectraforge.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

// ============================================================================
// INTERNAL TYPES
// ============================================================================

// Primitive type flags for BVH indices
#define PRIM_TYPE_SPHERE   0x00000000u
#define PRIM_TYPE_TRIANGLE 0x80000000u
#define PRIM_INDEX_MASK    0x7FFFFFFFu

typedef struct {
    float3 centroid;
    uint32_t index;      // Original primitive index
    uint32_t prim_type;  // PRIM_TYPE_SPHERE or PRIM_TYPE_TRIANGLE
    AABB bbox;
} PrimitiveInfo;

typedef struct {
    float3 min;
    float3 max;
} BBox;

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

static inline float min_f(float a, float b) { return a < b ? a : b; }
static inline float max_f(float a, float b) { return a > b ? a : b; }

static inline float3 min3(float3 a, float3 b) {
    return (float3){min_f(a.x, b.x), min_f(a.y, b.y), min_f(a.z, b.z)};
}

static inline float3 max3(float3 a, float3 b) {
    return (float3){max_f(a.x, b.x), max_f(a.y, b.y), max_f(a.z, b.z)};
}

static inline float3 add3(float3 a, float3 b) {
    return (float3){a.x + b.x, a.y + b.y, a.z + b.z};
}

static inline float3 sub3(float3 a, float3 b) {
    return (float3){a.x - b.x, a.y - b.y, a.z - b.z};
}

static inline float3 scale3(float3 v, float s) {
    return (float3){v.x * s, v.y * s, v.z * s};
}

/**
 * Compute AABB for a sphere.
 */
static AABB sphere_bbox(const Sphere* sphere) {
    float r = fabsf(sphere->radius);
    AABB bbox;
    bbox.min = (float3){sphere->center_x - r, sphere->center_y - r, sphere->center_z - r};
    bbox.max = (float3){sphere->center_x + r, sphere->center_y + r, sphere->center_z + r};
    return bbox;
}

/**
 * Compute AABB for a triangle.
 */
static AABB triangle_bbox(const Triangle* tri) {
    AABB bbox;
    bbox.min.x = min_f(min_f(tri->v0_x, tri->v1_x), tri->v2_x) - 0.0001f;
    bbox.min.y = min_f(min_f(tri->v0_y, tri->v1_y), tri->v2_y) - 0.0001f;
    bbox.min.z = min_f(min_f(tri->v0_z, tri->v1_z), tri->v2_z) - 0.0001f;
    bbox.max.x = max_f(max_f(tri->v0_x, tri->v1_x), tri->v2_x) + 0.0001f;
    bbox.max.y = max_f(max_f(tri->v0_y, tri->v1_y), tri->v2_y) + 0.0001f;
    bbox.max.z = max_f(max_f(tri->v0_z, tri->v1_z), tri->v2_z) + 0.0001f;
    return bbox;
}

/**
 * Compute centroid of AABB.
 */
static float3 bbox_centroid(const AABB* bbox) {
    return scale3(add3(bbox->min, bbox->max), 0.5f);
}

/**
 * Merge two AABBs.
 */
static AABB bbox_union(const AABB* a, const AABB* b) {
    AABB result;
    result.min = min3(a->min, b->min);
    result.max = max3(a->max, b->max);
    return result;
}

/**
 * Get surface area of AABB (for SAH).
 */
static float bbox_surface_area(const AABB* bbox) {
    float3 d = sub3(bbox->max, bbox->min);
    return 2.0f * (d.x * d.y + d.y * d.z + d.z * d.x);
}

/**
 * Comparison functions for sorting primitives.
 */
static int compare_x(const void* a, const void* b) {
    const PrimitiveInfo* pa = (const PrimitiveInfo*)a;
    const PrimitiveInfo* pb = (const PrimitiveInfo*)b;
    return (pa->centroid.x > pb->centroid.x) - (pa->centroid.x < pb->centroid.x);
}

static int compare_y(const void* a, const void* b) {
    const PrimitiveInfo* pa = (const PrimitiveInfo*)a;
    const PrimitiveInfo* pb = (const PrimitiveInfo*)b;
    return (pa->centroid.y > pb->centroid.y) - (pa->centroid.y < pb->centroid.y);
}

static int compare_z(const void* a, const void* b) {
    const PrimitiveInfo* pa = (const PrimitiveInfo*)a;
    const PrimitiveInfo* pb = (const PrimitiveInfo*)b;
    return (pa->centroid.z > pb->centroid.z) - (pa->centroid.z < pb->centroid.z);
}

// ============================================================================
// BVH CONSTRUCTION
// ============================================================================

/**
 * Internal BVH build state.
 */
typedef struct {
    BVHNode* nodes;
    uint32_t* ordered_prims;  // Reordered primitive indices
    uint32_t node_count;
    uint32_t node_capacity;
    uint32_t ordered_offset;
} BVHBuildState;

/**
 * Recursively build BVH node.
 * Returns the index of the created node.
 */
static uint32_t build_recursive(
    BVHBuildState* state,
    PrimitiveInfo* prim_info,
    uint32_t start,
    uint32_t end,
    uint32_t max_leaf_prims
) {
    // Allocate node
    if (state->node_count >= state->node_capacity) {
        state->node_capacity *= 2;
        state->nodes = realloc(state->nodes, sizeof(BVHNode) * state->node_capacity);
    }
    uint32_t node_idx = state->node_count++;
    BVHNode* node = &state->nodes[node_idx];

    // Compute bounds of all primitives in this subtree
    AABB bounds = prim_info[start].bbox;
    for (uint32_t i = start + 1; i < end; i++) {
        bounds = bbox_union(&bounds, &prim_info[i].bbox);
    }

    uint32_t n_prims = end - start;

    if (n_prims <= max_leaf_prims) {
        // Create leaf node
        uint32_t first_prim_offset = state->ordered_offset;
        for (uint32_t i = start; i < end; i++) {
            // Encode primitive type in high bit: triangles have PRIM_TYPE_TRIANGLE set
            uint32_t encoded_idx = prim_info[i].prim_type | prim_info[i].index;
            state->ordered_prims[state->ordered_offset++] = encoded_idx;
        }
        node->bbox_min_x = bounds.min.x;
        node->bbox_min_y = bounds.min.y;
        node->bbox_min_z = bounds.min.z;
        node->bbox_max_x = bounds.max.x;
        node->bbox_max_y = bounds.max.y;
        node->bbox_max_z = bounds.max.z;
        node->left_or_first = first_prim_offset;
        node->count = n_prims;
    } else {
        // Create interior node

        // Compute centroid bounds for split
        AABB centroid_bounds;
        centroid_bounds.min = centroid_bounds.max = prim_info[start].centroid;
        for (uint32_t i = start + 1; i < end; i++) {
            centroid_bounds.min = min3(centroid_bounds.min, prim_info[i].centroid);
            centroid_bounds.max = max3(centroid_bounds.max, prim_info[i].centroid);
        }

        // Choose split axis (largest extent)
        float3 extent = sub3(centroid_bounds.max, centroid_bounds.min);
        int axis = 0;
        if (extent.y > extent.x) axis = 1;
        if (extent.z > (axis == 0 ? extent.x : extent.y)) axis = 2;

        uint32_t mid = (start + end) / 2;

        // Check for degenerate case where all centroids are at same position
        float axis_extent = (axis == 0) ? extent.x : ((axis == 1) ? extent.y : extent.z);

        if (axis_extent < 0.0001f) {
            // All centroids at same position - just split in half
            mid = (start + end) / 2;
        } else if (n_prims <= 4) {
            // Small number of prims - use simple median split
            int (*compare_func)(const void*, const void*) =
                (axis == 0) ? compare_x : ((axis == 1) ? compare_y : compare_z);
            qsort(&prim_info[start], n_prims, sizeof(PrimitiveInfo), compare_func);
            mid = (start + end) / 2;
        } else {
            // Use SAH (Surface Area Heuristic) for larger nodes
            // Simplified: try 12 bucket splits

            #define N_BUCKETS 12

            struct Bucket {
                uint32_t count;
                AABB bounds;
            } buckets[N_BUCKETS];

            // Initialize buckets
            for (int i = 0; i < N_BUCKETS; i++) {
                buckets[i].count = 0;
                buckets[i].bounds.min = (float3){1e30f, 1e30f, 1e30f};
                buckets[i].bounds.max = (float3){-1e30f, -1e30f, -1e30f};
            }

            // Assign primitives to buckets
            for (uint32_t i = start; i < end; i++) {
                float centroid_val = (axis == 0) ? prim_info[i].centroid.x :
                                     ((axis == 1) ? prim_info[i].centroid.y : prim_info[i].centroid.z);
                float min_val = (axis == 0) ? centroid_bounds.min.x :
                                ((axis == 1) ? centroid_bounds.min.y : centroid_bounds.min.z);
                float max_val = (axis == 0) ? centroid_bounds.max.x :
                                ((axis == 1) ? centroid_bounds.max.y : centroid_bounds.max.z);

                int b = (int)(N_BUCKETS * (centroid_val - min_val) / (max_val - min_val));
                if (b >= N_BUCKETS) b = N_BUCKETS - 1;
                if (b < 0) b = 0;

                buckets[b].count++;
                if (buckets[b].count == 1) {
                    buckets[b].bounds = prim_info[i].bbox;
                } else {
                    buckets[b].bounds = bbox_union(&buckets[b].bounds, &prim_info[i].bbox);
                }
            }

            // Compute costs for splitting after each bucket
            float cost[N_BUCKETS - 1];
            for (int i = 0; i < N_BUCKETS - 1; i++) {
                AABB b0 = buckets[0].bounds;
                AABB b1 = buckets[i + 1].bounds;
                uint32_t count0 = buckets[0].count;
                uint32_t count1 = buckets[i + 1].count;

                for (int j = 1; j <= i; j++) {
                    if (buckets[j].count > 0) {
                        b0 = bbox_union(&b0, &buckets[j].bounds);
                        count0 += buckets[j].count;
                    }
                }
                for (int j = i + 2; j < N_BUCKETS; j++) {
                    if (buckets[j].count > 0) {
                        b1 = bbox_union(&b1, &buckets[j].bounds);
                        count1 += buckets[j].count;
                    }
                }

                float area0 = (count0 > 0) ? bbox_surface_area(&b0) : 0;
                float area1 = (count1 > 0) ? bbox_surface_area(&b1) : 0;
                cost[i] = 1.0f + (count0 * area0 + count1 * area1) / bbox_surface_area(&bounds);
            }

            // Find best split
            float min_cost = cost[0];
            int min_bucket = 0;
            for (int i = 1; i < N_BUCKETS - 1; i++) {
                if (cost[i] < min_cost) {
                    min_cost = cost[i];
                    min_bucket = i;
                }
            }

            // Partition primitives based on best split
            float leaf_cost = (float)n_prims;
            if (n_prims > max_leaf_prims || min_cost < leaf_cost) {
                // Partition
                uint32_t write_idx = start;
                for (uint32_t i = start; i < end; i++) {
                    float centroid_val = (axis == 0) ? prim_info[i].centroid.x :
                                         ((axis == 1) ? prim_info[i].centroid.y : prim_info[i].centroid.z);
                    float min_val = (axis == 0) ? centroid_bounds.min.x :
                                    ((axis == 1) ? centroid_bounds.min.y : centroid_bounds.min.z);
                    float max_val = (axis == 0) ? centroid_bounds.max.x :
                                    ((axis == 1) ? centroid_bounds.max.y : centroid_bounds.max.z);

                    int b = (int)(N_BUCKETS * (centroid_val - min_val) / (max_val - min_val));
                    if (b >= N_BUCKETS) b = N_BUCKETS - 1;
                    if (b < 0) b = 0;

                    if (b <= min_bucket) {
                        // Swap to front
                        PrimitiveInfo tmp = prim_info[write_idx];
                        prim_info[write_idx] = prim_info[i];
                        prim_info[i] = tmp;
                        write_idx++;
                    }
                }
                mid = write_idx;
                if (mid == start || mid == end) {
                    mid = (start + end) / 2;
                }
            }

            #undef N_BUCKETS
        }

        // Build children recursively
        uint32_t left_idx = build_recursive(state, prim_info, start, mid, max_leaf_prims);
        uint32_t right_idx = build_recursive(state, prim_info, mid, end, max_leaf_prims);

        // Fill in interior node
        // For interior nodes: left_or_first = left child, count has high bit set + right child index
        node = &state->nodes[node_idx];  // Reacquire pointer (may have moved due to realloc)
        node->bbox_min_x = bounds.min.x;
        node->bbox_min_y = bounds.min.y;
        node->bbox_min_z = bounds.min.z;
        node->bbox_max_x = bounds.max.x;
        node->bbox_max_y = bounds.max.y;
        node->bbox_max_z = bounds.max.z;
        node->left_or_first = left_idx;
        node->count = 0x80000000u | right_idx;  // High bit marks interior, low bits = right child
    }

    return node_idx;
}

// ============================================================================
// PUBLIC API
// ============================================================================

/**
 * Build BVH for spheres in the scene.
 */
void sf_scene_build_bvh(Scene* scene) {
    if (!scene || scene->num_spheres == 0) {
        scene->num_bvh_nodes = 0;
        return;
    }

    uint32_t num_prims = scene->num_spheres;

    // Initialize primitive info
    PrimitiveInfo* prim_info = malloc(sizeof(PrimitiveInfo) * num_prims);
    for (uint32_t i = 0; i < num_prims; i++) {
        prim_info[i].index = i;
        prim_info[i].prim_type = PRIM_TYPE_SPHERE;
        prim_info[i].bbox = sphere_bbox(&scene->spheres[i]);
        prim_info[i].centroid = bbox_centroid(&prim_info[i].bbox);
    }

    // Initialize build state
    BVHBuildState state;
    state.node_capacity = num_prims * 2;  // Reasonable initial capacity
    state.nodes = malloc(sizeof(BVHNode) * state.node_capacity);
    state.ordered_prims = malloc(sizeof(uint32_t) * num_prims);
    state.node_count = 0;
    state.ordered_offset = 0;

    // Build BVH
    uint32_t max_leaf_prims = 4;
    build_recursive(&state, prim_info, 0, num_prims, max_leaf_prims);

    // Reorder spheres according to BVH layout
    Sphere* new_spheres = malloc(sizeof(Sphere) * num_prims);
    for (uint32_t i = 0; i < num_prims; i++) {
        new_spheres[i] = scene->spheres[state.ordered_prims[i]];
    }
    memcpy(scene->spheres, new_spheres, sizeof(Sphere) * num_prims);
    free(new_spheres);

    // Copy BVH nodes to scene
    if (scene->bvh_nodes) {
        free(scene->bvh_nodes);
    }
    scene->bvh_nodes = state.nodes;
    scene->num_bvh_nodes = state.node_count;

    printf("Built BVH: %u nodes for %u primitives\n", state.node_count, num_prims);

    // Cleanup
    free(prim_info);
    free(state.ordered_prims);
}

/**
 * Build BVH for both spheres and triangles.
 * Uses encoded indices: high bit = triangle, low bits = index.
 */
void sf_scene_build_bvh_full(Scene* scene) {
    if (!scene) {
        return;
    }

    uint32_t num_spheres = scene->num_spheres;
    uint32_t num_triangles = scene->num_triangles;
    uint32_t total_prims = num_spheres + num_triangles;

    if (total_prims == 0) {
        scene->num_bvh_nodes = 0;
        return;
    }

    // If no triangles, fall back to sphere-only BVH
    if (num_triangles == 0) {
        sf_scene_build_bvh(scene);
        return;
    }

    // Initialize primitive info for all primitives
    PrimitiveInfo* prim_info = malloc(sizeof(PrimitiveInfo) * total_prims);

    // Add spheres
    for (uint32_t i = 0; i < num_spheres; i++) {
        prim_info[i].index = i;
        prim_info[i].prim_type = PRIM_TYPE_SPHERE;
        prim_info[i].bbox = sphere_bbox(&scene->spheres[i]);
        prim_info[i].centroid = bbox_centroid(&prim_info[i].bbox);
    }

    // Add triangles
    for (uint32_t i = 0; i < num_triangles; i++) {
        uint32_t idx = num_spheres + i;
        prim_info[idx].index = i;
        prim_info[idx].prim_type = PRIM_TYPE_TRIANGLE;
        prim_info[idx].bbox = triangle_bbox(&scene->triangles[i]);
        prim_info[idx].centroid = bbox_centroid(&prim_info[idx].bbox);
    }

    // Initialize build state
    BVHBuildState state;
    state.node_capacity = total_prims * 2;
    state.nodes = malloc(sizeof(BVHNode) * state.node_capacity);
    state.ordered_prims = malloc(sizeof(uint32_t) * total_prims);
    state.node_count = 0;
    state.ordered_offset = 0;

    // Build BVH
    uint32_t max_leaf_prims = 4;
    build_recursive(&state, prim_info, 0, total_prims, max_leaf_prims);

    // For mixed primitives, we DON'T reorder the original arrays.
    // Instead, store the primitive index buffer for indirect lookup.
    // The shader will decode type from high bit and look up in appropriate array.

    // Copy BVH nodes to scene
    if (scene->bvh_nodes) {
        free(scene->bvh_nodes);
    }
    scene->bvh_nodes = state.nodes;
    scene->num_bvh_nodes = state.node_count;

    // Store primitive indices buffer (don't free - transfer ownership)
    if (scene->primitive_indices) {
        free(scene->primitive_indices);
    }
    scene->primitive_indices = state.ordered_prims;
    scene->num_primitive_indices = state.ordered_offset;

    printf("Built mixed BVH: %u nodes for %u primitives (%u spheres, %u triangles)\n",
           state.node_count, total_prims, num_spheres, num_triangles);

    // Cleanup (don't free ordered_prims - transferred to scene)
    free(prim_info);
}
