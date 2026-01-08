/**
 * SpectraForge Metal - Main Entry Point
 *
 * Command-line interface for the GPU ray tracer.
 */

#import <Foundation/Foundation.h>
#include "../include/spectraforge.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

// ============================================================================
// COMMAND LINE PARSING
// ============================================================================

typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t samples;
    uint32_t max_depth;
    const char* output;
    const char* scene;
    const char* scene_file;  // JSON scene file path
    int benchmark;
    int preview;     // Real-time preview mode
    int debug_mode;  // 0=normal, 1=normals, 2=depth
    int help;
} Options;

static void print_usage(const char* program) {
    printf("SpectraForge Metal - GPU Ray Tracer for Apple Silicon\n");
    printf("\n");
    printf("Usage: %s [options]\n", program);
    printf("\n");
    printf("Options:\n");
    printf("  --width N       Image width in pixels (default: 800)\n");
    printf("  --height N      Image height in pixels (default: 600)\n");
    printf("  --samples N     Samples per pixel (default: 16)\n");
    printf("  --depth N       Maximum ray bounce depth (default: 10)\n");
    printf("  --output FILE   Output file path (default: output/render.png)\n");
    printf("  --scene NAME    Scene preset: demo, cornell, triangle, pbr, mesh, dof, motion (default: demo)\n");
    printf("  --scene-file F  Load scene from JSON file (overrides --scene)\n");
    printf("  --preview       Open interactive preview window\n");
    printf("  --benchmark     Run performance benchmark\n");
    printf("  --debug-normals Output surface normals as colors\n");
    printf("  --debug-depth   Output depth buffer\n");
    printf("  --help          Show this help message\n");
    printf("\n");
    printf("Examples:\n");
    printf("  %s                                    # Quick preview\n", program);
    printf("  %s --width 1920 --height 1080 --samples 256  # HD render\n", program);
    printf("  %s --benchmark                        # Performance test\n", program);
    printf("\n");
}

static Options parse_args(int argc, char* argv[]) {
    Options opts = {
        .width = 800,
        .height = 600,
        .samples = 16,
        .max_depth = 10,
        .output = "output/render.png",
        .scene = "demo",
        .scene_file = NULL,
        .benchmark = 0,
        .preview = 0,
        .debug_mode = 0,
        .help = 0
    };

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            opts.help = 1;
        } else if (strcmp(argv[i], "--width") == 0 && i + 1 < argc) {
            opts.width = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--height") == 0 && i + 1 < argc) {
            opts.height = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--samples") == 0 && i + 1 < argc) {
            opts.samples = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--depth") == 0 && i + 1 < argc) {
            opts.max_depth = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            opts.output = argv[++i];
        } else if (strcmp(argv[i], "--scene") == 0 && i + 1 < argc) {
            opts.scene = argv[++i];
        } else if (strcmp(argv[i], "--scene-file") == 0 && i + 1 < argc) {
            opts.scene_file = argv[++i];
        } else if (strcmp(argv[i], "--benchmark") == 0) {
            opts.benchmark = 1;
        } else if (strcmp(argv[i], "--preview") == 0) {
            opts.preview = 1;
        } else if (strcmp(argv[i], "--debug-normals") == 0) {
            opts.debug_mode = 1;
        } else if (strcmp(argv[i], "--debug-depth") == 0) {
            opts.debug_mode = 2;
        }
    }

    return opts;
}

// ============================================================================
// SCENE SETUP
// ============================================================================

/**
 * Create a demo scene with various materials and spheres.
 * Similar to "Ray Tracing in One Weekend" cover image.
 */
static void setup_demo_scene(Scene* scene) {
    // Ground material (large gray sphere)
    Material ground_mat = {
        .albedo = make_float3(0.5f, 0.5f, 0.5f),
        .type = MATERIAL_LAMBERTIAN
    };
    uint32_t ground_id = sf_scene_add_material(scene, &ground_mat);
    sf_scene_add_sphere(scene, make_float3(0.0f, -1000.0f, 0.0f), 1000.0f, ground_id);

    // Three main spheres
    // Center: Glass sphere
    Material glass_mat = {
        .albedo = make_float3(1.0f, 1.0f, 1.0f),
        .ior = 1.5f,
        .type = MATERIAL_DIELECTRIC
    };
    uint32_t glass_id = sf_scene_add_material(scene, &glass_mat);
    sf_scene_add_sphere(scene, make_float3(0.0f, 1.0f, 0.0f), 1.0f, glass_id);

    // Left: Matte sphere
    Material matte_mat = {
        .albedo = make_float3(0.4f, 0.2f, 0.1f),
        .type = MATERIAL_LAMBERTIAN
    };
    uint32_t matte_id = sf_scene_add_material(scene, &matte_mat);
    sf_scene_add_sphere(scene, make_float3(-4.0f, 1.0f, 0.0f), 1.0f, matte_id);

    // Right: Metal sphere
    Material metal_mat = {
        .albedo = make_float3(0.7f, 0.6f, 0.5f),
        .roughness = 0.0f,
        .type = MATERIAL_METAL
    };
    uint32_t metal_id = sf_scene_add_material(scene, &metal_mat);
    sf_scene_add_sphere(scene, make_float3(4.0f, 1.0f, 0.0f), 1.0f, metal_id);

    // Random small spheres
    srand(42);  // Fixed seed for reproducibility

    for (int a = -5; a < 5; a++) {
        for (int b = -5; b < 5; b++) {
            float choose_mat = (float)rand() / RAND_MAX;
            float3 center = make_float3(
                a + 0.9f * (float)rand() / RAND_MAX,
                0.2f,
                b + 0.9f * (float)rand() / RAND_MAX
            );

            // Skip if too close to main spheres
            float dx = center.x - 4.0f;
            float dz = center.z;
            if (sqrtf(dx * dx + dz * dz) < 0.9f) continue;

            dx = center.x;
            if (sqrtf(dx * dx + dz * dz) < 0.9f) continue;

            dx = center.x + 4.0f;
            if (sqrtf(dx * dx + dz * dz) < 0.9f) continue;

            Material mat;
            memset(&mat, 0, sizeof(mat));

            if (choose_mat < 0.65f) {
                // Diffuse
                mat.albedo = make_float3(
                    (float)rand() / RAND_MAX * (float)rand() / RAND_MAX,
                    (float)rand() / RAND_MAX * (float)rand() / RAND_MAX,
                    (float)rand() / RAND_MAX * (float)rand() / RAND_MAX
                );
                mat.type = MATERIAL_LAMBERTIAN;
            } else if (choose_mat < 0.85f) {
                // Metal
                mat.albedo = make_float3(
                    0.5f + 0.5f * (float)rand() / RAND_MAX,
                    0.5f + 0.5f * (float)rand() / RAND_MAX,
                    0.5f + 0.5f * (float)rand() / RAND_MAX
                );
                mat.roughness = 0.5f * (float)rand() / RAND_MAX;
                mat.type = MATERIAL_METAL;
            } else {
                // Glass
                mat.albedo = make_float3(1.0f, 1.0f, 1.0f);
                mat.ior = 1.5f;
                mat.type = MATERIAL_DIELECTRIC;
            }

            uint32_t mat_id = sf_scene_add_material(scene, &mat);
            sf_scene_add_sphere(scene, center, 0.2f, mat_id);
        }
    }

    printf("Demo scene: %u spheres, %u materials\n",
           scene->num_spheres, scene->num_materials);
}

/**
 * Create a stress test scene with thousands of spheres.
 */
static void setup_stress_scene(Scene* scene) {
    // Ground
    Material ground_mat = {
        .albedo = make_float3(0.48f, 0.83f, 0.53f),
        .type = MATERIAL_LAMBERTIAN
    };
    uint32_t ground_id = sf_scene_add_material(scene, &ground_mat);
    sf_scene_add_sphere(scene, make_float3(0.0f, -10000.0f, 0.0f), 10000.0f, ground_id);

    srand(12345);

    // Create grid of random spheres
    int grid_size = 20;  // 20x20 = 400 base positions
    for (int a = -grid_size; a < grid_size; a++) {
        for (int b = -grid_size; b < grid_size; b++) {
            float choose_mat = (float)rand() / RAND_MAX;
            float height = 0.2f + 0.3f * (float)rand() / RAND_MAX;
            float3 center = make_float3(
                a + 0.9f * (float)rand() / RAND_MAX,
                height,
                b + 0.9f * (float)rand() / RAND_MAX
            );

            Material mat;
            memset(&mat, 0, sizeof(mat));

            if (choose_mat < 0.7f) {
                mat.albedo = make_float3(
                    (float)rand() / RAND_MAX * (float)rand() / RAND_MAX,
                    (float)rand() / RAND_MAX * (float)rand() / RAND_MAX,
                    (float)rand() / RAND_MAX * (float)rand() / RAND_MAX
                );
                mat.type = MATERIAL_LAMBERTIAN;
            } else if (choose_mat < 0.9f) {
                mat.albedo = make_float3(
                    0.5f + 0.5f * (float)rand() / RAND_MAX,
                    0.5f + 0.5f * (float)rand() / RAND_MAX,
                    0.5f + 0.5f * (float)rand() / RAND_MAX
                );
                mat.roughness = 0.5f * (float)rand() / RAND_MAX;
                mat.type = MATERIAL_METAL;
            } else {
                mat.albedo = make_float3(1.0f, 1.0f, 1.0f);
                mat.ior = 1.5f;
                mat.type = MATERIAL_DIELECTRIC;
            }

            uint32_t mat_id = sf_scene_add_material(scene, &mat);
            sf_scene_add_sphere(scene, center, 0.2f, mat_id);
        }
    }

    // Add some larger feature spheres
    Material glass = {.albedo = make_float3(1,1,1), .ior = 1.5f, .type = MATERIAL_DIELECTRIC};
    uint32_t glass_id = sf_scene_add_material(scene, &glass);
    sf_scene_add_sphere(scene, make_float3(0.0f, 1.5f, 0.0f), 1.5f, glass_id);

    Material metal = {.albedo = make_float3(0.8f, 0.8f, 0.9f), .roughness = 0.0f, .type = MATERIAL_METAL};
    uint32_t metal_id = sf_scene_add_material(scene, &metal);
    sf_scene_add_sphere(scene, make_float3(4.0f, 1.0f, 4.0f), 1.0f, metal_id);

    printf("Stress scene: %u spheres, %u materials\n",
           scene->num_spheres, scene->num_materials);
}

/**
 * Create a PBR material test scene.
 * Shows a grid of spheres with varying metallic and roughness values.
 */
static void setup_pbr_scene(Scene* scene) {
    // Checker ground
    Material ground_mat = {
        .albedo = make_float3(0.5f, 0.5f, 0.5f),
        .type = MATERIAL_LAMBERTIAN
    };
    uint32_t ground_id = sf_scene_add_material(scene, &ground_mat);
    sf_scene_add_sphere(scene, make_float3(0.0f, -1000.0f, 0.0f), 1000.0f, ground_id);

    // Create a 5x5 grid of PBR spheres
    // X axis: roughness (0.0 to 1.0)
    // Z axis: metallic (0.0 to 1.0)
    int grid_size = 5;
    float spacing = 1.2f;
    float start_x = -spacing * (grid_size - 1) / 2.0f;
    float start_z = -spacing * (grid_size - 1) / 2.0f;

    for (int iz = 0; iz < grid_size; iz++) {
        for (int ix = 0; ix < grid_size; ix++) {
            float roughness = (float)ix / (grid_size - 1);
            float metallic = (float)iz / (grid_size - 1);

            float x = start_x + ix * spacing;
            float z = start_z + iz * spacing;

            Material pbr_mat = {
                .albedo = make_float3(0.9f, 0.2f, 0.2f),  // Red base color
                .metallic = metallic,
                .roughness = fmaxf(0.05f, roughness),  // Avoid 0 roughness (perfect mirror artifacts)
                .type = MATERIAL_PBR
            };

            uint32_t mat_id = sf_scene_add_material(scene, &pbr_mat);
            sf_scene_add_sphere(scene, make_float3(x, 0.5f, z), 0.5f, mat_id);
        }
    }

    // Add some reference materials around the grid

    // Pure dielectric (glass)
    Material glass_mat = {
        .albedo = make_float3(1.0f, 1.0f, 1.0f),
        .ior = 1.5f,
        .type = MATERIAL_DIELECTRIC
    };
    uint32_t glass_id = sf_scene_add_material(scene, &glass_mat);
    sf_scene_add_sphere(scene, make_float3(-4.0f, 0.7f, 0.0f), 0.7f, glass_id);

    // Pure gold metal (traditional metal shader)
    Material gold_mat = {
        .albedo = make_float3(1.0f, 0.84f, 0.0f),
        .roughness = 0.1f,
        .type = MATERIAL_METAL
    };
    uint32_t gold_id = sf_scene_add_material(scene, &gold_mat);
    sf_scene_add_sphere(scene, make_float3(4.0f, 0.7f, 0.0f), 0.7f, gold_id);

    // Add a light source
    Material light_mat = {
        .albedo = make_float3(1.0f, 1.0f, 1.0f),
        .emission_color = make_float3(12.0f, 12.0f, 12.0f),
        .emission_intensity = 1.0f,
        .type = MATERIAL_EMISSIVE
    };
    uint32_t light_id = sf_scene_add_material(scene, &light_mat);
    sf_scene_add_sphere(scene, make_float3(0.0f, 8.0f, 3.0f), 2.0f, light_id);

    printf("PBR scene: %u spheres, %u materials\n",
           scene->num_spheres, scene->num_materials);
    printf("  Grid shows roughness (X axis) vs metallic (Z axis)\n");
}

/**
 * Create a scene with triangles to test triangle rendering.
 */
static void setup_triangle_scene(Scene* scene) {
    // Ground sphere
    Material ground_mat = {
        .albedo = make_float3(0.4f, 0.4f, 0.4f),
        .type = MATERIAL_LAMBERTIAN
    };
    uint32_t ground_id = sf_scene_add_material(scene, &ground_mat);
    sf_scene_add_sphere(scene, make_float3(0.0f, -1000.0f, 0.0f), 1000.0f, ground_id);

    // Red triangle - a simple flat triangle
    Material red_mat = {
        .albedo = make_float3(0.8f, 0.1f, 0.1f),
        .type = MATERIAL_LAMBERTIAN
    };
    uint32_t red_id = sf_scene_add_material(scene, &red_mat);
    sf_scene_add_triangle(scene,
        make_float3(-2.0f, 0.0f, -1.0f),
        make_float3(0.0f, 3.0f, -1.0f),
        make_float3(2.0f, 0.0f, -1.0f),
        red_id);

    // Green triangle - tilted
    Material green_mat = {
        .albedo = make_float3(0.1f, 0.8f, 0.1f),
        .type = MATERIAL_LAMBERTIAN
    };
    uint32_t green_id = sf_scene_add_material(scene, &green_mat);
    sf_scene_add_triangle(scene,
        make_float3(-3.0f, 0.0f, 0.0f),
        make_float3(-1.0f, 2.0f, 1.0f),
        make_float3(-3.0f, 0.0f, 2.0f),
        green_id);

    // Blue metallic triangle
    Material blue_mat = {
        .albedo = make_float3(0.3f, 0.3f, 0.9f),
        .roughness = 0.3f,
        .type = MATERIAL_METAL
    };
    uint32_t blue_id = sf_scene_add_material(scene, &blue_mat);
    sf_scene_add_triangle(scene,
        make_float3(1.0f, 0.0f, 0.0f),
        make_float3(3.0f, 0.0f, 2.0f),
        make_float3(2.0f, 2.5f, 1.0f),
        blue_id);

    // Add a glass sphere next to the triangles
    Material glass_mat = {
        .albedo = make_float3(1.0f, 1.0f, 1.0f),
        .ior = 1.5f,
        .type = MATERIAL_DIELECTRIC
    };
    uint32_t glass_id = sf_scene_add_material(scene, &glass_mat);
    sf_scene_add_sphere(scene, make_float3(0.0f, 1.0f, 2.0f), 1.0f, glass_id);

    // Add a small emissive sphere as light
    Material light_mat = {
        .albedo = make_float3(1.0f, 1.0f, 1.0f),
        .emission_color = make_float3(8.0f, 8.0f, 8.0f),
        .emission_intensity = 1.0f,
        .type = MATERIAL_EMISSIVE
    };
    uint32_t light_id = sf_scene_add_material(scene, &light_mat);
    sf_scene_add_sphere(scene, make_float3(0.0f, 6.0f, 0.0f), 1.0f, light_id);

    printf("Triangle scene: %u spheres, %u triangles, %u materials\n",
           scene->num_spheres, scene->num_triangles, scene->num_materials);
}

/**
 * Create a mesh stress test scene with many triangles.
 * Tests combined BVH performance with larger triangle counts.
 */
static void setup_mesh_scene(Scene* scene) {
    // Materials
    Material ground_mat = {
        .albedo = make_float3(0.3f, 0.5f, 0.3f),
        .type = MATERIAL_LAMBERTIAN
    };
    uint32_t ground_id = sf_scene_add_material(scene, &ground_mat);

    Material mesh_mat = {
        .albedo = make_float3(0.9f, 0.6f, 0.2f),
        .roughness = 0.3f,
        .type = MATERIAL_METAL
    };
    uint32_t mesh_id = sf_scene_add_material(scene, &mesh_mat);

    Material glass_mat = {
        .albedo = make_float3(1.0f, 1.0f, 1.0f),
        .ior = 1.5f,
        .type = MATERIAL_DIELECTRIC
    };
    uint32_t glass_id = sf_scene_add_material(scene, &glass_mat);

    Material light_mat = {
        .albedo = make_float3(1.0f, 1.0f, 1.0f),
        .emission_color = make_float3(12.0f, 12.0f, 12.0f),
        .emission_intensity = 1.0f,
        .type = MATERIAL_EMISSIVE
    };
    uint32_t light_id = sf_scene_add_material(scene, &light_mat);

    // Create a ground plane made of triangles (grid)
    int grid_size = 20;  // 20x20 grid = 800 triangles
    float tile_size = 1.0f;
    float ground_y = 0.0f;
    float start_x = -grid_size * tile_size / 2.0f;
    float start_z = -grid_size * tile_size / 2.0f;

    for (int i = 0; i < grid_size; i++) {
        for (int j = 0; j < grid_size; j++) {
            float x0 = start_x + i * tile_size;
            float z0 = start_z + j * tile_size;
            float x1 = x0 + tile_size;
            float z1 = z0 + tile_size;

            // Two triangles per grid cell
            sf_scene_add_triangle(scene,
                make_float3(x0, ground_y, z0),
                make_float3(x1, ground_y, z0),
                make_float3(x1, ground_y, z1),
                ground_id);
            sf_scene_add_triangle(scene,
                make_float3(x0, ground_y, z0),
                make_float3(x1, ground_y, z1),
                make_float3(x0, ground_y, z1),
                ground_id);
        }
    }

    // Create a simple box mesh (12 triangles)
    float box_size = 1.5f;
    float bh = box_size / 2.0f;

    // Front face (+Z)
    sf_scene_add_triangle(scene,
        make_float3(-bh, 0, bh), make_float3(bh, 0, bh), make_float3(bh, box_size, bh), mesh_id);
    sf_scene_add_triangle(scene,
        make_float3(-bh, 0, bh), make_float3(bh, box_size, bh), make_float3(-bh, box_size, bh), mesh_id);

    // Back face (-Z)
    sf_scene_add_triangle(scene,
        make_float3(bh, 0, -bh), make_float3(-bh, 0, -bh), make_float3(-bh, box_size, -bh), mesh_id);
    sf_scene_add_triangle(scene,
        make_float3(bh, 0, -bh), make_float3(-bh, box_size, -bh), make_float3(bh, box_size, -bh), mesh_id);

    // Left face (-X)
    sf_scene_add_triangle(scene,
        make_float3(-bh, 0, -bh), make_float3(-bh, 0, bh), make_float3(-bh, box_size, bh), mesh_id);
    sf_scene_add_triangle(scene,
        make_float3(-bh, 0, -bh), make_float3(-bh, box_size, bh), make_float3(-bh, box_size, -bh), mesh_id);

    // Right face (+X)
    sf_scene_add_triangle(scene,
        make_float3(bh, 0, bh), make_float3(bh, 0, -bh), make_float3(bh, box_size, -bh), mesh_id);
    sf_scene_add_triangle(scene,
        make_float3(bh, 0, bh), make_float3(bh, box_size, -bh), make_float3(bh, box_size, bh), mesh_id);

    // Top face (+Y)
    sf_scene_add_triangle(scene,
        make_float3(-bh, box_size, bh), make_float3(bh, box_size, bh), make_float3(bh, box_size, -bh), mesh_id);
    sf_scene_add_triangle(scene,
        make_float3(-bh, box_size, bh), make_float3(bh, box_size, -bh), make_float3(-bh, box_size, -bh), mesh_id);

    // Add spheres for visual variety
    sf_scene_add_sphere(scene, make_float3(2.5f, 0.8f, 0.0f), 0.8f, glass_id);
    sf_scene_add_sphere(scene, make_float3(-2.5f, 0.6f, 1.0f), 0.6f, mesh_id);

    // Light sphere
    sf_scene_add_sphere(scene, make_float3(0.0f, 8.0f, 0.0f), 2.0f, light_id);

    printf("Mesh scene: %u spheres, %u triangles, %u materials\n",
           scene->num_spheres, scene->num_triangles, scene->num_materials);
    printf("  Ground grid: %dx%d = %d tiles (%d triangles)\n",
           grid_size, grid_size, grid_size * grid_size, grid_size * grid_size * 2);
}

/**
 * Create depth-of-field test scene.
 * Shows spheres at varying distances to demonstrate bokeh effect.
 */
static void setup_dof_scene(Scene* scene) {
    // Ground
    Material ground_mat = {
        .albedo = make_float3(0.3f, 0.3f, 0.35f),
        .type = MATERIAL_LAMBERTIAN
    };
    uint32_t ground_id = sf_scene_add_material(scene, &ground_mat);
    sf_scene_add_sphere(scene, make_float3(0.0f, -1000.0f, 0.0f), 1000.0f, ground_id);

    // Row of spheres at varying Z distances to show focus falloff
    // Focus will be set on the middle sphere (z = 0)
    float colors[][3] = {
        {0.9f, 0.2f, 0.2f},  // Red (far background)
        {0.9f, 0.5f, 0.2f},  // Orange
        {0.9f, 0.9f, 0.2f},  // Yellow
        {0.2f, 0.9f, 0.2f},  // Green (focus point)
        {0.2f, 0.9f, 0.9f},  // Cyan
        {0.2f, 0.2f, 0.9f},  // Blue
        {0.9f, 0.2f, 0.9f},  // Magenta (foreground)
    };

    float z_positions[] = {-8.0f, -5.0f, -2.5f, 0.0f, 2.5f, 5.0f, 8.0f};
    int num_spheres = 7;

    for (int i = 0; i < num_spheres; i++) {
        Material mat = {
            .albedo = make_float3(colors[i][0], colors[i][1], colors[i][2]),
            .type = MATERIAL_LAMBERTIAN
        };
        uint32_t mat_id = sf_scene_add_material(scene, &mat);
        sf_scene_add_sphere(scene, make_float3(0.0f, 0.6f, z_positions[i]), 0.6f, mat_id);
    }

    // Add some metal spheres for specular highlights (shows bokeh circles)
    Material chrome = {
        .albedo = make_float3(0.95f, 0.95f, 0.95f),
        .roughness = 0.05f,
        .type = MATERIAL_METAL
    };
    uint32_t chrome_id = sf_scene_add_material(scene, &chrome);

    // Background chrome spheres (will be blurry)
    sf_scene_add_sphere(scene, make_float3(-2.0f, 0.4f, -6.0f), 0.4f, chrome_id);
    sf_scene_add_sphere(scene, make_float3(2.0f, 0.4f, -7.0f), 0.4f, chrome_id);

    // Foreground chrome sphere (will be blurry)
    sf_scene_add_sphere(scene, make_float3(-1.5f, 0.5f, 6.0f), 0.5f, chrome_id);

    // Glass sphere at focus point
    Material glass = {
        .albedo = make_float3(1.0f, 1.0f, 1.0f),
        .ior = 1.5f,
        .type = MATERIAL_DIELECTRIC
    };
    uint32_t glass_id = sf_scene_add_material(scene, &glass);
    sf_scene_add_sphere(scene, make_float3(1.5f, 0.8f, 0.0f), 0.8f, glass_id);

    // Light source
    Material light = {
        .albedo = make_float3(1.0f, 1.0f, 1.0f),
        .emission_color = make_float3(15.0f, 15.0f, 15.0f),
        .emission_intensity = 1.0f,
        .type = MATERIAL_EMISSIVE
    };
    uint32_t light_id = sf_scene_add_material(scene, &light);
    sf_scene_add_sphere(scene, make_float3(0.0f, 10.0f, 0.0f), 3.0f, light_id);

    printf("DOF scene: %u spheres, %u materials\n",
           scene->num_spheres, scene->num_materials);
    printf("  Focus point: z=0 (green sphere)\n");
}

/**
 * Create motion blur test scene.
 * Shows spheres with different velocities to demonstrate motion blur.
 */
static void setup_motion_scene(Scene* scene) {
    // Ground
    Material ground_mat = {
        .albedo = make_float3(0.4f, 0.4f, 0.4f),
        .type = MATERIAL_LAMBERTIAN
    };
    uint32_t ground_id = sf_scene_add_material(scene, &ground_mat);
    sf_scene_add_sphere(scene, make_float3(0.0f, -1000.0f, 0.0f), 1000.0f, ground_id);

    // Stationary reference sphere
    Material red_mat = {
        .albedo = make_float3(0.8f, 0.2f, 0.2f),
        .type = MATERIAL_LAMBERTIAN
    };
    uint32_t red_id = sf_scene_add_material(scene, &red_mat);
    sf_scene_add_sphere(scene, make_float3(-3.0f, 0.5f, 0.0f), 0.5f, red_id);

    // Moving spheres with different velocities
    Material blue_mat = {
        .albedo = make_float3(0.2f, 0.2f, 0.9f),
        .type = MATERIAL_LAMBERTIAN
    };
    uint32_t blue_id = sf_scene_add_material(scene, &blue_mat);

    // Slow moving sphere
    sf_scene_add_sphere_moving(scene,
        make_float3(-1.0f, 0.5f, 0.0f), 0.5f,
        make_float3(1.0f, 0.0f, 0.0f),  // velocity: right
        blue_id);

    // Fast moving sphere
    Material green_mat = {
        .albedo = make_float3(0.2f, 0.9f, 0.2f),
        .type = MATERIAL_LAMBERTIAN
    };
    uint32_t green_id = sf_scene_add_material(scene, &green_mat);
    sf_scene_add_sphere_moving(scene,
        make_float3(1.0f, 0.5f, 0.0f), 0.5f,
        make_float3(3.0f, 0.0f, 0.0f),  // velocity: faster right
        green_id);

    // Falling sphere
    Material yellow_mat = {
        .albedo = make_float3(0.9f, 0.9f, 0.2f),
        .type = MATERIAL_LAMBERTIAN
    };
    uint32_t yellow_id = sf_scene_add_material(scene, &yellow_mat);
    sf_scene_add_sphere_moving(scene,
        make_float3(3.0f, 1.5f, 0.0f), 0.5f,
        make_float3(0.0f, -3.0f, 0.0f),  // velocity: down
        yellow_id);

    // Metal sphere moving diagonally
    Material chrome = {
        .albedo = make_float3(0.95f, 0.95f, 0.95f),
        .roughness = 0.1f,
        .type = MATERIAL_METAL
    };
    uint32_t chrome_id = sf_scene_add_material(scene, &chrome);
    sf_scene_add_sphere_moving(scene,
        make_float3(0.0f, 0.8f, -2.0f), 0.8f,
        make_float3(2.0f, 1.0f, 0.0f),  // velocity: up-right
        chrome_id);

    // Light
    Material light = {
        .albedo = make_float3(1.0f, 1.0f, 1.0f),
        .emission_color = make_float3(15.0f, 15.0f, 15.0f),
        .emission_intensity = 1.0f,
        .type = MATERIAL_EMISSIVE
    };
    uint32_t light_id = sf_scene_add_material(scene, &light);
    sf_scene_add_sphere(scene, make_float3(0.0f, 10.0f, 0.0f), 3.0f, light_id);

    printf("Motion blur scene: %u spheres, %u materials\n",
           scene->num_spheres, scene->num_materials);
    printf("  Red sphere: stationary reference\n");
    printf("  Blue/green/yellow: moving spheres with motion blur\n");
}

/**
 * Create Cornell box scene.
 */
static void setup_cornell_scene(Scene* scene) {
    // Simple Cornell box approximation with spheres

    // White material
    Material white_mat = {
        .albedo = make_float3(0.73f, 0.73f, 0.73f),
        .type = MATERIAL_LAMBERTIAN
    };
    uint32_t white_id = sf_scene_add_material(scene, &white_mat);

    // Red material
    Material red_mat = {
        .albedo = make_float3(0.65f, 0.05f, 0.05f),
        .type = MATERIAL_LAMBERTIAN
    };
    uint32_t red_id = sf_scene_add_material(scene, &red_mat);

    // Green material
    Material green_mat = {
        .albedo = make_float3(0.12f, 0.45f, 0.15f),
        .type = MATERIAL_LAMBERTIAN
    };
    uint32_t green_id = sf_scene_add_material(scene, &green_mat);

    // Light material
    Material light_mat = {
        .albedo = make_float3(1.0f, 1.0f, 1.0f),
        .emission_color = make_float3(15.0f, 15.0f, 15.0f),
        .emission_intensity = 1.0f,
        .type = MATERIAL_EMISSIVE
    };
    uint32_t light_id = sf_scene_add_material(scene, &light_mat);

    // Floor
    sf_scene_add_sphere(scene, make_float3(0.0f, -1000.0f, 0.0f), 1000.0f, white_id);

    // Ceiling
    sf_scene_add_sphere(scene, make_float3(0.0f, 1006.0f, 0.0f), 1000.0f, white_id);

    // Back wall
    sf_scene_add_sphere(scene, make_float3(0.0f, 3.0f, -1006.0f), 1000.0f, white_id);

    // Left wall (red)
    sf_scene_add_sphere(scene, make_float3(-1003.0f, 3.0f, 0.0f), 1000.0f, red_id);

    // Right wall (green)
    sf_scene_add_sphere(scene, make_float3(1003.0f, 3.0f, 0.0f), 1000.0f, green_id);

    // Light (small sphere at top)
    sf_scene_add_sphere(scene, make_float3(0.0f, 5.5f, 0.0f), 1.5f, light_id);

    // Two spheres in the box
    Material glass_mat = {
        .albedo = make_float3(1.0f, 1.0f, 1.0f),
        .ior = 1.5f,
        .type = MATERIAL_DIELECTRIC
    };
    uint32_t glass_id = sf_scene_add_material(scene, &glass_mat);

    Material metal_mat = {
        .albedo = make_float3(0.9f, 0.9f, 0.9f),
        .roughness = 0.1f,
        .type = MATERIAL_METAL
    };
    uint32_t metal_id = sf_scene_add_material(scene, &metal_mat);

    sf_scene_add_sphere(scene, make_float3(-1.2f, 1.0f, -1.0f), 1.0f, glass_id);
    sf_scene_add_sphere(scene, make_float3(1.2f, 1.5f, 0.5f), 1.5f, metal_id);

    printf("Cornell scene: %u spheres, %u materials\n",
           scene->num_spheres, scene->num_materials);
}

// ============================================================================
// MAIN
// ============================================================================

int main(int argc, char* argv[]) {
    @autoreleasepool {
        Options opts = parse_args(argc, argv);

        if (opts.help) {
            print_usage(argv[0]);
            return 0;
        }

        printf("SpectraForge Metal - GPU Ray Tracer\n");
        printf("====================================\n\n");

        // Create renderer
        MetalRenderer* renderer = sf_renderer_create();
        if (!renderer) {
            fprintf(stderr, "Failed to create renderer\n");
            return 1;
        }

        // Create scene
        Scene* scene = sf_scene_create();
        if (!scene) {
            fprintf(stderr, "Failed to create scene\n");
            sf_renderer_destroy(renderer);
            return 1;
        }

        // Configure initial settings (may be overridden by JSON)
        RenderSettings settings = {
            .width = opts.width,
            .height = opts.height,
            .samples_per_pixel = opts.samples,
            .max_depth = opts.max_depth,
            .frame_number = 0,
            .num_spheres = 0,
            .num_triangles = 0,
            .num_bvh_nodes = 0,
            .use_sky_gradient = 1,
            .background_r = 0.0f,
            .background_g = 0.0f,
            .background_b = 0.0f,
            .num_primitive_indices = 0
        };

        // Camera will be set up based on scene
        Camera camera;
        bool camera_from_json = false;
        bool use_motion_blur = false;

        // Setup scene - either from JSON file or preset
        if (opts.scene_file) {
            // Load scene from JSON file
            if (sf_scene_load_json(scene, opts.scene_file, &camera, &settings) != 0) {
                fprintf(stderr, "Failed to load scene from: %s\n", opts.scene_file);
                sf_scene_destroy(scene);
                sf_renderer_destroy(renderer);
                return 1;
            }
            camera_from_json = true;

            // Check if any spheres have velocity (motion blur)
            for (uint32_t i = 0; i < scene->num_spheres; i++) {
                Sphere* s = &scene->spheres[i];
                if (s->velocity_x != 0 || s->velocity_y != 0 || s->velocity_z != 0) {
                    use_motion_blur = true;
                    break;
                }
            }
        } else if (strcmp(opts.scene, "cornell") == 0) {
            setup_cornell_scene(scene);
        } else if (strcmp(opts.scene, "stress") == 0) {
            setup_stress_scene(scene);
        } else if (strcmp(opts.scene, "triangle") == 0) {
            setup_triangle_scene(scene);
        } else if (strcmp(opts.scene, "pbr") == 0) {
            setup_pbr_scene(scene);
        } else if (strcmp(opts.scene, "mesh") == 0) {
            setup_mesh_scene(scene);
        } else if (strcmp(opts.scene, "dof") == 0) {
            setup_dof_scene(scene);
        } else if (strcmp(opts.scene, "motion") == 0) {
            setup_motion_scene(scene);
            use_motion_blur = true;
        } else {
            setup_demo_scene(scene);
        }

        // Build BVH acceleration structure
        // Note: Skip BVH for motion blur (requires expanded bounds, not yet implemented)
        if (!use_motion_blur) {
            printf("Building BVH...\n");
            if (scene->num_triangles > 0) {
                // Use combined BVH for scenes with triangles
                sf_scene_build_bvh_full(scene);
            } else {
                // Use optimized sphere-only BVH
                sf_scene_build_bvh(scene);
            }
        } else {
            printf("Motion blur detected: skipping BVH (brute force for motion support)\n");
        }

        // Update settings with scene info
        settings.num_spheres = scene->num_spheres;
        settings.num_triangles = scene->num_triangles;
        settings.num_bvh_nodes = scene->num_bvh_nodes;
        settings.num_primitive_indices = scene->num_primitive_indices;

        sf_renderer_set_settings(renderer, &settings);

        // Setup camera (if not already loaded from JSON)
        float aspect_ratio = (float)opts.width / (float)opts.height;

        if (camera_from_json) {
            printf("  Camera loaded from JSON\n");
        } else if (strcmp(opts.scene, "cornell") == 0) {
            camera = sf_camera_create(
                make_float3(0.0f, 3.0f, 10.0f),   // look_from
                make_float3(0.0f, 2.5f, 0.0f),   // look_at
                make_float3(0.0f, 1.0f, 0.0f),   // vup
                40.0f,                            // vfov
                aspect_ratio,
                0.0f,                             // aperture (no DOF)
                10.0f                             // focus_dist
            );
        } else if (strcmp(opts.scene, "triangle") == 0) {
            camera = sf_camera_create(
                make_float3(0.0f, 3.0f, 8.0f),   // look_from
                make_float3(0.0f, 1.0f, 0.0f),   // look_at
                make_float3(0.0f, 1.0f, 0.0f),   // vup
                40.0f,                            // vfov
                aspect_ratio,
                0.0f,                             // aperture (no DOF)
                8.0f                              // focus_dist
            );
        } else if (strcmp(opts.scene, "pbr") == 0) {
            camera = sf_camera_create(
                make_float3(0.0f, 4.0f, 8.0f),   // look_from - elevated to see grid
                make_float3(0.0f, 0.5f, 0.0f),   // look_at - center of grid
                make_float3(0.0f, 1.0f, 0.0f),   // vup
                35.0f,                            // vfov
                aspect_ratio,
                0.0f,                             // aperture (no DOF for clarity)
                8.0f                              // focus_dist
            );
        } else if (strcmp(opts.scene, "stress") == 0) {
            camera = sf_camera_create(
                make_float3(26.0f, 6.0f, 6.0f),  // look_from (higher up, farther back)
                make_float3(0.0f, 0.0f, 0.0f),  // look_at
                make_float3(0.0f, 1.0f, 0.0f),  // vup
                25.0f,                           // vfov (wider view)
                aspect_ratio,
                0.0f,                            // no DOF for stress test
                26.0f                            // focus_dist
            );
        } else if (strcmp(opts.scene, "dof") == 0) {
            // DOF demo: camera looking down the row of spheres
            // Focus on middle (green) sphere at z=0, use wide aperture
            camera = sf_camera_create(
                make_float3(0.0f, 1.5f, 12.0f),  // look_from (behind, elevated)
                make_float3(0.0f, 0.5f, 0.0f),   // look_at (focus point)
                make_float3(0.0f, 1.0f, 0.0f),   // vup
                30.0f,                            // vfov
                aspect_ratio,
                0.3f,                             // aperture (wide for strong DOF)
                12.0f                             // focus_dist (distance to z=0)
            );
        } else if (strcmp(opts.scene, "motion") == 0) {
            // Motion blur demo
            camera = sf_camera_create(
                make_float3(0.0f, 3.0f, 10.0f),  // look_from
                make_float3(0.0f, 0.5f, 0.0f),   // look_at
                make_float3(0.0f, 1.0f, 0.0f),   // vup
                35.0f,                            // vfov
                aspect_ratio,
                0.0f,                             // no DOF to isolate motion blur effect
                10.0f                             // focus_dist
            );
            // Enable motion blur shutter
            camera.time0 = 0.0f;
            camera.time1 = 1.0f;
        } else {
            camera = sf_camera_create(
                make_float3(13.0f, 2.0f, 3.0f),  // look_from
                make_float3(0.0f, 0.0f, 0.0f),  // look_at
                make_float3(0.0f, 1.0f, 0.0f),  // vup
                20.0f,                           // vfov
                aspect_ratio,
                0.1f,                            // aperture
                10.0f                            // focus_dist
            );
        }

        sf_renderer_set_camera(renderer, &camera);

        // Upload scene to GPU
        sf_renderer_upload_scene(renderer, scene);

        // Preview mode - open interactive window
        if (opts.preview) {
            printf("\nOpening preview window...\n");
            int result = sf_preview_run(renderer, scene, opts.width, opts.height);
            sf_scene_destroy(scene);
            sf_renderer_destroy(renderer);
            return result;
        }

        // Allocate output buffer
        size_t pixel_count = opts.width * opts.height;
        float* output = (float*)malloc(pixel_count * 3 * sizeof(float));
        if (!output) {
            fprintf(stderr, "Failed to allocate output buffer\n");
            sf_scene_destroy(scene);
            sf_renderer_destroy(renderer);
            return 1;
        }

        printf("\nRendering %ux%u @ %u spp, max depth %u...\n",
               opts.width, opts.height, opts.samples, opts.max_depth);

        // Render with GPU timing
        sf_renderer_render(renderer, output);

        // Get precise GPU timing
        double gpu_time_ms = sf_renderer_get_render_time(renderer);
        double elapsed = gpu_time_ms / 1000.0;

        // Calculate statistics
        uint64_t total_rays = (uint64_t)opts.width * opts.height * opts.samples * opts.max_depth;
        double grays_per_sec = (double)total_rays / elapsed / 1000000000.0;
        double mrays_per_sec = grays_per_sec * 1000.0;

        printf("\nRender complete!\n");
        printf("  GPU Time: %.2f ms (%.3f sec)\n", gpu_time_ms, elapsed);
        printf("  Rays: ~%llu million\n", (unsigned long long)(total_rays / 1000000));
        printf("  Performance: %.2f Grays/sec (%.0f Mrays/sec)\n", grays_per_sec, mrays_per_sec);

        // Create output directory if needed
        [[NSFileManager defaultManager] createDirectoryAtPath:@"output"
                                  withIntermediateDirectories:YES
                                                   attributes:nil
                                                        error:nil];

        // Save output
        const char* ext = strrchr(opts.output, '.');
        if (ext && strcmp(ext, ".hdr") == 0) {
            sf_save_hdr(opts.output, output, opts.width, opts.height);
        } else {
            sf_save_png(opts.output, output, opts.width, opts.height);
        }

        // Benchmark mode - run multiple iterations with warmup
        if (opts.benchmark) {
            printf("\n--- Performance Benchmark ---\n");
            printf("Resolution: %ux%u, Samples: %u, Max depth: %u\n",
                   opts.width, opts.height, opts.samples, opts.max_depth);
            printf("Rays per frame: %llu million\n\n", (unsigned long long)(total_rays / 1000000));

            // Warmup iterations
            printf("Warmup (2 iterations)...\n");
            for (int i = 0; i < 2; i++) {
                sf_renderer_reset_accumulation(renderer);
                sf_renderer_render(renderer, output);
            }

            // Benchmark iterations
            int iterations = 5;
            double total_time = 0;
            double min_time = 1e30;
            double max_time = 0;

            printf("\nBenchmark iterations:\n");
            for (int i = 0; i < iterations; i++) {
                sf_renderer_reset_accumulation(renderer);
                sf_renderer_render(renderer, output);

                double iter_time_ms = sf_renderer_get_render_time(renderer);
                double iter_grays = (double)total_rays / (iter_time_ms / 1000.0) / 1000000000.0;

                total_time += iter_time_ms;
                min_time = fmin(min_time, iter_time_ms);
                max_time = fmax(max_time, iter_time_ms);

                printf("  [%d] %.2f ms (%.2f Grays/sec)\n", i + 1, iter_time_ms, iter_grays);
            }

            double avg_time = total_time / iterations;
            double avg_grays = (double)total_rays / (avg_time / 1000.0) / 1000000000.0;
            double peak_grays = (double)total_rays / (min_time / 1000.0) / 1000000000.0;

            printf("\n--- Results ---\n");
            printf("  Average: %.2f ms (%.2f Grays/sec)\n", avg_time, avg_grays);
            printf("  Best:    %.2f ms (%.2f Grays/sec)\n", min_time, peak_grays);
            printf("  Worst:   %.2f ms\n", max_time);
            printf("  Variance: %.2f ms\n", max_time - min_time);
        }

        // Cleanup
        free(output);
        sf_scene_destroy(scene);
        sf_renderer_destroy(renderer);

        printf("\nDone!\n");
        return 0;
    }
}
