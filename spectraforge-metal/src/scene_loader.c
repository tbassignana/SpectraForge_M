/**
 * SpectraForge Metal - JSON Scene Loader
 *
 * Loads scene definitions from JSON files.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../lib/cJSON.h"
#include "../include/spectraforge.h"

// Helper to read entire file into string
static char* read_file(const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Error: Cannot open file '%s'\n", filename);
        return NULL;
    }

    fseek(f, 0, SEEK_END);
    long length = ftell(f);
    fseek(f, 0, SEEK_SET);

    char* buffer = (char*)malloc(length + 1);
    if (!buffer) {
        fclose(f);
        return NULL;
    }

    fread(buffer, 1, length, f);
    buffer[length] = '\0';
    fclose(f);

    return buffer;
}

// Parse float3 from JSON array [x, y, z]
static float3 parse_float3(cJSON* arr, float3 default_val) {
    if (!arr || !cJSON_IsArray(arr) || cJSON_GetArraySize(arr) < 3) {
        return default_val;
    }
    return make_float3(
        (float)cJSON_GetArrayItem(arr, 0)->valuedouble,
        (float)cJSON_GetArrayItem(arr, 1)->valuedouble,
        (float)cJSON_GetArrayItem(arr, 2)->valuedouble
    );
}

// Parse material from JSON object
static Material parse_material(cJSON* mat_json) {
    Material mat = {0};

    // Albedo (default white)
    mat.albedo = parse_float3(cJSON_GetObjectItem(mat_json, "albedo"),
                              make_float3(1.0f, 1.0f, 1.0f));

    // Material type
    cJSON* type = cJSON_GetObjectItem(mat_json, "type");
    if (type && cJSON_IsString(type)) {
        const char* type_str = type->valuestring;
        if (strcmp(type_str, "lambertian") == 0) {
            mat.type = MATERIAL_LAMBERTIAN;
        } else if (strcmp(type_str, "metal") == 0) {
            mat.type = MATERIAL_METAL;
        } else if (strcmp(type_str, "dielectric") == 0) {
            mat.type = MATERIAL_DIELECTRIC;
        } else if (strcmp(type_str, "emissive") == 0) {
            mat.type = MATERIAL_EMISSIVE;
        } else if (strcmp(type_str, "pbr") == 0) {
            mat.type = MATERIAL_PBR;
        }
    }

    // Metallic (for PBR/metal)
    cJSON* metallic = cJSON_GetObjectItem(mat_json, "metallic");
    if (metallic) mat.metallic = (float)metallic->valuedouble;

    // Roughness (for metal/PBR)
    cJSON* roughness = cJSON_GetObjectItem(mat_json, "roughness");
    if (roughness) mat.roughness = (float)roughness->valuedouble;

    // IOR (for dielectric)
    cJSON* ior = cJSON_GetObjectItem(mat_json, "ior");
    if (ior) mat.ior = (float)ior->valuedouble;
    else mat.ior = 1.5f;  // Default glass

    // Emission
    cJSON* emission = cJSON_GetObjectItem(mat_json, "emission");
    if (emission) {
        mat.emission_color = parse_float3(cJSON_GetObjectItem(emission, "color"),
                                          make_float3(1.0f, 1.0f, 1.0f));
        cJSON* intensity = cJSON_GetObjectItem(emission, "intensity");
        if (intensity) mat.emission_intensity = (float)intensity->valuedouble;
    }

    return mat;
}

// Load scene from JSON file
int sf_scene_load_json(Scene* scene, const char* filename,
                       Camera* out_camera, RenderSettings* out_settings) {
    if (!scene || !filename) return -1;

    char* json_str = read_file(filename);
    if (!json_str) return -1;

    cJSON* root = cJSON_Parse(json_str);
    free(json_str);

    if (!root) {
        fprintf(stderr, "Error: Failed to parse JSON: %s\n", cJSON_GetErrorPtr());
        return -1;
    }

    printf("Loading scene from: %s\n", filename);

    // Parse materials first (referenced by index)
    cJSON* materials = cJSON_GetObjectItem(root, "materials");
    if (materials && cJSON_IsArray(materials)) {
        int num_materials = cJSON_GetArraySize(materials);
        printf("  Materials: %d\n", num_materials);

        cJSON* mat_json;
        cJSON_ArrayForEach(mat_json, materials) {
            Material mat = parse_material(mat_json);
            sf_scene_add_material(scene, &mat);
        }
    }

    // Parse spheres
    cJSON* spheres = cJSON_GetObjectItem(root, "spheres");
    if (spheres && cJSON_IsArray(spheres)) {
        int num_spheres = cJSON_GetArraySize(spheres);
        printf("  Spheres: %d\n", num_spheres);

        cJSON* sphere_json;
        cJSON_ArrayForEach(sphere_json, spheres) {
            float3 center = parse_float3(cJSON_GetObjectItem(sphere_json, "center"),
                                         make_float3(0, 0, 0));
            cJSON* radius_json = cJSON_GetObjectItem(sphere_json, "radius");
            float radius = radius_json ? (float)radius_json->valuedouble : 1.0f;

            cJSON* material_id_json = cJSON_GetObjectItem(sphere_json, "material");
            uint32_t material_id = material_id_json ? (uint32_t)material_id_json->valueint : 0;

            // Check for velocity (motion blur)
            cJSON* velocity_json = cJSON_GetObjectItem(sphere_json, "velocity");
            if (velocity_json) {
                float3 velocity = parse_float3(velocity_json, make_float3(0, 0, 0));
                sf_scene_add_sphere_moving(scene, center, radius, velocity, material_id);
            } else {
                sf_scene_add_sphere(scene, center, radius, material_id);
            }
        }
    }

    // Parse triangles
    cJSON* triangles = cJSON_GetObjectItem(root, "triangles");
    if (triangles && cJSON_IsArray(triangles)) {
        int num_triangles = cJSON_GetArraySize(triangles);
        printf("  Triangles: %d\n", num_triangles);

        cJSON* tri_json;
        cJSON_ArrayForEach(tri_json, triangles) {
            float3 v0 = parse_float3(cJSON_GetObjectItem(tri_json, "v0"),
                                     make_float3(0, 0, 0));
            float3 v1 = parse_float3(cJSON_GetObjectItem(tri_json, "v1"),
                                     make_float3(1, 0, 0));
            float3 v2 = parse_float3(cJSON_GetObjectItem(tri_json, "v2"),
                                     make_float3(0, 1, 0));

            cJSON* material_id_json = cJSON_GetObjectItem(tri_json, "material");
            uint32_t material_id = material_id_json ? (uint32_t)material_id_json->valueint : 0;

            sf_scene_add_triangle(scene, v0, v1, v2, material_id);
        }
    }

    // Parse camera (optional)
    cJSON* camera_json = cJSON_GetObjectItem(root, "camera");
    if (camera_json && out_camera) {
        float3 look_from = parse_float3(cJSON_GetObjectItem(camera_json, "position"),
                                        make_float3(13, 2, 3));
        float3 look_at = parse_float3(cJSON_GetObjectItem(camera_json, "look_at"),
                                      make_float3(0, 0, 0));
        float3 vup = parse_float3(cJSON_GetObjectItem(camera_json, "up"),
                                  make_float3(0, 1, 0));

        cJSON* fov_json = cJSON_GetObjectItem(camera_json, "fov");
        float fov = fov_json ? (float)fov_json->valuedouble : 40.0f;

        cJSON* aperture_json = cJSON_GetObjectItem(camera_json, "aperture");
        float aperture = aperture_json ? (float)aperture_json->valuedouble : 0.0f;

        cJSON* focus_json = cJSON_GetObjectItem(camera_json, "focus_distance");
        float focus_dist = focus_json ? (float)focus_json->valuedouble : 10.0f;

        // Aspect ratio from settings or default
        float aspect = 16.0f / 9.0f;
        if (out_settings) {
            aspect = (float)out_settings->width / (float)out_settings->height;
        }

        *out_camera = sf_camera_create(look_from, look_at, vup, fov, aspect, aperture, focus_dist);

        // Motion blur shutter
        cJSON* shutter_json = cJSON_GetObjectItem(camera_json, "shutter");
        if (shutter_json && cJSON_IsArray(shutter_json) && cJSON_GetArraySize(shutter_json) >= 2) {
            out_camera->time0 = (float)cJSON_GetArrayItem(shutter_json, 0)->valuedouble;
            out_camera->time1 = (float)cJSON_GetArrayItem(shutter_json, 1)->valuedouble;
        }

        printf("  Camera: pos=(%.1f,%.1f,%.1f) fov=%.1f aperture=%.2f\n",
               look_from.x, look_from.y, look_from.z, fov, aperture);
    }

    // Parse render settings (optional)
    cJSON* settings_json = cJSON_GetObjectItem(root, "settings");
    if (settings_json && out_settings) {
        cJSON* samples = cJSON_GetObjectItem(settings_json, "samples");
        if (samples) out_settings->samples_per_pixel = (uint32_t)samples->valueint;

        cJSON* depth = cJSON_GetObjectItem(settings_json, "max_depth");
        if (depth) out_settings->max_depth = (uint32_t)depth->valueint;

        cJSON* sky = cJSON_GetObjectItem(settings_json, "sky_gradient");
        if (sky) out_settings->use_sky_gradient = cJSON_IsTrue(sky) ? 1 : 0;

        cJSON* background = cJSON_GetObjectItem(settings_json, "background");
        if (background && cJSON_IsArray(background) && cJSON_GetArraySize(background) >= 3) {
            out_settings->background_r = (float)cJSON_GetArrayItem(background, 0)->valuedouble;
            out_settings->background_g = (float)cJSON_GetArrayItem(background, 1)->valuedouble;
            out_settings->background_b = (float)cJSON_GetArrayItem(background, 2)->valuedouble;
        }
    }

    cJSON_Delete(root);
    return 0;
}
