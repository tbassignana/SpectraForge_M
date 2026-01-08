/**
 * SpectraForge Metal - Renderer Implementation
 *
 * Objective-C implementation of the Metal renderer.
 * Handles device initialization, buffer management, and kernel dispatch.
 */

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "../include/spectraforge.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ============================================================================
// INTERNAL RENDERER STRUCTURE
// ============================================================================

struct MetalRenderer {
    // Metal objects
    id<MTLDevice> device;
    id<MTLCommandQueue> commandQueue;
    id<MTLLibrary> library;

    // Compute pipelines
    id<MTLComputePipelineState> pathTracePipeline;
    id<MTLComputePipelineState> simpleRenderPipeline;
    id<MTLComputePipelineState> debugNormalsPipeline;
    id<MTLComputePipelineState> debugDepthPipeline;
    id<MTLComputePipelineState> tonemapPipeline;

    // Bloom pipelines
    id<MTLComputePipelineState> bloomThresholdPipeline;
    id<MTLComputePipelineState> bloomBlurHPipeline;
    id<MTLComputePipelineState> bloomBlurVPipeline;
    id<MTLComputePipelineState> bloomCombinePipeline;

    // Buffers
    id<MTLBuffer> outputBuffer;
    id<MTLBuffer> accumulationBuffer;
    id<MTLBuffer> bloomBuffer;       // For bloom intermediate storage
    id<MTLBuffer> bloomBuffer2;      // For ping-pong blur
    id<MTLBuffer> sphereBuffer;
    id<MTLBuffer> triangleBuffer;
    id<MTLBuffer> materialBuffer;
    id<MTLBuffer> bvhBuffer;
    id<MTLBuffer> cameraBuffer;
    id<MTLBuffer> settingsBuffer;
    id<MTLBuffer> primitiveIndicesBuffer;

    // Current settings
    RenderSettings settings;
    Camera camera;
    uint32_t frameNumber;

    // Buffer capacities
    uint32_t sphereCapacity;
    uint32_t triangleCapacity;
    uint32_t materialCapacity;
    uint32_t bvhCapacity;
    uint32_t primitiveIndicesCapacity;

    // Performance timing
    double lastRenderTime;      // GPU render time in ms
    double lastPostProcessTime; // Post-process time in ms
    uint64_t totalRaysTraced;   // Cumulative rays
};

// High precision timing
#include <mach/mach_time.h>
static double getTimeInMs(void) {
    static mach_timebase_info_data_t timebase = {0};
    if (timebase.denom == 0) {
        mach_timebase_info(&timebase);
    }
    uint64_t time = mach_absolute_time();
    return (double)time * timebase.numer / timebase.denom / 1000000.0;
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

static id<MTLComputePipelineState> createPipeline(
    id<MTLDevice> device,
    id<MTLLibrary> library,
    const char* functionName
) {
    NSString* name = [NSString stringWithUTF8String:functionName];
    id<MTLFunction> function = [library newFunctionWithName:name];

    if (!function) {
        fprintf(stderr, "Failed to find function: %s\n", functionName);
        return nil;
    }

    NSError* error = nil;
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function
                                                                                 error:&error];

    if (error) {
        fprintf(stderr, "Failed to create pipeline for %s: %s\n",
                functionName, [[error localizedDescription] UTF8String]);
        return nil;
    }

    return pipeline;
}

/**
 * Load a shader file and return its contents as a string.
 */
static NSString* loadShaderSource(NSString* filename) {
    // Try multiple paths
    NSArray* searchPaths = @[
        @"shaders",
        @"../shaders",
        @"../../shaders",
        [[NSBundle mainBundle] resourcePath]
    ];

    for (NSString* basePath in searchPaths) {
        NSString* path = [basePath stringByAppendingPathComponent:filename];
        NSError* error = nil;
        NSString* source = [NSString stringWithContentsOfFile:path
                                                     encoding:NSUTF8StringEncoding
                                                        error:&error];
        if (source) {
            return source;
        }
    }

    return nil;
}

/**
 * Compile Metal shaders from source files at runtime.
 */
static id<MTLLibrary> compileShaders(id<MTLDevice> device) {
    // Load all shader source files and concatenate
    NSArray* shaderFiles = @[@"types.metal", @"textures.metal", @"intersect.metal", @"materials.metal", @"pathtracer.metal", @"tonemap.metal"];

    NSMutableString* combinedSource = [NSMutableString string];

    for (NSString* filename in shaderFiles) {
        NSString* source = loadShaderSource(filename);
        if (!source) {
            fprintf(stderr, "Failed to load shader: %s\n", [filename UTF8String]);
            return nil;
        }

        // Skip duplicate includes for combined source
        if ([filename isEqualToString:@"types.metal"]) {
            [combinedSource appendString:source];
            [combinedSource appendString:@"\n"];
        } else {
            // Remove #include directives as we're concatenating everything
            NSString* cleanedSource = source;
            cleanedSource = [cleanedSource stringByReplacingOccurrencesOfString:@"#include \"types.metal\""
                                                                      withString:@"// types.metal included above"];
            cleanedSource = [cleanedSource stringByReplacingOccurrencesOfString:@"#include \"textures.metal\""
                                                                      withString:@"// textures.metal included above"];
            cleanedSource = [cleanedSource stringByReplacingOccurrencesOfString:@"#include \"intersect.metal\""
                                                                      withString:@"// intersect.metal included above"];
            cleanedSource = [cleanedSource stringByReplacingOccurrencesOfString:@"#include \"materials.metal\""
                                                                      withString:@"// materials.metal included above"];
            [combinedSource appendString:cleanedSource];
            [combinedSource appendString:@"\n"];
        }
    }

    printf("Compiling shaders from source (%lu bytes)...\n", (unsigned long)[combinedSource length]);

    // Compile options
    MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
    if (@available(macOS 15.0, iOS 18.0, *)) {
        options.mathMode = MTLMathModeFast;
    } else {
        #pragma clang diagnostic push
        #pragma clang diagnostic ignored "-Wdeprecated-declarations"
        options.fastMathEnabled = YES;
        #pragma clang diagnostic pop
    }
    options.languageVersion = MTLLanguageVersion3_0;

    // Compile
    NSError* error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:combinedSource
                                                  options:options
                                                    error:&error];

    if (error) {
        fprintf(stderr, "Shader compilation failed: %s\n", [[error localizedDescription] UTF8String]);
        if ([error.userInfo objectForKey:@"MTLCompilerErrorDescription"]) {
            fprintf(stderr, "Compiler errors:\n%s\n",
                    [[error.userInfo objectForKey:@"MTLCompilerErrorDescription"] UTF8String]);
        }
        return nil;
    }

    printf("Shaders compiled successfully\n");
    return library;
}

// ============================================================================
// RENDERER API IMPLEMENTATION
// ============================================================================

MetalRenderer* sf_renderer_create(void) {
    MetalRenderer* renderer = (MetalRenderer*)calloc(1, sizeof(MetalRenderer));
    if (!renderer) {
        fprintf(stderr, "Failed to allocate renderer\n");
        return NULL;
    }

    // Get default Metal device (Apple Silicon GPU)
    renderer->device = MTLCreateSystemDefaultDevice();
    if (!renderer->device) {
        fprintf(stderr, "Metal is not supported on this device\n");
        free(renderer);
        return NULL;
    }

    printf("Using Metal device: %s\n", [[renderer->device name] UTF8String]);

    // Create command queue
    renderer->commandQueue = [renderer->device newCommandQueue];
    if (!renderer->commandQueue) {
        fprintf(stderr, "Failed to create command queue\n");
        free(renderer);
        return NULL;
    }

    // Compile shaders from source at runtime
    // This avoids requiring the Metal offline compiler toolchain
    renderer->library = compileShaders(renderer->device);

    if (!renderer->library) {
        // Fall back to trying pre-compiled metallib
        NSError* error = nil;
        NSString* libraryPath = @"build/shaders.metallib";

        if ([[NSFileManager defaultManager] fileExistsAtPath:libraryPath]) {
            NSURL* libraryURL = [NSURL fileURLWithPath:libraryPath];
            renderer->library = [renderer->device newLibraryWithURL:libraryURL error:&error];
        }

        if (!renderer->library) {
            renderer->library = [renderer->device newDefaultLibrary];
        }

        if (!renderer->library) {
            fprintf(stderr, "Failed to compile or load shaders\n");
            free(renderer);
            return NULL;
        }
    }

    // Create compute pipelines
    renderer->pathTracePipeline = createPipeline(renderer->device, renderer->library, "path_trace_kernel");
    renderer->simpleRenderPipeline = createPipeline(renderer->device, renderer->library, "simple_render_kernel");
    renderer->debugNormalsPipeline = createPipeline(renderer->device, renderer->library, "debug_normals_kernel");
    renderer->debugDepthPipeline = createPipeline(renderer->device, renderer->library, "debug_depth_kernel");
    renderer->tonemapPipeline = createPipeline(renderer->device, renderer->library, "tonemap_simple_kernel");

    // Bloom pipelines
    renderer->bloomThresholdPipeline = createPipeline(renderer->device, renderer->library, "bloom_threshold_kernel");
    renderer->bloomBlurHPipeline = createPipeline(renderer->device, renderer->library, "bloom_blur_h_kernel");
    renderer->bloomBlurVPipeline = createPipeline(renderer->device, renderer->library, "bloom_blur_v_kernel");
    renderer->bloomCombinePipeline = createPipeline(renderer->device, renderer->library, "bloom_combine_kernel");

    // Log GPU capabilities for optimization reference
    if (renderer->pathTracePipeline) {
        NSUInteger simdWidth = renderer->pathTracePipeline.threadExecutionWidth;
        NSUInteger maxThreads = renderer->pathTracePipeline.maxTotalThreadsPerThreadgroup;
        fprintf(stderr, "GPU Config: SIMD width=%lu, Max threads/group=%lu (%lux%lu)\n",
                (unsigned long)simdWidth, (unsigned long)maxThreads,
                (unsigned long)simdWidth, (unsigned long)(maxThreads / simdWidth));
    }

    if (!renderer->simpleRenderPipeline) {
        fprintf(stderr, "Failed to create required pipelines\n");
        free(renderer);
        return NULL;
    }

    // Initialize with default settings
    renderer->settings.width = 800;
    renderer->settings.height = 600;
    renderer->settings.samples_per_pixel = 4;
    renderer->settings.max_depth = 10;
    renderer->settings.frame_number = 0;
    renderer->settings.use_sky_gradient = 1;
    renderer->settings.background_r = 0.0f;
    renderer->settings.background_g = 0.0f;
    renderer->settings.background_b = 0.0f;

    renderer->frameNumber = 0;

    // Pre-allocate buffers with reasonable defaults
    renderer->sphereCapacity = 1024;
    renderer->triangleCapacity = 65536;
    renderer->materialCapacity = 256;
    renderer->bvhCapacity = 2048;

    renderer->sphereBuffer = [renderer->device newBufferWithLength:sizeof(Sphere) * renderer->sphereCapacity
                                                           options:MTLResourceStorageModeShared];
    renderer->triangleBuffer = [renderer->device newBufferWithLength:sizeof(Triangle) * renderer->triangleCapacity
                                                             options:MTLResourceStorageModeShared];
    renderer->materialBuffer = [renderer->device newBufferWithLength:sizeof(Material) * renderer->materialCapacity
                                                             options:MTLResourceStorageModeShared];
    renderer->bvhBuffer = [renderer->device newBufferWithLength:sizeof(BVHNode) * renderer->bvhCapacity
                                                        options:MTLResourceStorageModeShared];
    renderer->cameraBuffer = [renderer->device newBufferWithLength:sizeof(Camera)
                                                           options:MTLResourceStorageModeShared];
    renderer->settingsBuffer = [renderer->device newBufferWithLength:sizeof(RenderSettings)
                                                             options:MTLResourceStorageModeShared];

    printf("SpectraForge Metal renderer initialized successfully\n");
    return renderer;
}

void sf_renderer_destroy(MetalRenderer* renderer) {
    if (!renderer) return;

    // Metal objects are automatically released via ARC
    free(renderer);
}

void sf_renderer_set_settings(MetalRenderer* renderer, const RenderSettings* settings) {
    if (!renderer || !settings) return;

    renderer->settings = *settings;

    // Recreate output buffers if size changed
    size_t bufferSize = sizeof(float) * 4 * settings->width * settings->height;

    if (!renderer->outputBuffer ||
        [renderer->outputBuffer length] < bufferSize) {

        renderer->outputBuffer = [renderer->device newBufferWithLength:bufferSize
                                                               options:MTLResourceStorageModeShared];
        renderer->accumulationBuffer = [renderer->device newBufferWithLength:bufferSize
                                                                     options:MTLResourceStorageModeShared];
        // Bloom buffers for ping-pong blur
        renderer->bloomBuffer = [renderer->device newBufferWithLength:bufferSize
                                                              options:MTLResourceStorageModeShared];
        renderer->bloomBuffer2 = [renderer->device newBufferWithLength:bufferSize
                                                               options:MTLResourceStorageModeShared];
    }

    // Reset accumulation when settings change
    sf_renderer_reset_accumulation(renderer);
}

void sf_renderer_set_camera(MetalRenderer* renderer, const Camera* camera) {
    if (!renderer || !camera) return;
    renderer->camera = *camera;
}

void sf_renderer_upload_scene(MetalRenderer* renderer, const Scene* scene) {
    if (!renderer || !scene) return;

    // Update sphere buffer
    if (scene->num_spheres > 0 && scene->spheres) {
        if (scene->num_spheres > renderer->sphereCapacity) {
            renderer->sphereCapacity = scene->num_spheres * 2;
            renderer->sphereBuffer = [renderer->device newBufferWithLength:sizeof(Sphere) * renderer->sphereCapacity
                                                                   options:MTLResourceStorageModeShared];
        }
        memcpy([renderer->sphereBuffer contents], scene->spheres, sizeof(Sphere) * scene->num_spheres);
    }
    renderer->settings.num_spheres = scene->num_spheres;

    // Update triangle buffer
    if (scene->num_triangles > 0 && scene->triangles) {
        if (scene->num_triangles > renderer->triangleCapacity) {
            renderer->triangleCapacity = scene->num_triangles * 2;
            renderer->triangleBuffer = [renderer->device newBufferWithLength:sizeof(Triangle) * renderer->triangleCapacity
                                                                     options:MTLResourceStorageModeShared];
        }
        memcpy([renderer->triangleBuffer contents], scene->triangles, sizeof(Triangle) * scene->num_triangles);
    }
    renderer->settings.num_triangles = scene->num_triangles;

    // Update material buffer
    if (scene->num_materials > 0 && scene->materials) {
        if (scene->num_materials > renderer->materialCapacity) {
            renderer->materialCapacity = scene->num_materials * 2;
            renderer->materialBuffer = [renderer->device newBufferWithLength:sizeof(Material) * renderer->materialCapacity
                                                                     options:MTLResourceStorageModeShared];
        }
        memcpy([renderer->materialBuffer contents], scene->materials, sizeof(Material) * scene->num_materials);
    }

    // Update BVH buffer
    if (scene->num_bvh_nodes > 0 && scene->bvh_nodes) {
        if (scene->num_bvh_nodes > renderer->bvhCapacity) {
            renderer->bvhCapacity = scene->num_bvh_nodes * 2;
            renderer->bvhBuffer = [renderer->device newBufferWithLength:sizeof(BVHNode) * renderer->bvhCapacity
                                                                options:MTLResourceStorageModeShared];
        }
        memcpy([renderer->bvhBuffer contents], scene->bvh_nodes, sizeof(BVHNode) * scene->num_bvh_nodes);
    }
    renderer->settings.num_bvh_nodes = scene->num_bvh_nodes;

    // Update primitive indices buffer (for combined BVH)
    if (scene->num_primitive_indices > 0 && scene->primitive_indices) {
        if (scene->num_primitive_indices > renderer->primitiveIndicesCapacity) {
            renderer->primitiveIndicesCapacity = scene->num_primitive_indices * 2;
            renderer->primitiveIndicesBuffer = [renderer->device newBufferWithLength:sizeof(uint32_t) * renderer->primitiveIndicesCapacity
                                                                             options:MTLResourceStorageModeShared];
        }
        memcpy([renderer->primitiveIndicesBuffer contents], scene->primitive_indices, sizeof(uint32_t) * scene->num_primitive_indices);
    }
    renderer->settings.num_primitive_indices = scene->num_primitive_indices;
}

void sf_renderer_render(MetalRenderer* renderer, float* output) {
    if (!renderer || !output) return;

    double startTime = getTimeInMs();

    // Update frame number
    renderer->settings.frame_number = renderer->frameNumber++;

    // Copy camera and settings to GPU buffers
    memcpy([renderer->cameraBuffer contents], &renderer->camera, sizeof(Camera));
    memcpy([renderer->settingsBuffer contents], &renderer->settings, sizeof(RenderSettings));

    // Create command buffer
    id<MTLCommandBuffer> commandBuffer = [renderer->commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

    // Calculate thread groups (needed for all pipelines)
    MTLSize gridSize = MTLSizeMake(renderer->settings.width, renderer->settings.height, 1);

    // Choose pipeline based on debug mode or normal rendering
    id<MTLComputePipelineState> pipeline;
    bool skipPostProcessing = false;

    if (renderer->settings.debug_mode == 1 && renderer->debugNormalsPipeline) {
        // Debug normals mode
        pipeline = renderer->debugNormalsPipeline;
        [encoder setComputePipelineState:pipeline];

        // Set buffers for debug_normals_kernel
        [encoder setBuffer:renderer->outputBuffer offset:0 atIndex:0];
        [encoder setBuffer:renderer->sphereBuffer offset:0 atIndex:1];
        [encoder setBuffer:renderer->cameraBuffer offset:0 atIndex:2];
        [encoder setBuffer:renderer->settingsBuffer offset:0 atIndex:3];

        skipPostProcessing = true;  // Debug output is already final

    } else if (renderer->settings.debug_mode == 2 && renderer->debugDepthPipeline) {
        // Debug depth mode
        pipeline = renderer->debugDepthPipeline;
        [encoder setComputePipelineState:pipeline];

        // Set buffers for debug_depth_kernel
        [encoder setBuffer:renderer->outputBuffer offset:0 atIndex:0];
        [encoder setBuffer:renderer->sphereBuffer offset:0 atIndex:1];
        [encoder setBuffer:renderer->cameraBuffer offset:0 atIndex:2];
        [encoder setBuffer:renderer->settingsBuffer offset:0 atIndex:3];

        skipPostProcessing = true;  // Debug output is already final

    } else if (renderer->settings.num_bvh_nodes > 0 && renderer->pathTracePipeline) {
        // Use full path tracer with BVH
        pipeline = renderer->pathTracePipeline;
        [encoder setComputePipelineState:pipeline];

        // Set buffers for path_trace_kernel
        [encoder setBuffer:renderer->outputBuffer offset:0 atIndex:0];
        [encoder setBuffer:renderer->accumulationBuffer offset:0 atIndex:1];
        [encoder setBuffer:renderer->sphereBuffer offset:0 atIndex:2];
        [encoder setBuffer:renderer->triangleBuffer offset:0 atIndex:3];
        [encoder setBuffer:renderer->materialBuffer offset:0 atIndex:4];
        [encoder setBuffer:renderer->bvhBuffer offset:0 atIndex:5];
        [encoder setBuffer:renderer->cameraBuffer offset:0 atIndex:6];
        [encoder setBuffer:renderer->settingsBuffer offset:0 atIndex:7];
        [encoder setBuffer:renderer->primitiveIndicesBuffer offset:0 atIndex:8];
    } else {
        // Use simple render pipeline (no BVH)
        pipeline = renderer->simpleRenderPipeline;
        [encoder setComputePipelineState:pipeline];

        // Set buffers for simple_render_kernel
        [encoder setBuffer:renderer->outputBuffer offset:0 atIndex:0];
        [encoder setBuffer:renderer->accumulationBuffer offset:0 atIndex:1];  // For bloom
        [encoder setBuffer:renderer->sphereBuffer offset:0 atIndex:2];
        [encoder setBuffer:renderer->materialBuffer offset:0 atIndex:3];
        [encoder setBuffer:renderer->cameraBuffer offset:0 atIndex:4];
        [encoder setBuffer:renderer->settingsBuffer offset:0 atIndex:5];
    }

    // Calculate thread groups
    NSUInteger threadGroupWidth = pipeline.threadExecutionWidth;
    NSUInteger threadGroupHeight = pipeline.maxTotalThreadsPerThreadgroup / threadGroupWidth;
    MTLSize threadGroupSize = MTLSizeMake(threadGroupWidth, threadGroupHeight, 1);

    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
    [encoder endEncoding];

    // Post-processing with bloom (skip for debug modes)
    uint32_t width = renderer->settings.width;
    uint32_t height = renderer->settings.height;
    float exposure = 0.0f;  // exposure in stops

    // Bloom settings
    float bloom_threshold = 1.0f;      // Luminance threshold for bloom
    float soft_threshold = 0.5f;       // Soft knee
    float bloom_intensity = 0.3f;      // Bloom strength
    int bloom_blur_passes = 3;         // Number of blur iterations (wider = more glow)

    if (!skipPostProcessing && renderer->bloomThresholdPipeline && renderer->bloomBlurHPipeline &&
        renderer->bloomBlurVPipeline && renderer->bloomCombinePipeline) {

        NSUInteger ppWidth = renderer->bloomThresholdPipeline.threadExecutionWidth;
        NSUInteger ppHeight = renderer->bloomThresholdPipeline.maxTotalThreadsPerThreadgroup / ppWidth;
        MTLSize ppThreadGroupSize = MTLSizeMake(ppWidth, ppHeight, 1);

        // Pass 1: Extract bright pixels
        id<MTLComputeCommandEncoder> thresholdEncoder = [commandBuffer computeCommandEncoder];
        [thresholdEncoder setComputePipelineState:renderer->bloomThresholdPipeline];
        [thresholdEncoder setBuffer:renderer->bloomBuffer offset:0 atIndex:0];
        [thresholdEncoder setBuffer:renderer->accumulationBuffer offset:0 atIndex:1];
        [thresholdEncoder setBytes:&width length:sizeof(uint32_t) atIndex:2];
        [thresholdEncoder setBytes:&height length:sizeof(uint32_t) atIndex:3];
        [thresholdEncoder setBytes:&bloom_threshold length:sizeof(float) atIndex:4];
        [thresholdEncoder setBytes:&soft_threshold length:sizeof(float) atIndex:5];
        [thresholdEncoder dispatchThreads:gridSize threadsPerThreadgroup:ppThreadGroupSize];
        [thresholdEncoder endEncoding];

        // Pass 2+: Blur passes (ping-pong between buffers)
        for (int pass = 0; pass < bloom_blur_passes; pass++) {
            // Horizontal blur: bloomBuffer -> bloomBuffer2
            id<MTLComputeCommandEncoder> blurHEncoder = [commandBuffer computeCommandEncoder];
            [blurHEncoder setComputePipelineState:renderer->bloomBlurHPipeline];
            [blurHEncoder setBuffer:renderer->bloomBuffer2 offset:0 atIndex:0];
            [blurHEncoder setBuffer:renderer->bloomBuffer offset:0 atIndex:1];
            [blurHEncoder setBytes:&width length:sizeof(uint32_t) atIndex:2];
            [blurHEncoder setBytes:&height length:sizeof(uint32_t) atIndex:3];
            [blurHEncoder dispatchThreads:gridSize threadsPerThreadgroup:ppThreadGroupSize];
            [blurHEncoder endEncoding];

            // Vertical blur: bloomBuffer2 -> bloomBuffer
            id<MTLComputeCommandEncoder> blurVEncoder = [commandBuffer computeCommandEncoder];
            [blurVEncoder setComputePipelineState:renderer->bloomBlurVPipeline];
            [blurVEncoder setBuffer:renderer->bloomBuffer offset:0 atIndex:0];
            [blurVEncoder setBuffer:renderer->bloomBuffer2 offset:0 atIndex:1];
            [blurVEncoder setBytes:&width length:sizeof(uint32_t) atIndex:2];
            [blurVEncoder setBytes:&height length:sizeof(uint32_t) atIndex:3];
            [blurVEncoder dispatchThreads:gridSize threadsPerThreadgroup:ppThreadGroupSize];
            [blurVEncoder endEncoding];
        }

        // Pass 3: Combine original + bloom with tone mapping
        id<MTLComputeCommandEncoder> combineEncoder = [commandBuffer computeCommandEncoder];
        [combineEncoder setComputePipelineState:renderer->bloomCombinePipeline];
        [combineEncoder setBuffer:renderer->outputBuffer offset:0 atIndex:0];
        [combineEncoder setBuffer:renderer->accumulationBuffer offset:0 atIndex:1];
        [combineEncoder setBuffer:renderer->bloomBuffer offset:0 atIndex:2];
        [combineEncoder setBytes:&width length:sizeof(uint32_t) atIndex:3];
        [combineEncoder setBytes:&height length:sizeof(uint32_t) atIndex:4];
        [combineEncoder setBytes:&bloom_intensity length:sizeof(float) atIndex:5];
        [combineEncoder setBytes:&exposure length:sizeof(float) atIndex:6];
        [combineEncoder dispatchThreads:gridSize threadsPerThreadgroup:ppThreadGroupSize];
        [combineEncoder endEncoding];

    } else if (!skipPostProcessing && renderer->tonemapPipeline) {
        // Fallback: simple tone mapping without bloom
        id<MTLComputeCommandEncoder> ppEncoder = [commandBuffer computeCommandEncoder];
        [ppEncoder setComputePipelineState:renderer->tonemapPipeline];
        [ppEncoder setBuffer:renderer->outputBuffer offset:0 atIndex:0];
        [ppEncoder setBuffer:renderer->accumulationBuffer offset:0 atIndex:1];
        [ppEncoder setBytes:&width length:sizeof(uint32_t) atIndex:2];
        [ppEncoder setBytes:&height length:sizeof(uint32_t) atIndex:3];
        [ppEncoder setBytes:&exposure length:sizeof(float) atIndex:4];

        NSUInteger ppWidth = renderer->tonemapPipeline.threadExecutionWidth;
        NSUInteger ppHeight = renderer->tonemapPipeline.maxTotalThreadsPerThreadgroup / ppWidth;
        MTLSize ppThreadGroupSize = MTLSizeMake(ppWidth, ppHeight, 1);

        [ppEncoder dispatchThreads:gridSize threadsPerThreadgroup:ppThreadGroupSize];
        [ppEncoder endEncoding];
    }

    // Execute and wait
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    double endTime = getTimeInMs();
    renderer->lastRenderTime = endTime - startTime;

    // Track rays traced
    uint64_t raysThisFrame = (uint64_t)renderer->settings.width *
                             renderer->settings.height *
                             renderer->settings.samples_per_pixel *
                             renderer->settings.max_depth;
    renderer->totalRaysTraced += raysThisFrame;

    // Copy result to output buffer (now tone-mapped)
    float* gpuOutput = (float*)[renderer->outputBuffer contents];
    size_t pixelCount = renderer->settings.width * renderer->settings.height;

    // Convert from RGBA to RGB
    for (size_t i = 0; i < pixelCount; i++) {
        output[i * 3 + 0] = gpuOutput[i * 4 + 0];  // R
        output[i * 3 + 1] = gpuOutput[i * 4 + 1];  // G
        output[i * 3 + 2] = gpuOutput[i * 4 + 2];  // B
    }
}

// Get last render time in milliseconds
double sf_renderer_get_render_time(MetalRenderer* renderer) {
    return renderer ? renderer->lastRenderTime : 0.0;
}

// Get total rays traced
uint64_t sf_renderer_get_total_rays(MetalRenderer* renderer) {
    return renderer ? renderer->totalRaysTraced : 0;
}

void sf_renderer_get_accumulation(MetalRenderer* renderer, float* output) {
    if (!renderer || !output) return;

    float* accum = (float*)[renderer->accumulationBuffer contents];
    size_t pixelCount = renderer->settings.width * renderer->settings.height;

    for (size_t i = 0; i < pixelCount; i++) {
        output[i * 3 + 0] = accum[i * 4 + 0];
        output[i * 3 + 1] = accum[i * 4 + 1];
        output[i * 3 + 2] = accum[i * 4 + 2];
    }
}

void sf_renderer_reset_accumulation(MetalRenderer* renderer) {
    if (!renderer) return;
    renderer->frameNumber = 0;

    // Clear accumulation buffer
    if (renderer->accumulationBuffer) {
        memset([renderer->accumulationBuffer contents], 0, [renderer->accumulationBuffer length]);
    }
}

// ============================================================================
// SCENE MANAGEMENT
// ============================================================================

Scene* sf_scene_create(void) {
    Scene* scene = (Scene*)calloc(1, sizeof(Scene));
    if (!scene) return NULL;

    scene->capacity_spheres = 64;
    scene->capacity_triangles = 1024;
    scene->capacity_materials = 32;

    scene->spheres = (Sphere*)calloc(scene->capacity_spheres, sizeof(Sphere));
    scene->triangles = (Triangle*)calloc(scene->capacity_triangles, sizeof(Triangle));
    scene->materials = (Material*)calloc(scene->capacity_materials, sizeof(Material));

    return scene;
}

void sf_scene_destroy(Scene* scene) {
    if (!scene) return;
    free(scene->spheres);
    free(scene->triangles);
    free(scene->materials);
    free(scene->bvh_nodes);
    free(scene->primitive_indices);
    free(scene);
}

uint32_t sf_scene_add_sphere(Scene* scene, float3 center, float radius, uint32_t material_id) {
    if (!scene) return UINT32_MAX;

    // Grow array if needed
    if (scene->num_spheres >= scene->capacity_spheres) {
        scene->capacity_spheres *= 2;
        scene->spheres = (Sphere*)realloc(scene->spheres, sizeof(Sphere) * scene->capacity_spheres);
    }

    uint32_t idx = scene->num_spheres++;
    scene->spheres[idx].center_x = center.x;
    scene->spheres[idx].center_y = center.y;
    scene->spheres[idx].center_z = center.z;
    scene->spheres[idx].radius = radius;
    scene->spheres[idx].material_id = material_id;
    // Default: no motion blur
    scene->spheres[idx].velocity_x = 0.0f;
    scene->spheres[idx].velocity_y = 0.0f;
    scene->spheres[idx].velocity_z = 0.0f;

    return idx;
}

uint32_t sf_scene_add_sphere_moving(Scene* scene, float3 center, float radius, float3 velocity, uint32_t material_id) {
    uint32_t idx = sf_scene_add_sphere(scene, center, radius, material_id);
    if (idx != UINT32_MAX) {
        scene->spheres[idx].velocity_x = velocity.x;
        scene->spheres[idx].velocity_y = velocity.y;
        scene->spheres[idx].velocity_z = velocity.z;
    }
    return idx;
}

uint32_t sf_scene_add_triangle(Scene* scene, float3 v0, float3 v1, float3 v2, uint32_t material_id) {
    if (!scene) return UINT32_MAX;

    if (scene->num_triangles >= scene->capacity_triangles) {
        scene->capacity_triangles *= 2;
        scene->triangles = (Triangle*)realloc(scene->triangles, sizeof(Triangle) * scene->capacity_triangles);
    }

    uint32_t idx = scene->num_triangles++;
    scene->triangles[idx].v0_x = v0.x;
    scene->triangles[idx].v0_y = v0.y;
    scene->triangles[idx].v0_z = v0.z;
    scene->triangles[idx].v1_x = v1.x;
    scene->triangles[idx].v1_y = v1.y;
    scene->triangles[idx].v1_z = v1.z;
    scene->triangles[idx].v2_x = v2.x;
    scene->triangles[idx].v2_y = v2.y;
    scene->triangles[idx].v2_z = v2.z;
    scene->triangles[idx].material_id = material_id;

    // Compute normal
    float3 e1 = {v1.x - v0.x, v1.y - v0.y, v1.z - v0.z};
    float3 e2 = {v2.x - v0.x, v2.y - v0.y, v2.z - v0.z};
    float3 n = {
        e1.y * e2.z - e1.z * e2.y,
        e1.z * e2.x - e1.x * e2.z,
        e1.x * e2.y - e1.y * e2.x
    };
    float len = sqrtf(n.x * n.x + n.y * n.y + n.z * n.z);
    if (len > 0.0001f) {
        n.x /= len; n.y /= len; n.z /= len;
    }
    scene->triangles[idx].normal_x = n.x;
    scene->triangles[idx].normal_y = n.y;
    scene->triangles[idx].normal_z = n.z;

    return idx;
}

uint32_t sf_scene_add_material(Scene* scene, const Material* material) {
    if (!scene || !material) return UINT32_MAX;

    if (scene->num_materials >= scene->capacity_materials) {
        scene->capacity_materials *= 2;
        scene->materials = (Material*)realloc(scene->materials, sizeof(Material) * scene->capacity_materials);
    }

    uint32_t idx = scene->num_materials++;
    scene->materials[idx] = *material;

    return idx;
}

// sf_scene_build_bvh is implemented in bvh.c

// ============================================================================
// CAMERA HELPERS
// ============================================================================

Camera sf_camera_create(
    float3 look_from,
    float3 look_at,
    float3 vup,
    float vfov,
    float aspect_ratio,
    float aperture,
    float focus_dist
) {
    Camera camera;
    memset(&camera, 0, sizeof(Camera));

    float theta = vfov * 3.14159265358979323846f / 180.0f;
    float h = tanf(theta / 2.0f);
    float viewport_height = 2.0f * h;
    float viewport_width = aspect_ratio * viewport_height;

    // Calculate camera basis
    float3 w = {look_from.x - look_at.x, look_from.y - look_at.y, look_from.z - look_at.z};
    float w_len = sqrtf(w.x * w.x + w.y * w.y + w.z * w.z);
    w.x /= w_len; w.y /= w_len; w.z /= w_len;

    float3 u = {
        vup.y * w.z - vup.z * w.y,
        vup.z * w.x - vup.x * w.z,
        vup.x * w.y - vup.y * w.x
    };
    float u_len = sqrtf(u.x * u.x + u.y * u.y + u.z * u.z);
    u.x /= u_len; u.y /= u_len; u.z /= u_len;

    float3 v = {
        w.y * u.z - w.z * u.y,
        w.z * u.x - w.x * u.z,
        w.x * u.y - w.y * u.x
    };

    // Store camera origin
    camera.origin_x = look_from.x;
    camera.origin_y = look_from.y;
    camera.origin_z = look_from.z;

    // Horizontal viewport vector
    camera.horizontal_x = focus_dist * viewport_width * u.x;
    camera.horizontal_y = focus_dist * viewport_width * u.y;
    camera.horizontal_z = focus_dist * viewport_width * u.z;

    // Vertical viewport vector
    camera.vertical_x = focus_dist * viewport_height * v.x;
    camera.vertical_y = focus_dist * viewport_height * v.y;
    camera.vertical_z = focus_dist * viewport_height * v.z;

    // Lower-left corner of the viewport
    camera.lower_left_x = camera.origin_x - camera.horizontal_x / 2.0f - camera.vertical_x / 2.0f - focus_dist * w.x;
    camera.lower_left_y = camera.origin_y - camera.horizontal_y / 2.0f - camera.vertical_y / 2.0f - focus_dist * w.y;
    camera.lower_left_z = camera.origin_z - camera.horizontal_z / 2.0f - camera.vertical_z / 2.0f - focus_dist * w.z;

    camera.lens_radius = aperture / 2.0f;
    camera.focus_dist = focus_dist;

    // Store basis vectors for DOF
    camera.u_x = u.x; camera.u_y = u.y; camera.u_z = u.z;
    camera.v_x = v.x; camera.v_y = v.y; camera.v_z = v.z;
    camera.w_x = w.x; camera.w_y = w.y; camera.w_z = w.z;

    return camera;
}
