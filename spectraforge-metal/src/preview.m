/**
 * SpectraForge Metal - Real-time Preview Window
 *
 * Interactive preview using Metal Kit with progressive rendering.
 */

#import <Cocoa/Cocoa.h>
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>
#include "../include/spectraforge.h"

// Forward declarations
@class PreviewView;
@class PreviewWindowDelegate;

// ============================================================================
// PREVIEW STATE
// ============================================================================

typedef struct {
    MetalRenderer* renderer;
    Scene* scene;
    Camera camera;
    RenderSettings settings;

    // Camera control state
    float camera_yaw;      // Horizontal angle
    float camera_pitch;    // Vertical angle
    float camera_distance; // Distance from look_at point
    float3 camera_target;  // Look-at point

    // Rendering state
    uint32_t accumulated_frames;
    bool needs_reset;
    bool is_moving;

    // Performance
    double last_frame_time;
    double fps;
} PreviewState;

static PreviewState g_preview = {0};

// ============================================================================
// CAMERA CONTROLS
// ============================================================================

static void update_camera_from_orbit(void) {
    // Convert spherical coordinates to Cartesian
    float cos_pitch = cosf(g_preview.camera_pitch);
    float sin_pitch = sinf(g_preview.camera_pitch);
    float cos_yaw = cosf(g_preview.camera_yaw);
    float sin_yaw = sinf(g_preview.camera_yaw);

    float3 offset = make_float3(
        g_preview.camera_distance * cos_pitch * sin_yaw,
        g_preview.camera_distance * sin_pitch,
        g_preview.camera_distance * cos_pitch * cos_yaw
    );

    float3 look_from = make_float3(
        g_preview.camera_target.x + offset.x,
        g_preview.camera_target.y + offset.y,
        g_preview.camera_target.z + offset.z
    );

    float aspect = (float)g_preview.settings.width / (float)g_preview.settings.height;
    g_preview.camera = sf_camera_create(
        look_from,
        g_preview.camera_target,
        make_float3(0, 1, 0),
        40.0f,  // FOV
        aspect,
        0.0f,   // No DOF in preview for speed
        g_preview.camera_distance
    );

    g_preview.needs_reset = true;
}

// ============================================================================
// METAL KIT VIEW DELEGATE
// ============================================================================

@interface PreviewView : MTKView <MTKViewDelegate>
@property (nonatomic) NSPoint lastMouseLocation;
@property (nonatomic) BOOL isDragging;
@end

@implementation PreviewView

- (instancetype)initWithFrame:(NSRect)frameRect device:(id<MTLDevice>)device {
    self = [super initWithFrame:frameRect device:device];
    if (self) {
        self.delegate = self;
        self.preferredFramesPerSecond = 60;
        self.enableSetNeedsDisplay = NO;
        self.paused = NO;
        self.colorPixelFormat = MTLPixelFormatBGRA8Unorm;
        _isDragging = NO;
    }
    return self;
}

- (BOOL)acceptsFirstResponder {
    return YES;
}

- (void)keyDown:(NSEvent *)event {
    float move_speed = 0.5f;
    float rotate_speed = 0.05f;

    switch ([event keyCode]) {
        case 0:  // A - rotate left
            g_preview.camera_yaw -= rotate_speed;
            g_preview.needs_reset = true;
            break;
        case 2:  // D - rotate right
            g_preview.camera_yaw += rotate_speed;
            g_preview.needs_reset = true;
            break;
        case 1:  // S - move back
            g_preview.camera_distance += move_speed;
            g_preview.needs_reset = true;
            break;
        case 13: // W - move forward
            g_preview.camera_distance = fmaxf(1.0f, g_preview.camera_distance - move_speed);
            g_preview.needs_reset = true;
            break;
        case 12: // Q - rotate up
            g_preview.camera_pitch = fminf(1.5f, g_preview.camera_pitch + rotate_speed);
            g_preview.needs_reset = true;
            break;
        case 14: // E - rotate down
            g_preview.camera_pitch = fmaxf(-1.5f, g_preview.camera_pitch - rotate_speed);
            g_preview.needs_reset = true;
            break;
        case 49: // Space - reset view
            g_preview.camera_yaw = 0.4f;
            g_preview.camera_pitch = 0.2f;
            g_preview.camera_distance = 13.0f;
            g_preview.camera_target = make_float3(0, 0, 0);
            g_preview.needs_reset = true;
            break;
        case 53: // Escape - close
            [[self window] close];
            return;
        default:
            [super keyDown:event];
            return;
    }

    update_camera_from_orbit();
}

- (void)mouseDown:(NSEvent *)event {
    self.lastMouseLocation = [event locationInWindow];
    self.isDragging = YES;
    g_preview.is_moving = YES;
}

- (void)mouseUp:(NSEvent *)event {
    self.isDragging = NO;
    g_preview.is_moving = NO;
}

- (void)mouseDragged:(NSEvent *)event {
    if (!self.isDragging) return;

    NSPoint currentLocation = [event locationInWindow];
    float dx = (currentLocation.x - self.lastMouseLocation.x) * 0.005f;
    float dy = (currentLocation.y - self.lastMouseLocation.y) * 0.005f;

    g_preview.camera_yaw += dx;
    g_preview.camera_pitch = fminf(1.5f, fmaxf(-1.5f, g_preview.camera_pitch + dy));

    self.lastMouseLocation = currentLocation;
    g_preview.needs_reset = true;
    update_camera_from_orbit();
}

- (void)scrollWheel:(NSEvent *)event {
    float delta = [event deltaY] * 0.5f;
    g_preview.camera_distance = fmaxf(1.0f, g_preview.camera_distance - delta);
    g_preview.needs_reset = true;
    update_camera_from_orbit();
}

- (void)mtkView:(MTKView *)view drawableSizeWillChange:(CGSize)size {
    // Handle resize if needed
}

- (void)drawInMTKView:(MTKView *)view {
    // Performance timing
    double current_time = CFAbsoluteTimeGetCurrent();
    if (g_preview.last_frame_time > 0) {
        double delta = current_time - g_preview.last_frame_time;
        g_preview.fps = 1.0 / delta;
    }
    g_preview.last_frame_time = current_time;

    // Reset accumulation if camera moved
    if (g_preview.needs_reset) {
        sf_renderer_reset_accumulation(g_preview.renderer);
        g_preview.accumulated_frames = 0;
        g_preview.needs_reset = false;
    }

    // Adaptive quality: low samples while moving, high when still
    uint32_t samples = g_preview.is_moving ? 1 : 4;
    g_preview.settings.samples_per_pixel = samples;
    g_preview.settings.frame_number = g_preview.accumulated_frames;

    sf_renderer_set_settings(g_preview.renderer, &g_preview.settings);
    sf_renderer_set_camera(g_preview.renderer, &g_preview.camera);

    // Render frame
    uint32_t pixel_count = g_preview.settings.width * g_preview.settings.height;
    float* rgb_buffer = (float*)malloc(pixel_count * 3 * sizeof(float));

    sf_renderer_render(g_preview.renderer, rgb_buffer);
    g_preview.accumulated_frames++;

    // Get drawable and copy to texture
    id<CAMetalDrawable> drawable = [view currentDrawable];
    if (!drawable) {
        free(rgb_buffer);
        return;
    }

    id<MTLTexture> texture = drawable.texture;

    // Convert RGB float to BGRA8
    uint8_t* bgra_buffer = (uint8_t*)malloc(pixel_count * 4);
    for (uint32_t i = 0; i < pixel_count; i++) {
        // Already tone mapped by renderer
        float r = fminf(1.0f, fmaxf(0.0f, rgb_buffer[i * 3 + 0]));
        float g = fminf(1.0f, fmaxf(0.0f, rgb_buffer[i * 3 + 1]));
        float b = fminf(1.0f, fmaxf(0.0f, rgb_buffer[i * 3 + 2]));

        bgra_buffer[i * 4 + 0] = (uint8_t)(b * 255.0f);
        bgra_buffer[i * 4 + 1] = (uint8_t)(g * 255.0f);
        bgra_buffer[i * 4 + 2] = (uint8_t)(r * 255.0f);
        bgra_buffer[i * 4 + 3] = 255;
    }

    MTLRegion region = MTLRegionMake2D(0, 0, g_preview.settings.width, g_preview.settings.height);
    [texture replaceRegion:region
               mipmapLevel:0
                 withBytes:bgra_buffer
               bytesPerRow:g_preview.settings.width * 4];

    free(rgb_buffer);
    free(bgra_buffer);

    // Present
    id<MTLCommandBuffer> commandBuffer = [[view.device newCommandQueue] commandBuffer];
    [commandBuffer presentDrawable:drawable];
    [commandBuffer commit];

    // Update window title with stats
    NSString* title = [NSString stringWithFormat:@"SpectraForge Preview - %d frames, %.1f FPS | WASD/QE: rotate | Scroll: zoom | Space: reset | Esc: quit",
                       g_preview.accumulated_frames, g_preview.fps];
    [[view window] setTitle:title];
}

@end

// ============================================================================
// WINDOW DELEGATE
// ============================================================================

@interface PreviewWindowDelegate : NSObject <NSWindowDelegate>
@property (nonatomic) BOOL shouldClose;
@end

@implementation PreviewWindowDelegate

- (BOOL)windowShouldClose:(NSWindow *)sender {
    self.shouldClose = YES;
    [NSApp stop:nil];
    return YES;
}

@end

// ============================================================================
// PUBLIC API
// ============================================================================

int sf_preview_run(MetalRenderer* renderer, Scene* scene, uint32_t width, uint32_t height) {
    @autoreleasepool {
        // Initialize preview state
        g_preview.renderer = renderer;
        g_preview.scene = scene;
        g_preview.camera_yaw = 0.4f;
        g_preview.camera_pitch = 0.2f;
        g_preview.camera_distance = 13.0f;
        g_preview.camera_target = make_float3(0, 0, 0);
        g_preview.accumulated_frames = 0;
        g_preview.needs_reset = true;
        g_preview.is_moving = false;
        g_preview.last_frame_time = 0;
        g_preview.fps = 0;

        // Setup settings
        g_preview.settings.width = width;
        g_preview.settings.height = height;
        g_preview.settings.samples_per_pixel = 4;
        g_preview.settings.max_depth = 8;  // Lower for preview speed
        g_preview.settings.frame_number = 0;
        g_preview.settings.num_spheres = scene->num_spheres;
        g_preview.settings.num_triangles = scene->num_triangles;
        g_preview.settings.num_bvh_nodes = scene->num_bvh_nodes;
        g_preview.settings.num_primitive_indices = scene->num_primitive_indices;
        g_preview.settings.use_sky_gradient = 1;

        // Initial camera
        update_camera_from_orbit();

        // Create application if needed
        [NSApplication sharedApplication];
        [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];

        // Get Metal device
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            fprintf(stderr, "Error: Metal not available\n");
            return -1;
        }

        // Create window
        NSRect frame = NSMakeRect(100, 100, width, height);
        NSWindow* window = [[NSWindow alloc]
            initWithContentRect:frame
                      styleMask:NSWindowStyleMaskTitled |
                               NSWindowStyleMaskClosable |
                               NSWindowStyleMaskMiniaturizable
                        backing:NSBackingStoreBuffered
                          defer:NO];

        [window setTitle:@"SpectraForge Preview - Loading..."];

        PreviewWindowDelegate* windowDelegate = [[PreviewWindowDelegate alloc] init];
        [window setDelegate:windowDelegate];

        // Create Metal view
        PreviewView* metalView = [[PreviewView alloc] initWithFrame:frame device:device];
        [window setContentView:metalView];

        // Show window
        [window makeKeyAndOrderFront:nil];
        [window makeFirstResponder:metalView];
        [NSApp activateIgnoringOtherApps:YES];

        printf("\nPreview Controls:\n");
        printf("  Mouse drag: Orbit camera\n");
        printf("  Scroll: Zoom in/out\n");
        printf("  W/S: Move forward/back\n");
        printf("  A/D: Rotate left/right\n");
        printf("  Q/E: Rotate up/down\n");
        printf("  Space: Reset view\n");
        printf("  Escape: Close preview\n\n");

        // Run event loop
        [NSApp run];

        return 0;
    }
}
