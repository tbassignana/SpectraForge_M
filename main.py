#!/usr/bin/env python3
"""
SpectraForge - A Python Ray Tracing Renderer

Main entry point for rendering scenes.
"""

import argparse
import sys
import time
from pathlib import Path

from spectraforge.vec3 import Vec3, Color, Point3
from spectraforge.ray import Ray
from spectraforge.camera import Camera
from spectraforge.shapes import Sphere, Plane, HittableList, Triangle
from spectraforge.materials import Lambertian, Metal, Dielectric, Emissive, PBRMaterial
from spectraforge.renderer import Renderer, RenderSettings, get_platform_info


def create_demo_scene() -> HittableList:
    """Create a demo scene with various materials."""
    world = HittableList()

    # Ground plane
    ground_material = Lambertian(Color(0.5, 0.5, 0.5))
    world.add(Sphere(Point3(0, -1000, 0), 1000, ground_material))

    # Center sphere - glass
    glass = Dielectric(1.5)
    world.add(Sphere(Point3(0, 1, 0), 1.0, glass))

    # Left sphere - diffuse
    diffuse = Lambertian(Color(0.4, 0.2, 0.1))
    world.add(Sphere(Point3(-4, 1, 0), 1.0, diffuse))

    # Right sphere - metal
    metal = Metal(Color(0.7, 0.6, 0.5), 0.0)
    world.add(Sphere(Point3(4, 1, 0), 1.0, metal))

    # Small spheres with PBR materials
    pbr_gold = PBRMaterial(
        albedo=Color(1.0, 0.766, 0.336),
        metallic=1.0,
        roughness=0.3
    )
    world.add(Sphere(Point3(1.5, 0.5, 2), 0.5, pbr_gold))

    pbr_rough = PBRMaterial(
        albedo=Color(0.8, 0.1, 0.1),
        metallic=0.0,
        roughness=0.8
    )
    world.add(Sphere(Point3(-1.5, 0.5, 2), 0.5, pbr_rough))

    # Emissive sphere (light source)
    light = Emissive(Color(1, 1, 1), 5.0)
    world.add(Sphere(Point3(0, 5, 0), 1.0, light))

    return world


def create_cornell_box() -> HittableList:
    """Create a Cornell box scene."""
    world = HittableList()

    # Materials
    red = Lambertian(Color(0.65, 0.05, 0.05))
    white = Lambertian(Color(0.73, 0.73, 0.73))
    green = Lambertian(Color(0.12, 0.45, 0.15))
    light = Emissive(Color(1, 1, 1), 15.0)

    # Room walls (using large spheres as approximation - proper boxes would use quads)
    # Left wall (red)
    world.add(Sphere(Point3(-1005, 0, 0), 1000, red))
    # Right wall (green)
    world.add(Sphere(Point3(1005, 0, 0), 1000, green))
    # Back wall
    world.add(Sphere(Point3(0, 0, -1005), 1000, white))
    # Floor
    world.add(Sphere(Point3(0, -1000, 0), 1000, white))
    # Ceiling
    world.add(Sphere(Point3(0, 1010, 0), 1000, white))

    # Light on ceiling
    world.add(Sphere(Point3(0, 9.9, 0), 1.5, light))

    # Objects in the box
    glass = Dielectric(1.5)
    world.add(Sphere(Point3(-2, 1.5, -2), 1.5, glass))

    metal = Metal(Color(0.8, 0.8, 0.8), 0.1)
    world.add(Sphere(Point3(2, 1, 1), 1.0, metal))

    return world


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='SpectraForge - A Python Ray Tracing Renderer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python main.py --scene demo --output render.png
  python main.py --width 1920 --height 1080 --samples 500 --output hd_render.png
  python main.py --scene cornell --samples 1000 --output cornell.exr
        '''
    )

    parser.add_argument('--width', type=int, default=800, help='Image width (default: 800)')
    parser.add_argument('--height', type=int, default=600, help='Image height (default: 600)')
    parser.add_argument('--samples', type=int, default=100, help='Samples per pixel (default: 100)')
    parser.add_argument('--depth', type=int, default=50, help='Max ray depth (default: 50)')
    parser.add_argument('--threads', type=int, default=0, help='Number of threads (0=auto)')
    parser.add_argument('--output', type=str, default='output/render.png', help='Output filename')
    parser.add_argument('--scene', type=str, default='demo', choices=['demo', 'cornell'],
                        help='Scene to render (default: demo)')
    parser.add_argument('--info', action='store_true', help='Show platform info and exit')

    args = parser.parse_args()

    # Show platform info
    if args.info:
        info = get_platform_info()
        print("SpectraForge Platform Info:")
        print(f"  System: {info['system']}")
        print(f"  Machine: {info['machine']}")
        print(f"  Processor: {info['processor']}")
        print(f"  Python: {info['python_version']}")
        print(f"  CPU Cores: {info['cpu_count']}")
        print(f"  ARM: {info['is_arm']}")
        print(f"  x86: {info['is_x86']}")
        print(f"  Apple Silicon: {info['is_apple_silicon']}")
        return 0

    # Print header
    print("=" * 60)
    print("SpectraForge Ray Tracer")
    print("=" * 60)

    info = get_platform_info()
    print(f"Platform: {info['system']} {info['machine']}")
    print(f"CPU Cores: {info['cpu_count']}")

    # Create render settings
    settings = RenderSettings(
        width=args.width,
        height=args.height,
        samples_per_pixel=args.samples,
        max_depth=args.depth,
        num_threads=args.threads
    )

    print(f"\nRender Settings:")
    print(f"  Resolution: {settings.width}x{settings.height}")
    print(f"  Samples: {settings.samples_per_pixel}")
    print(f"  Max Depth: {settings.max_depth}")
    print(f"  Threads: {settings.num_threads}")

    # Create scene
    print(f"\nCreating scene: {args.scene}")
    if args.scene == 'cornell':
        world = create_cornell_box()
        camera = Camera(
            look_from=Point3(0, 5, 15),
            look_at=Point3(0, 5, 0),
            vup=Vec3(0, 1, 0),
            vfov=40,
            aspect_ratio=settings.width / settings.height,
            aperture=0.0,
            focus_dist=15.0
        )
    else:
        world = create_demo_scene()
        camera = Camera(
            look_from=Point3(13, 2, 3),
            look_at=Point3(0, 0, 0),
            vup=Vec3(0, 1, 0),
            vfov=20,
            aspect_ratio=settings.width / settings.height,
            aperture=0.1,
            focus_dist=10.0
        )

    print(f"  Objects in scene: {len(world)}")

    # Create renderer
    renderer = Renderer(settings)

    # Progress tracking
    last_progress = [0]

    def progress_callback(progress: float):
        pct = int(progress * 100)
        if pct > last_progress[0]:
            last_progress[0] = pct
            bar_len = 40
            filled = int(bar_len * progress)
            bar = '█' * filled + '░' * (bar_len - filled)
            print(f'\rRendering: [{bar}] {pct}%', end='', flush=True)

    renderer.set_progress_callback(progress_callback)

    # Render
    print("\nRendering...")
    start_time = time.time()

    image = renderer.render(world, camera)

    elapsed = time.time() - start_time
    print(f"\nRender completed in {elapsed:.2f} seconds")
    print(f"  Rays per second: {(settings.width * settings.height * settings.samples_per_pixel) / elapsed:.0f}")

    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save image
    print(f"\nSaving to: {args.output}")
    renderer.save_image(image, args.output)

    print("\nDone!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
