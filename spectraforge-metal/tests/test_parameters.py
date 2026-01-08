#!/usr/bin/env python3
"""
SpectraForge Metal - Parameter Verification Tests

Verifies that Web UI parameters actually affect the rendered output.
Renders images with different parameter values and compares them to detect changes.

Usage:
    python tests/test_parameters.py          # Run all tests
    python tests/test_parameters.py --quick  # Run quick subset
    python tests/test_parameters.py --verbose # Show detailed diffs
"""

import argparse
import hashlib
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Test configuration - small size for speed, deterministic seed
BASE_WIDTH = 200
BASE_HEIGHT = 150
BASE_SAMPLES = 4  # Low samples for speed, still shows differences


@dataclass
class TestResult:
    """Result of a parameter test."""
    name: str
    passed: bool
    baseline_hash: str
    test_hash: str
    difference_percent: float
    message: str


def find_binary() -> Path:
    """Find the spectraforge binary."""
    candidates = [
        Path(__file__).parent.parent / 'build' / 'spectraforge',
        Path('build/spectraforge'),
    ]
    for path in candidates:
        if path.exists():
            return path.resolve()
    raise FileNotFoundError("spectraforge binary not found. Run 'make' first.")


def render_image(
    binary: Path,
    output_path: Path,
    width: int = BASE_WIDTH,
    height: int = BASE_HEIGHT,
    samples: int = BASE_SAMPLES,
    depth: int = 10,
    scene: str = 'demo',
    extra_args: Optional[List[str]] = None
) -> bool:
    """Render an image with specified parameters."""
    cmd = [
        str(binary),
        '--width', str(width),
        '--height', str(height),
        '--samples', str(samples),
        '--depth', str(depth),
        '--scene', scene,
        '--output', str(output_path),
    ]
    if extra_args:
        cmd.extend(extra_args)

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(binary.parent.parent)
    )
    return result.returncode == 0 and output_path.exists()


def compute_image_hash(path: Path) -> str:
    """Compute SHA256 hash of image file."""
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def compare_images_pixels(path1: Path, path2: Path) -> Tuple[bool, float]:
    """
    Compare two PNG images pixel by pixel.
    Returns (are_different, difference_percentage).
    """
    try:
        # Try to use PIL if available for pixel comparison
        from PIL import Image
        import numpy as np

        img1 = np.array(Image.open(path1))
        img2 = np.array(Image.open(path2))

        if img1.shape != img2.shape:
            return True, 100.0

        # Calculate percentage of different pixels
        diff = np.abs(img1.astype(float) - img2.astype(float))
        # Consider a pixel different if any channel differs by more than 1
        different_pixels = np.any(diff > 1, axis=-1) if len(diff.shape) > 2 else diff > 1
        diff_percent = (np.sum(different_pixels) / different_pixels.size) * 100

        return diff_percent > 0.1, diff_percent

    except ImportError:
        # Fallback to hash comparison
        hash1 = compute_image_hash(path1)
        hash2 = compute_image_hash(path2)
        return hash1 != hash2, 100.0 if hash1 != hash2 else 0.0


def test_parameter_change(
    binary: Path,
    tmpdir: Path,
    param_name: str,
    baseline_kwargs: Dict,
    test_kwargs: Dict,
    expected_different: bool = True
) -> TestResult:
    """Test that changing a parameter produces a different image."""
    baseline_path = tmpdir / f'baseline_{param_name}.png'
    test_path = tmpdir / f'test_{param_name}.png'

    # Render baseline
    if not render_image(binary, baseline_path, **baseline_kwargs):
        return TestResult(
            name=param_name,
            passed=False,
            baseline_hash='',
            test_hash='',
            difference_percent=0,
            message='Baseline render failed'
        )

    # Render with changed parameter
    if not render_image(binary, test_path, **test_kwargs):
        return TestResult(
            name=param_name,
            passed=False,
            baseline_hash='',
            test_hash='',
            difference_percent=0,
            message='Test render failed'
        )

    baseline_hash = compute_image_hash(baseline_path)
    test_hash = compute_image_hash(test_path)

    are_different, diff_percent = compare_images_pixels(baseline_path, test_path)

    if expected_different:
        passed = are_different
        if passed:
            message = f'Images differ by {diff_percent:.1f}%'
        else:
            message = 'Images are identical - parameter may not be working'
    else:
        passed = not are_different
        message = 'Images match as expected' if passed else f'Images differ by {diff_percent:.1f}%'

    return TestResult(
        name=param_name,
        passed=passed,
        baseline_hash=baseline_hash,
        test_hash=test_hash,
        difference_percent=diff_percent,
        message=message
    )


def run_all_tests(binary: Path, verbose: bool = False, quick: bool = False) -> List[TestResult]:
    """Run all parameter verification tests."""
    results = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        print("\n" + "=" * 60)
        print("SpectraForge Metal - Parameter Verification Tests")
        print("=" * 60)
        print(f"Binary: {binary}")
        print(f"Test resolution: {BASE_WIDTH}x{BASE_HEIGHT} @ {BASE_SAMPLES} spp")
        print("=" * 60 + "\n")

        # =================================================================
        # SCENE TESTS - Different scenes should produce different images
        # =================================================================
        print("Testing Scene Presets...")

        scenes = ['demo', 'cornell', 'dof', 'motion']
        if quick:
            scenes = ['demo', 'cornell']

        baseline_scene = 'demo'
        for scene in scenes:
            if scene == baseline_scene:
                continue
            result = test_parameter_change(
                binary, tmpdir,
                f'scene_{scene}',
                {'scene': baseline_scene},
                {'scene': scene}
            )
            results.append(result)
            print_result(result, verbose)

        # =================================================================
        # RESOLUTION TESTS - Different sizes produce different images
        # =================================================================
        print("\nTesting Resolution Parameters...")

        # Width change
        result = test_parameter_change(
            binary, tmpdir,
            'width',
            {'width': 200, 'height': 150},
            {'width': 300, 'height': 150}
        )
        results.append(result)
        print_result(result, verbose)

        # Height change
        result = test_parameter_change(
            binary, tmpdir,
            'height',
            {'width': 200, 'height': 150},
            {'width': 200, 'height': 200}
        )
        results.append(result)
        print_result(result, verbose)

        # =================================================================
        # SAMPLES TEST - More samples = less noise (different pixels)
        # =================================================================
        print("\nTesting Sample Count...")

        result = test_parameter_change(
            binary, tmpdir,
            'samples',
            {'samples': 2},
            {'samples': 16}
        )
        results.append(result)
        print_result(result, verbose)

        # =================================================================
        # MAX DEPTH TEST - Affects reflections and refractions
        # =================================================================
        print("\nTesting Max Depth (Ray Bounces)...")

        # Use a scene with reflective/refractive materials
        result = test_parameter_change(
            binary, tmpdir,
            'max_depth',
            {'depth': 2, 'scene': 'demo'},
            {'depth': 10, 'scene': 'demo'}
        )
        results.append(result)
        print_result(result, verbose)

        if not quick:
            # =================================================================
            # DOF SCENE SPECIFIC TESTS
            # =================================================================
            print("\nTesting Depth of Field Scene...")

            # The DOF scene has built-in aperture settings
            result = test_parameter_change(
                binary, tmpdir,
                'dof_scene',
                {'scene': 'demo'},
                {'scene': 'dof'}
            )
            results.append(result)
            print_result(result, verbose)

            # =================================================================
            # MOTION BLUR TEST
            # =================================================================
            print("\nTesting Motion Blur Scene...")

            result = test_parameter_change(
                binary, tmpdir,
                'motion_blur',
                {'scene': 'demo'},
                {'scene': 'motion'}
            )
            results.append(result)
            print_result(result, verbose)

            # =================================================================
            # DEBUG MODE TESTS
            # =================================================================
            print("\nTesting Debug Modes...")

            # Debug normals - visualizes surface normals as RGB colors
            result = test_parameter_change(
                binary, tmpdir,
                'debug_normals',
                {},
                {'extra_args': ['--debug-normals']}
            )
            results.append(result)
            print_result(result, verbose)

            # Debug depth - visualizes depth buffer as grayscale
            result = test_parameter_change(
                binary, tmpdir,
                'debug_depth',
                {},
                {'extra_args': ['--debug-depth']}
            )
            results.append(result)
            print_result(result, verbose)

    return results


def print_result(result: TestResult, verbose: bool = False):
    """Print a single test result."""
    status = "PASS" if result.passed else "FAIL"
    color = "\033[92m" if result.passed else "\033[91m"
    reset = "\033[0m"

    print(f"  [{color}{status}{reset}] {result.name}: {result.message}")

    if verbose and result.baseline_hash:
        print(f"        Baseline hash: {result.baseline_hash}")
        print(f"        Test hash:     {result.test_hash}")


def print_summary(results: List[TestResult]):
    """Print test summary."""
    passed = sum(1 for r in results if r.passed)
    total = len(results)

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    # Group by category
    categories = {}
    for r in results:
        cat = r.name.split('_')[0]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r)

    for cat, cat_results in categories.items():
        cat_passed = sum(1 for r in cat_results if r.passed)
        print(f"  {cat.title()}: {cat_passed}/{len(cat_results)} passed")

    print("-" * 60)
    color = "\033[92m" if passed == total else "\033[91m"
    reset = "\033[0m"
    print(f"  {color}Total: {passed}/{total} tests passed{reset}")
    print("=" * 60)

    if passed < total:
        print("\nFailed tests may indicate:")
        print("  - Parameter not implemented in renderer")
        print("  - Parameter only affects certain scenes")
        print("  - Difference too subtle at low sample count")

    return passed == total


def main():
    parser = argparse.ArgumentParser(
        description='Verify SpectraForge parameters affect rendered output'
    )
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed output including hashes')
    parser.add_argument('--quick', '-q', action='store_true',
                        help='Run only essential tests')
    args = parser.parse_args()

    try:
        binary = find_binary()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    results = run_all_tests(binary, args.verbose, args.quick)
    success = print_summary(results)

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
