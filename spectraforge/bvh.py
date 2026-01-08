"""
Bounding Volume Hierarchy (BVH) for accelerating ray-object intersection.

BVH is a tree structure where each node contains an AABB and either:
- Two child nodes (interior node)
- A list of primitives (leaf node)

This implementation supports multi-threaded construction.
"""

from __future__ import annotations
import random
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor
import platform
import os

from .vec3 import Vec3, Point3
from .ray import Ray
from .shapes import Hittable, HitRecord, AABB, HittableList


class BVHNode(Hittable):
    """A node in the Bounding Volume Hierarchy tree.

    Interior nodes have two children; leaf nodes contain primitives.
    """

    def __init__(
        self,
        objects: List[Hittable],
        start: int = 0,
        end: int = None,
        max_leaf_size: int = 4
    ):
        """Build a BVH from a list of objects.

        Args:
            objects: List of hittable objects
            start: Start index in the objects list
            end: End index (exclusive) in the objects list
            max_leaf_size: Maximum objects in a leaf node before splitting
        """
        if end is None:
            end = len(objects)

        self.left: Optional[Hittable] = None
        self.right: Optional[Hittable] = None
        self.bbox: Optional[AABB] = None

        object_span = end - start

        if object_span <= 0:
            return

        if object_span == 1:
            # Single object - make it a leaf
            self.left = objects[start]
            self.bbox = self.left.bounding_box()

        elif object_span == 2:
            # Two objects - left and right children
            self.left = objects[start]
            self.right = objects[start + 1]
            box_left = self.left.bounding_box()
            box_right = self.right.bounding_box()
            if box_left and box_right:
                self.bbox = AABB.surrounding_box(box_left, box_right)

        elif object_span <= max_leaf_size:
            # Small number of objects - create leaf with HittableList
            self.left = HittableList(objects[start:end])
            self.bbox = self.left.bounding_box()

        else:
            # Sort objects along random axis and split
            axis = random.randint(0, 2)

            # Sort by centroid of bounding box
            def get_centroid(obj: Hittable, axis: int) -> float:
                bbox = obj.bounding_box()
                if bbox is None:
                    return 0.0
                return (bbox.minimum[axis] + bbox.maximum[axis]) / 2

            objects[start:end] = sorted(
                objects[start:end],
                key=lambda obj: get_centroid(obj, axis)
            )

            mid = start + object_span // 2

            # Recursively build children
            self.left = BVHNode(objects, start, mid, max_leaf_size)
            self.right = BVHNode(objects, mid, end, max_leaf_size)

            # Compute bounding box
            box_left = self.left.bounding_box() if self.left else None
            box_right = self.right.bounding_box() if self.right else None

            if box_left and box_right:
                self.bbox = AABB.surrounding_box(box_left, box_right)
            elif box_left:
                self.bbox = box_left
            elif box_right:
                self.bbox = box_right

    def hit(self, ray: Ray, t_min: float, t_max: float) -> Optional[HitRecord]:
        """Test ray intersection with BVH node."""
        # Early exit if ray misses bounding box
        if self.bbox is None or not self.bbox.hit(ray, t_min, t_max):
            return None

        # Check children
        hit_left = self.left.hit(ray, t_min, t_max) if self.left else None

        # If left hit, use its t as new t_max for right
        if hit_left:
            hit_right = self.right.hit(ray, t_min, hit_left.t) if self.right else None
        else:
            hit_right = self.right.hit(ray, t_min, t_max) if self.right else None

        # Return closest hit
        if hit_right:
            return hit_right
        return hit_left

    def bounding_box(self) -> Optional[AABB]:
        """Return the bounding box for this node."""
        return self.bbox


class BVH(Hittable):
    """Bounding Volume Hierarchy acceleration structure.

    Provides O(log n) ray intersection instead of O(n) for n objects.
    """

    def __init__(
        self,
        objects: List[Hittable],
        max_leaf_size: int = 4,
        use_parallel: bool = True
    ):
        """Build a BVH from a list of objects.

        Args:
            objects: List of hittable objects to accelerate
            max_leaf_size: Maximum objects per leaf node
            use_parallel: Whether to use parallel construction (for large scenes)
        """
        self.objects = list(objects)  # Make a copy

        if len(self.objects) == 0:
            self.root = None
        elif use_parallel and len(self.objects) > 1000:
            self.root = self._build_parallel(max_leaf_size)
        else:
            self.root = BVHNode(self.objects, 0, len(self.objects), max_leaf_size)

    def _build_parallel(self, max_leaf_size: int) -> BVHNode:
        """Build BVH in parallel for large scenes.

        Splits work at the top levels and builds subtrees in parallel.
        """
        num_workers = os.cpu_count() or 4

        # Sort and split into chunks
        axis = random.randint(0, 2)

        def get_centroid(obj: Hittable) -> float:
            bbox = obj.bounding_box()
            if bbox is None:
                return 0.0
            return (bbox.minimum[axis] + bbox.maximum[axis]) / 2

        self.objects.sort(key=get_centroid)

        # Split into chunks for parallel processing
        chunk_size = len(self.objects) // num_workers
        if chunk_size < max_leaf_size:
            # Not worth parallelizing
            return BVHNode(self.objects, 0, len(self.objects), max_leaf_size)

        chunks = []
        for i in range(num_workers):
            start = i * chunk_size
            end = len(self.objects) if i == num_workers - 1 else (i + 1) * chunk_size
            chunks.append((start, end))

        # Build subtrees in parallel
        def build_subtree(start_end):
            start, end = start_end
            return BVHNode(self.objects, start, end, max_leaf_size)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            subtrees = list(executor.map(build_subtree, chunks))

        # Combine subtrees into final BVH
        return self._combine_subtrees(subtrees)

    def _combine_subtrees(self, subtrees: List[BVHNode]) -> BVHNode:
        """Combine multiple BVH subtrees into a single tree."""
        if len(subtrees) == 0:
            return BVHNode([])
        if len(subtrees) == 1:
            return subtrees[0]

        # Build a balanced tree from subtrees
        while len(subtrees) > 1:
            new_subtrees = []
            for i in range(0, len(subtrees), 2):
                if i + 1 < len(subtrees):
                    # Combine two subtrees
                    combined = BVHNode.__new__(BVHNode)
                    combined.left = subtrees[i]
                    combined.right = subtrees[i + 1]

                    box_left = combined.left.bounding_box()
                    box_right = combined.right.bounding_box()
                    if box_left and box_right:
                        combined.bbox = AABB.surrounding_box(box_left, box_right)
                    elif box_left:
                        combined.bbox = box_left
                    else:
                        combined.bbox = box_right

                    new_subtrees.append(combined)
                else:
                    new_subtrees.append(subtrees[i])
            subtrees = new_subtrees

        return subtrees[0]

    def hit(self, ray: Ray, t_min: float, t_max: float) -> Optional[HitRecord]:
        """Test ray intersection using the BVH."""
        if self.root is None:
            return None
        return self.root.hit(ray, t_min, t_max)

    def bounding_box(self) -> Optional[AABB]:
        """Return the bounding box for the entire BVH."""
        if self.root is None:
            return None
        return self.root.bounding_box()

    def __len__(self) -> int:
        """Return the number of objects in the BVH."""
        return len(self.objects)


def build_bvh(scene: HittableList, max_leaf_size: int = 4) -> BVH:
    """Convenience function to build a BVH from a HittableList.

    Args:
        scene: The scene as a HittableList
        max_leaf_size: Maximum objects per leaf node

    Returns:
        A BVH acceleration structure
    """
    return BVH(list(scene.objects), max_leaf_size)
