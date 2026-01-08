"""
Arbitrary Output Variables (AOV) / Render Pass support.

Implements various render passes commonly used in production:
- Beauty (final color)
- Depth (Z-depth for compositing)
- Normal (world-space normals)
- Albedo (unlit surface color)
- Object ID (for masking)
- Material ID
- UV coordinates
- Emission
- Direct/indirect lighting

These passes enable post-production compositing and selective adjustments.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any
from enum import Enum
import math

import numpy as np

from .vec3 import Vec3, Color, Point3


class AOVType(Enum):
    """Standard AOV/render pass types."""

    BEAUTY = "beauty"  # Final rendered color (RGB)
    DEPTH = "depth"  # Z-depth from camera (float)
    NORMAL = "normal"  # World-space normals (RGB)
    NORMAL_CAMERA = "normal_camera"  # Camera-space normals (RGB)
    ALBEDO = "albedo"  # Unlit surface color (RGB)
    EMISSION = "emission"  # Emissive contribution (RGB)
    DIRECT = "direct"  # Direct lighting only (RGB)
    INDIRECT = "indirect"  # Indirect/GI lighting only (RGB)
    DIFFUSE = "diffuse"  # Diffuse component (RGB)
    SPECULAR = "specular"  # Specular component (RGB)
    OBJECT_ID = "object_id"  # Per-object ID (int)
    MATERIAL_ID = "material_id"  # Per-material ID (int)
    UV = "uv"  # UV coordinates (RG)
    POSITION = "position"  # World position (RGB)
    MOTION = "motion"  # Motion vectors (RG)
    AO = "ambient_occlusion"  # Ambient occlusion (float)
    SHADOW = "shadow"  # Shadow mask (float)


@dataclass
class AOVSample:
    """Sample data for a single ray hit containing all AOV information."""

    depth: float = float('inf')  # Distance from camera
    normal: Optional[Vec3] = None  # World-space normal
    albedo: Optional[Color] = None  # Surface albedo
    emission: Optional[Color] = None  # Emissive color
    direct: Optional[Color] = None  # Direct lighting
    indirect: Optional[Color] = None  # Indirect lighting
    object_id: int = 0  # Object identifier
    material_id: int = 0  # Material identifier
    uv: Optional[Tuple[float, float]] = None  # UV coordinates
    position: Optional[Point3] = None  # World position
    motion: Optional[Tuple[float, float]] = None  # Motion vector (screen-space)


@dataclass
class AOVBuffer:
    """Buffer for storing AOV data during rendering."""

    width: int
    height: int
    aov_type: AOVType

    # Data storage
    data: np.ndarray = field(init=False)
    sample_count: np.ndarray = field(init=False)

    def __post_init__(self):
        """Initialize buffer arrays based on AOV type."""
        channels = self._get_channel_count()
        self.data = np.zeros((self.height, self.width, channels), dtype=np.float32)
        self.sample_count = np.zeros((self.height, self.width), dtype=np.int32)

    def _get_channel_count(self) -> int:
        """Get number of channels for this AOV type."""
        single_channel = {
            AOVType.DEPTH, AOVType.OBJECT_ID, AOVType.MATERIAL_ID,
            AOVType.AO, AOVType.SHADOW
        }
        two_channel = {AOVType.UV, AOVType.MOTION}

        if self.aov_type in single_channel:
            return 1
        elif self.aov_type in two_channel:
            return 2
        else:
            return 3  # RGB

    def add_sample(self, x: int, y: int, value: Any) -> None:
        """Add a sample to the buffer.

        Args:
            x: Pixel x coordinate
            y: Pixel y coordinate
            value: Sample value (scalar, tuple, Vec3, or Color)
        """
        if not (0 <= x < self.width and 0 <= y < self.height):
            return

        # Convert value to array
        if isinstance(value, (Vec3, Color)):
            arr_value = np.array([value.x, value.y, value.z])
        elif isinstance(value, tuple):
            arr_value = np.array(value)
        else:
            arr_value = np.array([value])

        # Accumulate with running average
        count = self.sample_count[y, x]
        new_count = count + 1

        channels = self._get_channel_count()
        self.data[y, x, :channels] = (
            self.data[y, x, :channels] * count + arr_value[:channels]
        ) / new_count

        self.sample_count[y, x] = new_count

    def get_image(self) -> np.ndarray:
        """Get the final image from accumulated samples.

        Returns:
            Image array (H, W, C)
        """
        return self.data.copy()

    def clear(self) -> None:
        """Clear buffer to start fresh."""
        self.data.fill(0)
        self.sample_count.fill(0)


class AOVManager:
    """Manages multiple AOV buffers for a render."""

    def __init__(self, width: int, height: int):
        """Initialize AOV manager.

        Args:
            width: Image width
            height: Image height
        """
        self.width = width
        self.height = height
        self.buffers: Dict[AOVType, AOVBuffer] = {}

    def enable_aov(self, aov_type: AOVType) -> None:
        """Enable an AOV pass.

        Args:
            aov_type: Type of AOV to enable
        """
        if aov_type not in self.buffers:
            self.buffers[aov_type] = AOVBuffer(
                width=self.width,
                height=self.height,
                aov_type=aov_type
            )

    def enable_aovs(self, aov_types: List[AOVType]) -> None:
        """Enable multiple AOV passes.

        Args:
            aov_types: List of AOV types to enable
        """
        for aov_type in aov_types:
            self.enable_aov(aov_type)

    def disable_aov(self, aov_type: AOVType) -> None:
        """Disable an AOV pass.

        Args:
            aov_type: Type of AOV to disable
        """
        if aov_type in self.buffers:
            del self.buffers[aov_type]

    def is_enabled(self, aov_type: AOVType) -> bool:
        """Check if an AOV is enabled.

        Args:
            aov_type: Type of AOV to check

        Returns:
            True if enabled
        """
        return aov_type in self.buffers

    def add_sample(self, x: int, y: int, sample: AOVSample) -> None:
        """Add sample data for all enabled AOVs.

        Args:
            x: Pixel x coordinate
            y: Pixel y coordinate
            sample: AOV sample data
        """
        for aov_type, buffer in self.buffers.items():
            value = self._extract_value(sample, aov_type)
            if value is not None:
                buffer.add_sample(x, y, value)

    def _extract_value(self, sample: AOVSample, aov_type: AOVType) -> Any:
        """Extract the appropriate value from sample for given AOV type.

        Args:
            sample: AOV sample data
            aov_type: AOV type to extract

        Returns:
            Value for the AOV, or None if not available
        """
        if aov_type == AOVType.DEPTH:
            return sample.depth if sample.depth != float('inf') else 0.0
        elif aov_type == AOVType.NORMAL:
            return sample.normal
        elif aov_type == AOVType.ALBEDO:
            return sample.albedo
        elif aov_type == AOVType.EMISSION:
            return sample.emission
        elif aov_type == AOVType.DIRECT:
            return sample.direct
        elif aov_type == AOVType.INDIRECT:
            return sample.indirect
        elif aov_type == AOVType.OBJECT_ID:
            return sample.object_id
        elif aov_type == AOVType.MATERIAL_ID:
            return sample.material_id
        elif aov_type == AOVType.UV:
            return sample.uv
        elif aov_type == AOVType.POSITION:
            return sample.position
        elif aov_type == AOVType.MOTION:
            return sample.motion
        else:
            return None

    def get_aov(self, aov_type: AOVType) -> Optional[np.ndarray]:
        """Get the final image for an AOV.

        Args:
            aov_type: AOV type to retrieve

        Returns:
            Image array or None if not enabled
        """
        if aov_type in self.buffers:
            return self.buffers[aov_type].get_image()
        return None

    def get_all_aovs(self) -> Dict[str, np.ndarray]:
        """Get all enabled AOVs.

        Returns:
            Dictionary mapping AOV names to image arrays
        """
        return {
            aov_type.value: buffer.get_image()
            for aov_type, buffer in self.buffers.items()
        }

    def clear(self) -> None:
        """Clear all buffers."""
        for buffer in self.buffers.values():
            buffer.clear()


def normalize_depth(
    depth_buffer: np.ndarray,
    near: float = 0.0,
    far: Optional[float] = None,
    invert: bool = False
) -> np.ndarray:
    """Normalize depth buffer to [0, 1] range.

    Args:
        depth_buffer: Raw depth values (H, W, 1) or (H, W)
        near: Near clipping plane
        far: Far clipping plane (None = auto from max)
        invert: Invert depth (1 = near, 0 = far)

    Returns:
        Normalized depth buffer (H, W)
    """
    depth = depth_buffer.squeeze()

    # Handle infinite values
    mask = np.isfinite(depth)

    if far is None:
        far = np.max(depth[mask]) if np.any(mask) else 1.0

    # Normalize
    normalized = np.clip((depth - near) / (far - near + 1e-8), 0.0, 1.0)

    # Set infinite depths to 1.0 (far plane)
    normalized[~mask] = 1.0

    if invert:
        normalized = 1.0 - normalized

    return normalized


def pack_normal(normal_buffer: np.ndarray) -> np.ndarray:
    """Pack world-space normals to [0, 1] range for display.

    Normals are in range [-1, 1], we remap to [0, 1]:
    R = (Nx + 1) / 2
    G = (Ny + 1) / 2
    B = (Nz + 1) / 2

    Args:
        normal_buffer: Normal buffer (H, W, 3) in range [-1, 1]

    Returns:
        Packed normals (H, W, 3) in range [0, 1]
    """
    return (normal_buffer + 1.0) / 2.0


def unpack_normal(packed_buffer: np.ndarray) -> np.ndarray:
    """Unpack normals from [0, 1] range to [-1, 1].

    Args:
        packed_buffer: Packed normals (H, W, 3) in range [0, 1]

    Returns:
        World-space normals (H, W, 3) in range [-1, 1]
    """
    return packed_buffer * 2.0 - 1.0


def create_object_id_colormap(
    id_buffer: np.ndarray,
    seed: int = 42
) -> np.ndarray:
    """Create colorful visualization of object ID buffer.

    Args:
        id_buffer: Object ID buffer (H, W, 1) or (H, W)
        seed: Random seed for consistent colors

    Returns:
        Colored visualization (H, W, 3)
    """
    ids = id_buffer.squeeze().astype(np.int32)
    unique_ids = np.unique(ids)

    # Generate consistent colors for each ID
    rng = np.random.default_rng(seed)
    colors = {}
    colors[0] = np.array([0, 0, 0])  # Background is black

    for obj_id in unique_ids:
        if obj_id != 0:
            # Generate saturated colors using HSV
            hue = rng.random()
            saturation = 0.7 + rng.random() * 0.3
            value = 0.8 + rng.random() * 0.2

            # HSV to RGB conversion
            h_i = int(hue * 6)
            f = hue * 6 - h_i
            p = value * (1 - saturation)
            q = value * (1 - f * saturation)
            t = value * (1 - (1 - f) * saturation)

            if h_i == 0:
                rgb = [value, t, p]
            elif h_i == 1:
                rgb = [q, value, p]
            elif h_i == 2:
                rgb = [p, value, t]
            elif h_i == 3:
                rgb = [p, q, value]
            elif h_i == 4:
                rgb = [t, p, value]
            else:
                rgb = [value, p, q]

            colors[obj_id] = np.array(rgb)

    # Map IDs to colors
    h, w = ids.shape
    result = np.zeros((h, w, 3), dtype=np.float32)

    for obj_id, color in colors.items():
        mask = ids == obj_id
        result[mask] = color

    return result


def combine_direct_indirect(
    direct: np.ndarray,
    indirect: np.ndarray,
    albedo: Optional[np.ndarray] = None
) -> np.ndarray:
    """Combine direct and indirect lighting passes.

    If albedo is provided, reconstruct beauty pass:
    beauty = albedo * (direct + indirect)

    Otherwise just sum:
    beauty = direct + indirect

    Args:
        direct: Direct lighting (H, W, 3)
        indirect: Indirect lighting (H, W, 3)
        albedo: Optional albedo (H, W, 3)

    Returns:
        Combined result (H, W, 3)
    """
    combined = direct + indirect

    if albedo is not None:
        combined = albedo * combined

    return combined


def depth_to_world_position(
    depth_buffer: np.ndarray,
    camera_origin: np.ndarray,
    camera_forward: np.ndarray,
    camera_right: np.ndarray,
    camera_up: np.ndarray,
    fov: float,
    aspect_ratio: float
) -> np.ndarray:
    """Reconstruct world positions from depth buffer.

    Args:
        depth_buffer: Depth buffer (H, W)
        camera_origin: Camera position (3,)
        camera_forward: Camera forward vector (3,)
        camera_right: Camera right vector (3,)
        camera_up: Camera up vector (3,)
        fov: Field of view in degrees
        aspect_ratio: Width / Height

    Returns:
        World positions (H, W, 3)
    """
    h, w = depth_buffer.shape[:2]
    depth = depth_buffer.squeeze()

    # Create normalized screen coordinates
    y_coords = np.linspace(1, -1, h)  # Top to bottom
    x_coords = np.linspace(-1, 1, w)  # Left to right
    xx, yy = np.meshgrid(x_coords, y_coords)

    # Scale by FOV and aspect ratio
    half_height = np.tan(np.radians(fov) / 2)
    half_width = half_height * aspect_ratio

    xx = xx * half_width
    yy = yy * half_height

    # Create ray directions
    directions = (
        camera_forward[np.newaxis, np.newaxis, :] +
        xx[:, :, np.newaxis] * camera_right[np.newaxis, np.newaxis, :] +
        yy[:, :, np.newaxis] * camera_up[np.newaxis, np.newaxis, :]
    )

    # Normalize directions
    lengths = np.linalg.norm(directions, axis=2, keepdims=True)
    directions = directions / (lengths + 1e-8)

    # Calculate world positions
    positions = camera_origin + directions * depth[:, :, np.newaxis]

    return positions.astype(np.float32)


class RenderPassCompositor:
    """Compositor for combining render passes."""

    def __init__(self, aov_manager: AOVManager):
        """Initialize compositor.

        Args:
            aov_manager: AOV manager with render passes
        """
        self.aov_manager = aov_manager

    def get_beauty(self) -> Optional[np.ndarray]:
        """Get beauty/final render pass."""
        return self.aov_manager.get_aov(AOVType.BEAUTY)

    def get_depth_visualization(
        self,
        near: float = 0.0,
        far: Optional[float] = None
    ) -> Optional[np.ndarray]:
        """Get depth pass as grayscale visualization.

        Args:
            near: Near clipping plane
            far: Far clipping plane

        Returns:
            Grayscale depth image (H, W, 3)
        """
        depth = self.aov_manager.get_aov(AOVType.DEPTH)
        if depth is None:
            return None

        normalized = normalize_depth(depth, near, far, invert=True)
        return np.stack([normalized, normalized, normalized], axis=-1)

    def get_normal_visualization(self) -> Optional[np.ndarray]:
        """Get normal pass as RGB visualization."""
        normal = self.aov_manager.get_aov(AOVType.NORMAL)
        if normal is None:
            return None

        return pack_normal(normal)

    def get_object_id_visualization(self) -> Optional[np.ndarray]:
        """Get object ID pass as colored visualization."""
        object_id = self.aov_manager.get_aov(AOVType.OBJECT_ID)
        if object_id is None:
            return None

        return create_object_id_colormap(object_id)

    def reconstruct_beauty(self) -> Optional[np.ndarray]:
        """Reconstruct beauty from direct, indirect, and albedo passes.

        Returns:
            Reconstructed beauty or None if passes not available
        """
        direct = self.aov_manager.get_aov(AOVType.DIRECT)
        indirect = self.aov_manager.get_aov(AOVType.INDIRECT)

        if direct is None or indirect is None:
            return None

        albedo = self.aov_manager.get_aov(AOVType.ALBEDO)
        return combine_direct_indirect(direct, indirect, albedo)
