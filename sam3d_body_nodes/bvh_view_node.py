from __future__ import annotations

import json
import math
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, ControlNode

try:
    from griptape_nodes.traits.widget import Widget

    _HAS_WIDGET = True
except ImportError:
    Widget = None  # type: ignore[assignment]
    _HAS_WIDGET = False

try:
    from sam3d_body_nodes._common import extract_input_text, publish_bytes
except ImportError:
    from _common import extract_input_text, publish_bytes  # type: ignore

LIBRARY_NAME = "GTN SAM 3D Body"


def _rot_matrix(axis: str, degrees: np.ndarray) -> np.ndarray:
    """Per-frame rotation matrices (F, 3, 3) for one Euler axis."""
    rad = np.radians(degrees)
    c, s = np.cos(rad), np.sin(rad)
    n = len(rad)
    m = np.zeros((n, 3, 3), dtype=np.float64)
    if axis == "X":
        m[:, 0, 0] = 1
        m[:, 1, 1] = c
        m[:, 1, 2] = -s
        m[:, 2, 1] = s
        m[:, 2, 2] = c
    elif axis == "Y":
        m[:, 0, 0] = c
        m[:, 0, 2] = s
        m[:, 1, 1] = 1
        m[:, 2, 0] = -s
        m[:, 2, 2] = c
    else:
        m[:, 0, 0] = c
        m[:, 0, 1] = -s
        m[:, 1, 0] = s
        m[:, 1, 1] = c
        m[:, 2, 2] = 1
    return m


def parse_bvh(text: str) -> dict:
    """Parse a BVH file and run forward kinematics.

    Returns {"positions": (F, J, 3) float32 world-space, "parents": list[int],
    "names": list[str], "fps": float}. Handles both standard hierarchical BVH
    (offsets + rotation channels) and the flat "star" BVH this library exports
    (every joint a child of root with absolute position channels).
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    joints: list[dict[str, Any]] = []
    channel_owner: list[tuple[int, str]] = []
    stack: list[int] = []
    pending = -1
    motion_start = None
    for i, ln in enumerate(lines):
        parts = ln.split()
        kw = parts[0].upper()
        if kw in ("ROOT", "JOINT"):
            joints.append(
                {
                    "name": parts[1] if len(parts) > 1 else f"joint_{len(joints)}",
                    "parent": stack[-1] if stack else -1,
                    "offset": np.zeros(3),
                    "channels": [],
                    "end": False,
                }
            )
            pending = len(joints) - 1
        elif kw == "END":  # End Site
            joints.append(
                {
                    "name": (joints[stack[-1]]["name"] + "_end") if stack else "end",
                    "parent": stack[-1] if stack else -1,
                    "offset": np.zeros(3),
                    "channels": [],
                    "end": True,
                }
            )
            pending = len(joints) - 1
        elif kw == "{":
            stack.append(pending)
        elif kw == "}":
            stack.pop()
        elif kw == "OFFSET" and stack:
            joints[stack[-1]]["offset"] = np.array([float(v) for v in parts[1:4]])
        elif kw == "CHANNELS" and stack:
            chans = parts[2 : 2 + int(parts[1])]
            joints[stack[-1]]["channels"] = chans
            for c in chans:
                channel_owner.append((stack[-1], c))
        elif kw == "MOTION":
            motion_start = i + 1
            break
    if not joints or motion_start is None:
        raise ValueError("Not a valid BVH file (no HIERARCHY/MOTION sections).")

    frame_time = 1.0 / 30.0
    rows: list[list[float]] = []
    for ln in lines[motion_start:]:
        low = ln.lower()
        if low.startswith("frames"):
            continue
        if low.startswith("frame time"):
            try:
                frame_time = float(ln.split(":")[1])
            except (IndexError, ValueError):
                pass
            continue
        try:
            rows.append([float(v) for v in ln.split()])
        except ValueError:
            continue
    data = np.asarray(rows, dtype=np.float64)
    if data.ndim != 2 or not len(data):
        raise ValueError("BVH has no motion frames.")
    if data.shape[1] < len(channel_owner):
        raise ValueError(f"BVH motion rows have {data.shape[1]} values, hierarchy declares {len(channel_owner)} channels.")

    n_frames = data.shape[0]
    n_joints = len(joints)
    # Column indices per joint, in channel order.
    joint_cols: list[list[tuple[int, str]]] = [[] for _ in range(n_joints)]
    for col, (j, chan) in enumerate(channel_owner):
        joint_cols[j].append((col, chan))

    # Star BVH (this library's exporter): every non-root joint hangs off the
    # root and carries absolute position channels (End Sites excluded).
    # Positions are world-space already, so skip the parent accumulation FK
    # would apply, and drop the placeholder End Sites entirely.
    real = [i for i, j in enumerate(joints) if not j["end"]]
    non_root_real = [i for i in real if joints[i]["parent"] == 0]
    star = (
        len(real) > 4
        and len(non_root_real) == len(real) - 1
        and all(c[1].lower().endswith("position") for i in real for c in joint_cols[i])
    )

    if star:
        positions = np.zeros((n_frames, len(real), 3), dtype=np.float64)
        for out_idx, j in enumerate(real):
            for col, chan in joint_cols[j]:
                axis = chan[0].upper()
                positions[:, out_idx, "XYZ".index(axis)] = data[:, col]
        return {
            "positions": positions.astype(np.float32),
            "parents": [-1] + [0] * (len(real) - 1),
            "names": [str(joints[i]["name"]) for i in real],
            "star": True,
            "fps": float(1.0 / frame_time if frame_time > 1e-6 else 30.0),
        }

    # Standard hierarchical BVH: forward kinematics through the joint tree.
    positions = np.zeros((n_frames, n_joints, 3), dtype=np.float64)
    world_rot = np.zeros((n_frames, n_joints, 3, 3), dtype=np.float64)
    for j, joint in enumerate(joints):
        local_rot = np.broadcast_to(np.eye(3), (n_frames, 3, 3)).copy()
        local_pos = np.broadcast_to(joint["offset"], (n_frames, 3)).copy()
        for col, chan in joint_cols[j]:
            lc = chan.lower()
            axis = chan[0].upper()
            if lc.endswith("position"):
                local_pos[:, "XYZ".index(axis)] += data[:, col]
            elif lc.endswith("rotation"):
                local_rot = np.einsum("fij,fjk->fik", local_rot, _rot_matrix(axis, data[:, col]))
        p = joint["parent"]
        if p < 0:
            world_rot[:, j] = local_rot
            positions[:, j] = local_pos
        else:
            world_rot[:, j] = np.einsum("fij,fjk->fik", world_rot[:, p], local_rot)
            positions[:, j] = positions[:, p] + np.einsum("fij,fj->fi", world_rot[:, p], local_pos)

    return {
        "positions": positions.astype(np.float32),
        "parents": [int(j["parent"]) for j in joints],
        "names": [str(j["name"]) for j in joints],
        "star": False,
        "fps": float(1.0 / frame_time if frame_time > 1e-6 else 30.0),
    }


def infer_bones_mst(positions: np.ndarray) -> list[tuple[int, int]]:
    """Bone connectivity via a minimum spanning tree over mean joint positions.

    Used for flat/star BVHs whose hierarchy carries no structure: human joints
    chain to their nearest neighbors, so the MST recovers a plausible skeleton.
    """
    mean = positions.mean(axis=0)  # (J, 3)
    n = len(mean)
    if n < 2:
        return []
    dist = np.linalg.norm(mean[:, None, :] - mean[None, :, :], axis=2)
    in_tree = {0}
    edges: list[tuple[int, int]] = []
    best = dist[0].copy()
    best_from = np.zeros(n, dtype=int)
    best[0] = np.inf
    for _ in range(n - 1):
        j = int(np.argmin(best))
        if not math.isfinite(best[j]):
            break
        edges.append((int(best_from[j]), j))
        in_tree.add(j)
        best[j] = np.inf
        closer = dist[j] < best
        best_from[closer] = j
        best = np.minimum(best, dist[j])
        best[list(in_tree)] = np.inf
    return edges


class ViewBVHSkeletonNode(ControlNode):
    """Play a .bvh mocap file as an animated bone skeleton in the 3D viewer.

    Works with the BVH this library exports and with external files (Mixamo,
    Rokoko, etc.). Standard hierarchical BVHs use their own bone tree; flat
    star-hierarchy files get their bones inferred from joint proximity.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add_parameter(
            Parameter(
                name="bvh_path",
                input_types=["str"],
                type="str",
                tooltip="Path to a .bvh file - wire bvh_path from One-Click / Export, or point at any BVH.",
            )
        )
        self.add_parameter(
            Parameter(
                name="lock_root",
                input_types=["bool"],
                type="bool",
                default_value=False,
                tooltip="Keep the skeleton centered at the origin (ignore root travel).",
            )
        )
        viewer_kwargs = {}
        if _HAS_WIDGET:
            viewer_kwargs["traits"] = {Widget(name="View3DBodyWidget", library=LIBRARY_NAME)}
        self.add_parameter(
            Parameter(
                name="viewer",
                input_types=["dict"],
                type="dict",
                output_type="dict",
                default_value={"skeletonUrl": "", "fps": 30},
                allowed_modes={ParameterMode.PROPERTY, ParameterMode.OUTPUT},
                tooltip="Skeleton playback: orbit / pan / zoom, play and scrub.",
                **viewer_kwargs,
            )
        )
        self.add_parameter(
            Parameter(
                name="joint_count",
                output_type="int",
                allowed_modes={ParameterMode.OUTPUT},
                tooltip="Number of joints in the BVH.",
            )
        )

    def process(self) -> AsyncResult:
        yield lambda: self._process()

    def _process(self) -> None:
        bvh_path = extract_input_text(self.get_parameter_value("bvh_path"))
        if not bvh_path:
            raise ValueError("bvh_path is required. Wire it from One-Click / Export, or set a .bvh file path.")
        path = Path(bvh_path)
        if not path.exists():
            raise ValueError(f"BVH file not found: {path}")

        parsed = parse_bvh(path.read_text(encoding="utf-8", errors="replace"))
        positions = parsed["positions"]  # (F, J, 3)

        # Units: BVH files are often authored in centimeters; the viewer's
        # grid is meters. Scale when the skeleton is implausibly tall.
        height = float(positions[0, :, 1].max() - positions[0, :, 1].min())
        if height > 10.0:
            positions = positions * 0.01

        if bool(self.get_parameter_value("lock_root")):
            root_xz = positions[:, 0:1, [0, 2]]
            positions = positions.copy()
            positions[:, :, [0, 2]] -= root_xz

        # Ground the clip: lowest point across all frames sits on the grid.
        positions = positions - np.array([0.0, float(positions[:, :, 1].min()), 0.0], dtype=np.float32)

        if parsed["star"]:
            bones = infer_bones_mst(positions)
        else:
            bones = [(p, j) for j, p in enumerate(parsed["parents"]) if p >= 0]

        payload = {
            "fps": parsed["fps"],
            "nJoints": int(positions.shape[1]),
            "bones": [[int(a), int(b)] for a, b in bones],
            "frames": [np.round(frame.reshape(-1), 4).tolist() for frame in positions],
        }
        url = publish_bytes(json.dumps(payload).encode("utf-8"), f"bvh_skeleton_{uuid.uuid4().hex[:8]}.json")

        viewer_state = {"skeletonUrl": url, "fps": parsed["fps"]}
        self.set_parameter_value("viewer", viewer_state)
        try:
            self.publish_update_to_parameter("viewer", viewer_state)
        except Exception:
            pass
        self.parameter_output_values["viewer"] = viewer_state
        self.parameter_output_values["joint_count"] = int(positions.shape[1])
