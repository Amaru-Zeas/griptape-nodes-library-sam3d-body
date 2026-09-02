from __future__ import annotations

from pathlib import Path

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, ControlNode
from griptape_nodes.traits.options import Options

import importlib

try:
    from sam3d_body_nodes import _common as _common_module
except ImportError:
    import _common as _common_module  # type: ignore

# The engine caches imported modules across library refreshes, so a stale
# _common (without the newer keyword arguments) can survive a refresh. Reload
# it here so this node always binds the current helpers.
_common_module = importlib.reload(_common_module)
build_skeleton_state = _common_module.build_skeleton_state
build_viewer_state = _common_module.build_viewer_state
extract_input_text = _common_module.extract_input_text


class CreateSAM3DAnimationNode(ControlNode):
    """Comfy-style settings node: choose HOW the recovered motion is shown in 3D.

    Sits before the 3D Body Viewer, like ComfyUI's "Create 3D Animation File":
    body mesh (default, with face close-up inset) or octahedral bone skeleton,
    plus bone look, smoothing, fps, and camera-translation knobs. Wire the
    `viewer` output into a 3D Body Viewer node.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add_parameter(Parameter(name="pose_path", input_types=["str"], type="str", tooltip="Pose pack (.npz) from Predict, Smooth, Face, or One-Click."))
        self.add_parameter(
            Parameter(
                name="mesh_style",
                input_types=["str"],
                type="str",
                default_value="body_mesh",
                traits={Options(choices=["body_mesh", "bones_only"])},
                tooltip="body_mesh = colored mesh with face close-up inset. bones_only = octahedral mocap bone skeleton.",
            )
        )
        self.add_parameter(
            Parameter(
                name="bone_vis",
                input_types=["str"],
                type="str",
                default_value="octahedrons",
                traits={Options(choices=["octahedrons", "lines"])},
                tooltip="Bone look for bones_only: solid mocap octahedrons, or thin sticks with joint dots.",
            )
        )
        self.add_parameter(
            Parameter(
                name="bone_vis_color",
                input_types=["str"],
                type="str",
                default_value="rainbow_y",
                traits={Options(choices=["rainbow_y", "solid"])},
                tooltip="rainbow_y = hue gradient by height (blue feet, red head). solid = single gray.",
            )
        )
        self.add_parameter(
            Parameter(
                name="bone_smooth_window",
                input_types=["int"],
                type="int",
                default_value=0,
                tooltip="Temporal moving average over the bones, in frames (odd, e.g. 5). 0 = off. Kills residual jitter, softens fast motion.",
            )
        )
        self.add_parameter(
            Parameter(
                name="fps",
                input_types=["int"],
                type="int",
                default_value=0,
                tooltip="Playback fps. 0 = use the source video's fps.",
            )
        )
        self.add_parameter(
            Parameter(
                name="camera_translation",
                input_types=["str"],
                type="str",
                default_value="off",
                traits={Options(choices=["off", "on"])},
                tooltip="off = body stays centered on the grid (camera motion removed). on = keep the estimated world translation.",
            )
        )
        self.add_parameter(Parameter(name="show_hands", input_types=["bool"], type="bool", default_value=True, tooltip="Include the finger bones (bones_only)."))
        self.add_parameter(
            Parameter(
                name="viewer",
                output_type="dict",
                allowed_modes={ParameterMode.OUTPUT},
                tooltip="Animation state. Wire into a 3D Body Viewer node.",
            )
        )

    def process(self) -> AsyncResult:
        yield lambda: self._process()

    def _process(self) -> None:
        pose_path = extract_input_text(self.get_parameter_value("pose_path"))
        if not pose_path:
            raise ValueError("pose_path is required. Wire it from One-Click, Predict, or Smooth.")
        lock_root = str(self.get_parameter_value("camera_translation") or "off") != "on"
        fps_override = float(int(self.get_parameter_value("fps") or 0))
        style = str(self.get_parameter_value("mesh_style") or "body_mesh")
        if style == "bones_only":
            viewer_state = build_skeleton_state(
                Path(pose_path),
                lock_root=lock_root,
                include_hands=bool(self.get_parameter_value("show_hands")),
                smooth_window=int(self.get_parameter_value("bone_smooth_window") or 0),
                fps_override=fps_override,
            )
            bone_vis = str(self.get_parameter_value("bone_vis") or "octahedrons")
            viewer_state["skeletonStyle"] = {
                "bones": "lines" if bone_vis == "lines" else "octahedron",
                "color": str(self.get_parameter_value("bone_vis_color") or "rainbow_y"),
            }
        else:
            viewer_state = build_viewer_state(Path(pose_path), lock_root=lock_root, fps_override=fps_override)
        self.parameter_output_values["viewer"] = viewer_state
