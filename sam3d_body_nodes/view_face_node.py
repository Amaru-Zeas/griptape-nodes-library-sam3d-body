from __future__ import annotations

from pathlib import Path

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, ControlNode

try:
    from griptape_nodes.traits.widget import Widget

    _HAS_WIDGET = True
except ImportError:
    Widget = None  # type: ignore[assignment]
    _HAS_WIDGET = False

try:
    from sam3d_body_nodes._common import build_viewer_state, extract_input_text
except ImportError:
    from _common import build_viewer_state, extract_input_text  # type: ignore

LIBRARY_NAME = "GTN SAM 3D Body"


class ViewSAM3DFaceNode(ControlNode):
    """Face mocap close-up: a virtual camera rigged to the head, like a VFX head-mounted cam.

    The camera is fully locked to the skull: it follows the head position and
    facing direction every frame so the face stays centered and framed while
    the body moves. Viewport mouse control is disabled in face-cam mode.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add_parameter(Parameter(name="pose_path", input_types=["str"], type="str", tooltip="Pose pack (.npz) from Predict, Smooth, Face, or One-Click."))
        viewer_kwargs = {}
        if _HAS_WIDGET:
            viewer_kwargs["traits"] = {Widget(name="View3DBodyWidget", library=LIBRARY_NAME)}
        self.add_parameter(
            Parameter(
                name="viewer",
                input_types=["dict"],
                type="dict",
                output_type="dict",
                default_value={"jointsUrl": "", "meshUrl": "", "fps": 24, "faceCam": True},
                allowed_modes={ParameterMode.PROPERTY, ParameterMode.OUTPUT},
                tooltip="Face close-up playback. Camera is locked to the head and follows it; mouse control is disabled.",
                **viewer_kwargs,
            )
        )

    def process(self) -> AsyncResult:
        yield lambda: self._process()

    def _process(self) -> None:
        pose_path = extract_input_text(self.get_parameter_value("pose_path"))
        if not pose_path:
            raise ValueError("pose_path is required. Wire it from Predict, Smooth, Face, or One-Click.")
        viewer_state = build_viewer_state(Path(pose_path), lock_root=True, face_cam=True)
        self.set_parameter_value("viewer", viewer_state)
        try:
            self.publish_update_to_parameter("viewer", viewer_state)
        except Exception:
            pass
        self.parameter_output_values["viewer"] = viewer_state
