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


class ViewSAM3DBodyNode(ControlNode):
    """Interactive 3D viewer: plays the recovered joint animation with the exported mesh."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add_parameter(Parameter(name="pose_path", input_types=["str"], type="str", tooltip="Pose pack (.npz) from Predict, Smooth, Face, or One-Click."))
        self.add_parameter(Parameter(name="glb_path", input_types=["str"], type="str", default_value="", tooltip="Unused for display (mesh streams from the pose pack); kept for wiring compatibility."))
        self.add_parameter(
            Parameter(
                name="lock_root",
                input_types=["bool"],
                type="bool",
                default_value=True,
                tooltip="Keep the body centered on the grid (removes camera-motion drift). Off = keep estimated world translation.",
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
                default_value={"jointsUrl": "", "meshUrl": "", "fps": 24},
                allowed_modes={ParameterMode.PROPERTY, ParameterMode.OUTPUT},
                tooltip="3D playback. Drag to orbit, wheel to zoom, Play to animate.",
                **viewer_kwargs,
            )
        )
        self.add_parameter(
            Parameter(
                name="joints_url",
                output_type="str",
                allowed_modes={ParameterMode.OUTPUT},
                tooltip="Published joint-animation JSON.",
                ui_options={"hide_property": True},
            )
        )

    def process(self) -> AsyncResult:
        yield lambda: self._process()

    def _process(self) -> None:
        pose_path = extract_input_text(self.get_parameter_value("pose_path"))
        if not pose_path:
            raise ValueError("pose_path is required. Wire it from Predict, Smooth, Face, or One-Click.")
        viewer_state = build_viewer_state(Path(pose_path), lock_root=bool(self.get_parameter_value("lock_root")))
        self.set_parameter_value("viewer", viewer_state)
        try:
            self.publish_update_to_parameter("viewer", viewer_state)
        except Exception:
            pass
        self.parameter_output_values["viewer"] = viewer_state
        self.parameter_output_values["joints_url"] = viewer_state["jointsUrl"]
