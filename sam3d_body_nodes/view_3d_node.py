from __future__ import annotations

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, ControlNode

try:
    from griptape_nodes.traits.widget import Widget

    _HAS_WIDGET = True
except ImportError:
    Widget = None  # type: ignore[assignment]
    _HAS_WIDGET = False

LIBRARY_NAME = "GTN SAM 3D Body"


class ViewSAM3DBodyNode(ControlNode):
    """Interactive 3D viewer: plays whatever a Create 3D Animation node wires in.

    Body mesh mode shows the face mocap close-up inset (head-locked camera);
    bones_only mode shows the octahedral skeleton. Orbit / pan / zoom / play /
    scrub / fullscreen.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
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
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY, ParameterMode.OUTPUT},
                tooltip="3D playback. Wire from Create 3D Animation. Drag to orbit, wheel to zoom, Play to animate; the Face button toggles the head-locked close-up inset.",
                **viewer_kwargs,
            )
        )

    def process(self) -> AsyncResult:
        yield lambda: self._process()

    def _process(self) -> None:
        viewer_state = self.get_parameter_value("viewer") or {}
        self.set_parameter_value("viewer", viewer_state)
        try:
            self.publish_update_to_parameter("viewer", viewer_state)
        except Exception:
            pass
        self.parameter_output_values["viewer"] = viewer_state
