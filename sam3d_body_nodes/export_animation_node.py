from __future__ import annotations

from pathlib import Path

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult
from griptape_nodes.traits.options import Options

try:
    from sam3d_body_nodes._common import extract_input_text, publish_file, run_worker
    from sam3d_body_nodes.base_node import SAM3DBodyWorkerNode
except ImportError:
    from _common import extract_input_text, publish_file, run_worker  # type: ignore
    from base_node import SAM3DBodyWorkerNode  # type: ignore


class Create3DAnimationFileNode(SAM3DBodyWorkerNode):
    """Export pose data as GLB mesh, BVH mocap, or both."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add_parameter(Parameter(name="pose_path", input_types=["str"], type="str", tooltip="Pose pack from Predict / Smooth / Face."))
        self.add_parameter(
            Parameter(
                name="format",
                input_types=["str"],
                type="str",
                default_value="both",
                traits={Options(choices=["glb", "bvh", "both"])},
                tooltip="GLB = body mesh. BVH = joint mocap for Blender. both = write both files.",
            )
        )
        self.add_parameter(Parameter(name="frame_index", input_types=["int"], type="int", default_value=0, tooltip="Which frame to bake into the GLB mesh."))
        self.add_parameter(Parameter(name="glb_path", output_type="str", allowed_modes={ParameterMode.OUTPUT}, tooltip="Local GLB/OBJ path."))
        self.add_parameter(Parameter(name="bvh_path", output_type="str", allowed_modes={ParameterMode.OUTPUT}, tooltip="Local BVH path."))
        self.add_parameter(Parameter(name="glb_url", output_type="str", allowed_modes={ParameterMode.OUTPUT}, tooltip="Published GLB static URL."))
        self.add_parameter(Parameter(name="bvh_url", output_type="str", allowed_modes={ParameterMode.OUTPUT}, tooltip="Published BVH static URL."))
        self.move_logs_last()

    def process(self) -> AsyncResult:
        yield lambda: self._process()

    def _process(self) -> None:
        self.log_params.clear_logs()
        pose_path = extract_input_text(self.get_parameter_value("pose_path"))
        if not pose_path:
            raise ValueError("pose_path is required.")
        export_dir = self.out_dir() / "export"
        payload = run_worker(
            self.python_path(),
            [
                "export",
                "--pose-path",
                pose_path,
                "--output",
                str(export_dir),
                "--format",
                str(self.get_parameter_value("format") or "both"),
                "--frame-index",
                str(int(self.get_parameter_value("frame_index") or 0)),
                *self.handle_args(),
            ],
            cwd=self.repo_path(),
            log=self.log,
        )
        glb_path = Path(payload.get("glb_path") or "")
        bvh_path = Path(payload.get("bvh_path") or "")
        self.parameter_output_values["glb_path"] = str(glb_path) if glb_path.exists() else ""
        self.parameter_output_values["bvh_path"] = str(bvh_path) if bvh_path.exists() else ""
        self.parameter_output_values["glb_url"] = publish_file(glb_path) if glb_path.exists() else ""
        self.parameter_output_values["bvh_url"] = publish_file(bvh_path) if bvh_path.exists() else ""
