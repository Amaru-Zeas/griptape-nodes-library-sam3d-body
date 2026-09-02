from __future__ import annotations

from pathlib import Path

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult

try:
    from sam3d_body_nodes._common import artifact_to_local_path, extract_input_text, publish_video, run_worker
    from sam3d_body_nodes.base_node import SAM3DBodyWorkerNode
except ImportError:
    from _common import artifact_to_local_path, extract_input_text, publish_video, run_worker  # type: ignore
    from base_node import SAM3DBodyWorkerNode  # type: ignore


class RenderSAM3DBodyPoseNode(SAM3DBodyWorkerNode):
    """Render the recovered 3D body mesh over the source video."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add_parameter(Parameter(name="pose_path", input_types=["str"], type="str", tooltip="Pose pack from Predict or Smooth."))
        self.add_parameter(
            Parameter(
                name="media_input",
                input_types=["str", "VideoArtifact", "VideoUrlArtifact", "ImageArtifact", "ImageUrlArtifact"],
                type="str",
                tooltip="Background image/video. Same source used for prediction.",
            )
        )
        self.add_parameter(Parameter(name="overlay_only", input_types=["bool"], type="bool", default_value=True, tooltip="If true, output the mesh composited over the source. If false, output source and overlay side by side."))
        self.add_parameter(Parameter(name="max_frames", input_types=["int"], type="int", default_value=-1, tooltip="Stop after N frames. -1 = all."))
        self.add_parameter(Parameter(name="overlay_video", output_type="VideoUrlArtifact", allowed_modes={ParameterMode.OUTPUT}, tooltip="Rendered overlay published to GTN static files."))
        self.add_parameter(Parameter(name="overlay_path", output_type="str", allowed_modes={ParameterMode.OUTPUT}, tooltip="Local overlay mp4 path."))
        self.move_logs_last()

    def process(self) -> AsyncResult:
        yield lambda: self._process()

    def _process(self) -> None:
        self.log_params.clear_logs()
        pose_path = extract_input_text(self.get_parameter_value("pose_path"))
        if not pose_path:
            raise ValueError("pose_path is required.")
        media = artifact_to_local_path(self.required_media(self.get_parameter_value("media_input"), "media_input"))
        overlay_path = self.out_dir() / "overlay.mp4"
        args = [
            "render",
            "--input",
            str(media),
            "--pose-path",
            pose_path,
            "--output",
            str(overlay_path),
            "--max-frames",
            str(int(self.get_parameter_value("max_frames") or -1)),
            *self.handle_args(),
        ]
        if bool(self.get_parameter_value("overlay_only")):
            args.append("--overlay-only")
        payload = run_worker(self.python_path(), args, cwd=self.repo_path(), log=self.log)
        local = Path(payload.get("overlay_path") or overlay_path)
        self.parameter_output_values["overlay_path"] = str(local)
        if local.exists():
            self.parameter_output_values["overlay_video"] = publish_video(local)
        else:
            self.parameter_output_values["overlay_video"] = None
