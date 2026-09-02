from __future__ import annotations

from pathlib import Path

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult

try:
    from sam3d_body_nodes._common import artifact_to_local_path, extract_input_text, run_worker
    from sam3d_body_nodes.base_node import SAM3DBodyWorkerNode
except ImportError:
    from _common import artifact_to_local_path, extract_input_text, run_worker  # type: ignore
    from base_node import SAM3DBodyWorkerNode  # type: ignore


class SAM3DBodyFaceExpressionNode(SAM3DBodyWorkerNode):
    """SAM 3D Body does not detect face expressions; this adds MediaPipe blendshapes."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add_parameter(Parameter(name="pose_path", input_types=["str"], type="str", tooltip="Pose pack to enrich with face blendshapes."))
        self.add_parameter(
            Parameter(
                name="media_input",
                input_types=["str", "VideoArtifact", "VideoUrlArtifact", "ImageArtifact", "ImageUrlArtifact"],
                type="str",
                tooltip="Same source media used for prediction.",
            )
        )
        self.add_parameter(
            Parameter(
                name="face_model_path",
                input_types=["str"],
                type="str",
                default_value="",
                tooltip="MediaPipe Face Landmarker .task file. Required for blendshape extraction.",
            )
        )
        self.add_parameter(
            Parameter(
                name="face_strength",
                input_types=["float"],
                type="float",
                default_value=1.0,
                tooltip="Expression intensity (0.5 subtle, 1 natural, 2 exaggerated).",
            )
        )
        self.add_parameter(Parameter(name="max_frames", input_types=["int"], type="int", default_value=-1, tooltip="Stop after N frames. -1 = all."))
        self.add_parameter(Parameter(name="pose_path_out", output_type="str", allowed_modes={ParameterMode.OUTPUT}, tooltip="Pose pack with face_blendshapes / expr_params filled in."))
        self.move_logs_last()

    def process(self) -> AsyncResult:
        yield lambda: self._process()

    def _process(self) -> None:
        self.log_params.clear_logs()
        pose_path = extract_input_text(self.get_parameter_value("pose_path"))
        if not pose_path:
            raise ValueError("pose_path is required.")
        media = artifact_to_local_path(self.required_media(self.get_parameter_value("media_input"), "media_input"))
        face_model = extract_input_text(self.get_parameter_value("face_model_path"))
        output = Path(pose_path).with_name(Path(pose_path).stem + "_face.npz")
        args = [
            "face",
            "--input",
            str(media),
            "--pose-path",
            pose_path,
            "--output",
            str(output),
            "--max-frames",
            str(int(self.get_parameter_value("max_frames") or -1)),
            "--face-strength",
            str(float(self.get_parameter_value("face_strength") or 1.0)),
            *self.handle_args(),
        ]
        if face_model:
            args.extend(["--face-model-path", face_model])
        payload = run_worker(self.python_path(), args, cwd=self.repo_path(), log=self.log)
        self.parameter_output_values["pose_path_out"] = payload.get("pose_path") or str(output)
