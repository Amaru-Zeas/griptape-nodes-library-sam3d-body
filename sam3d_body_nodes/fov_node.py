from __future__ import annotations

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult

try:
    from sam3d_body_nodes._common import artifact_to_local_path, run_worker
    from sam3d_body_nodes.base_node import SAM3DBodyWorkerNode
except ImportError:
    from _common import artifact_to_local_path, run_worker  # type: ignore
    from base_node import SAM3DBodyWorkerNode  # type: ignore


class ExtractFoVFromMoGeNode(SAM3DBodyWorkerNode):
    """Estimate camera vertical FoV with MoGe for better source-video alignment."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add_parameter(
            Parameter(
                name="media_input",
                input_types=["str", "VideoArtifact", "VideoUrlArtifact", "ImageArtifact", "ImageUrlArtifact"],
                type="str",
                tooltip="Image or video. First frame is used. MoGe is optional but improves alignment.",
            )
        )
        self.add_parameter(Parameter(name="fallback_fov", input_types=["float"], type="float", default_value=53.0, tooltip="Used when MoGe is unavailable. ~53° matches 16:9."))
        self.add_parameter(Parameter(name="fov", output_type="float", allowed_modes={ParameterMode.OUTPUT}, tooltip="Vertical field of view in degrees."))
        self.move_logs_last()

    def process(self) -> AsyncResult:
        yield lambda: self._process()

    def _process(self) -> None:
        self.log_params.clear_logs()
        media = artifact_to_local_path(self.required_media(self.get_parameter_value("media_input"), "media_input"), default_suffix=".png")
        handle_args = self.handle_args()
        if "--fov-name" not in handle_args:
            handle_args.extend(["--fov-name", "moge2"])
        payload = run_worker(
            self.python_path(),
            [
                "fov",
                "--input",
                str(media),
                "--fallback-fov",
                str(float(self.get_parameter_value("fallback_fov") or 53.0)),
                *handle_args,
            ],
            cwd=self.repo_path(),
            log=self.log,
        )
        self.parameter_output_values["fov"] = float(payload.get("fov") or 53.0)
