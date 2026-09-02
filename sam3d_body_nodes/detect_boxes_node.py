from __future__ import annotations

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult

try:
    from sam3d_body_nodes._common import artifact_to_local_path, run_worker
    from sam3d_body_nodes.base_node import SAM3DBodyWorkerNode
except ImportError:
    from _common import artifact_to_local_path, run_worker  # type: ignore
    from base_node import SAM3DBodyWorkerNode  # type: ignore


class DetectPersonBoxesNode(SAM3DBodyWorkerNode):
    """Detect per-frame person bounding boxes."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add_parameter(
            Parameter(
                name="media_input",
                input_types=["str", "VideoArtifact", "VideoUrlArtifact", "ImageArtifact", "ImageUrlArtifact"],
                type="str",
                tooltip="Source image or video. Boxes are technically optional, but they improve detection a lot.",
            )
        )
        self.add_parameter(Parameter(name="bbox_thresh", input_types=["float"], type="float", default_value=0.8, tooltip="Person box confidence threshold."))
        self.add_parameter(Parameter(name="max_frames", input_types=["int"], type="int", default_value=-1, tooltip="Stop after N frames. -1 = all."))
        self.add_parameter(Parameter(name="boxes_path", output_type="str", allowed_modes={ParameterMode.OUTPUT}, tooltip="JSON file of per-frame boxes / track ids."))
        self.add_parameter(Parameter(name="n_frames", output_type="int", allowed_modes={ParameterMode.OUTPUT}, tooltip="Number of frames written."))
        self.add_parameter(Parameter(name="max_people", output_type="int", allowed_modes={ParameterMode.OUTPUT}, tooltip="Peak person count in any frame."))
        self.move_logs_last()

    def process(self) -> AsyncResult:
        yield lambda: self._process()

    def _process(self) -> None:
        self.log_params.clear_logs()
        media = artifact_to_local_path(self.required_media(self.get_parameter_value("media_input"), "media_input"))
        output = self.out_dir() / "person_boxes.json"
        payload = run_worker(
            self.python_path(),
            [
                "detect",
                "--input",
                str(media),
                "--output",
                str(output),
                "--bbox-thresh",
                str(float(self.get_parameter_value("bbox_thresh") or 0.8)),
                "--max-frames",
                str(int(self.get_parameter_value("max_frames") or -1)),
                *self.handle_args(),
            ],
            cwd=self.repo_path(),
            log=self.log,
        )
        self.parameter_output_values["boxes_path"] = payload.get("boxes_path") or str(output)
        self.parameter_output_values["n_frames"] = int(payload.get("n_frames") or 0)
        self.parameter_output_values["max_people"] = int(payload.get("max_people") or 0)
