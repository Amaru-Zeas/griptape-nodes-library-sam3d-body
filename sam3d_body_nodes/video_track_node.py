from __future__ import annotations

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult

try:
    from sam3d_body_nodes._common import artifact_to_local_path, extract_input_text, run_worker
    from sam3d_body_nodes.base_node import SAM3DBodyWorkerNode
except ImportError:
    from _common import artifact_to_local_path, extract_input_text, run_worker  # type: ignore
    from base_node import SAM3DBodyWorkerNode  # type: ignore


class SAM3DBodyVideoTrackNode(SAM3DBodyWorkerNode):
    """Track people across frames. Practically required for multi-person clips."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add_parameter(
            Parameter(
                name="media_input",
                input_types=["str", "VideoArtifact", "VideoUrlArtifact", "ImageArtifact", "ImageUrlArtifact"],
                type="str",
                tooltip="Source video. For multiple people in the same clip, tracking is practically required.",
            )
        )
        self.add_parameter(
            Parameter(
                name="boxes_path",
                input_types=["str"],
                type="str",
                default_value="",
                tooltip="Optional boxes JSON from Detect Bounding Boxes. If empty, this node detects and tracks in one pass.",
            )
        )
        self.add_parameter(Parameter(name="bbox_thresh", input_types=["float"], type="float", default_value=0.8, tooltip="Detector threshold used when boxes_path is empty."))
        self.add_parameter(Parameter(name="max_frames", input_types=["int"], type="int", default_value=-1, tooltip="Stop after N frames. -1 = all."))
        self.add_parameter(Parameter(name="track_data_path", output_type="str", allowed_modes={ParameterMode.OUTPUT}, tooltip="Tracked person boxes with stable track_id values."))
        self.add_parameter(Parameter(name="max_people", output_type="int", allowed_modes={ParameterMode.OUTPUT}, tooltip="Peak tracked person count."))
        self.move_logs_last()

    def process(self) -> AsyncResult:
        yield lambda: self._process()

    def _process(self) -> None:
        self.log_params.clear_logs()
        existing = extract_input_text(self.get_parameter_value("boxes_path"))
        if existing:
            self.log("Reusing existing boxes JSON; IoU ids are assigned during detect.\n")
            self.parameter_output_values["track_data_path"] = existing
            self.parameter_output_values["max_people"] = 0
            return
        media = artifact_to_local_path(self.required_media(self.get_parameter_value("media_input"), "media_input"))
        output = self.out_dir() / "person_tracks.json"
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
        self.parameter_output_values["track_data_path"] = payload.get("boxes_path") or str(output)
        self.parameter_output_values["max_people"] = int(payload.get("max_people") or 0)
