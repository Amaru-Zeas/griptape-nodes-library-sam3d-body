from __future__ import annotations

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult
from griptape_nodes.traits.options import Options

try:
    from sam3d_body_nodes._common import artifact_to_local_path, extract_input_text, run_worker
    from sam3d_body_nodes.base_node import SAM3DBodyWorkerNode
except ImportError:
    from _common import artifact_to_local_path, extract_input_text, run_worker  # type: ignore
    from base_node import SAM3DBodyWorkerNode  # type: ignore


class RunSAM3DBodyPredictionNode(SAM3DBodyWorkerNode):
    """Run SAM 3D Body mesh recovery on an image or video."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add_parameter(
            Parameter(
                name="media_input",
                input_types=["str", "VideoArtifact", "VideoUrlArtifact", "ImageArtifact", "ImageUrlArtifact"],
                type="str",
                tooltip="Source image or video.",
            )
        )
        self.add_parameter(
            Parameter(
                name="track_data_path",
                input_types=["str"],
                type="str",
                default_value="",
                tooltip="Optional tracked boxes JSON. Required for reliable multi-person detection.",
            )
        )
        self.add_parameter(
            Parameter(
                name="boxes_path",
                input_types=["str"],
                type="str",
                default_value="",
                tooltip="Optional per-frame boxes JSON if you are not using video tracks.",
            )
        )
        self.add_parameter(
            Parameter(
                name="precision",
                input_types=["str"],
                type="str",
                default_value="bf16",
                tooltip="Inference precision. bf16 is ~2x faster with near-identical results; fp32 is the slow reference; fp16 if bf16 misbehaves.",
                traits={Options(choices=["bf16", "fp32", "fp16"])},
            )
        )
        self.add_parameter(
            Parameter(
                name="batch_size",
                input_types=["int"],
                type="int",
                default_value=16,
                tooltip="Frames per GPU batch (hand refinement off only). Higher = faster until VRAM runs out; 16-32 is a good range.",
            )
        )
        self.add_parameter(
            Parameter(
                name="run_hand_refinement",
                input_types=["bool"],
                type="bool",
                default_value=False,
                tooltip="Extra hand detail pass - but ~4x slower overall (it is 78% of inference time). Off still gives full body with decent hands.",
            )
        )
        self.add_parameter(Parameter(name="use_mask", input_types=["bool"], type="bool", default_value=False, tooltip="Mask-conditioned prediction from the official segmentor."))
        self.add_parameter(Parameter(name="fov", input_types=["float"], type="float", default_value=0.0, tooltip="Vertical FoV in degrees. 0 = model default."))
        self.add_parameter(Parameter(name="bbox_thresh", input_types=["float"], type="float", default_value=0.8, tooltip="Detector threshold when no boxes/tracks are wired."))
        self.add_parameter(Parameter(name="max_frames", input_types=["int"], type="int", default_value=-1, tooltip="Stop after N frames. -1 = all."))
        self.add_parameter(Parameter(name="pose_path", output_type="str", allowed_modes={ParameterMode.OUTPUT}, tooltip="Pose pack (.npz) for Smooth / Face / Render / Export."))
        self.add_parameter(Parameter(name="n_frames", output_type="int", allowed_modes={ParameterMode.OUTPUT}, tooltip="Frames written into the pose pack."))
        self.add_parameter(Parameter(name="max_people", output_type="int", allowed_modes={ParameterMode.OUTPUT}, tooltip="Peak person count."))
        self.move_logs_last()

    def process(self) -> AsyncResult:
        yield lambda: self._process()

    def _process(self) -> None:
        self.log_params.clear_logs()
        media = artifact_to_local_path(self.required_media(self.get_parameter_value("media_input"), "media_input"))
        boxes = extract_input_text(self.get_parameter_value("track_data_path")) or extract_input_text(self.get_parameter_value("boxes_path"))
        pose_path = self.out_dir() / "pose_pack.npz"
        args = [
            "predict",
            "--input",
            str(media),
            "--output",
            str(pose_path),
            "--bbox-thresh",
            str(float(self.get_parameter_value("bbox_thresh") or 0.8)),
            "--fov",
            str(float(self.get_parameter_value("fov") or 0.0)),
            "--max-frames",
            str(int(self.get_parameter_value("max_frames") or -1)),
            *self.handle_args(),
        ]
        if boxes:
            args.extend(["--boxes-path", boxes])
        if bool(self.get_parameter_value("run_hand_refinement")):
            args.append("--run-hand-refinement")
        if bool(self.get_parameter_value("use_mask")):
            args.append("--use-mask")
        args.extend(["--precision", str(self.get_parameter_value("precision") or "bf16")])
        args.extend(["--batch-size", str(int(self.get_parameter_value("batch_size") or 16))])
        payload = run_worker(self.python_path(), args, cwd=self.repo_path(), log=self.log)
        self.parameter_output_values["pose_path"] = payload.get("pose_path") or str(pose_path)
        self.parameter_output_values["n_frames"] = int(payload.get("n_frames") or 0)
        self.parameter_output_values["max_people"] = int(payload.get("max_people") or 0)
