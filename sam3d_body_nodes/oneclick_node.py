from __future__ import annotations

from pathlib import Path

from griptape_nodes.exe_types.core_types import Parameter, ParameterGroup, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult
from griptape_nodes.traits.options import Options

try:
    from sam3d_body_nodes._common import artifact_to_local_path, publish_file, publish_video, run_worker
    from sam3d_body_nodes.base_node import SAM3DBodyWorkerNode
except ImportError:
    from _common import artifact_to_local_path, publish_file, publish_video, run_worker  # type: ignore
    from base_node import SAM3DBodyWorkerNode  # type: ignore


class SAM3DBodyOneClickNode(SAM3DBodyWorkerNode):
    """Run the full SAM 3D Body video pipeline in one node."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add_parameter(
            Parameter(
                name="media_input",
                input_types=["str", "VideoArtifact", "VideoUrlArtifact", "ImageArtifact", "ImageUrlArtifact"],
                type="str",
                tooltip="Source image or video of a person.",
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
                name="foot_lock",
                input_types=["bool"],
                type="bool",
                default_value=True,
                tooltip="Pin planted feet to the floor (kills foot sliding/flapping in the 3D result). Jumps keep their height.",
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
        self.add_parameter(Parameter(name="smooth", input_types=["bool"], type="bool", default_value=True, tooltip="Apply temporal smoothing before overlay/export."))
        self.add_parameter(
            Parameter(
                name="smooth_strength",
                input_types=["float"],
                type="float",
                default_value=1.0,
                tooltip="Temporal smoothing amount (0.5 subtle, 1 default, 2-3 strong). Higher = less jitter but softer fast motion.",
            )
        )
        self.add_parameter(
            Parameter(
                name="smooth_window",
                input_types=["int"],
                type="int",
                default_value=7,
                tooltip="Smoothing window in frames (odd number). Larger = smoother but laggier. 5-15 is sensible.",
            )
        )
        self.add_parameter(
            Parameter(
                name="face_expressions",
                input_types=["bool"],
                type="bool",
                default_value=True,
                tooltip="Track mouth/brow/eye motion with MediaPipe and bake it into the mesh.",
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
        self.add_parameter(Parameter(name="overlay_only", input_types=["bool"], type="bool", default_value=True, tooltip="Write mesh overlay instead of the 4-panel visualization."))
        self.add_parameter(Parameter(name="fov", input_types=["float"], type="float", default_value=0.0, tooltip="Optional vertical FoV. Wire Extract FoV from MoGe here."))
        self.add_parameter(Parameter(name="max_frames", input_types=["int"], type="int", default_value=-1, tooltip="Stop after N frames. -1 = all. Use a small number for the first test."))
        self.add_parameter(
            Parameter(
                name="overlay_video",
                output_type="VideoUrlArtifact",
                allowed_modes={ParameterMode.OUTPUT},
                tooltip="Rendered overlay video. Wire to a Display Video node to watch it.",
                ui_options={"hide_property": True},
            )
        )
        # File outputs live in a collapsed group so the node stays compact;
        # connections to them (pose_path -> viewers) work regardless.
        with ParameterGroup(name="Output Files", collapsed=True) as outputs_group:
            Parameter(name="pose_path", output_type="str", allowed_modes={ParameterMode.OUTPUT}, tooltip="Pose pack written by the pipeline.")
            Parameter(name="overlay_path", output_type="str", allowed_modes={ParameterMode.OUTPUT}, tooltip="Local overlay mp4.")
            Parameter(name="glb_path", output_type="str", allowed_modes={ParameterMode.OUTPUT}, tooltip="Exported GLB/OBJ.")
            Parameter(name="bvh_path", output_type="str", allowed_modes={ParameterMode.OUTPUT}, tooltip="Exported BVH.")
            Parameter(name="glb_url", output_type="str", allowed_modes={ParameterMode.OUTPUT}, tooltip="Published GLB URL.")
            Parameter(name="bvh_url", output_type="str", allowed_modes={ParameterMode.OUTPUT}, tooltip="Published BVH URL.")
        self.add_node_element(outputs_group)
        # Logs stay visible but always render last; the base class re-anchors it.
        self.move_logs_last()

    def process(self) -> AsyncResult:
        yield lambda: self._process()

    def _process(self) -> None:
        self.log_params.clear_logs()
        # Publish the source video as an output IMMEDIATELY: media_input is a
        # pass-through port (wired on to Render Body Video). Without this, the
        # engine sees a connected output with no published value and re-runs
        # this whole node (~1 min) every time a downstream node re-resolves.
        self.parameter_output_values["media_input"] = self.get_parameter_value("media_input")
        media = artifact_to_local_path(self.required_media(self.get_parameter_value("media_input"), "media_input"))
        out_dir = self.out_dir() / "oneclick"
        args = [
            "pipeline",
            "--input",
            str(media),
            "--output",
            str(out_dir),
            "--fov",
            str(float(self.get_parameter_value("fov") or 0.0)),
            "--max-frames",
            str(int(self.get_parameter_value("max_frames") or -1)),
            "--format",
            "both",
            *self.handle_args(),
        ]
        if bool(self.get_parameter_value("run_hand_refinement")):
            args.append("--run-hand-refinement")
        if bool(self.get_parameter_value("smooth")):
            args.append("--smooth")
            args.extend(["--strength", str(float(self.get_parameter_value("smooth_strength") or 1.0))])
            args.extend(["--window", str(int(self.get_parameter_value("smooth_window") or 7))])
        if bool(self.get_parameter_value("face_expressions")):
            args.append("--face-expressions")
            args.extend(["--face-strength", str(float(self.get_parameter_value("face_strength") or 1.0))])
        if bool(self.get_parameter_value("overlay_only")):
            args.append("--overlay-only")
        args.extend(["--precision", str(self.get_parameter_value("precision") or "bf16")])
        args.extend(["--batch-size", str(int(self.get_parameter_value("batch_size") or 16))])
        if bool(self.get_parameter_value("foot_lock")):
            args.append("--foot-lock")
        payload = run_worker(self.python_path(), args, cwd=self.repo_path(), log=self.log)
        pose_path = Path(payload.get("pose_path") or (out_dir / "pose_pack.npz"))
        overlay_path = Path(payload.get("overlay_path") or (out_dir / "overlay.mp4"))
        glb_path = Path(payload.get("glb_path") or (out_dir / "sam3d_body.glb"))
        bvh_path = Path(payload.get("bvh_path") or (out_dir / "sam3d_body.bvh"))
        self.parameter_output_values["pose_path"] = str(pose_path)
        self.parameter_output_values["overlay_path"] = str(overlay_path) if overlay_path.exists() else ""
        self.parameter_output_values["overlay_video"] = publish_video(overlay_path) if overlay_path.exists() else None
        self.parameter_output_values["glb_path"] = str(glb_path) if glb_path.exists() else ""
        self.parameter_output_values["bvh_path"] = str(bvh_path) if bvh_path.exists() else ""
        self.parameter_output_values["glb_url"] = publish_file(glb_path) if glb_path.exists() else ""
        self.parameter_output_values["bvh_url"] = publish_file(bvh_path) if bvh_path.exists() else ""
