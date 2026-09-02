from __future__ import annotations

from pathlib import Path

from griptape_nodes.exe_types.core_types import Parameter, ParameterGroup, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult
from griptape_nodes.traits.options import Options

try:
    from sam3d_body_nodes._common import artifact_to_local_path, extract_input_text, publish_video, run_worker
    from sam3d_body_nodes.base_node import SAM3DBodyWorkerNode
except ImportError:
    from _common import artifact_to_local_path, extract_input_text, publish_video, run_worker  # type: ignore
    from base_node import SAM3DBodyWorkerNode  # type: ignore


class RenderSAM3DPoseVideoNode(SAM3DBodyWorkerNode):
    """Render the recovered motion as a video, ComfyUI style: one node, one style dropdown.

    mesh     = 3D body composited over the source video (the classic overlay).
    openpose = ControlNet pose-map colors.
    mhr      = Meta's skeleton colors.
    white    = plain white sticks.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add_parameter(Parameter(name="pose_path", input_types=["str"], type="str", tooltip="Pose pack from Predict, Smooth, or One-Click."))
        self.add_parameter(
            Parameter(
                name="media_input",
                input_types=["str", "VideoArtifact", "VideoUrlArtifact", "ImageArtifact", "ImageUrlArtifact"],
                type="str",
                default_value="",
                tooltip="Source video. Needed for mesh style and for background = source.",
            )
        )
        self.add_parameter(
            Parameter(
                name="render_style",
                input_types=["str"],
                type="str",
                default_value="mesh",
                traits={Options(choices=["mesh", "openpose", "mhr", "white"])},
                tooltip="mesh = 3D body over the source video. openpose = classic ControlNet pose colors. mhr = Meta's skeleton colors. white = plain white sticks.",
            )
        )
        self.add_parameter(
            Parameter(
                name="background",
                input_types=["str"],
                type="str",
                default_value="black",
                traits={Options(choices=["black", "source"])},
                tooltip="Skeleton styles only: draw on black (ControlNet-ready) or over the source video. Mesh always composites over the source.",
            )
        )
        # Fine-tuning knobs live in a collapsed group to keep the node small.
        with ParameterGroup(name="Advanced", collapsed=True) as advanced_group:
            Parameter(
                name="overlay_path",
                input_types=["str"],
                type="str",
                default_value="",
                tooltip="Optional: an already-rendered overlay mp4 (One-Click makes one). Reused for mesh style so nothing re-renders.",
            )
            Parameter(name="stick_width_px", input_types=["int"], type="int", default_value=4, tooltip="Limb stick thickness in pixels.")
            Parameter(name="marker_radius_px", input_types=["int"], type="int", default_value=4, tooltip="Joint marker radius in pixels.")
            Parameter(name="limb_alpha", input_types=["float"], type="float", default_value=0.6, tooltip="Limb opacity (openpose style).")
            Parameter(name="draw_hands", input_types=["bool"], type="bool", default_value=True, tooltip="Draw finger sticks (skeleton styles).")
            Parameter(name="draw_face", input_types=["bool"], type="bool", default_value=True, tooltip="Draw the eye/ear/nose links (skeleton styles).")
            Parameter(name="max_frames", input_types=["int"], type="int", default_value=-1, tooltip="Stop after N frames. -1 = all.")
            Parameter(name="video_path", output_type="str", allowed_modes={ParameterMode.OUTPUT}, tooltip="Local mp4 path.")
        self.add_node_element(advanced_group)
        self.add_parameter(
            Parameter(
                name="video",
                output_type="VideoUrlArtifact",
                allowed_modes={ParameterMode.OUTPUT},
                tooltip="Rendered video. Wire to a Display Video node to watch it.",
                ui_options={"hide_property": True},
            )
        )
        self.move_logs_last()
        # Keep this node compact, like ComfyUI's Render 3D Body Pose: the
        # worker plumbing params stay functional but hidden.
        try:
            self.hide_parameter_by_name(["repo_dir", "python_executable", "model_handle", "output_dir"])
        except Exception:
            pass

    def process(self) -> AsyncResult:
        yield lambda: self._process()

    def _process(self) -> None:
        self.log_params.clear_logs()
        pose_path = extract_input_text(self.get_parameter_value("pose_path"))
        if not pose_path:
            raise ValueError("pose_path is required. Wire it from One-Click, Predict, or Smooth.")
        style = str(self.get_parameter_value("render_style") or "mesh")
        if style == "mesh":
            local = self._render_mesh(pose_path)
        else:
            local = self._render_skeleton(pose_path, style)
        self.parameter_output_values["video_path"] = str(local)
        self.parameter_output_values["video"] = publish_video(local) if local.exists() else None

    def _render_mesh(self, pose_path: str) -> Path:
        # One-Click already renders this exact overlay; reuse it when wired so
        # flipping the style dropdown back to mesh is instant.
        existing = extract_input_text(self.get_parameter_value("overlay_path"))
        if existing:
            existing_path = Path(existing)
            if existing_path.exists():
                self.log("Reusing overlay from One-Click (no re-render).")
                return existing_path
        media = artifact_to_local_path(self.required_media(self.get_parameter_value("media_input"), "media_input"))
        out_path = self.out_dir() / "overlay.mp4"
        args = [
            "render",
            "--input",
            str(media),
            "--pose-path",
            pose_path,
            "--output",
            str(out_path),
            "--overlay-only",
            "--max-frames",
            str(int(self.get_parameter_value("max_frames") or -1)),
            *self.handle_args(),
        ]
        payload = run_worker(self.python_path(), args, cwd=self.repo_path(), log=self.log)
        return Path(payload.get("overlay_path") or out_path)

    def _render_skeleton(self, pose_path: str, style: str) -> Path:
        background = str(self.get_parameter_value("background") or "black")
        out_path = self.out_dir() / "pose_video.mp4"
        args = [
            "render_pose",
            "--pose-path",
            pose_path,
            "--output",
            str(out_path),
            "--pose-style",
            style,
            "--pose-background",
            background,
            "--stick-width",
            str(int(self.get_parameter_value("stick_width_px") or 4)),
            "--marker-radius",
            str(int(self.get_parameter_value("marker_radius_px") or 4)),
            "--limb-alpha",
            str(float(self.get_parameter_value("limb_alpha") or 0.6)),
            "--max-frames",
            str(int(self.get_parameter_value("max_frames") or -1)),
        ]
        if background == "source":
            media = artifact_to_local_path(self.required_media(self.get_parameter_value("media_input"), "media_input"))
            args += ["--input", str(media)]
        if not bool(self.get_parameter_value("draw_hands")):
            args.append("--no-hands")
        if not bool(self.get_parameter_value("draw_face")):
            args.append("--no-face")
        payload = run_worker(self.python_path(), args, cwd=self.repo_path(), log=self.log)
        return Path(payload.get("pose_video_path") or out_path)
