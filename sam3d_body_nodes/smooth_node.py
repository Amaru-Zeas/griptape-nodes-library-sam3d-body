from __future__ import annotations

from pathlib import Path

import numpy as np
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, ControlNode
from griptape_nodes.exe_types.param_components.log_parameter import LogParameter
from griptape_nodes.traits.options import Options

try:
    from sam3d_body_nodes._common import extract_input_text, load_pose_pack, save_pose_pack
except ImportError:
    from _common import extract_input_text, load_pose_pack, save_pose_pack  # type: ignore


class SmoothSAM3DBodyPoseNode(ControlNode):
    """Reduce frame-to-frame jitter via temporal smoothing."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.log_params = LogParameter(self)
        self.add_parameter(Parameter(name="pose_path", input_types=["str"], type="str", tooltip="Pose pack from Run SAM 3D Body Prediction."))
        self.add_parameter(Parameter(name="strength", input_types=["float"], type="float", default_value=1.0, tooltip="0 = raw, 1 = fully smoothed."))
        self.add_parameter(
            Parameter(
                name="method",
                input_types=["str"],
                type="str",
                default_value="savgol",
                traits={Options(choices=["gaussian", "savgol"])},
                tooltip="gaussian = weighted average. savgol = polynomial fit that keeps sharp peaks.",
            )
        )
        self.add_parameter(Parameter(name="window", input_types=["int"], type="int", default_value=7, tooltip="Temporal window in frames. Odd values work best."))
        self.add_parameter(Parameter(name="smoothed_pose_path", output_type="str", allowed_modes={ParameterMode.OUTPUT}, tooltip="Smoothed pose pack path."))
        self.log_params.add_output_parameters()

    def process(self) -> AsyncResult:
        yield lambda: self._process()

    def _process(self) -> None:
        self.log_params.clear_logs()
        pose_path = Path(extract_input_text(self.get_parameter_value("pose_path")))
        if not pose_path.exists():
            raise ValueError(f"pose_path not found: {pose_path}")
        strength = float(self.get_parameter_value("strength") or 0.0)
        window = int(self.get_parameter_value("window") or 7)
        method = str(self.get_parameter_value("method") or "savgol").strip().lower()
        if window % 2 == 0:
            window += 1
        pack = load_pose_pack(pose_path)
        n_frames = int(pack["meta"]["n_frames"])
        if strength <= 0.0 or window <= 1 or n_frames < 2:
            self.parameter_output_values["smoothed_pose_path"] = str(pose_path)
            self.log_params.append_to_logs("Smoothing skipped.\n")
            return
        present = np.asarray(pack["present"])
        savgol_filter = None
        if method == "savgol":
            try:
                from scipy.signal import savgol_filter as _savgol

                savgol_filter = _savgol
            except ImportError:
                self.log_params.append_to_logs("scipy not installed in the GTN env; using gaussian smoothing.\n")
        radius = window // 2
        sigma = max(0.5, radius / 2.0)
        kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
        kernel = kernel / kernel.sum()
        for key in ("pred_vertices", "pred_joint_coords", "pred_cam_t", "pred_keypoints_2d", "pred_keypoints_3d"):
            if key not in pack:
                continue
            data = np.asarray(pack[key], dtype=np.float32)
            smoothed = data.copy()
            for person_idx in range(data.shape[1]):
                mask = present[:, person_idx]
                if int(mask.sum()) < 2:
                    continue
                series = data[:, person_idx]
                if savgol_filter is not None and n_frames >= window:
                    poly = 2 if window > 2 else 1
                    win = min(window, n_frames if n_frames % 2 else n_frames - 1)
                    win = max(3, win)
                    filtered = savgol_filter(series, window_length=win, polyorder=min(poly, win - 1), axis=0)
                else:
                    padded = np.pad(series, ((radius, radius),) + ((0, 0),) * (series.ndim - 1), mode="edge")
                    filtered = np.zeros_like(series)
                    for frame_idx in range(n_frames):
                        chunk = padded[frame_idx : frame_idx + window]
                        filtered[frame_idx] = np.tensordot(kernel, chunk, axes=(0, 0))
                blend = series * (1.0 - strength) + filtered * strength
                mask_shape = (-1,) + (1,) * (series.ndim - 1)
                smoothed[:, person_idx] = np.where(mask.reshape(mask_shape), blend, series)
            pack[key] = smoothed
        output = pose_path.with_name(pose_path.stem + "_smooth.npz")
        save_pose_pack(output, pack)
        self.parameter_output_values["smoothed_pose_path"] = str(output)
        self.log_params.append_to_logs(f"Wrote smoothed pose pack: {output}\n")
