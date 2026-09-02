from __future__ import annotations

import json
from pathlib import Path

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, ControlNode
from griptape_nodes.exe_types.param_components.log_parameter import LogParameter

try:
    from sam3d_body_nodes._common import (
        DEFAULT_HF_REPO,
        default_checkpoint_dir,
        default_repo_dir,
        default_venv_path,
        first_existing,
        venv_python,
        write_json,
    )
except ImportError:
    from _common import (  # type: ignore
        DEFAULT_HF_REPO,
        default_checkpoint_dir,
        default_repo_dir,
        default_venv_path,
        first_existing,
        venv_python,
        write_json,
    )


class LoadSAM3DBodyModelNode(ControlNode):
    """Resolve checkpoint + MHR paths into a reusable model handle."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.log_params = LogParameter(self)
        checkpoint_dir = default_checkpoint_dir()
        default_ckpt = first_existing([checkpoint_dir / "model.ckpt"], checkpoint_dir / "model.ckpt")
        default_mhr = first_existing(
            [checkpoint_dir / "assets" / "mhr_model.pt"],
            checkpoint_dir / "assets" / "mhr_model.pt",
        )
        self.add_parameter(Parameter(name="checkpoint_path", input_types=["str"], type="str", default_value=str(default_ckpt), tooltip="Path to model.ckpt from facebook/sam-3d-body-dinov3."))
        self.add_parameter(Parameter(name="mhr_path", input_types=["str"], type="str", default_value=str(default_mhr), tooltip="Path to assets/mhr_model.pt."))
        self.add_parameter(Parameter(name="hf_repo", input_types=["str"], type="str", default_value=DEFAULT_HF_REPO, tooltip="Hugging Face repo id used if files still need downloading."))
        self.add_parameter(Parameter(name="detector_name", input_types=["str"], type="str", default_value="vitdet", tooltip="Official demo detector. vitdet is the default; sam3 matches the playground."))
        self.add_parameter(Parameter(name="fov_name", input_types=["str"], type="str", default_value="moge2", tooltip="FoV estimator name. moge2 is the official default. Leave empty to skip."))
        self.add_parameter(Parameter(name="segmentor_name", input_types=["str"], type="str", default_value="", tooltip="Optional human segmentor (sam2). Leave empty unless you want mask-conditioned prediction."))
        self.add_parameter(Parameter(name="repo_dir", input_types=["str"], type="str", default_value=str(default_repo_dir()), tooltip="Local sam-3d-body checkout."))
        self.add_parameter(Parameter(name="python_executable", input_types=["str"], type="str", default_value=str(venv_python(default_venv_path())), tooltip="Python from the SAM 3D Body venv."))
        self.add_parameter(Parameter(name="model_handle", output_type="str", allowed_modes={ParameterMode.OUTPUT}, tooltip="JSON handle consumed by Predict / Detect / FoV nodes."))
        self.add_parameter(Parameter(name="status", output_type="str", allowed_modes={ParameterMode.OUTPUT}, tooltip="Resolved paths."))
        self.log_params.add_output_parameters()

    def process(self) -> AsyncResult:
        yield lambda: self._process()

    def _process(self) -> None:
        self.log_params.clear_logs()
        checkpoint_path = Path(str(self.get_parameter_value("checkpoint_path") or "").strip())
        mhr_path = Path(str(self.get_parameter_value("mhr_path") or "").strip())
        handle = {
            "checkpoint_path": str(checkpoint_path),
            "mhr_path": str(mhr_path),
            "hf_repo": str(self.get_parameter_value("hf_repo") or DEFAULT_HF_REPO).strip(),
            "detector_name": str(self.get_parameter_value("detector_name") or "").strip(),
            "fov_name": str(self.get_parameter_value("fov_name") or "").strip(),
            "segmentor_name": str(self.get_parameter_value("segmentor_name") or "").strip(),
            "repo_dir": str(self.get_parameter_value("repo_dir") or default_repo_dir()).strip(),
            "python_executable": str(self.get_parameter_value("python_executable") or "").strip(),
        }
        missing = [key for key in ("checkpoint_path", "mhr_path") if not Path(handle[key]).exists()]
        if missing:
            raise ValueError(
                "SAM 3D Body checkpoint files are missing: "
                + ", ".join(missing)
                + ". Run SAM 3D Body Setup first, and request Hugging Face access to facebook/sam-3d-body-dinov3."
            )
        handle_path = checkpoint_path.parent / "model_handle.json"
        write_json(handle_path, handle)
        self.parameter_output_values["model_handle"] = json.dumps(handle)
        status = f"Loaded model handle from {checkpoint_path}"
        self.parameter_output_values["status"] = status
        self.log_params.append_to_logs(status + "\n")
