from __future__ import annotations

from pathlib import Path
from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import ControlNode
from griptape_nodes.exe_types.param_components.log_parameter import LogParameter

try:
    from sam3d_body_nodes._common import (
        default_output_dir,
        default_repo_dir,
        default_venv_path,
        extract_input_text,
        model_handle_cli_args,
        parse_model_handle,
        venv_python,
    )
except ImportError:
    from _common import (  # type: ignore
        default_output_dir,
        default_repo_dir,
        default_venv_path,
        extract_input_text,
        model_handle_cli_args,
        parse_model_handle,
        venv_python,
    )


class SAM3DBodyWorkerNode(ControlNode):
    """Shared runtime wiring for SAM 3D Body worker nodes."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.log_params = LogParameter(self)
        self.add_parameter(
            Parameter(
                name="repo_dir",
                input_types=["str"],
                type="str",
                default_value=str(default_repo_dir()),
                tooltip="Local facebookresearch/sam-3d-body checkout.",
            )
        )
        self.add_parameter(
            Parameter(
                name="python_executable",
                input_types=["str"],
                type="str",
                default_value=str(venv_python(default_venv_path())),
                tooltip="Python from the SAM 3D Body virtualenv.",
            )
        )
        self.add_parameter(
            Parameter(
                name="model_handle",
                input_types=["str", "dict"],
                type="str",
                default_value="",
                tooltip="JSON handle from Load SAM 3D Body Model. Optional if checkpoint paths are already set on that node.",
            )
        )
        self.add_parameter(
            Parameter(
                name="output_dir",
                input_types=["str"],
                type="str",
                default_value=str(default_output_dir()),
                tooltip="Folder for pose packs, overlays, and exports.",
            )
        )
        self.log_params.add_output_parameters()

    def move_logs_last(self) -> None:
        """Re-anchor the logs parameter to the bottom of the node.

        The base class adds "logs" before subclass parameters; call this at the
        end of a subclass __init__ so logs render last.
        """
        try:
            element = self.root_ui_element.find_element_by_name("logs")
            if element is not None:
                self.root_ui_element.remove_child(element)
                self.root_ui_element.add_child(element)
        except Exception:  # noqa: BLE001 - purely cosmetic, never fail node init
            pass

    def repo_path(self) -> Path:
        return Path(str(self.get_parameter_value("repo_dir") or default_repo_dir()).strip())

    def python_path(self) -> str:
        return str(self.get_parameter_value("python_executable") or venv_python(default_venv_path())).strip()

    def handle(self) -> dict[str, Any]:
        return parse_model_handle(self.get_parameter_value("model_handle"))

    def handle_args(self) -> list[str]:
        return model_handle_cli_args(self.handle())

    def out_dir(self) -> Path:
        path = Path(str(self.get_parameter_value("output_dir") or default_output_dir()).strip())
        path.mkdir(parents=True, exist_ok=True)
        return path

    def log(self, message: str) -> None:
        self.log_params.append_to_logs(message if message.endswith("\n") else message + "\n")

    def text(self, name: str, fallback: str = "") -> str:
        return str(self.get_parameter_value(name) or fallback).strip()

    @staticmethod
    def required_media(value: Any, label: str) -> Any:
        if not extract_input_text(value):
            raise ValueError(f"{label} is required.")
        return value
