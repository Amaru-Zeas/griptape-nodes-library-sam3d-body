from __future__ import annotations

import os
from pathlib import Path

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, ControlNode
from griptape_nodes.exe_types.param_components.log_parameter import LogParameter
from griptape_nodes.traits.options import Options

try:
    from sam3d_body_nodes._common import (
        DEFAULT_HF_REPO,
        default_checkpoint_dir,
        default_repo_dir,
        default_venv_path,
        huggingface_token,
        run_command,
        run_worker,
        venv_python,
        write_json,
    )
except ImportError:
    from _common import (  # type: ignore
        DEFAULT_HF_REPO,
        default_checkpoint_dir,
        default_repo_dir,
        default_venv_path,
        huggingface_token,
        run_command,
        run_worker,
        venv_python,
        write_json,
    )


class SAM3DBodySetupNode(ControlNode):
    """Clone Meta SAM 3D Body, create a CUDA venv, and download gated checkpoints."""

    DEFAULT_REPO_URL = "https://github.com/facebookresearch/sam-3d-body.git"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.log_params = LogParameter(self)
        self.add_parameter(Parameter(name="repo_url", input_types=["str"], type="str", default_value=self.DEFAULT_REPO_URL, tooltip="Git URL for facebookresearch/sam-3d-body."))
        self.add_parameter(Parameter(name="repo_dir", input_types=["str"], type="str", default_value=str(default_repo_dir()), tooltip="Local clone directory."))
        self.add_parameter(Parameter(name="python_executable", input_types=["str"], type="str", default_value="python", tooltip="Base Python used to create the venv."))
        self.add_parameter(Parameter(name="venv_path", input_types=["str"], type="str", default_value=str(default_venv_path()), tooltip="Virtualenv path for the heavy SAM 3D Body deps."))
        self.add_parameter(Parameter(name="hf_repo", input_types=["str"], type="str", default_value=DEFAULT_HF_REPO, tooltip="Gated Hugging Face repo id."))
        self.add_parameter(Parameter(name="checkpoint_dir", input_types=["str"], type="str", default_value=str(default_checkpoint_dir()), tooltip="Where to download model.ckpt and assets/mhr_model.pt."))
        self.add_parameter(
            Parameter(
                name="torch_index_url",
                input_types=["str"],
                type="str",
                default_value="https://download.pytorch.org/whl/cu128",
                tooltip="PyTorch wheel index. Use cu121/cu124/cu128 to match your CUDA, or leave empty for CPU wheels.",
            )
        )
        self.add_parameter(
            Parameter(
                name="install_mode",
                input_types=["str"],
                type="str",
                default_value="full",
                traits={Options(choices=["full", "clone_only", "download_only"])},
                tooltip="full = clone + venv + pip + checkpoints. clone_only skips pip/download. download_only skips clone/pip.",
            )
        )
        self.add_parameter(Parameter(name="pull_latest", input_types=["bool"], type="bool", default_value=False, tooltip="If the repo exists, run git pull --ff-only."))
        self.add_parameter(Parameter(name="reinstall_torch", input_types=["bool"], type="bool", default_value=False, tooltip="Reinstall CUDA PyTorch even if it is already present. Leave off after the first successful torch install."))
        self.add_parameter(Parameter(name="status", output_type="str", allowed_modes={ParameterMode.OUTPUT}, tooltip="Setup status summary."))
        self.add_parameter(Parameter(name="repo_dir_out", output_type="str", allowed_modes={ParameterMode.OUTPUT}, tooltip="Resolved repository path."))
        self.add_parameter(Parameter(name="python_executable_out", output_type="str", allowed_modes={ParameterMode.OUTPUT}, tooltip="Python executable for the other SAM 3D Body nodes."))
        self.add_parameter(Parameter(name="model_handle", output_type="str", allowed_modes={ParameterMode.OUTPUT}, tooltip="JSON model handle for Load / Predict nodes."))
        self.log_params.add_output_parameters()

    def process(self) -> AsyncResult:
        yield lambda: self._process()

    def _process(self) -> None:
        self.log_params.clear_logs()
        repo_url = str(self.get_parameter_value("repo_url") or self.DEFAULT_REPO_URL).strip()
        repo_dir = Path(str(self.get_parameter_value("repo_dir") or default_repo_dir()).strip())
        python_executable = str(self.get_parameter_value("python_executable") or "python").strip() or "python"
        venv_path = Path(str(self.get_parameter_value("venv_path") or default_venv_path()).strip())
        hf_repo = str(self.get_parameter_value("hf_repo") or DEFAULT_HF_REPO).strip()
        checkpoint_dir = Path(str(self.get_parameter_value("checkpoint_dir") or default_checkpoint_dir()).strip())
        torch_index_url = str(self.get_parameter_value("torch_index_url") or "").strip()
        install_mode = str(self.get_parameter_value("install_mode") or "full").strip().lower()
        pull_latest = bool(self.get_parameter_value("pull_latest") or False)

        def log(message: str) -> None:
            self.log_params.append_to_logs(message if message.endswith("\n") else message + "\n")

        selected_python = python_executable
        if install_mode in {"full", "clone_only"}:
            if not repo_dir.exists():
                repo_dir.parent.mkdir(parents=True, exist_ok=True)
                log(f"Cloning SAM 3D Body into {repo_dir}")
                run_command(["git", "clone", repo_url, str(repo_dir)], log=log)
            elif pull_latest:
                log(f"Pulling latest changes in {repo_dir}")
                run_command(["git", "pull", "--ff-only"], cwd=repo_dir, log=log)
            else:
                log(f"Using existing repo: {repo_dir}")

        if install_mode == "full":
            venv_path.parent.mkdir(parents=True, exist_ok=True)
            venv_py = venv_python(venv_path)
            if venv_py.exists():
                log(f"Using existing venv: {venv_path}")
            else:
                log(f"Creating venv at {venv_path}")
                run_command([python_executable, "-m", "venv", str(venv_path)], cwd=repo_dir, log=log)
            selected_python = str(venv_py)
            run_command([selected_python, "-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools"], cwd=repo_dir, log=log)

            reinstall_torch = bool(self.get_parameter_value("reinstall_torch") or False)
            if (not reinstall_torch) and self._module_ok(selected_python, "torch"):
                log("PyTorch already installed; skipping the 2.5GB download. Set reinstall_torch=true to force it.")
            else:
                torch_cmd = [selected_python, "-m", "pip", "install", "torch", "torchvision"]
                if torch_index_url:
                    torch_cmd.extend(["--index-url", torch_index_url])
                log("Installing PyTorch.")
                run_command(torch_cmd, cwd=repo_dir, log=log)

            log("Installing build helpers for Windows packages that compile against numpy.")
            run_command([selected_python, "-m", "pip", "install", "numpy", "cython"], cwd=repo_dir, log=log)

            log("Installing SAM 3D Body Python dependencies.")
            self._pip_install(
                selected_python,
                repo_dir,
                log,
                [
                    "pytorch-lightning",
                    "pyrender",
                    "opencv-python",
                    "yacs",
                    "scikit-image",
                    "einops",
                    "timm",
                    "dill",
                    "pandas",
                    "rich",
                    "hydra-core",
                    "pyrootutils",
                    "networkx==3.2.1",
                    "roma",
                    "joblib",
                    "braceexpand",
                    "webdataset",
                    "jsonlines",
                    "optree",
                    "appdirs",
                    "chump",
                    "seaborn",
                    "imageio",
                    "imageio-ffmpeg",
                    "ultralytics",
                    "huggingface_hub",
                    "loguru",
                    "fvcore",
                    "trimesh",
                    "scipy",
                    "mediapipe",
                ],
            )
            self._install_coco_tools(selected_python, repo_dir, log)
            self._install_detectron2(selected_python, repo_dir, log)
            log("Installing MoGe (optional FoV).")
            self._pip_install(
                selected_python,
                repo_dir,
                log,
                ["git+https://github.com/microsoft/MoGe.git"],
                required=False,
            )
            # Meta's repo has no setup.py/pyproject.toml; it is used straight from
            # source. The worker adds repo_dir to sys.path at runtime.
            if not (repo_dir / "sam_3d_body").is_dir():
                raise ValueError(f"sam_3d_body package folder not found in {repo_dir}. Clone may be incomplete.")
            log("sam-3d-body source verified (used directly from the repo, no pip install needed).")

        if install_mode in {"full", "download_only"}:
            if not huggingface_token():
                log("WARNING: HF_TOKEN is not set. facebook/sam-3d-body-dinov3 is gated — request access on Hugging Face, then set HF_TOKEN in Griptape secrets.")
            selected_python = selected_python if Path(selected_python).exists() else str(venv_python(venv_path))
            log(f"Downloading checkpoints from {hf_repo} into {checkpoint_dir}")
            run_worker(
                selected_python,
                ["download", "--hf-repo", hf_repo, "--checkpoint-dir", str(checkpoint_dir)],
                cwd=repo_dir if repo_dir.exists() else Path.cwd(),
                log=log,
            )

        ckpt = checkpoint_dir / "model.ckpt"
        mhr = checkpoint_dir / "assets" / "mhr_model.pt"
        handle = {
            "checkpoint_path": str(ckpt) if ckpt.exists() else "",
            "mhr_path": str(mhr) if mhr.exists() else "",
            "hf_repo": hf_repo,
            "detector_name": "vitdet",
            "fov_name": "moge2",
            "repo_dir": str(repo_dir),
            "python_executable": selected_python,
        }
        handle_path = checkpoint_dir / "model_handle.json"
        write_json(handle_path, handle)

        status = "SAM 3D Body setup complete."
        self.parameter_output_values["status"] = status
        self.parameter_output_values["repo_dir_out"] = str(repo_dir)
        self.parameter_output_values["python_executable_out"] = selected_python
        self.parameter_output_values["model_handle"] = json_dumps(handle)
        log(status)
        if not os.environ.get("HF_TOKEN") and not huggingface_token():
            log("Request model access: https://huggingface.co/facebook/sam-3d-body-dinov3")

    def _module_ok(self, python_executable: str, module_name: str) -> bool:
        try:
            run_command([python_executable, "-c", f"import {module_name}"], log=None)
            return True
        except Exception:
            return False

    def _pip_install(
        self,
        python_executable: str,
        repo_dir: Path,
        log,
        packages: list[str],
        extra_flags: list[str] | None = None,
        required: bool = True,
        env: dict[str, str] | None = None,
    ) -> None:
        command = [python_executable, "-m", "pip", "install", *(extra_flags or []), *packages]
        try:
            run_command(command, cwd=repo_dir, env=env, log=log)
        except ValueError as exc:
            if required:
                raise
            log(f"Optional install skipped ({' '.join(packages)}): {exc}")

    def _install_detectron2(self, python_executable: str, repo_dir: Path, log) -> None:
        if self._module_ok(python_executable, "detectron2"):
            log("Detectron2 already installed; skipping.")
            return
        log("Installing Detectron2 without CUDA extensions (your system CUDA is 13.1, PyTorch is cu124).")
        env = os.environ.copy()
        env["FORCE_CUDA"] = "0"
        env.pop("CUDA_HOME", None)
        env.pop("CUDA_PATH", None)
        self._pip_install(
            python_executable,
            repo_dir,
            log,
            ["git+https://github.com/facebookresearch/detectron2.git@a1ce2f9"],
            extra_flags=["--no-build-isolation", "--no-deps"],
            required=False,
            env=env,
        )
        if self._module_ok(python_executable, "detectron2"):
            log("Detectron2 installed (CPU ops). ViTDet still runs the model on GPU via PyTorch.")
        else:
            log("Detectron2 skipped. Predict will fall back to a full-frame box, which is fine for single-person clips.")

    def _install_coco_tools(self, python_executable: str, repo_dir: Path, log) -> None:
        log("Installing xtcocotools with --no-build-isolation so the build can see numpy.")
        try:
            run_command(
                [python_executable, "-m", "pip", "install", "--no-build-isolation", "xtcocotools"],
                cwd=repo_dir,
                log=log,
            )
        except ValueError:
            log("PyPI xtcocotools failed; trying git+https://github.com/jin-s13/xtcocoapi.git")
            try:
                run_command(
                    [
                        python_executable,
                        "-m",
                        "pip",
                        "install",
                        "--no-build-isolation",
                        "git+https://github.com/jin-s13/xtcocoapi.git",
                    ],
                    cwd=repo_dir,
                    log=log,
                )
            except ValueError as exc:
                log(f"xtcocotools skipped (COCO eval only; inference can continue): {exc}")

        log("Installing pycocotools.")
        coco_candidates = ["pycocotools"]
        installed = False
        for package in coco_candidates:
            try:
                flags = ["--no-build-isolation"] if package == "pycocotools" else []
                run_command(
                    [python_executable, "-m", "pip", "install", *flags, package],
                    cwd=repo_dir,
                    log=log,
                )
                installed = True
                break
            except ValueError as exc:
                log(f"{package} failed: {exc}")
        if not installed:
            log("pycocotools skipped; person detection still works via Detectron2.")


def json_dumps(payload: dict) -> str:
    import json

    return json.dumps(payload)
