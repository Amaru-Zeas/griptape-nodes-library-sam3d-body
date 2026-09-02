from __future__ import annotations

import json
import os
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any, Callable

import numpy as np
from griptape.artifacts.video_url_artifact import VideoUrlArtifact
from griptape_nodes.files.file import File
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

WORKER_SCRIPT = Path(__file__).resolve().parent / "worker" / "run_pipeline.py"
DEFAULT_HF_REPO = "facebook/sam-3d-body-dinov3"


def default_workspace() -> Path:
    sketchfab = Path(r"A:\GriptapeSketchFab\sam-3d-body")
    if sketchfab.exists():
        return sketchfab
    return Path.home() / "GriptapeNodes" / "sam-3d-body"


def default_repo_dir() -> Path:
    return default_workspace()


def default_venv_path() -> Path:
    return default_workspace() / ".venv"


def default_checkpoint_dir() -> Path:
    return default_workspace() / "checkpoints" / "sam-3d-body-dinov3"


def _project_output_root() -> Path | None:
    """Ask the engine where node outputs belong for the current project.

    Honors the project template ("situations" yaml) when one is loaded by
    resolving the standard save_node_output situation; falls back to the
    {outputs} directory macro. Returns None when the project system is
    unavailable so the caller can use the legacy location.
    """
    try:
        from griptape_nodes.common.macro_parser import ParsedMacro
        from griptape_nodes.retained_mode.events.project_events import (
            GetPathForMacroRequest,
            GetPathForMacroResultSuccess,
            GetSituationRequest,
            GetSituationResultSuccess,
        )
    except Exception:  # noqa: BLE001 - engine predates the project system
        return None

    def resolve(macro: str, variables: dict[str, str | int]) -> Path | None:
        try:
            result = GriptapeNodes.handle_request(
                GetPathForMacroRequest(parsed_macro=ParsedMacro(macro), variables=variables)
            )
        except Exception:  # noqa: BLE001
            return None
        if isinstance(result, GetPathForMacroResultSuccess):
            return Path(result.absolute_path)
        return None

    try:
        situation = GriptapeNodes.handle_request(GetSituationRequest(situation_name="save_node_output"))
    except Exception:  # noqa: BLE001
        situation = None
    if isinstance(situation, GetSituationResultSuccess):
        resolved = resolve(
            situation.situation.macro,
            {
                "file_name_base": "output",
                "file_extension": "txt",
                "node_name": "sam3d_body",
                "sub_dirs": "sam3d_body",
            },
        )
        if resolved is not None:
            return resolved.parent
    resolved = resolve("{outputs}", {})
    if resolved is not None:
        return resolved / "sam3d_body"
    return None


def default_output_dir() -> Path:
    root = _project_output_root()
    if root is not None:
        return root
    return default_workspace() / "gtn_output"


def venv_python(venv_path: Path) -> Path:
    if os.name == "nt":
        return venv_path / "Scripts" / "python.exe"
    return venv_path / "bin" / "python"


def first_existing(candidates: list[Path], fallback: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return fallback


def extract_input_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("value", "path", "url", "uri"):
            if value.get(key):
                return str(value[key]).strip()
        return ""
    for attr in ("value", "path", "url", "uri"):
        attr_value = getattr(value, attr, None)
        if attr_value:
            return str(attr_value).strip()
    return str(value).strip()


def artifact_to_local_path(value: Any, default_suffix: str = ".mp4") -> Path:
    text = extract_input_text(value)
    if not text:
        raise ValueError("Input path/artifact is empty.")

    local_candidate = Path(os.path.expandvars(os.path.expanduser(text)))
    if local_candidate.exists():
        return local_candidate

    suffix = Path(text).suffix or default_suffix
    fd, temp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    temp_file = Path(temp_path)
    temp_file.write_bytes(File(text).read_bytes())
    return temp_file


def publish_bytes(data: bytes, filename: str) -> str:
    return GriptapeNodes.StaticFilesManager().save_static_file(data, filename)


def publish_video(video_path: Path) -> VideoUrlArtifact:
    filename = f"{uuid.uuid4()}{video_path.suffix or '.mp4'}"
    url = publish_bytes(video_path.read_bytes(), filename)
    return VideoUrlArtifact(url)


def publish_file(path: Path) -> str:
    filename = f"{uuid.uuid4()}{path.suffix}"
    return publish_bytes(path.read_bytes(), filename)


def build_viewer_state(pose_path: Path, glb_path: Path | None = None, lock_root: bool = True, face_cam: bool = False) -> dict:  # noqa: ARG001 - glb kept for API compat
    """Read a pose pack, publish joint animation JSON + per-frame mesh binary for the 3D widget.

    lock_root keeps the body centered at the origin every frame, so camera motion
    in the source clip does not drag the body across the grid.
    """
    import json as _json
    import struct

    import numpy as np

    pose_file = Path(pose_path)
    if pose_file.is_dir():
        pose_file = pose_file / "pose_pack.npz"
    if not pose_file.exists():
        raise ValueError(f"Pose pack not found: {pose_file}")

    with np.load(pose_file, allow_pickle=True) as packed:
        meta = _json.loads(str(packed["meta_json"]))
        joints = np.asarray(packed["pred_joint_coords"], dtype=np.float32)  # (F, P, J, 3)
        cam_t = np.asarray(packed["pred_cam_t"], dtype=np.float32)  # (F, P, 3)
        present = np.asarray(packed["present"]).astype(bool)  # (F, P)
        verts = np.asarray(packed["pred_vertices"], dtype=np.float32) if "pred_vertices" in packed.files else None  # (F, P, V, 3)
        faces = np.asarray(packed["faces"], dtype=np.int64) if "faces" in packed.files else None  # (T, 3)
        colors = np.asarray(packed["vertex_colors"], dtype=np.float32) if "vertex_colors" in packed.files else None  # (V, 3)
        head_idx = np.asarray(packed["face_track_head_idx"], dtype=np.int64) if "face_track_head_idx" in packed.files else None
        nose_idx = int(np.asarray(packed["face_track_nose_idx"]).reshape(-1)[0]) if "face_track_nose_idx" in packed.files else -1
        head_mask = np.asarray(packed["face_track_head_mask"], dtype=np.uint8) if "face_track_head_mask" in packed.files else None

    n_frames = min(int(meta.get("n_frames") or joints.shape[0]), joints.shape[0])
    fps = float(meta.get("fps") or 24.0) or 24.0
    flip = np.array([1.0, -1.0, -1.0], dtype=np.float32)

    world_j = (joints + cam_t[:, :, None, :]) * flip
    world_v = (verts + cam_t[:, :, None, :]) * flip if verts is not None else None

    # Per-frame XZ offset from the primary person's pelvis (joint 0) - steadier
    # than the whole-body centroid, which slides as the limbs move.
    primary = np.full(n_frames, -1, dtype=np.int64)
    offsets = np.zeros((n_frames, 3), dtype=np.float32)
    held = np.zeros(3, dtype=np.float32)
    have_held = False
    for f in range(n_frames):
        idx = np.argwhere(present[f]).reshape(-1)
        if idx.size:
            p0 = int(idx[0])
            primary[f] = p0
            held = np.array([world_j[f, p0, 0, 0], 0.0, world_j[f, p0, 0, 2]], dtype=np.float32)
            have_held = True
        offsets[f] = held if have_held else 0.0
    if not lock_root:
        # Single fixed offset from the first visible frame keeps real locomotion.
        valid = np.argwhere(primary >= 0).reshape(-1)
        if valid.size:
            offsets[:] = offsets[valid[0]]

    world_j = world_j - offsets[:n_frames, None, None, :]
    if world_v is not None:
        world_v = world_v - offsets[:n_frames, None, None, :]

    # Global ground: robust low percentile of per-frame foot height, so a single
    # bad frame cannot sink the floor and leave the body floating. Measure from
    # mesh vertices when available - joints sit inside the body (the ankle joint
    # is well above the foot sole), which used to leave the mesh hovering.
    ground = 0.0
    ground_src = world_v if world_v is not None else world_j
    ys = [ground_src[f, p, :, 1].min() for f in range(n_frames) for p in np.argwhere(present[f]).reshape(-1)]
    if ys:
        ground = float(np.percentile(np.asarray(ys, dtype=np.float32), 5.0))
    world_j[..., 1] -= ground
    if world_v is not None:
        world_v[..., 1] -= ground

    # Per-frame contact snap: temporal smoothing lifts the lowest poses a few
    # centimeters, leaving the body hovering. When the lowest point is near the
    # floor (< 12 cm) pull it flush; clearly airborne frames (jumps) keep their
    # height. Lightly smoothed so the snap itself cannot add vertical jitter.
    mins = np.zeros(n_frames, dtype=np.float32)
    for f in range(n_frames):
        idx = np.argwhere(present[f]).reshape(-1)
        mins[f] = min((float(ground_src[f, p, :, 1].min()) for p in idx), default=0.0)
    snap = np.where(mins < 0.12, mins, 0.0).astype(np.float32)
    if n_frames >= 5:
        kernel = np.exp(-0.5 * (np.arange(-2, 3) / 1.0) ** 2)
        kernel /= kernel.sum()
        snap = np.convolve(np.pad(snap, 2, mode="edge"), kernel, mode="valid").astype(np.float32)
    world_j[..., 1] -= snap[:n_frames, None, None]
    if world_v is not None:
        world_v[..., 1] -= snap[:n_frames, None, None]

    frames: list[list[list[float]]] = []
    for f in range(n_frames):
        people: list[list[float]] = []
        for p in range(world_j.shape[1]):
            if not present[f, p]:
                continue
            people.append([round(float(v), 3) for v in world_j[f, p].reshape(-1)])
        frames.append(people)
    joints_payload = {"fps": fps, "n_frames": len(frames), "frames": frames}
    joints_url = publish_bytes(_json.dumps(joints_payload).encode("utf-8"), f"sam3d_joints_{uuid.uuid4().hex}.json")

    mesh_url = ""
    if world_v is not None and faces is not None and faces.size:
        # Primary person mesh, one vertex block per frame; hold last pose when absent.
        n_verts = world_v.shape[2]
        mesh_frames = np.zeros((n_frames, n_verts, 3), dtype=np.float32)
        last = None
        for f in range(n_frames):
            p0 = int(primary[f])
            if p0 >= 0:
                last = world_v[f, p0]
            if last is not None:
                mesh_frames[f] = last
        has_colors = 1 if (colors is not None and colors.shape == (n_verts, 3)) else 0
        has_head_mask = 1 if (head_mask is not None and head_mask.shape == (n_verts,)) else 0
        blob = bytearray()
        # v3 header: counts, fps, flags for optional per-vertex RGB and head-mask
        # (u8 per vertex; lets the viewer hide the body in face close-up mode).
        blob += struct.pack("<IIIfII", n_frames, n_verts, int(faces.shape[0]), fps, has_colors, has_head_mask)
        blob += faces.astype("<u4").tobytes()
        if has_colors:
            blob += np.clip(colors, 0.0, 1.0).astype("<f4").tobytes()
        if has_head_mask:
            blob += head_mask.astype("u1").tobytes()
            # Pad to 4-byte alignment so the Float32Array vertex view that
            # follows can be mapped directly onto the buffer in the browser.
            blob += b"\x00" * ((-n_verts) % 4)
        blob += mesh_frames.astype("<f4").tobytes()
        mesh_url = publish_bytes(bytes(blob), f"sam3d_mesh_{uuid.uuid4().hex}.bin")

    state: dict[str, Any] = {"jointsUrl": joints_url, "meshUrl": mesh_url, "fps": fps, "faceCam": bool(face_cam)}

    # Face-mocap camera anchors. Old packs (pre face-track) fall back to a
    # geometric guess from the first visible frame: topmost 10% of vertices is
    # the head, the head vertex farthest (horizontally) from its centroid is
    # the nose. Good enough unless frame 0 is an extreme pose.
    if world_v is not None:
        if head_idx is None:
            f0 = next((f for f in range(n_frames) if primary[f] >= 0), None)
            if f0 is not None:
                v0 = world_v[f0, int(primary[f0])]
                cutoff = np.percentile(v0[:, 1], 90.0)
                head_idx = np.argwhere(v0[:, 1] >= cutoff).reshape(-1)
                center = v0[head_idx].mean(axis=0)
                horiz = v0[head_idx][:, [0, 2]] - center[[0, 2]]
                nose_idx = int(head_idx[int(np.argmax(np.linalg.norm(horiz, axis=1)))])
                if head_idx.size > 256:
                    head_idx = head_idx[np.linspace(0, head_idx.size - 1, 256).astype(np.int64)]
        if head_idx is not None and nose_idx >= 0:
            state["faceTrack"] = {"headIdx": [int(i) for i in head_idx], "noseIdx": nose_idx}

    return state


def command_to_pretty_string(command: list[str]) -> str:
    pretty: list[str] = []
    for part in command:
        part = str(part)
        if " " in part or "\t" in part:
            pretty.append(f'"{part}"')
        else:
            pretty.append(part)
    return " ".join(pretty)


def huggingface_token() -> str:
    env_token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or ""
    ).strip()
    if env_token:
        return env_token
    try:
        token = GriptapeNodes.SecretsManager().get_secret("HF_TOKEN", should_error_on_not_found=False)
        return str(token or "").strip()
    except Exception:
        return ""


def run_command(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    log: Callable[[str], None] | None = None,
) -> subprocess.CompletedProcess[str]:
    if log:
        log("$ " + command_to_pretty_string(command) + "\n")
    completed = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if log:
        if completed.stdout:
            log(completed.stdout + ("\n" if not completed.stdout.endswith("\n") else ""))
        if completed.stderr:
            log(completed.stderr + ("\n" if not completed.stderr.endswith("\n") else ""))
    if completed.returncode != 0:
        tail_source = (completed.stderr or completed.stdout or "").strip().splitlines()[-40:]
        tail = "\n".join(tail_source)
        raise ValueError(
            f"Command failed with exit code {completed.returncode}: {command[0]}"
            + (f"\nLast output:\n{tail}" if tail else "")
        )
    return completed


def worker_env() -> dict[str, str]:
    env = os.environ.copy()
    token = huggingface_token()
    if token:
        env["HF_TOKEN"] = token
        env["HUGGING_FACE_HUB_TOKEN"] = token
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def _daemon_state_path(cwd: Path) -> Path:
    return Path(cwd) / ".gtn_worker_daemon.json"


def _spawn_daemon(python_executable: str, cwd: Path) -> None:
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    subprocess.Popen(
        [python_executable, str(WORKER_SCRIPT), "serve", "--daemon-state", str(_daemon_state_path(cwd))],
        cwd=str(cwd),
        env=worker_env(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        creationflags=creationflags,
    )


def _run_via_daemon(
    python_executable: str,
    args: list[str],
    *,
    cwd: Path,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any] | None:
    """Run a job on the warm worker daemon (keeps the model loaded in VRAM).

    Returns the result payload, or None when the daemon path is unusable and
    the caller should fall back to a one-shot subprocess. Raises when the job
    itself ran and failed - that is a real error, not a transport problem.
    """
    import socket
    import time as _time

    state_path = _daemon_state_path(cwd)
    env = worker_env()
    env_subset = {key: env[key] for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN") if env.get(key)}

    for attempt in range(2):
        sock = None
        info = None
        if state_path.exists():
            try:
                info = json.loads(state_path.read_text(encoding="utf-8"))
                sock = socket.create_connection(("127.0.0.1", int(info["port"])), timeout=3)
            except Exception:  # noqa: BLE001 - stale state file or dead daemon
                sock = None
        if sock is None:
            try:
                state_path.unlink(missing_ok=True)
                _spawn_daemon(python_executable, cwd)
            except Exception:  # noqa: BLE001
                return None
            deadline = _time.time() + 90
            while _time.time() < deadline:
                if state_path.exists():
                    try:
                        info = json.loads(state_path.read_text(encoding="utf-8"))
                        sock = socket.create_connection(("127.0.0.1", int(info["port"])), timeout=3)
                        break
                    except Exception:  # noqa: BLE001 - daemon still booting
                        sock = None
                _time.sleep(0.5)
            if sock is None:
                return None

        try:
            with sock:
                sock.settimeout(None)
                stream = sock.makefile("rwb")
                stream.write((json.dumps({"token": info["token"], "args": [str(a) for a in args], "env": env_subset}) + "\n").encode("utf-8"))
                stream.flush()
                stdout_lines: list[str] = []
                exit_code: int | None = None
                stale = False
                for raw in stream:
                    try:
                        message = json.loads(raw.decode("utf-8"))
                    except Exception:  # noqa: BLE001
                        continue
                    if "out" in message:
                        line = str(message["out"])
                        stdout_lines.append(line)
                        if log:
                            log(line + "\n")
                    if "exit" in message:
                        exit_code = int(message["exit"])
                        stale = bool(message.get("stale"))
                        break
        except Exception:  # noqa: BLE001 - transport dropped mid-job
            return None

        if stale:
            # Worker code changed on disk; the old daemon exits itself. Retry
            # once, which spawns a fresh daemon with the new code.
            state_path.unlink(missing_ok=True)
            continue
        if exit_code is None:
            return None
        stdout_text = "\n".join(stdout_lines)
        if exit_code != 0:
            tail = "\n".join(stdout_text.strip().splitlines()[-40:])
            raise ValueError(f"Worker failed with exit code {exit_code}." + (f"\nLast output:\n{tail}" if tail else ""))
        payload = _parse_worker_json(stdout_text)
        if payload is None:
            raise ValueError("Worker finished but did not print a JSON result payload.")
        return payload
    return None


def run_worker(
    python_executable: str,
    args: list[str],
    *,
    cwd: Path,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    if not WORKER_SCRIPT.exists():
        raise ValueError(f"Worker script not found: {WORKER_SCRIPT}")
    # Warm daemon first: it keeps the model in VRAM, so repeat runs skip the
    # ~18 s of imports + checkpoint loading. Any transport problem falls back
    # to the proven one-shot subprocess. GTN_SAM3D_NO_DAEMON=1 disables it.
    if not os.environ.get("GTN_SAM3D_NO_DAEMON"):
        try:
            payload = _run_via_daemon(python_executable, args, cwd=cwd, log=log)
        except ValueError:
            raise
        except Exception:  # noqa: BLE001 - never let daemon plumbing break a run
            payload = None
        if payload is not None:
            return payload
        if log:
            log("Warm worker unavailable; running one-shot subprocess.\n")
    command = [python_executable, str(WORKER_SCRIPT), *args]
    completed = run_command(command, cwd=cwd, env=worker_env(), log=log)
    payload = _parse_worker_json(completed.stdout)
    if payload is None:
        raise ValueError("Worker finished but did not print a JSON result payload.")
    return payload


def _parse_worker_json(stdout: str) -> dict[str, Any] | None:
    marker = "GTN_SAM3D_RESULT:"
    for line in reversed((stdout or "").splitlines()):
        stripped = line.strip()
        if stripped.startswith(marker):
            return json.loads(stripped[len(marker) :].strip())
    return None


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_pose_pack(pose_path: Path) -> dict[str, Any]:
    pose_path = Path(pose_path)
    if pose_path.is_dir():
        pose_path = pose_path / "pose_pack.npz"
    if not pose_path.exists():
        raise ValueError(f"Pose pack not found: {pose_path}")
    with np.load(pose_path, allow_pickle=True) as packed:
        data = {key: packed[key] for key in packed.files}
    meta = json.loads(str(data.pop("meta_json")))
    data["meta"] = meta
    return data


def save_pose_pack(output_path: Path, pack: dict[str, Any]) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    meta = pack.get("meta") or {}
    arrays = {key: value for key, value in pack.items() if key != "meta"}
    np.savez_compressed(output_path, meta_json=json.dumps(meta), **arrays)
    return output_path


def default_model_handle() -> dict[str, Any]:
    """Handle written by Setup, or a minimal one built from the default checkpoint dir."""
    checkpoint_dir = default_checkpoint_dir()
    handle_path = checkpoint_dir / "model_handle.json"
    if handle_path.exists():
        return read_json(handle_path)
    handle: dict[str, Any] = {}
    ckpt = checkpoint_dir / "model.ckpt"
    mhr = checkpoint_dir / "assets" / "mhr_model.pt"
    if ckpt.exists():
        handle["checkpoint_path"] = str(ckpt)
    if mhr.exists():
        handle["mhr_path"] = str(mhr)
    return handle


def parse_model_handle(value: Any) -> dict[str, Any]:
    if value is None:
        return default_model_handle()
    if isinstance(value, dict):
        return value
    text = extract_input_text(value)
    if not text:
        return default_model_handle()
    path = Path(os.path.expandvars(os.path.expanduser(text)))
    if path.exists() and path.suffix.lower() == ".json":
        return read_json(path)
    if text.startswith("{"):
        return json.loads(text)
    return {"checkpoint_path": text}


def model_handle_cli_args(handle: dict[str, Any]) -> list[str]:
    args: list[str] = []
    mapping = {
        "checkpoint_path": "--checkpoint-path",
        "mhr_path": "--mhr-path",
        "detector_name": "--detector-name",
        "segmentor_name": "--segmentor-name",
        "fov_name": "--fov-name",
        "detector_path": "--detector-path",
        "segmentor_path": "--segmentor-path",
        "fov_path": "--fov-path",
        "hf_repo": "--hf-repo",
    }
    for key, flag in mapping.items():
        value = str(handle.get(key) or "").strip()
        if value:
            args.extend([flag, value])
    return args
