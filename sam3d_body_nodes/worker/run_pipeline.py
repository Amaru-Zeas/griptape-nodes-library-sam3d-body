from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# The sam-3d-body repo is not pip-installable (no setup.py/pyproject.toml).
# The nodes always launch this worker with cwd=repo_dir, so put it on sys.path
# to make `import sam_3d_body` work.
_REPO_DIR = str(Path.cwd())
if _REPO_DIR not in sys.path:
    sys.path.insert(0, _REPO_DIR)

# Meta's renderer defaults PYOPENGL_PLATFORM to "egl" (headless Linux). EGL does
# not exist on Windows, so pin the native WGL backend before pyrender is imported.
if os.name == "nt":
    os.environ.setdefault("PYOPENGL_PLATFORM", "win32")


POSE_KEYS = (
    "pred_vertices",
    "pred_joint_coords",
    "pred_cam_t",
    "pred_keypoints_2d",
    "pred_keypoints_3d",
    "bbox",
    "focal_length",
    "global_rot",
    "body_pose_params",
    "hand_pose_params",
    "shape_params",
    "expr_params",
    "scale_params",
    "face_blendshapes",
)


def emit_result(payload: dict[str, Any]) -> None:
    print("GTN_SAM3D_RESULT: " + json.dumps(payload), flush=True)


def _as_float_array(value: Any, fallback_shape: tuple[int, ...] | None = None) -> np.ndarray:
    if value is None:
        if fallback_shape is None:
            return np.zeros((0,), dtype=np.float32)
        return np.zeros(fallback_shape, dtype=np.float32)
    array = np.asarray(value)
    if array.dtype == object:
        array = np.asarray(array.tolist())
    return array.astype(np.float32, copy=False)


def load_frames(media_path: Path, max_frames: int = -1) -> tuple[list[np.ndarray], float, tuple[int, int]]:
    suffix = media_path.suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}:
        image = cv2.imread(str(media_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read image: {media_path}")
        height, width = image.shape[:2]
        return [image], 24.0, (height, width)

    capture = cv2.VideoCapture(str(media_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {media_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 24.0) or 24.0
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
        if max_frames > 0 and len(frames) >= max_frames:
            break
    capture.release()
    if not frames:
        raise ValueError(f"No frames decoded from: {media_path}")
    height, width = frames[0].shape[:2]
    return frames, fps, (height, width)


def write_video(path: Path, frames: list[np.ndarray], fps: float) -> Path:
    if not frames:
        raise ValueError("No frames to write.")
    path.parent.mkdir(parents=True, exist_ok=True)
    # H.264 + yuv420p so browsers can play the result (cv2's mp4v cannot be
    # decoded by the Griptape editor's <video> element). Try NVENC (GPU) first;
    # it encodes 1080p several times faster than libx264 on this pipeline.
    for codec, params in (("h264_nvenc", ["-preset", "p4"]), ("libx264", ["-preset", "veryfast"])):
        try:
            import imageio.v2 as iio

            writer = iio.get_writer(
                str(path),
                format="FFMPEG",
                mode="I",
                fps=max(1.0, float(fps)),
                codec=codec,
                pixelformat="yuv420p",
                macro_block_size=2,
                output_params=params,
            )
            for frame in frames:
                writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            writer.close()
            return path
        except Exception as exc:  # noqa: BLE001
            print(f"[write_video] {codec} unavailable ({exc}); trying fallback")
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(1.0, float(fps)),
        (width, height),
    )
    if not writer.isOpened():
        raise ValueError(f"Could not open video writer: {path}")
    for frame in frames:
        writer.write(frame)
    writer.release()
    return path


def save_pose_pack(output_path: Path, pack: dict[str, Any]) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    meta = pack.get("meta") or {}
    arrays = {key: value for key, value in pack.items() if key != "meta"}
    np.savez_compressed(output_path, meta_json=json.dumps(meta), **arrays)
    return output_path


def load_pose_pack(pose_path: Path) -> dict[str, Any]:
    pose_path = Path(pose_path)
    if pose_path.is_dir():
        pose_path = pose_path / "pose_pack.npz"
    with np.load(pose_path, allow_pickle=True) as packed:
        data = {key: packed[key] for key in packed.files}
    data["meta"] = json.loads(str(data.pop("meta_json")))
    return data


def empty_person(n_verts: int = 1, n_joints: int = 127) -> dict[str, np.ndarray]:
    return {
        "pred_vertices": np.zeros((n_verts, 3), dtype=np.float32),
        "pred_joint_coords": np.zeros((n_joints, 3), dtype=np.float32),
        "pred_cam_t": np.zeros((3,), dtype=np.float32),
        "pred_keypoints_2d": np.zeros((n_joints, 2), dtype=np.float32),
        "pred_keypoints_3d": np.zeros((n_joints, 3), dtype=np.float32),
        "bbox": np.zeros((4,), dtype=np.float32),
        "focal_length": np.array([0.0], dtype=np.float32),
        "global_rot": np.zeros((3,), dtype=np.float32),
        "body_pose_params": np.zeros((1,), dtype=np.float32),
        "hand_pose_params": np.zeros((1,), dtype=np.float32),
        "shape_params": np.zeros((1,), dtype=np.float32),
        "expr_params": np.zeros((1,), dtype=np.float32),
        "scale_params": np.zeros((1,), dtype=np.float32),
        "face_blendshapes": np.zeros((52,), dtype=np.float32),
    }


def outputs_to_frame_people(outputs: list[dict[str, Any]]) -> list[dict[str, np.ndarray]]:
    people: list[dict[str, np.ndarray]] = []
    for person in outputs or []:
        packed = empty_person()
        for key in POSE_KEYS:
            if key not in person:
                continue
            value = person[key]
            if key == "focal_length":
                packed[key] = np.array([float(np.asarray(value).reshape(-1)[0])], dtype=np.float32)
            else:
                packed[key] = _as_float_array(value)
        people.append(packed)
    return people


def pack_sequence(
    frames_people: list[list[dict[str, np.ndarray]]],
    *,
    image_size: tuple[int, int],
    fps: float,
    faces: np.ndarray | None,
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    max_people = max((len(people) for people in frames_people), default=0)
    n_frames = len(frames_people)
    present = np.zeros((n_frames, max(max_people, 1)), dtype=np.bool_)
    template = empty_person()
    if frames_people and frames_people[0]:
        template = frames_people[0][0]
    arrays: dict[str, np.ndarray] = {}
    for key, sample in template.items():
        shape = (n_frames, max(max_people, 1), *np.asarray(sample).shape)
        arrays[key] = np.zeros(shape, dtype=np.float32)
    for frame_idx, people in enumerate(frames_people):
        for person_idx, person in enumerate(people):
            present[frame_idx, person_idx] = True
            for key, value in person.items():
                arrays[key][frame_idx, person_idx] = np.asarray(value, dtype=np.float32)
    meta = {
        "image_size": [int(image_size[0]), int(image_size[1])],
        "fps": float(fps),
        "n_frames": int(n_frames),
        "max_people": int(max(max_people, 1)),
        "keys": list(template.keys()),
    }
    if extra_meta:
        meta.update(extra_meta)
    pack = {"meta": meta, "present": present, **arrays}
    if faces is not None:
        pack["faces"] = np.asarray(faces)
    return pack


def unpack_frame_people(pack: dict[str, Any]) -> list[list[dict[str, np.ndarray]]]:
    meta = pack["meta"]
    n_frames = int(meta["n_frames"])
    max_people = int(meta["max_people"])
    present = np.asarray(pack["present"])
    keys = [key for key in meta.get("keys", POSE_KEYS) if key in pack]
    frames: list[list[dict[str, np.ndarray]]] = []
    for frame_idx in range(n_frames):
        people: list[dict[str, np.ndarray]] = []
        for person_idx in range(max_people):
            if not bool(present[frame_idx, person_idx]):
                continue
            person = {key: np.asarray(pack[key][frame_idx, person_idx]) for key in keys}
            people.append(person)
        frames.append(people)
    return frames


def iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a[:4]]
    bx1, by1, bx2, by2 = [float(v) for v in b[:4]]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def track_boxes(frame_boxes: list[list[list[float]]], iou_threshold: float = 0.3) -> list[list[dict[str, Any]]]:
    next_id = 0
    prev: list[dict[str, Any]] = []
    tracked: list[list[dict[str, Any]]] = []
    for boxes in frame_boxes:
        used_prev: set[int] = set()
        current: list[dict[str, Any]] = []
        for box in boxes:
            best_idx = -1
            best_iou = iou_threshold
            for idx, previous in enumerate(prev):
                if idx in used_prev:
                    continue
                score = iou(np.asarray(box), np.asarray(previous["bbox"]))
                if score > best_iou:
                    best_iou = score
                    best_idx = idx
            if best_idx >= 0:
                person_id = int(prev[best_idx]["track_id"])
                used_prev.add(best_idx)
            else:
                person_id = next_id
                next_id += 1
            x1, y1, x2, y2 = [float(v) for v in box[:4]]
            current.append(
                {
                    "track_id": person_id,
                    "bbox": [x1, y1, x2, y2],
                    "x": x1,
                    "y": y1,
                    "width": max(0.0, x2 - x1),
                    "height": max(0.0, y2 - y1),
                }
            )
        tracked.append(current)
        prev = current
    return tracked


def load_boxes_file(path: str | None) -> list[list[list[float]]] | None:
    if not path:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    frames = payload.get("frames", payload)
    out: list[list[list[float]]] = []
    for frame in frames:
        boxes: list[list[float]] = []
        for item in frame:
            if isinstance(item, dict):
                if "bbox" in item:
                    boxes.append([float(v) for v in item["bbox"][:4]])
                else:
                    x, y = float(item.get("x", 0)), float(item.get("y", 0))
                    w, h = float(item.get("width", 0)), float(item.get("height", 0))
                    boxes.append([x, y, x + w, y + h])
            else:
                boxes.append([float(v) for v in item[:4]])
        out.append(boxes)
    return out


class YoloHumanDetector:
    """Person detector with the same interface Meta's HumanDetector exposes.

    Used when Detectron2/ViTDet is unavailable (its CUDA extension does not
    build on this machine). Stable per-frame person crops matter a lot for
    temporal stability: full-frame boxes make the pose estimate jitter.
    """

    def __init__(self, device: Any, weights: str = "yolov8m.pt") -> None:
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.device = str(device)

    def run_human_detection(
        self,
        img: np.ndarray,
        det_cat_id: int = 0,  # noqa: ARG002 - kept for interface parity
        bbox_thr: float = 0.5,
        nms_thr: float = 0.5,
        default_to_full_image: bool = False,
        **_: Any,
    ) -> np.ndarray:
        # Meta's default bbox_thr (0.8) is tuned for ViTDet scores; YOLO
        # confidences run lower, so cap the threshold to keep recall.
        conf = min(float(bbox_thr or 0.5), 0.5)
        results = self.model.predict(
            img, classes=[0], conf=conf, iou=float(nms_thr or 0.5), device=self.device, verbose=False
        )
        boxes: list[np.ndarray] = []
        for result in results:
            if result.boxes is not None and len(result.boxes):
                boxes.append(result.boxes.xyxy.cpu().numpy().astype(np.float32))
        out = np.concatenate(boxes, axis=0) if boxes else np.zeros((0, 4), dtype=np.float32)
        if out.shape[0] == 0 and default_to_full_image:
            height, width = img.shape[:2]
            out = np.array([[0.0, 0.0, float(width), float(height)]], dtype=np.float32)
        return out


# One cached estimator (daemon mode): keyed by the args that change the loaded
# weights. A single slot bounds VRAM to one resident model.
_ESTIMATOR_CACHE: dict[str, Any] = {}


def build_estimator(args: argparse.Namespace):
    import torch
    from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body

    cache_key = "|".join(
        str(getattr(args, name, "") or "")
        for name in ("checkpoint_path", "mhr_path", "detector_name", "detector_path", "segmentor_name", "segmentor_path", "fov_name", "fov_path")
    )
    cached = _ESTIMATOR_CACHE.get(cache_key)
    if cached is not None:
        print("WORKER_INFO: reusing warm model (daemon).", flush=True)
        return cached

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model, model_cfg = load_sam_3d_body(args.checkpoint_path, device=device, mhr_path=args.mhr_path or "")
    human_detector = None
    human_segmentor = None
    fov_estimator = None
    if args.detector_name:
        try:
            from tools.build_detector import HumanDetector

            human_detector = HumanDetector(name=args.detector_name, device=device, path=args.detector_path or "")
        except Exception as exc:  # Detectron2 may be absent (CUDA-mismatch skip on Windows)
            print(
                f"WORKER_WARN: detector '{args.detector_name}' unavailable ({exc}); trying YOLO.",
                file=sys.stderr,
                flush=True,
            )
        if human_detector is None:
            try:
                human_detector = YoloHumanDetector(device)
                print("WORKER_INFO: using YOLO person detector (ultralytics).", flush=True)
            except Exception as exc:
                print(
                    f"WORKER_WARN: YOLO detector unavailable ({exc}); falling back to full-frame boxes.",
                    file=sys.stderr,
                    flush=True,
                )
    if args.segmentor_name:
        try:
            from tools.build_sam import HumanSegmentor

            human_segmentor = HumanSegmentor(name=args.segmentor_name, device=device, path=args.segmentor_path or "")
        except Exception as exc:
            print(f"WORKER_WARN: segmentor '{args.segmentor_name}' unavailable ({exc}); continuing without masks.", file=sys.stderr, flush=True)
    if args.fov_name:
        from tools.build_fov_estimator import FOVEstimator

        fov_estimator = FOVEstimator(name=args.fov_name, device=device, path=args.fov_path or "")
    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=human_detector,
        human_segmentor=human_segmentor,
        fov_estimator=fov_estimator,
    )
    _ESTIMATOR_CACHE.clear()
    _ESTIMATOR_CACHE[cache_key] = estimator
    return estimator


def cmd_download(args: argparse.Namespace) -> None:
    from huggingface_hub import snapshot_download

    local_dir = Path(args.checkpoint_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=args.hf_repo, local_dir=str(local_dir), token=os.environ.get("HF_TOKEN") or True)
    ckpt = local_dir / "model.ckpt"
    mhr = local_dir / "assets" / "mhr_model.pt"
    emit_result(
        {
            "checkpoint_dir": str(local_dir),
            "checkpoint_path": str(ckpt) if ckpt.exists() else "",
            "mhr_path": str(mhr) if mhr.exists() else "",
        }
    )


def cmd_detect(args: argparse.Namespace) -> None:
    estimator = build_estimator(args)
    frames, fps, image_size = load_frames(Path(args.input), max_frames=args.max_frames)
    all_boxes: list[list[list[float]]] = []
    for frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes: list[list[float]] = []
        detector = getattr(estimator, "human_detector", None)
        if detector is not None:
            if hasattr(detector, "run_human_detection"):
                detections = detector.run_human_detection(rgb, bbox_thr=float(args.bbox_thresh))
            else:
                detections = detector(rgb)
            if isinstance(detections, dict):
                raw = detections.get("boxes") or detections.get("bboxes") or []
            else:
                raw = detections or []
            for item in raw:
                arr = np.asarray(item).reshape(-1)
                if arr.size >= 4:
                    boxes.append([float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])])
        if not boxes:
            height, width = frame.shape[:2]
            boxes = [[0.0, 0.0, float(width), float(height)]]
        all_boxes.append(boxes)
    tracked = track_boxes(all_boxes)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fps": fps,
        "image_size": [int(image_size[0]), int(image_size[1])],
        "frames": tracked,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    emit_result({"boxes_path": str(output_path), "n_frames": len(tracked), "max_people": max((len(f) for f in tracked), default=0)})


def cmd_fov(args: argparse.Namespace) -> None:
    estimator = build_estimator(args)
    frames, _, image_size = load_frames(Path(args.input), max_frames=1)
    rgb = cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB)
    fov = float(args.fallback_fov)
    fov_estimator = getattr(estimator, "fov_estimator", None)
    if fov_estimator is not None:
        try:
            # Meta's FOVEstimator returns a (1, 3, 3) camera intrinsics matrix.
            intrinsics = fov_estimator.get_cam_intrinsics(rgb)
            k = np.asarray(intrinsics).reshape(-1, 3, 3)[0]
            focal_y = float(k[1, 1])
            height = float(rgb.shape[0])
            if focal_y > 0:
                fov = float(np.degrees(2.0 * np.arctan(height / (2.0 * focal_y))))
        except Exception as exc:
            print(f"WORKER_WARN: FoV estimation failed ({exc}); using fallback {fov} degrees.", file=sys.stderr, flush=True)
    emit_result({"fov": fov, "image_size": [int(image_size[0]), int(image_size[1])]})


FACE_REGION_RGB_URL = "https://huggingface.co/Comfy-Org/sam-3d-body/resolve/main/detection/sam_3d_body_dinov3_bf16.safetensors"


def _ensure_face_region_rgb(n_verts: int) -> np.ndarray | None:
    """Painted per-vertex face-region colors (hair/lips/brows) from Comfy-Org's
    repackaged SAM 3D Body weights (SAM license). Fetched once via HTTP range
    requests (~220 KB, not the full checkpoint) and cached beside our checkpoints.
    """
    cache = Path.cwd() / "checkpoints" / "face_region_rgb.npy"
    if cache.exists():
        arr = np.load(cache)
        return arr if arr.shape == (n_verts, 3) else None
    try:
        import requests

        head = requests.get(FACE_REGION_RGB_URL, headers={"Range": "bytes=0-7"}, timeout=30)
        head.raise_for_status()
        header_len = int.from_bytes(head.content[:8], "little")
        meta = requests.get(FACE_REGION_RGB_URL, headers={"Range": f"bytes=8-{8 + header_len - 1}"}, timeout=60)
        meta.raise_for_status()
        info = json.loads(meta.content).get("head_pose.face_region_rgb")
        if not info or info.get("dtype") != "F32" or info.get("shape") != [n_verts, 3]:
            return None
        start, end = info["data_offsets"]
        blob = requests.get(
            FACE_REGION_RGB_URL,
            headers={"Range": f"bytes={8 + header_len + start}-{8 + header_len + end - 1}"},
            timeout=120,
        )
        blob.raise_for_status()
        arr = np.frombuffer(blob.content, dtype="<f4").reshape(n_verts, 3).astype(np.float32)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, arr)
        return arr
    except Exception as exc:  # noqa: BLE001 - colors are cosmetic, never fail the run
        print(f"WORKER_WARN: face region colors unavailable ({exc}).", file=sys.stderr, flush=True)
        return None


def _rest_pose_vertices(head: Any, device: Any) -> np.ndarray:
    """Canonical (zero-parameter) MHR vertices, y-up, meters."""
    import torch

    def zeros(*shape: int) -> Any:
        return torch.zeros(1, *shape, device=device)

    with torch.no_grad():
        verts = head.mhr_forward(
            global_trans=zeros(3),
            global_rot=zeros(3),
            body_pose_params=zeros(130),
            hand_pose_params=zeros(head.num_hand_comps * 2),
            scale_params=zeros(head.num_scale_comps),
            shape_params=zeros(head.num_shape_comps),
            expr_params=zeros(head.num_face_comps),
        )
    return verts[0].float().cpu().numpy()


def compute_face_track(estimator: Any) -> dict[str, np.ndarray] | None:
    """Vertex indices the 3D viewer needs to anchor a face-mocap camera.

    head_idx: subsampled head-region vertices (head center per frame).
    nose_idx: nose-tip vertex; (nose - head center) gives the facing direction.
    Both are stable indices into the mesh, valid for every frame.
    """
    try:
        head = estimator.model.head_pose
        device = next(estimator.model.parameters()).device
        canon = _rest_pose_vertices(head, device)
    except Exception as exc:  # noqa: BLE001
        print(f"WORKER_WARN: canonical pose for face track failed ({exc}).", file=sys.stderr, flush=True)
        return None
    head_mask = canon[:, 1] > 1.43
    idx = np.argwhere(head_mask).reshape(-1)
    if idx.size < 32:
        return None
    center = canon[idx].mean(axis=0)
    # Nose tip: MHR's canonical pose faces +z (verified: the strongest forward
    # protrusion at nose height is +0.16 m, the skull back only -0.11 m), so
    # within the narrow center strip (|x| tiny, avoids ears/hair) the nose is
    # simply the max-z vertex.
    strip = idx[np.abs(canon[idx, 0] - center[0]) < 0.02]
    if strip.size == 0:
        strip = idx
    nose = int(strip[int(np.argmax(canon[strip, 2]))])
    if idx.size > 256:
        idx = idx[np.linspace(0, idx.size - 1, 256).astype(np.int64)]
    # Full-resolution mask for "hide the body" in the face viewer: head plus a
    # bit of neck (1.35 m) so the close-up doesn't cut off at the jaw.
    head_mask = (canon[:, 1] > 1.35).astype(np.uint8)
    return {
        "face_track_head_idx": idx.astype(np.int32),
        "face_track_nose_idx": np.array([nose], dtype=np.int32),
        "face_track_head_mask": head_mask,
    }


def compute_vertex_colors(estimator: Any) -> np.ndarray | None:
    """Per-vertex display colors: rainbow over canonical body height, with the
    painted face-region palette applied to the head so lips/brows/hair read."""
    try:
        import torch

        head = estimator.model.head_pose
        device = next(estimator.model.parameters()).device
        canon = _rest_pose_vertices(head, device)
    except Exception as exc:  # noqa: BLE001
        print(f"WORKER_WARN: canonical pose for vertex colors failed ({exc}).", file=sys.stderr, flush=True)
        return None

    y = canon[:, 1]
    t = (y - y.min()) / max(float(y.max() - y.min()), 1e-6)
    # Rainbow: hue blue (feet) -> red (head) through HSV.
    hue = (1.0 - t) * (2.0 / 3.0)
    h6 = hue * 6.0
    i = np.floor(h6).astype(np.int64) % 6
    f = h6 - np.floor(h6)
    one = np.ones_like(f)
    q = 1.0 - f
    lut = np.stack(
        [
            np.stack([one, f, np.zeros_like(f)], axis=1),
            np.stack([q, one, np.zeros_like(f)], axis=1),
            np.stack([np.zeros_like(f), one, f], axis=1),
            np.stack([np.zeros_like(f), q, one], axis=1),
            np.stack([f, np.zeros_like(f), one], axis=1),
            np.stack([one, np.zeros_like(f), q], axis=1),
        ],
        axis=0,
    )
    colors = lut[i, np.arange(canon.shape[0])].astype(np.float32)
    colors = 0.25 + 0.75 * colors  # lift shadows so the mesh doesn't go black

    # Head region (canonical y > 1.43 m, |x| < 0.11 m) takes the painted palette.
    face_rgb = _ensure_face_region_rgb(canon.shape[0])
    if face_rgb is not None:
        head_mask = (canon[:, 1] > 1.43) & (np.abs(canon[:, 0]) < 0.11)
        colors[head_mask] = face_rgb[head_mask]
    return colors


def _speed_patches(precision: str):
    """Context manager with video-loop speed fixes for Meta's estimator.

    - Disables torch.cuda.empty_cache: process_one_image calls it EVERY frame,
      which forces a GPU sync + allocator flush and dominates per-frame time in
      video loops. Safe to skip in a one-shot worker process.
    - Makes tensor->numpy conversion dtype-safe so bf16/fp16 autocast outputs
      don't crash Meta's recursive_to (torch bf16 has no numpy equivalent).
    - Returns the autocast context for the requested precision.
    """
    import contextlib

    import torch

    @contextlib.contextmanager
    def patched():
        orig_empty_cache = torch.cuda.empty_cache
        torch.cuda.empty_cache = lambda: None
        orig_numpy = torch.Tensor.numpy

        def safe_numpy(tensor, *a, **k):  # noqa: ANN001
            if tensor.dtype in (torch.bfloat16, torch.float16):
                tensor = tensor.float()
            return orig_numpy(tensor, *a, **k)

        torch.Tensor.numpy = safe_numpy

        # The MHR skinning head uses sparse matmuls that CUDA does not implement
        # in half precision ("addmm_sparse_cuda not implemented for BFloat16").
        # Run that head as an fp32 island; the ViT backbone (the actual cost)
        # still gets autocast.
        mhr_originals: dict[str, Any] = {}
        mhr_cls = None
        if precision in {"bf16", "fp16"}:
            try:
                from sam_3d_body.models.heads.mhr_head import MHRHead as mhr_cls  # noqa: N813

                def _to_fp32(value):  # noqa: ANN001
                    if torch.is_tensor(value) and value.is_floating_point():
                        return value.float()
                    if isinstance(value, dict):
                        return {k: _to_fp32(v) for k, v in value.items()}
                    if isinstance(value, (list, tuple)):
                        return type(value)(_to_fp32(v) for v in value)
                    return value

                def make_fp32_island(orig):  # noqa: ANN001
                    def wrapped(self, *fargs, **fkwargs):  # noqa: ANN001
                        with torch.autocast(device_type="cuda", enabled=False):
                            fargs = [_to_fp32(v) for v in fargs]
                            fkwargs = {k: _to_fp32(v) for k, v in fkwargs.items()}
                            return orig(self, *fargs, **fkwargs)

                    return wrapped

                # run_inference calls both forward AND mhr_forward directly
                # (hand-merge re-skinning), so both need the fp32 wrap.
                for method in ("forward", "mhr_forward"):
                    mhr_originals[method] = getattr(mhr_cls, method)
                    setattr(mhr_cls, method, make_fp32_island(mhr_originals[method]))
            except Exception as exc:  # noqa: BLE001
                print(f"WORKER_WARN: could not pin MHR head to fp32 ({exc}).", file=sys.stderr, flush=True)
                mhr_originals = {}
        try:
            yield
        finally:
            torch.cuda.empty_cache = orig_empty_cache
            torch.Tensor.numpy = orig_numpy
            if mhr_cls is not None:
                for method, orig in mhr_originals.items():
                    setattr(mhr_cls, method, orig)

    if precision in {"bf16", "fp16"} and torch.cuda.is_available():
        dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        autocast = lambda: torch.autocast(device_type="cuda", dtype=dtype)  # noqa: E731
    else:
        autocast = contextlib.nullcontext
    return patched, autocast


def _predict_frames_batched(
    estimator: Any,
    frames_bgr: list[np.ndarray],
    box_frames: list[list[list[float]]] | None,
    *,
    bbox_thr: float,
    batch_size: int,
    autocast: Any,
) -> list[list[dict[str, np.ndarray]]]:
    """Body-only inference with many frames per GPU forward pass.

    Meta's model flattens (batch, person) into one crop axis, so crops from
    different frames of the same video can share a single forward_step call -
    this is how ComfyUI's integration gets its video throughput. Only valid for
    the body decoder; hand refinement needs the original image per crop.
    """
    import torch
    from sam_3d_body.data.utils.prepare_batch import NoCollate, prepare_batch
    from sam_3d_body.utils.dist import recursive_to

    model = estimator.model
    n_frames = len(frames_bgr)

    # Person boxes for every frame first (YOLO is ~14 ms/frame).
    boxes_all: list[np.ndarray] = []
    for idx, frame in enumerate(frames_bgr):
        if box_frames is not None and idx < len(box_frames) and box_frames[idx]:
            boxes_all.append(np.asarray(box_frames[idx], dtype=np.float32).reshape(-1, 4))
        elif estimator.detector is not None:
            boxes_all.append(
                np.asarray(
                    estimator.detector.run_human_detection(frame, bbox_thr=bbox_thr, default_to_full_image=False),
                    dtype=np.float32,
                ).reshape(-1, 4)
            )
        else:
            height, width = frame.shape[:2]
            boxes_all.append(np.array([[0.0, 0.0, float(width), float(height)]], dtype=np.float32))
    print(f"detected_boxes {sum(len(b) for b in boxes_all)} across {n_frames} frames", flush=True)

    # Same key mapping the estimator uses for its per-person outputs.
    out_key_map = {
        "focal_length": "focal_length",
        "pred_keypoints_3d": "pred_keypoints_3d",
        "pred_keypoints_2d": "pred_keypoints_2d",
        "pred_vertices": "pred_vertices",
        "pred_cam_t": "pred_cam_t",
        "global_rot": "global_rot",
        "body_pose": "body_pose_params",
        "hand": "hand_pose_params",
        "scale": "scale_params",
        "shape": "shape_params",
        "face": "expr_params",
        "pred_joint_coords": "pred_joint_coords",
    }
    cat_keys = ("img", "img_size", "ori_img_size", "bbox_center", "bbox_scale", "bbox", "affine_trans", "mask", "mask_score")

    results: list[list[dict[str, np.ndarray]]] = [[] for _ in range(n_frames)]
    chunk: list[int] = []
    chunk_crops = 0
    done_frames = 0

    def flush_chunk() -> None:
        nonlocal chunk, chunk_crops, done_frames
        if not chunk:
            return
        per_frame = [prepare_batch(cv2.cvtColor(frames_bgr[f], cv2.COLOR_BGR2RGB), estimator.transform, boxes_all[f]) for f in chunk]
        combined: dict[str, Any] = {}
        for key in cat_keys:
            if key in per_frame[0]:
                combined[key] = torch.cat([b[key] for b in per_frame], dim=1)
        combined["person_valid"] = torch.cat([b["person_valid"] for b in per_frame], dim=1)
        combined["cam_int"] = per_frame[0]["cam_int"]
        combined["img_ori"] = [NoCollate(frames_bgr[chunk[0]])]  # unused by the body decoder
        combined = recursive_to(combined, "cuda")
        model._initialize_batch(combined)  # noqa: SLF001 - same call the estimator makes
        with torch.no_grad(), autocast():
            pose_output = model.forward_step(combined, decoder_type="body")
        out = recursive_to(recursive_to(pose_output["mhr"], "cpu"), "numpy")
        crop = 0
        for f in chunk:
            for box in boxes_all[f]:
                person: dict[str, np.ndarray] = {"bbox": np.asarray(box, dtype=np.float32)}
                for src, dst in out_key_map.items():
                    if src in out:
                        person[dst] = np.asarray(out[src][crop])
                results[f].append(person)
                crop += 1
        done_frames += len(chunk)
        print(f"predicted_frame {done_frames}/{n_frames} (batched x{len(chunk)})", flush=True)
        chunk = []
        chunk_crops = 0

    for f in range(n_frames):
        n_boxes = len(boxes_all[f])
        if n_boxes == 0:
            done_frames += 1
            continue
        if chunk_crops + n_boxes > batch_size and chunk:
            flush_chunk()
        chunk.append(f)
        chunk_crops += n_boxes
        if chunk_crops >= batch_size:
            flush_chunk()
    flush_chunk()
    return results


def cmd_predict(args: argparse.Namespace) -> None:
    estimator = build_estimator(args)
    frames, fps, image_size = load_frames(Path(args.input), max_frames=args.max_frames)
    box_frames = load_boxes_file(args.boxes_path)
    if box_frames is not None and len(box_frames) == 1 and len(frames) > 1:
        box_frames = box_frames * len(frames)
    frames_people: list[list[dict[str, np.ndarray]]] = []
    faces = np.asarray(getattr(estimator, "faces", np.zeros((0, 3), dtype=np.int32)))
    precision = str(getattr(args, "precision", "fp32") or "fp32").lower()
    patched, autocast = _speed_patches(precision)
    if precision != "fp32":
        print(f"WORKER_INFO: running inference in {precision} (autocast).", flush=True)
    with patched():
        # The hand-refinement decoder is ~78% of per-frame inference time
        # (851 ms vs 185 ms measured on an RTX 6000). "body" still produces the
        # full body mesh including hands, just without the extra refinement pass.
        inference_type = "full" if bool(getattr(args, "run_hand_refinement", False)) else "body"
        batched_done = False
        if inference_type == "body":
            try:
                raw_frames = _predict_frames_batched(
                    estimator,
                    frames,
                    box_frames,
                    bbox_thr=float(args.bbox_thresh),
                    batch_size=max(1, int(getattr(args, "batch_size", 16) or 16)),
                    autocast=autocast,
                )
                frames_people = [outputs_to_frame_people(people) for people in raw_frames]
                batched_done = True
            except Exception as exc:  # noqa: BLE001 - fall back to the slow, proven path
                print(f"WORKER_WARN: batched inference failed ({exc!r}); falling back to per-frame.", flush=True)
                frames_people = []
        for idx, frame in enumerate([] if batched_done else frames):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            kwargs: dict[str, Any] = {
                "bbox_thr": float(args.bbox_thresh),
                "use_mask": bool(args.use_mask),
                "inference_type": inference_type,
            }
            if box_frames is not None and idx < len(box_frames) and box_frames[idx]:
                kwargs["bboxes"] = np.asarray(box_frames[idx], dtype=np.float32)
            try:
                with autocast():
                    outputs = estimator.process_one_image(rgb, **kwargs)
            except TypeError:
                # Older estimator builds only accept a path-like / image positional.
                tmp = Path(args.output).with_name(f"_frame_{idx:06d}.png")
                tmp.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(tmp), frame)
                with autocast():
                    outputs = estimator.process_one_image(str(tmp), bbox_thr=float(args.bbox_thresh), use_mask=bool(args.use_mask))
                tmp.unlink(missing_ok=True)
            if isinstance(outputs, dict):
                outputs = [outputs]
            frames_people.append(outputs_to_frame_people(list(outputs or [])))
            print(f"predicted_frame {idx + 1}/{len(frames)} people={len(frames_people[-1])}", flush=True)
    pack = pack_sequence(
        frames_people,
        image_size=image_size,
        fps=fps,
        faces=faces,
        extra_meta={
            "source": str(args.input),
            "fov": float(args.fov or 0.0),
            "run_hand_refinement": bool(args.run_hand_refinement),
        },
    )
    colors = compute_vertex_colors(estimator)
    if colors is not None:
        pack["vertex_colors"] = colors
    face_track = compute_face_track(estimator)
    if face_track is not None:
        pack.update(face_track)
    pose_path = Path(args.output)
    if pose_path.suffix.lower() != ".npz":
        pose_path = pose_path / "pose_pack.npz"
    save_pose_pack(pose_path, pack)
    emit_result(
        {
            "pose_path": str(pose_path),
            "n_frames": int(pack["meta"]["n_frames"]),
            "max_people": int(pack["meta"]["max_people"]),
            "fps": float(fps),
        }
    )


def cmd_render(args: argparse.Namespace) -> None:
    # Use Meta's Renderer directly instead of tools.vis_utils, whose skeleton
    # visualizer pulls in Detectron2 (not installed on this machine).
    from sam_3d_body.visualization.renderer import Renderer

    # PYOPENGL_PLATFORM=win32 was needed while pyrender imported (see module top),
    # but pyrender's OffscreenRenderer only accepts egl/osmesa/unset. Unset it so
    # the renderer falls back to its hidden-window Pyglet backend on Windows.
    if os.name == "nt" and os.environ.get("PYOPENGL_PLATFORM") == "win32":
        del os.environ["PYOPENGL_PLATFORM"]

    pack = load_pose_pack(Path(args.pose_path))
    frames, fps, _ = load_frames(Path(args.input), max_frames=args.max_frames)
    people_frames = unpack_frame_people(pack)
    faces = np.asarray(pack.get("faces", np.zeros((0, 3), dtype=np.int32)))
    if faces.size == 0:
        raise RuntimeError("Pose pack has no mesh faces; re-run prediction before rendering.")
    import pyrender

    rendered: list[np.ndarray] = []
    n = min(len(frames), len(people_frames))
    height, width = frames[0].shape[:2]

    # Meta's render_rgba_multiple creates/destroys a GL context per call
    # (~300 ms/frame on Windows). Build one offscreen renderer + scene and only
    # swap the mesh nodes per frame.
    first_people = next((p for p in people_frames if p), None)
    focal = 5000.0
    if first_people:
        focal = float(np.asarray(first_people[0]["focal_length"]).reshape(-1)[0] or 5000.0)
    helper = Renderer(focal_length=focal, faces=faces)
    offscreen = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height, point_size=1.0)
    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.3, 0.3, 0.3))
    camera = pyrender.IntrinsicsCamera(fx=focal, fy=focal, cx=width / 2.0, cy=height / 2.0, zfar=1e12)
    camera_node = pyrender.Node(camera=camera, matrix=np.eye(4))
    scene.add_node(camera_node)
    helper.add_point_lighting(scene, camera_node)
    helper.add_lighting(scene, camera_node)
    from sam_3d_body.visualization.renderer import create_raymond_lights

    for node in create_raymond_lights():
        scene.add_node(node)

    mesh_color = (1.0, 1.0, 0.9)
    mesh_nodes: list[Any] = []
    for idx in range(n):
        outputs = people_frames[idx]
        frame = frames[idx]
        if not outputs:
            rendered.append(frame)
            continue
        for node in mesh_nodes:
            scene.remove_node(node)
        mesh_nodes = []
        for person in outputs:
            verts = np.asarray(person["pred_vertices"], dtype=np.float32)
            cam_t = np.asarray(person["pred_cam_t"], dtype=np.float32).reshape(-1)
            tri = helper.vertices_to_trimesh(verts, cam_t.copy(), mesh_color, [1, 0, 0], 0)
            mesh_nodes.append(scene.add(pyrender.Mesh.from_trimesh(tri)))
        color, _depth = offscreen.render(scene, flags=pyrender.RenderFlags.RGBA)
        if (idx + 1) % 50 == 0:
            print(f"rendered_frame {idx + 1}/{n}", flush=True)
        # Composite only the rows/cols the mesh actually covers - the body is a
        # fraction of the frame, so this cuts the per-frame blend cost a lot.
        composite = frame.copy()
        a8 = color[:, :, 3]
        rows = np.flatnonzero(a8.any(axis=1))
        cols = np.flatnonzero(a8.any(axis=0))
        if rows.size and cols.size:
            r0, r1 = int(rows[0]), int(rows[-1]) + 1
            c0, c1 = int(cols[0]), int(cols[-1]) + 1
            alpha = color[r0:r1, c0:c1, 3:4].astype(np.float32) / 255.0
            # Renderer outputs RGB uint8; frames are BGR uint8.
            mesh_bgr = color[r0:r1, c0:c1, 2::-1].astype(np.float32)
            roi = mesh_bgr * alpha + frame[r0:r1, c0:c1].astype(np.float32) * (1.0 - alpha)
            composite[r0:r1, c0:c1] = np.clip(roi, 0, 255).astype(np.uint8)
        if args.overlay_only:
            rendered.append(composite)
        else:
            # Side-by-side: original | mesh overlay.
            rendered.append(np.hstack([frame, composite]))
    offscreen.delete()
    output_path = Path(args.output)
    write_video(output_path, rendered, fps)
    emit_result({"overlay_path": str(output_path), "n_frames": len(rendered), "fps": fps})


# MHR expr axis -> ARKit blendshape driver(s). MHR's 72 expression axes are
# unnamed upstream; which axis maps to which ARKit shape is a fact about
# Meta's model, established empirically by the ComfyUI SAM 3D Body integration.
MHR_AXIS_DRIVERS: dict[int, list[tuple[str, float]]] = {
    0: [("browDownLeft", 1.0)],
    1: [("browDownRight", 1.0)],
    2: [("cheekPuff", 1.0)],
    3: [("cheekPuff", 1.0)],
    4: [("cheekSquintLeft", 1.0)],
    5: [("cheekSquintRight", 1.0)],
    6: [("mouthStretchLeft", 1.0)],
    7: [("mouthStretchRight", 1.0)],
    8: [("mouthShrugLower", 1.0)],
    9: [("mouthShrugUpper", 1.0)],
    10: [("mouthDimpleLeft", 1.0)],
    11: [("mouthDimpleRight", 1.0)],
    12: [("eyeLookDownLeft", 0.3)],
    13: [("eyeLookDownRight", 0.3)],
    14: [("eyeBlinkLeft", 1.0)],
    15: [("eyeBlinkRight", 1.0)],
    16: [("eyeLookOutLeft", 1.0)],
    17: [("eyeLookInRight", 1.0)],
    18: [("eyeLookInLeft", 1.0)],
    19: [("eyeLookOutRight", 1.0)],
    22: [("eyeLookUpLeft", 1.0), ("browInnerUp", 0.5)],
    23: [("eyeLookUpRight", 1.0), ("browInnerUp", 0.5)],
    24: [("jawOpen", 1.0), ("mouthLowerDownLeft", 0.3), ("mouthLowerDownRight", 0.3)],
    25: [("jawLeft", 1.0)],
    26: [("jawRight", 1.0)],
    27: [("jawForward", 1.0)],
    28: [("eyeSquintLeft", 1.0)],
    29: [("eyeSquintRight", 1.0)],
    32: [("mouthSmileLeft", 1.0)],
    33: [("mouthSmileRight", 1.0)],
    40: [("mouthLeft", 1.0)],
    41: [("mouthRight", 1.0)],
    42: [("mouthFrownLeft", 1.0)],
    43: [("mouthFrownRight", 1.0)],
    54: [("mouthLowerDownLeft", 1.0)],
    55: [("mouthLowerDownRight", 1.0)],
    60: [("noseSneerLeft", 1.0)],
    61: [("noseSneerRight", 1.0)],
    66: [("browOuterUpLeft", 1.0)],
    67: [("browOuterUpRight", 1.0)],
    68: [("eyeWideLeft", 1.0)],
    69: [("eyeWideRight", 1.0)],
    70: [("mouthUpperUpLeft", 1.0)],
    71: [("mouthUpperUpRight", 1.0)],
}

# MediaPipe magnitudes differ per facial region (jaw reaches 1.0, brows rarely
# pass 0.3), so upper-face signals get extra gain.
_REGION_GAIN = {"eye": 2.0, "brow": 2.0, "cheek": 2.0, "nose": 2.0}
# Jaw signals are clean; everything else gets a small deadzone against MP noise.
_CLEAN_PREFIXES = ("jaw",)
_DEADZONE = 0.02


def _blendshape_gain(name: str) -> float:
    for prefix, gain in _REGION_GAIN.items():
        if name.startswith(prefix):
            return gain
    return 1.0


def _map_blendshapes_to_expr(coefs: dict[str, float], strength: float, n_axes: int) -> np.ndarray:
    expr = np.zeros(n_axes, dtype=np.float32)
    for axis, drivers in MHR_AXIS_DRIVERS.items():
        if axis >= n_axes:
            continue
        best = 0.0
        for name, weight in drivers:
            raw = float(coefs.get(name, 0.0))
            if not name.startswith(_CLEAN_PREFIXES):
                if raw <= _DEADZONE:
                    raw = 0.0
                else:
                    raw = (raw - _DEADZONE) / (1.0 - _DEADZONE)
            best = max(best, raw * weight * _blendshape_gain(name))
        expr[axis] = best * strength
    return expr


def _smooth_series(series: np.ndarray, window: int) -> np.ndarray:
    """Gaussian smoothing along axis 0. MediaPipe scores swing frame-to-frame
    even on a static face; smoothing coefficients beats smoothing vertices."""
    if window <= 1 or series.shape[0] < 3:
        return series
    if window % 2 == 0:
        window += 1
    sigma = max(1.0, window / 5.0)
    x = np.arange(window) - (window - 1) / 2.0
    kernel = np.exp(-(x**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    pad = window // 2
    padded = np.concatenate([np.repeat(series[:1], pad, axis=0), series, np.repeat(series[-1:], pad, axis=0)], axis=0)
    out = np.zeros_like(series)
    for k, w in enumerate(kernel):
        out += w * padded[k : k + series.shape[0]]
    return out


def _reskin_pack(pack: dict[str, Any], args: argparse.Namespace, chunk: int = 32) -> bool:
    """Re-run MHR forward with the pack's (updated) params and rewrite vertices
    and joints. Verified to reproduce the original pack to ~1mm when expr is
    unchanged; the y/z flip matches the estimator's camera convention."""
    import torch
    from sam_3d_body import load_sam_3d_body

    if not args.checkpoint_path:
        print("WORKER_WARN: no checkpoint for re-skinning; expressions saved but mesh unchanged.", file=sys.stderr, flush=True)
        return False
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model, _ = load_sam_3d_body(args.checkpoint_path, device=device, mhr_path=args.mhr_path or "")
    head = model.head_pose

    present = np.asarray(pack["present"]).astype(bool)
    n_frames, n_people = present.shape
    verts_out = np.asarray(pack["pred_vertices"], dtype=np.float32)
    joints_out = np.asarray(pack["pred_joint_coords"], dtype=np.float32)

    def to_t(key: str, sel: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.asarray(pack[key], dtype=np.float32)[sel]).to(device)

    flat = [(f, p) for f in range(n_frames) for p in range(n_people) if present[f, p]]
    for start in range(0, len(flat), chunk):
        batch = flat[start : start + chunk]
        sel = (np.array([f for f, _ in batch]), np.array([p for _, p in batch]))
        with torch.no_grad():
            verts, joints = head.mhr_forward(
                global_trans=torch.zeros(len(batch), 3, device=device),
                global_rot=to_t("global_rot", sel),
                body_pose_params=to_t("body_pose_params", sel),
                hand_pose_params=to_t("hand_pose_params", sel),
                scale_params=to_t("scale_params", sel),
                shape_params=to_t("shape_params", sel),
                expr_params=to_t("expr_params", sel),
                return_joint_coords=True,
            )
        verts = verts.float().cpu().numpy()
        joints = joints.float().cpu().numpy()
        verts[..., 1:] *= -1.0  # camera-space y-down/z-forward convention
        joints[..., 1:] *= -1.0
        for row, (f, p) in enumerate(batch):
            verts_out[f, p] = verts[row]
            joints_out[f, p] = joints[row]
        print(f"reskinned {min(start + chunk, len(flat))}/{len(flat)}", flush=True)
    pack["pred_vertices"] = verts_out
    pack["pred_joint_coords"] = joints_out
    return True


def _head_crop(rgb: np.ndarray, bbox: np.ndarray | None) -> tuple[np.ndarray, bool]:
    """Upper 40% of the person bbox, expanded 1.4x. MediaPipe's face detector
    downsamples full frames so far that distant faces vanish; a tight head crop
    is what makes per-frame detection reliable."""
    height, width = rgb.shape[:2]
    if bbox is None:
        return rgb, False
    x1, y1, x2, y2 = (float(v) for v in bbox)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return rgb, False
    head_y2 = y1 + 0.4 * (y2 - y1)
    cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + head_y2)
    half_w, half_h = 0.7 * (x2 - x1), 0.7 * (head_y2 - y1)
    cx1, cy1 = max(0, int(cx - half_w)), max(0, int(cy - half_h))
    cx2, cy2 = min(width, int(cx + half_w)), min(height, int(cy + half_h))
    if cx2 - cx1 < 16 or cy2 - cy1 < 16:
        return rgb, False
    return np.ascontiguousarray(rgb[cy1:cy2, cx1:cx2]), True


def cmd_face(args: argparse.Namespace) -> None:
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError("mediapipe is not installed in the SAM 3D Body environment.") from exc

    pack = load_pose_pack(Path(args.pose_path))
    frames, _, _ = load_frames(Path(args.input), max_frames=args.max_frames)
    present = np.asarray(pack["present"])
    n_frames = min(int(pack["meta"]["n_frames"]), len(frames))
    n_people = present.shape[1]
    bboxes = np.asarray(pack["bbox"], dtype=np.float32) if "bbox" in pack else None  # (F, P, 4)

    from mediapipe.tasks.python.core.base_options import BaseOptions
    from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

    # Per-frame named coefficients per person slot; NaN rows mean "no face seen".
    names: list[str] = []
    coefs = np.full((n_frames, n_people, 52), np.nan, dtype=np.float32)
    landmarker = None
    try:
        model_path = args.face_model_path or _ensure_face_landmarker_model()
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.IMAGE,
            output_face_blendshapes=True,
            num_faces=1,
        )
        landmarker = FaceLandmarker.create_from_options(options)
        for idx in range(n_frames):
            rgb = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2RGB)
            for person_idx in range(n_people):
                if not present[idx, person_idx]:
                    continue
                bbox = bboxes[idx, person_idx] if bboxes is not None else None
                crop, cropped = _head_crop(rgb, bbox)
                result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=crop))
                if not result.face_blendshapes and cropped:
                    result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                if not result.face_blendshapes:
                    continue
                face = result.face_blendshapes[0]
                if not names:
                    names = [cat.category_name for cat in face]
                scores = np.array([cat.score for cat in face], dtype=np.float32)
                coefs[idx, person_idx, : min(52, scores.size)] = scores[:52]
            if (idx + 1) % 25 == 0:
                print(f"face_frame {idx + 1}/{n_frames}", flush=True)
    finally:
        if landmarker is not None:
            landmarker.close()

    detected = int(np.sum(~np.isnan(coefs[:, :, 0])))
    print(f"face detections: {detected}/{n_frames * n_people} frame-person slots", flush=True)
    strength = float(getattr(args, "face_strength", 1.0) or 1.0)
    expr = np.asarray(pack["expr_params"], dtype=np.float32).copy()
    n_axes = expr.shape[-1]
    any_face = False
    for p in range(n_people):
        series = coefs[:, p, :]
        valid = ~np.isnan(series[:, 0])
        if not valid.any():
            continue
        any_face = True
        # Interpolate gaps so brief misses don't snap the face to neutral.
        idx_all = np.arange(n_frames)
        for c in range(series.shape[1]):
            series[:, c] = np.interp(idx_all, idx_all[valid], series[valid, c])
        # Per-clip baseline: MediaPipe rests some shapes well above zero for
        # some faces; subtract the 5th percentile so neutral stays neutral.
        baseline = np.percentile(series, 5.0, axis=0)
        series = np.clip(series - baseline[None, :], 0.0, None)
        series = _smooth_series(series, int(args.window or 7))
        for f in range(n_frames):
            named = {names[c]: float(series[f, c]) for c in range(min(len(names), series.shape[1]))}
            expr[f, p] = _map_blendshapes_to_expr(named, strength, n_axes)

    pack["face_blendshapes"] = np.nan_to_num(coefs, nan=0.0)
    reskinned = False
    if any_face:
        pack["expr_params"] = expr
        reskinned = _reskin_pack(pack, args)
    else:
        print("WORKER_WARN: no faces detected; expressions unchanged.", file=sys.stderr, flush=True)
    output_path = Path(args.output or args.pose_path)
    save_pose_pack(output_path, pack)
    emit_result({"pose_path": str(output_path), "face_detections": detected, "reskinned": bool(reskinned)})


FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)


def _ensure_face_landmarker_model() -> str:
    """Download Google's Face Landmarker .task model once and cache it beside the checkpoints."""
    cache_dir = Path.cwd() / "checkpoints"  # worker always runs with cwd=repo_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / "face_landmarker.task"
    if target.exists() and target.stat().st_size > 1_000_000:
        return str(target)
    import urllib.request

    print(f"Downloading MediaPipe Face Landmarker model to {target}...", flush=True)
    tmp = target.with_suffix(".task.part")
    urllib.request.urlretrieve(FACE_LANDMARKER_URL, tmp)
    tmp.replace(target)
    return str(target)


def _animated_glb(
    person_seqs: list[np.ndarray],
    faces: np.ndarray,
    colors: np.ndarray | None,
    fps: float,
    path: Path,
) -> Path:
    """Write a GLB with the whole mesh sequence baked in as morph-target keyframes.

    One node per person. Frame 0 is the base mesh; every later frame is a morph
    target driven by a one-hot STEP weight track, so Blender imports the file as
    shape keys with keyframes and plays the animation back directly. Vertex
    colors ride along as COLOR_0.
    """
    import struct as _struct

    path.parent.mkdir(parents=True, exist_ok=True)
    bin_parts: list[bytes] = []
    views: list[dict[str, Any]] = []
    accessors: list[dict[str, Any]] = []
    size = 0

    def add_view(data: bytes) -> int:
        nonlocal size
        if size % 4:
            pad = b"\x00" * (4 - size % 4)
            bin_parts.append(pad)
            size += len(pad)
        views.append({"buffer": 0, "byteOffset": size, "byteLength": len(data)})
        bin_parts.append(data)
        size += len(data)
        return len(views) - 1

    FLOAT, UINT = 5126, 5125

    def add_accessor(arr: np.ndarray, comp_type: int, type_: str, with_bounds: bool = False) -> int:
        acc: dict[str, Any] = {
            "bufferView": add_view(arr.tobytes()),
            "componentType": comp_type,
            "count": int(arr.shape[0]),
            "type": type_,
        }
        if with_bounds:
            lo = arr.min(axis=0) if arr.ndim > 1 else [arr.min()]
            hi = arr.max(axis=0) if arr.ndim > 1 else [arr.max()]
            acc["min"] = [float(v) for v in lo]
            acc["max"] = [float(v) for v in hi]
        accessors.append(acc)
        return len(accessors) - 1

    idx_acc = add_accessor(np.ascontiguousarray(faces, dtype="<u4").reshape(-1), UINT, "SCALAR")
    color_acc = None
    if colors is not None:
        color_acc = add_accessor(np.clip(np.ascontiguousarray(colors, dtype="<f4"), 0.0, 1.0), FLOAT, "VEC3")

    n_frames = int(person_seqs[0].shape[0])
    times_acc = None
    if n_frames > 1:
        times = (np.arange(n_frames, dtype="<f4") / max(fps, 1.0)).astype("<f4")
        times_acc = add_accessor(times, FLOAT, "SCALAR", with_bounds=True)

    meshes: list[dict[str, Any]] = []
    nodes: list[dict[str, Any]] = []
    channels: list[dict[str, Any]] = []
    samplers: list[dict[str, Any]] = []
    for p_idx, seq in enumerate(person_seqs):
        base = np.ascontiguousarray(seq[0], dtype="<f4")
        attrs: dict[str, int] = {"POSITION": add_accessor(base, FLOAT, "VEC3", with_bounds=True)}
        if color_acc is not None:
            attrs["COLOR_0"] = color_acc
        prim: dict[str, Any] = {"attributes": attrs, "indices": idx_acc, "material": 0, "mode": 4}
        mesh: dict[str, Any] = {"name": f"sam3d_body_{p_idx}", "primitives": [prim]}
        if n_frames > 1:
            targets = []
            for f in range(1, n_frames):
                delta = np.ascontiguousarray(seq[f] - seq[0], dtype="<f4")
                targets.append({"POSITION": add_accessor(delta, FLOAT, "VEC3", with_bounds=True)})
            prim["targets"] = targets
            mesh["weights"] = [0.0] * (n_frames - 1)
            # Blender reads shape-key names from mesh.extras.targetNames.
            mesh["extras"] = {"targetNames": [f"frame_{f:04d}" for f in range(1, n_frames)]}
            weights = np.zeros((n_frames, n_frames - 1), dtype="<f4")
            for f in range(1, n_frames):
                weights[f, f - 1] = 1.0
            out_acc = add_accessor(weights.reshape(-1), FLOAT, "SCALAR")
            samplers.append({"input": times_acc, "output": out_acc, "interpolation": "STEP"})
            channels.append({"sampler": len(samplers) - 1, "target": {"node": p_idx, "path": "weights"}})
        meshes.append(mesh)
        nodes.append({"mesh": p_idx, "name": f"SAM3DBody_{p_idx}"})

    gltf: dict[str, Any] = {
        "asset": {"version": "2.0", "generator": "GTN SAM 3D Body"},
        "scene": 0,
        "scenes": [{"nodes": list(range(len(nodes)))}],
        "nodes": nodes,
        "meshes": meshes,
        "materials": [
            {
                "name": "sam3d_body",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.6,
                },
                "doubleSided": True,
            }
        ],
        "accessors": accessors,
        "bufferViews": views,
    }
    if channels:
        gltf["animations"] = [{"name": "sam3d_body", "channels": channels, "samplers": samplers}]

    bin_blob = b"".join(bin_parts)
    if len(bin_blob) % 4:
        bin_blob += b"\x00" * (4 - len(bin_blob) % 4)
    gltf["buffers"] = [{"byteLength": len(bin_blob)}]
    json_blob = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    if len(json_blob) % 4:
        json_blob += b" " * (4 - len(json_blob) % 4)
    with path.open("wb") as handle:
        handle.write(_struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(json_blob) + 8 + len(bin_blob)))
        handle.write(_struct.pack("<II", len(json_blob), 0x4E4F534A))
        handle.write(json_blob)
        handle.write(_struct.pack("<II", len(bin_blob), 0x004E4942))
        handle.write(bin_blob)
    return path


def _joints_to_bvh(joints_seq: np.ndarray, present: np.ndarray, fps: float, path: Path) -> Path:
    """Write a star-hierarchy BVH from MHR joints so Blender can import the mocap."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n_frames, _, n_joints, _ = joints_seq.shape
    names = ["body_world"] + [f"joint_{idx:03d}" for idx in range(1, n_joints)]
    lines = ["HIERARCHY", f"ROOT {names[0]}", "{", "  OFFSET 0.00 0.00 0.00", "  CHANNELS 3 Xposition Yposition Zposition"]
    for name in names[1:]:
        lines.extend(
            [
                f"  JOINT {name}",
                "  {",
                "    OFFSET 0.00 0.00 0.00",
                "    CHANNELS 3 Xposition Yposition Zposition",
                "    End Site",
                "    {",
                "      OFFSET 0.00 1.00 0.00",
                "    }",
                "  }",
            ]
        )
    lines.append("}")
    lines.extend(["MOTION", f"Frames: {n_frames}", f"Frame Time: {1.0 / max(fps, 1.0):.6f}"])
    for frame_idx in range(n_frames):
        person_idx = int(np.argmax(present[frame_idx])) if present[frame_idx].any() else 0
        joints = joints_seq[frame_idx, person_idx]
        values: list[str] = []
        for joint in joints:
            values.extend([f"{float(joint[0]):.6f}", f"{float(joint[1]):.6f}", f"{float(joint[2]):.6f}"])
        lines.append(" ".join(values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def cmd_export(args: argparse.Namespace) -> None:
    pack = load_pose_pack(Path(args.pose_path))
    present = np.asarray(pack["present"]).astype(bool)
    faces = np.asarray(pack.get("faces", np.zeros((0, 3), dtype=np.int32)))
    verts = np.asarray(pack["pred_vertices"], dtype=np.float32)
    cam_t = np.asarray(pack["pred_cam_t"], dtype=np.float32)
    joints = np.asarray(pack["pred_joint_coords"], dtype=np.float32)
    colors = np.asarray(pack["vertex_colors"], dtype=np.float32) if "vertex_colors" in pack else None
    fps = float(pack["meta"].get("fps", 24.0))

    # Same world transform as the 3D viewer: predictions are in camera space
    # (Y down, Z forward), so flip Y/Z for a Y-up scene and drop the take onto
    # the floor. Without this, DCC imports come in upside down.
    flip = np.array([1.0, -1.0, -1.0], dtype=np.float32)
    world_v = (verts + cam_t[:, :, None, :]) * flip
    world_j = (joints + cam_t[:, :, None, :]) * flip
    n_frames, max_people = present.shape[:2]
    ys = [world_v[f, p, :, 1].min() for f in range(n_frames) for p in range(max_people) if present[f, p]]
    if ys:
        ground = float(np.percentile(np.asarray(ys, dtype=np.float32), 5.0))
        world_v[..., 1] -= ground
        world_j[..., 1] -= ground

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {"output_dir": str(output_dir)}
    fmt = str(args.format).lower()
    if fmt in {"glb", "both"}:
        if faces.size == 0:
            print("WORKER_WARN: pose pack has no mesh faces; skipping GLB export.", file=sys.stderr, flush=True)
        else:
            person_seqs: list[np.ndarray] = []
            for p in range(max_people):
                if not present[:, p].any():
                    continue
                seq = world_v[:, p].copy()
                # Hold the last seen pose through detection gaps so every
                # morph target references a sensible mesh.
                last = seq[int(np.argmax(present[:, p]))]
                for f in range(n_frames):
                    if present[f, p]:
                        last = seq[f]
                    else:
                        seq[f] = last
                person_seqs.append(np.ascontiguousarray(seq, dtype=np.float32))
            glb_path = _animated_glb(person_seqs, faces, colors, fps, output_dir / "sam3d_body.glb")
            result["glb_path"] = str(glb_path)
    if fmt in {"bvh", "both"}:
        bvh_path = _joints_to_bvh(world_j, present, fps, output_dir / "sam3d_body.bvh")
        result["bvh_path"] = str(bvh_path)
    emit_result(result)


def _lock_feet_in_pack(pose_path: Path) -> None:
    """Pin planted feet: remove the slide/bob of the body while a foot is on the floor.

    SAM 3D Body estimates every frame independently, so absolute position
    (especially depth and height) jitters frame to frame - visually the planted
    foot skates and flaps. This detects frames where the body's lowest vertex
    cluster is in ground contact, holds that cluster's position fixed for the
    duration of each contact, and applies the correction as a smooth rigid
    shift of pred_cam_t (so mesh AND joints move together). Airborne frames
    (jumps) keep their trajectory; corrections are interpolated through them.

    Runs AFTER the overlay render on purpose: the overlay must stay aligned to
    the source pixels, while the viewer/GLB/BVH want stable world motion.
    """
    pack = load_pose_pack(pose_path)
    present = np.asarray(pack["present"]).astype(bool)
    verts = np.asarray(pack["pred_vertices"], dtype=np.float32)
    cam_t = np.asarray(pack["pred_cam_t"], dtype=np.float32)
    n_frames, max_people = present.shape[:2]
    if n_frames < 5:
        return
    flip = np.array([1.0, -1.0, -1.0], dtype=np.float32)
    contact_height = 0.05  # lowest point within 5 cm of floor = contact
    cluster_band = 0.03  # vertices within 3 cm of the lowest point = the planted cluster
    switch_dist = 0.10  # cluster centroid jumping >10 cm = weight moved to the other foot
    max_correction = 0.15

    for person in range(max_people):
        if int(present[:, person].sum()) < 5:
            continue
        world = (verts[:, person] + cam_t[:, person][:, None, :]) * flip  # (T, V, 3)
        y_min = world[:, :, 1].min(axis=1)
        valid = present[:, person]
        ground = float(np.percentile(y_min[valid], 5.0))
        rel = y_min - ground
        contact = (rel < contact_height) & valid

        centroid = np.full((n_frames, 2), np.nan, dtype=np.float32)  # (x, z) of planted cluster
        for f in np.where(contact)[0]:
            sel = world[f, :, 1] < (y_min[f] + cluster_band)
            centroid[f, 0] = float(world[f, sel, 0].mean())
            centroid[f, 1] = float(world[f, sel, 2].mean())

        # Target correction per frame, NaN where we have no contact info.
        delta = np.full((n_frames, 3), np.nan, dtype=np.float32)
        f = 0
        while f < n_frames:
            if not contact[f]:
                f += 1
                continue
            start = f
            while (
                f + 1 < n_frames
                and contact[f + 1]
                and abs(centroid[f + 1, 0] - centroid[f, 0]) < switch_dist
                and abs(centroid[f + 1, 1] - centroid[f, 1]) < switch_dist
            ):
                f += 1
            end = f
            f += 1
            if end - start + 1 < 3:
                continue  # too short to be a confident plant
            seg = slice(start, end + 1)
            anchor_x = float(np.median(centroid[seg, 0]))
            anchor_z = float(np.median(centroid[seg, 1]))
            delta[seg, 0] = anchor_x - centroid[seg, 0]
            delta[seg, 1] = -rel[seg]  # planted cluster sits flush on the floor
            delta[seg, 2] = anchor_z - centroid[seg, 1]

        if not np.isfinite(delta).any():
            continue
        np.clip(delta, -max_correction, max_correction, out=delta)
        # Fill gaps (airborne / no-contact stretches) by interpolating between
        # neighboring plants, then low-pass the whole track so the correction
        # itself can never pop.
        idx = np.arange(n_frames, dtype=np.float32)
        for c in range(3):
            col = delta[:, c]
            known = np.isfinite(col)
            col[:] = np.interp(idx, idx[known], col[known])
        kernel = np.exp(-0.5 * (np.arange(-2, 3) / 1.0) ** 2).astype(np.float32)
        kernel /= kernel.sum()
        for c in range(3):
            delta[:, c] = np.convolve(np.pad(delta[:, c], 2, mode="edge"), kernel, mode="valid")

        cam_t[:, person] += delta * flip  # world-space fix -> camera-space storage
    pack["pred_cam_t"] = cam_t
    save_pose_pack(pose_path, pack)


def cmd_smooth(args: argparse.Namespace) -> None:
    pack = load_pose_pack(Path(args.pose_path))
    strength = float(args.strength)
    window = int(args.window)
    if window % 2 == 0:
        window += 1
    if strength <= 0 or window <= 1 or int(pack["meta"]["n_frames"]) < 2:
        output_path = Path(args.output or args.pose_path)
        save_pose_pack(output_path, pack)
        emit_result({"pose_path": str(output_path)})
        return
    present = np.asarray(pack["present"])
    radius = window // 2
    if args.method == "savgol":
        try:
            from scipy.signal import savgol_filter
        except ImportError:
            savgol_filter = None
    else:
        savgol_filter = None
    sigma = max(0.5, radius / 2.0)
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    kernel = kernel / kernel.sum()
    for key in ("pred_vertices", "pred_joint_coords", "pred_cam_t", "pred_keypoints_2d", "pred_keypoints_3d"):
        if key not in pack:
            continue
        data = np.asarray(pack[key], dtype=np.float32)
        smoothed = data.copy()
        n_frames = data.shape[0]
        for person_idx in range(data.shape[1]):
            mask = present[:, person_idx]
            if int(mask.sum()) < 2:
                continue
            series = data[:, person_idx]

            def apply_filter(x: np.ndarray) -> np.ndarray:
                if savgol_filter is not None and n_frames >= window:
                    poly = 2 if window > 2 else 1
                    wl = min(window, n_frames if n_frames % 2 else n_frames - 1 or 3)
                    return savgol_filter(x, window_length=wl, polyorder=poly, axis=0)
                padded = np.pad(x, ((radius, radius),) + ((0, 0),) * (x.ndim - 1), mode="edge")
                out = np.zeros_like(x)
                for frame_idx in range(n_frames):
                    out[frame_idx] = np.tensordot(kernel, padded[frame_idx : frame_idx + window], axes=(0, 0))
                return out

            # strength = number of filter passes. 1 = one pass, 3 = three passes
            # (progressively smoother), 0.5 = half-blend of one pass. The old
            # linear blend extrapolated past the filtered curve for strength > 1,
            # which AMPLIFIED jitter instead of removing it.
            filtered = series
            for _ in range(int(strength)):
                filtered = apply_filter(filtered)
            frac = strength - int(strength)
            if frac > 1e-6:
                filtered = filtered * (1.0 - frac) + apply_filter(filtered) * frac
            smoothed[:, person_idx] = np.where(mask.reshape((-1,) + (1,) * (series.ndim - 1)), filtered, series)
        pack[key] = smoothed
    output_path = Path(args.output or args.pose_path)
    save_pose_pack(output_path, pack)
    emit_result({"pose_path": str(output_path)})


def cmd_pipeline(args: argparse.Namespace) -> None:
    import time

    stage_times: list[tuple[str, float]] = []

    def timed(label: str, fn: Callable[[], None]) -> None:
        started = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - started
        stage_times.append((label, elapsed))
        print(f"STAGE_TIME {label}: {elapsed:.1f}s", flush=True)

    out_dir = Path(args.output)
    if out_dir.suffix:
        out_dir = out_dir.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    pose_path = out_dir / "pose_pack.npz"
    args.output = str(pose_path)
    timed("predict", lambda: cmd_predict(args))
    args.pose_path = str(pose_path)
    # Expressions before smoothing: re-skinning rebuilds vertices from raw
    # params, so smoothing must come after to keep its effect on the mesh.
    if getattr(args, "face_expressions", False):
        try:
            args.output = str(pose_path)
            timed("face_expressions", lambda: cmd_face(args))
        except Exception as exc:  # noqa: BLE001 - expressions are optional polish
            print(f"WORKER_WARN: face expression step failed ({exc}); continuing without it.", file=sys.stderr, flush=True)
    if args.smooth:
        args.output = str(pose_path)
        timed("smooth", lambda: cmd_smooth(args))
    overlay_path = out_dir / "overlay.mp4"
    args.output = str(overlay_path)
    timed("render_overlay", lambda: cmd_render(args))
    if getattr(args, "foot_lock", False):
        try:
            timed("foot_lock", lambda: _lock_feet_in_pack(pose_path))
        except Exception as exc:  # noqa: BLE001 - stabilization is optional polish
            print(f"WORKER_WARN: foot locking failed ({exc}); continuing without it.", file=sys.stderr, flush=True)
    args.output = str(out_dir)
    timed("export", lambda: cmd_export(args))
    total = sum(t for _, t in stage_times)
    summary = ", ".join(f"{label} {elapsed:.0f}s" for label, elapsed in stage_times)
    print(f"STAGE_TIME total: {total:.1f}s ({summary})", flush=True)
    glb_path = out_dir / "sam3d_body.glb"
    if not glb_path.exists():
        glb_path = out_dir / "sam3d_body.obj"
    bvh_path = out_dir / "sam3d_body.bvh"
    emit_result(
        {
            "pose_path": str(pose_path),
            "overlay_path": str(overlay_path),
            "output_dir": str(out_dir),
            "glb_path": str(glb_path) if glb_path.exists() else "",
            "bvh_path": str(bvh_path) if bvh_path.exists() else "",
        }
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GTN SAM 3D Body worker")
    parser.add_argument("command", choices=["download", "detect", "fov", "predict", "smooth", "face", "render", "export", "pipeline", "serve"])
    parser.add_argument("--input", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--pose-path", dest="pose_path", default="")
    parser.add_argument("--boxes-path", dest="boxes_path", default="")
    parser.add_argument("--checkpoint-path", dest="checkpoint_path", default="")
    parser.add_argument("--checkpoint-dir", dest="checkpoint_dir", default="")
    parser.add_argument("--mhr-path", dest="mhr_path", default="")
    parser.add_argument("--hf-repo", dest="hf_repo", default="facebook/sam-3d-body-dinov3")
    parser.add_argument("--detector-name", dest="detector_name", default="vitdet")
    parser.add_argument("--segmentor-name", dest="segmentor_name", default="")
    parser.add_argument("--fov-name", dest="fov_name", default="")
    parser.add_argument("--detector-path", dest="detector_path", default="")
    parser.add_argument("--segmentor-path", dest="segmentor_path", default="")
    parser.add_argument("--fov-path", dest="fov_path", default="")
    parser.add_argument("--bbox-thresh", dest="bbox_thresh", type=float, default=0.8)
    parser.add_argument("--use-mask", dest="use_mask", action="store_true")
    parser.add_argument("--run-hand-refinement", dest="run_hand_refinement", action="store_true")
    parser.add_argument("--fov", type=float, default=0.0)
    parser.add_argument("--fallback-fov", dest="fallback_fov", type=float, default=53.0)
    parser.add_argument("--max-frames", dest="max_frames", type=int, default=-1)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--method", default="savgol")
    parser.add_argument("--window", type=int, default=7)
    parser.add_argument("--format", default="both")
    parser.add_argument("--frame-index", dest="frame_index", type=int, default=0)
    parser.add_argument("--overlay-only", dest="overlay_only", action="store_true")
    parser.add_argument("--smooth", action="store_true")
    parser.add_argument("--face-expressions", dest="face_expressions", action="store_true")
    parser.add_argument("--face-strength", dest="face_strength", type=float, default=1.0)
    parser.add_argument("--face-model-path", dest="face_model_path", default="")
    parser.add_argument("--precision", default="fp32", choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=16)
    parser.add_argument("--foot-lock", dest="foot_lock", action="store_true")
    parser.add_argument("--daemon-state", dest="daemon_state", default="")
    return parser


class _JsonLineWriter:
    """stdout/stderr replacement that ships output to the daemon client as
    {"out": "..."} JSON lines, so log text can never be confused with the
    framing protocol."""

    def __init__(self, stream: Any) -> None:
        self._stream = stream
        self._buffer = ""

    def write(self, text: str) -> int:
        self._buffer += str(text)
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._send(line)
        return len(text)

    def flush(self) -> None:
        if self._buffer:
            self._send(self._buffer)
            self._buffer = ""
        try:
            self._stream.flush()
        except Exception:  # noqa: BLE001 - client may have disconnected
            pass

    def _send(self, line: str) -> None:
        try:
            self._stream.write((json.dumps({"out": line}) + "\n").encode("utf-8"))
            self._stream.flush()
        except Exception:  # noqa: BLE001 - keep the job running even if the client dropped
            pass


DAEMON_IDLE_SECONDS = 30 * 60


def cmd_serve(args: argparse.Namespace) -> None:
    """Warm worker daemon: keeps the model in VRAM between runs.

    Listens on localhost, one job at a time. Protocol: client sends one JSON
    line {"token", "args", "env"}; output streams back as {"out": line} JSON
    lines followed by an {"exit": code} sentinel. Exits after 30 idle minutes
    or when this script file changes on disk (client respawns a fresh one).
    """
    import contextlib
    import secrets
    import socket
    import traceback

    state_path = Path(args.daemon_state or (Path.cwd() / ".gtn_worker_daemon.json"))
    token = secrets.token_hex(16)
    script_mtime = os.path.getmtime(os.path.abspath(__file__))

    server = socket.create_server(("127.0.0.1", 0))
    port = server.getsockname()[1]
    server.settimeout(15.0)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({"port": port, "token": token, "pid": os.getpid()}), encoding="utf-8")
    print(f"GTN_SAM3D_DAEMON: listening on 127.0.0.1:{port}", flush=True)

    idle_deadline = time.time() + DAEMON_IDLE_SECONDS
    try:
        while True:
            try:
                conn, _addr = server.accept()
            except TimeoutError:
                if time.time() > idle_deadline:
                    break
                continue
            except OSError:
                continue
            with conn:
                conn.settimeout(None)
                stream = conn.makefile("rwb")
                try:
                    request = json.loads(stream.readline().decode("utf-8"))
                except Exception:  # noqa: BLE001
                    continue
                if request.get("token") != token:
                    continue
                if os.path.getmtime(os.path.abspath(__file__)) != script_mtime:
                    # Worker code changed on disk: tell the client to respawn.
                    stream.write((json.dumps({"exit": -42, "stale": True}) + "\n").encode("utf-8"))
                    stream.flush()
                    return
                for key, value in (request.get("env") or {}).items():
                    os.environ[str(key)] = str(value)
                writer = _JsonLineWriter(stream)
                exit_code = 0
                with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):  # type: ignore[arg-type]
                    try:
                        job_args = build_parser().parse_args([str(a) for a in request.get("args") or []])
                        if job_args.command == "serve":
                            raise ValueError("Nested serve is not allowed.")
                        COMMANDS[job_args.command](job_args)
                    except SystemExit as se:
                        exit_code = int(se.code or 0)
                    except Exception as exc:  # noqa: BLE001
                        print(f"WORKER_ERROR: {exc}", flush=True)
                        traceback.print_exc()
                        exit_code = 1
                    finally:
                        writer.flush()
                try:
                    stream.write((json.dumps({"exit": exit_code}) + "\n").encode("utf-8"))
                    stream.flush()
                except Exception:  # noqa: BLE001
                    pass
            idle_deadline = time.time() + DAEMON_IDLE_SECONDS
    finally:
        try:
            state_path.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass


COMMANDS = {
    "download": cmd_download,
    "detect": cmd_detect,
    "fov": cmd_fov,
    "predict": cmd_predict,
    "smooth": cmd_smooth,
    "face": cmd_face,
    "render": cmd_render,
    "export": cmd_export,
    "pipeline": cmd_pipeline,
    "serve": cmd_serve,
}


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        COMMANDS[args.command](args)
    except Exception as exc:
        print(f"WORKER_ERROR: {exc}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()
