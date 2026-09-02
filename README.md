# GTN SAM 3D Body

Turn a video of a person into a **3D body animation** inside [Griptape Nodes](https://www.griptapenodes.com/) — powered by Meta's [SAM 3D Body](https://github.com/facebookresearch/sam-3d-body) (the same model behind the ComfyUI SAM 3D Body graph, Kijai / Comfy-Org).

You get:

- a **mesh-overlay video** (recovered body composited over the source),
- an **interactive 3D viewer** node — orbit / pan / zoom, play & scrub, fullscreen, rainbow body colors + MediaPipe face expressions,
- a **3D face close-up viewer** — a virtual camera rigidly attached to the head, like a VFX helmet-mounted mocap cam (with a body on/off toggle),
- **GLB / BVH export** — animated (morph targets), vertex-colored, Y-up and grounded, ready for Blender or Unreal.

Inference runs in its own isolated CUDA venv; Griptape Nodes only orchestrates. A warm worker daemon keeps the model in VRAM between runs, so repeat runs skip the ~20 s model load.

## Install

1. Register `sam3d_body_nodes/griptape_nodes_library.json` in Griptape Nodes → Settings → Libraries.
2. Accept the gated model license: [facebook/sam-3d-body-dinov3](https://huggingface.co/facebook/sam-3d-body-dinov3).
3. Create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and save it in Griptape secrets as `HF_TOKEN`.
4. Drop the **Instructions** node for the step-by-step guide, then run **SAM 3D Body Setup** once (10–20 min: clones Meta's repo, builds the venv, downloads checkpoints).
5. Drop **Build SAM 3D Body Graph** and click the button — it spawns the complete wired graph (video in → overlay video + 3D viewer + face close-up) inside a group and removes itself.

## Nodes

**SAM 3D Body** (main)

| Node | What it does |
| --- | --- |
| **Instructions** | Quick-start guide as a note on the canvas |
| **Build SAM 3D Body Graph** | One click spawns the full ready-to-run graph, then deletes itself |
| **SAM 3D Body Setup** | One-time install: repo, CUDA venv, checkpoints |
| **SAM 3D Body (One-Click)** | The whole pipeline in one node: predict → face expressions → smooth → overlay render → GLB/BVH export |
| **3D Body Viewer** | Interactive three.js playback of the colored body animation |
| **3D Face Close-Up Viewer** | Head-locked mocap cam framing the face every frame |

**SAM 3D Body – Advanced** (every pipeline stage as its own node, for custom graphs): Load Model, Detect Person Boxes (YOLO), Track People Across Frames, Estimate Camera FoV (MoGe), Predict 3D Body Pose, Smooth Pose Animation, Add Face Expressions, Render Mesh Overlay Video, Export Animation (GLB / BVH).

## Speed & quality knobs

- `batch_size` — frames per GPU batch (16–32 is good).
- `precision` — `bf16` is the fast default; `fp32` if you see artifacts.
- `run_hand_refinement` — ~4x runtime; leave off unless you need perfect fingers.
- `smooth_strength` / `smooth_window` — raise to 2–3 / 9–13 for jittery clips.
- `foot_lock` — keeps planted feet from sliding.
- `face_expressions` + `face_strength` — MediaPipe-driven mouth/brow/eye motion.
- Multi-person clips work — add the tracking node (advanced) for stable IDs.

## Outputs

Saved to the Griptape workspace/project outputs folder under `sam3d_body/`:

- `pose_pack.npz` — MHR vertices, joints, camera, vertex colors, blendshapes
- overlay `mp4` — mesh composited on the source video (H.264)
- `sam3d_body.glb` — animated, vertex-colored mesh (morph-target animation)
- `sam3d_body.bvh` — joint mocap for Blender / retargeting

## Requirements

- Windows / Linux with an NVIDIA GPU (CUDA)
- Git
- Hugging Face access to `facebook/sam-3d-body-dinov3`

## Upstream

- Meta model: https://github.com/facebookresearch/sam-3d-body
- ComfyUI native PR: https://github.com/Comfy-Org/ComfyUI/pull/14370
- Comfy-Org weights: https://huggingface.co/Comfy-Org/sam-3d-body
