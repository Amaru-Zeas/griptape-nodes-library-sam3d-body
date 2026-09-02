from __future__ import annotations

from typing import Any

from griptape_nodes.exe_types.node_types import BaseNode
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString

INSTRUCTIONS = """# SAM 3D Body - Quick Start

Turn a video of a person into a 3D body animation.

**1. Hugging Face token (one time)**
Accept the model license here:
[huggingface.co/facebook/sam-3d-body-dinov3](https://huggingface.co/facebook/sam-3d-body-dinov3)
Then create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
and paste it into Settings -> Secrets -> `HF_TOKEN`.
(The FoV helper model [Ruicheng/moge-2-vitl-normal](https://huggingface.co/Ruicheng/moge-2-vitl-normal)
downloads with the same token.)

**2. Run "SAM 3D Body Setup" (one time)**
Installs Meta's model in its own isolated environment and downloads the checkpoints.
Takes 10-20 minutes and needs an NVIDIA GPU. Watch the logs on the node.

**3. Drop "Build SAM 3D Body Graph" and click the button**
It spawns a ready-wired graph (video in -> overlay video + 3D body viewer +
3D face close-up viewer) and removes itself.

**4. Load a video and run**
Pick a clip with one clearly visible person, then run the One-Click node.

**You get:** a mesh-overlay video, an interactive 3D viewer (orbit / pan / zoom,
fullscreen, rainbow body colors + face expressions), a face mocap close-up viewer
(camera locked to the head, like a VFX head-cam), and GLB / BVH files for Blender
or Unreal.

**Speed**
- The first run loads the model (~20 s extra); after that a warm worker keeps it
  in VRAM and repeat runs start instantly.
- `batch_size` = frames per GPU batch (16-32 is good). `precision` bf16 is the
  fast default.
- `run_hand_refinement` costs ~4x runtime - leave it off unless you need
  perfect fingers.

**Viewers**
- Fullscreen button on every viewer; "Face" button switches any viewer to the
  head-locked mocap cam (camera follows the head, mouse control disabled);
  the "body" checkbox hides everything below the neck.
- Normal mode: drag = orbit, middle-mouse or Shift+drag = pan, wheel = zoom.

**Tips**
- Jittery result? Raise `smooth_strength` to 2-3 and `smooth_window` to 9-13.
  Keep `foot_lock` on to stop foot sliding.
- Subtle face? Raise `face_strength` to 1.5-2.
- Advanced folder has every pipeline stage if you want a custom graph.
"""


class SAM3DBodyInstructionsNode(BaseNode):
    """Read-me note: how to set up and use the SAM 3D Body library."""

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)
        # Default to a size that shows the whole guide without scrolling.
        if "size" not in self.metadata:
            self.metadata["size"] = {"width": 680, "height": 1230}
        self.add_parameter(
            ParameterString(
                name="note",
                default_value=INSTRUCTIONS,
                allow_input=False,
                allow_property=True,
                allow_output=False,
                multiline=True,
                markdown=True,
                is_full_width=True,
                tooltip="How to set up and use the SAM 3D Body library.",
            )
        )

    def process(self) -> None:
        pass
