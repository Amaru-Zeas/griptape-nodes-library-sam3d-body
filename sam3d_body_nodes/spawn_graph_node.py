from __future__ import annotations

import uuid
from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, ControlNode
from griptape_nodes.exe_types.param_components.log_parameter import LogParameter
from griptape_nodes.traits.options import Options

try:
    from griptape_nodes.exe_types.core_types import ParameterButtonGroup
    from griptape_nodes.exe_types.param_types.parameter_button import ParameterButton

    _HAS_BUTTONS = True
except ImportError:
    ParameterButtonGroup = None  # type: ignore[assignment]
    ParameterButton = None  # type: ignore[assignment]
    _HAS_BUTTONS = False

LIBRARY_NAME = "GTN SAM 3D Body"

PIPELINE_NODES = (
    ("load", "LoadSAM3DBodyModelNode"),
    ("boxes", "DetectPersonBoxesNode"),
    ("track", "SAM3DBodyVideoTrackNode"),
    ("fov", "ExtractFoVFromMoGeNode"),
    ("predict", "RunSAM3DBodyPredictionNode"),
    ("smooth", "SmoothSAM3DBodyPoseNode"),
    ("face", "SAM3DBodyFaceExpressionNode"),
    ("render", "RenderSAM3DBodyPoseNode"),
    ("export", "Create3DAnimationFileNode"),
)

CORE_KEYS = ("load", "predict", "smooth", "render", "export")
FULL_KEYS = tuple(key for key, _ in PIPELINE_NODES)

VIDEO_SOURCE_PARAMS = ("video", "video_url", "output", "video_output", "media", "image")
VIDEO_NODE_CANDIDATES = (
    ("Griptape Nodes Library", "LoadVideo"),
    ("Griptape Nodes Library", "Load Video"),
    ("Griptape Nodes Library", "LoadVideoFromFile"),
    ("Griptape Nodes Library", "LoadVideoUrl"),
)
DISPLAY_VIDEO_CANDIDATES = (
    ("Griptape Nodes Library", "DisplayVideo"),
    ("Griptape Nodes Library", "Display Video"),
    ("Griptape Nodes Library", "VideoDisplay"),
)
DISPLAY_VIDEO_INPUT_PARAMS = ("video", "video_url", "media", "input", "value")

# Canvas offsets and sizes (relative to the drop node) mirroring the hand-arranged
# layout: Load Video left, One-Click center, Display Video top-right, View 3D
# bottom-right. Format: (dx, dy, width, height); size None keeps the node default.
COMPACT_LAYOUT = {
    "setup": (0, -640, None, None),
    "video": (0, 140, 520, 660),
    "oneclick": (620, 0, 460, None),
    # Right column sits well clear of the One-Click node so the connection
    # lines between them stay visible.
    "display": (1460, -40, 640, 620),
    # Wide viewer: the 16:9 viewport is width-driven, so extra width = a much
    # bigger 3D view. It intentionally dwarfs the video nodes.
    "view3d": (1460, 660, 980, 880),
    # Face mocap close-up viewer, below the body viewer.
    "viewface": (1460, 1600, 700, 660),
}

# Group frame padding around the members and headroom for its title bar.
GROUP_PAD = 80
GROUP_HEADER = 120
class DropSAM3DBodyGraphNode(ControlNode):
    """Drop this node, then click the button (or run it) to spawn the full wired pipeline."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.log_params = LogParameter(self)
        self.add_parameter(
            Parameter(
                name="graph_variant",
                input_types=["str"],
                type="str",
                default_value="compact",
                traits={Options(choices=["compact", "full", "core"])},
                tooltip="compact = video → One-Click → video + 3D preview (recommended). full = every pipeline stage. core = load → predict → smooth → render → export.",
            )
        )
        self.add_parameter(
            Parameter(
                name="include_setup",
                input_types=["bool"],
                type="bool",
                default_value=False,
                tooltip="Also spawn Setup. Leave off if you already ran SAM 3D Body Setup on this canvas.",
            )
        )
        self.add_parameter(
            Parameter(
                name="include_load_video",
                input_types=["bool"],
                type="bool",
                default_value=True,
                tooltip="Try to spawn a Load Video node and wire it into every media_input.",
            )
        )
        self.add_parameter(
            Parameter(
                name="reuse_existing_setup",
                input_types=["bool"],
                type="bool",
                default_value=True,
                tooltip="If a Setup node is already on the canvas, wire its outputs into the new graph.",
            )
        )
        if _HAS_BUTTONS:
            with ParameterButtonGroup(name="spawn_buttons") as spawn_buttons:
                ParameterButton(
                    name="drop_graph_btn",
                    label="Drop Full Graph",
                    icon="layout-grid",
                    on_click=self._on_drop_clicked,
                )
            self.add_node_element(spawn_buttons)
        self.add_parameter(
            Parameter(
                name="status",
                output_type="str",
                allowed_modes={ParameterMode.OUTPUT},
                tooltip="What got created and wired.",
            )
        )
        self.add_parameter(
            Parameter(
                name="created_nodes",
                output_type="str",
                allowed_modes={ParameterMode.OUTPUT},
                tooltip="Created node names.",
            )
        )
        self.log_params.add_output_parameters()

    def _on_drop_clicked(self, *args, **kwargs) -> None:  # noqa: ARG002 - button callbacks receive extra args
        self._spawn()

    def process(self) -> AsyncResult:
        yield lambda: self._spawn()

    def _spawn(self) -> None:
        self.log_params.clear_logs()
        engine = self._engine()
        if engine is None:
            raise ValueError("Griptape engine is not available. Drop this node onto a live canvas and click Drop Full Graph.")

        node_events, flow_events, parameter_events, connection_cls, autolayout_cls = self._event_types()
        flow_name = self._current_flow_name(engine, flow_events)
        tag = uuid.uuid4().hex[:4]
        variant = str(self.get_parameter_value("graph_variant") or "compact").strip().lower()
        if variant == "compact":
            self._spawn_compact(engine, node_events, connection_cls, autolayout_cls, flow_name, tag)
            return
        wanted_keys = FULL_KEYS if variant != "core" else CORE_KEYS
        created: dict[str, str] = {}
        notes: list[str] = []

        if bool(self.get_parameter_value("include_setup")):
            created["setup"] = self._create_node(
                engine,
                node_events,
                node_type="SAM3DBodySetupNode",
                node_name=f"SAM3D_Setup_{tag}",
                flow_name=flow_name,
            )

        for key, node_type in PIPELINE_NODES:
            if key not in wanted_keys:
                continue
            created[key] = self._create_node(
                engine,
                node_events,
                node_type=node_type,
                node_name=f"SAM3D_{key}_{tag}",
                flow_name=flow_name,
            )

        if bool(self.get_parameter_value("include_load_video")):
            video_name = self._create_candidate_node(engine, node_events, VIDEO_NODE_CANDIDATES, f"SAM3D_video_{tag}", flow_name)
            if video_name:
                created["video"] = video_name
            else:
                notes.append("No Load Video node type found; wire media_input yourself.")

        existing_setup = created.get("setup")
        if not existing_setup and bool(self.get_parameter_value("reuse_existing_setup")):
            existing_setup = self._find_existing_setup(engine)

        worker_keys = [key for key in wanted_keys if key not in {"load", "smooth"}]
        if existing_setup:
            if "load" in created:
                self._connect(engine, connection_cls, existing_setup, "model_handle", created["load"], "model_handle", notes)
                self._connect(engine, connection_cls, existing_setup, "repo_dir_out", created["load"], "repo_dir", notes)
                self._connect(engine, connection_cls, existing_setup, "python_executable_out", created["load"], "python_executable", notes)
            for key in worker_keys:
                if key not in created:
                    continue
                self._connect(engine, connection_cls, existing_setup, "repo_dir_out", created[key], "repo_dir", notes)
                self._connect(engine, connection_cls, existing_setup, "python_executable_out", created[key], "python_executable", notes)

        if "load" in created:
            for key in worker_keys:
                if key not in created:
                    continue
                self._connect(engine, connection_cls, created["load"], "model_handle", created[key], "model_handle", notes)
                self._copy_runtime_from_load(engine, parameter_events, created["load"], created[key], notes)

        if "video" in created:
            for key in ("boxes", "track", "fov", "predict", "face", "render"):
                if key in created:
                    self._connect_first(
                        engine,
                        connection_cls,
                        created["video"],
                        VIDEO_SOURCE_PARAMS,
                        created[key],
                        "media_input",
                        notes,
                    )

        if "boxes" in created and "predict" in created:
            self._connect(engine, connection_cls, created["boxes"], "boxes_path", created["predict"], "boxes_path", notes)
        if "track" in created and "predict" in created:
            self._connect(engine, connection_cls, created["track"], "track_data_path", created["predict"], "track_data_path", notes)
        if "fov" in created and "predict" in created:
            self._connect(engine, connection_cls, created["fov"], "fov", created["predict"], "fov", notes)
        if "predict" in created and "smooth" in created:
            self._connect(engine, connection_cls, created["predict"], "pose_path", created["smooth"], "pose_path", notes)
        pose_source = None
        pose_param = None
        if "face" in created and "smooth" in created:
            self._connect(engine, connection_cls, created["smooth"], "smoothed_pose_path", created["face"], "pose_path", notes)
            pose_source, pose_param = created["face"], "pose_path_out"
        elif "smooth" in created:
            pose_source, pose_param = created["smooth"], "smoothed_pose_path"
        elif "predict" in created:
            pose_source, pose_param = created["predict"], "pose_path"
        if pose_source and pose_param:
            if "render" in created:
                self._connect(engine, connection_cls, pose_source, pose_param, created["render"], "pose_path", notes)
            if "export" in created:
                self._connect(engine, connection_cls, pose_source, pose_param, created["export"], "pose_path", notes)
        if "face" in created and "video" not in created:
            notes.append("Face node needs the same source media as Predict.")

        chain = [key for key in ("setup", "load", "boxes", "track", "fov", "predict", "smooth", "face", "render", "export") if key in created]
        # If setup was reused rather than created, still start the control chain from it.
        if existing_setup and "setup" not in created and "load" in created:
            chain = ["_existing_setup", *chain]
            created["_existing_setup"] = existing_setup
        for left, right in zip(chain, chain[1:]):
            self._connect(engine, connection_cls, created[left], "exec_out", created[right], "exec_in", notes)

        if autolayout_cls is not None:
            try:
                kwargs: dict[str, Any] = {}
                if flow_name:
                    kwargs["flow_name"] = flow_name
                engine.handle_request(autolayout_cls(**kwargs))
                notes.append("Auto-layout applied.")
            except Exception as exc:
                notes.append(f"Auto-layout skipped: {exc}")

        member_names = [name for key, name in created.items() if not key.startswith("_")]
        group_name = self._group_nodes(engine, node_events, "SAM 3D Body", member_names, flow_name, notes)
        if group_name:
            created["_group"] = group_name

        created_text = ", ".join(f"{key}={name}" for key, name in created.items() if not key.startswith("_"))
        status = f"Dropped {variant} SAM 3D Body graph ({len(created)} nodes)."
        self.parameter_output_values["status"] = status
        self.parameter_output_values["created_nodes"] = created_text
        self.log_params.append_to_logs(status + "\n")
        self.log_params.append_to_logs(created_text + "\n")
        for note in notes:
            self.log_params.append_to_logs(note + "\n")

    def _spawn_compact(
        self,
        engine: Any,
        node_events: Any,
        connection_cls: Any,
        autolayout_cls: Any,
        flow_name: str | None,
        tag: str,
    ) -> None:
        created: dict[str, str] = {}
        notes: list[str] = []
        base_x, base_y = self._self_position(engine)

        include_setup = bool(self.get_parameter_value("include_setup"))

        # Group frame FIRST, sized from the planned layout, so every node can be
        # created directly inside it (parent_group_name) - that is how the
        # editor parents nodes, and the only way the frame drags them along.
        planned = [key for key in COMPACT_LAYOUT if key != "setup" or include_setup]
        group_name, group_origin = self._create_group_frame(engine, node_events, "SAM 3D Body", planned, base_x, base_y, flow_name, notes)
        if group_name:
            created["_group"] = group_name

        def meta(key: str, absolute: bool = False) -> dict[str, Any]:
            dx, dy, width, height = COMPACT_LAYOUT.get(key, (0, 0, None, None))
            x, y = base_x + dx, base_y + dy
            if not absolute and group_name and group_origin:
                # The editor positions group members RELATIVE to the group
                # frame, not in canvas coordinates.
                x -= group_origin[0]
                y -= group_origin[1]
            data: dict[str, Any] = {"position": {"x": x, "y": y}}
            if width and height:
                data["size"] = {"width": width, "height": height}
            return data

        def spawn(key: str, node_type: str) -> str:
            name = self._create_node(
                engine,
                node_events,
                node_type=node_type,
                node_name=f"SAM3D_{key}_{tag}",
                flow_name=flow_name,
                metadata=meta(key),
                parent_group=group_name or None,
            )
            created[key] = name
            return name

        if include_setup:
            spawn("setup", "SAM3DBodySetupNode")
        # One-Click runs the whole pipeline; previews are separate connected nodes.
        spawn("oneclick", "SAM3DBodyOneClickNode")
        spawn("view3d", "ViewSAM3DBodyNode")
        spawn("viewface", "ViewSAM3DFaceNode")

        # The video nodes are created inside the group like everything else:
        # only creation-time parenting makes the editor truly attach a node to
        # the frame (a later membership add updates the backend but the open
        # canvas does not pick it up).
        if bool(self.get_parameter_value("include_load_video")):
            video_name = self._create_candidate_node(
                engine, node_events, VIDEO_NODE_CANDIDATES, f"SAM3D_video_{tag}", flow_name, metadata=meta("video"), parent_group=group_name or None
            )
            if video_name:
                created["video"] = video_name
            else:
                notes.append("No Load Video node type found; set media_input on the One-Click node yourself.")

        display_name = self._create_candidate_node(
            engine, node_events, DISPLAY_VIDEO_CANDIDATES, f"SAM3D_overlay_{tag}", flow_name, metadata=meta("display"), parent_group=group_name or None
        )
        if display_name:
            created["display"] = display_name
        else:
            notes.append("No Display Video node type found; connect overlay_video to your own display node.")

        # Ground truth: ask the GROUP who its members are, rather than trusting
        # per-create results. Anything the engine failed to parent gets moved
        # to its ABSOLUTE canvas position (the editor converts canvas ->
        # group-relative while processing the add) and one membership request.
        if group_name:
            members = self._group_members(engine, group_name)
            missing = [name for key, name in created.items() if not key.startswith("_") and name not in members]
            if missing:
                name_to_key = {name: key for key, name in created.items()}
                for member in missing:
                    key = name_to_key.get(member)
                    if key in COMPACT_LAYOUT:
                        self._set_node_position(engine, node_events, member, meta(key, absolute=True)["position"], notes)
                self._add_nodes_to_group(engine, node_events, group_name, missing, flow_name, notes)
                notes.append(f"Grouped after creation (may need a canvas reload to drag along): {', '.join(missing)}")

        existing_setup = created.get("setup")
        if not existing_setup and bool(self.get_parameter_value("reuse_existing_setup")):
            existing_setup = self._find_existing_setup(engine)
        if existing_setup:
            self._connect(engine, connection_cls, existing_setup, "repo_dir_out", created["oneclick"], "repo_dir", notes)
            self._connect(engine, connection_cls, existing_setup, "python_executable_out", created["oneclick"], "python_executable", notes)

        # Data connections only; upstream nodes resolve automatically, no exec chain needed.
        if "video" in created:
            self._connect_first(engine, connection_cls, created["video"], VIDEO_SOURCE_PARAMS, created["oneclick"], "media_input", notes)
        if "display" in created:
            for target_param in DISPLAY_VIDEO_INPUT_PARAMS:
                if self._connect(engine, connection_cls, created["oneclick"], "overlay_video", created["display"], target_param, notes, optional=True):
                    break
            else:
                notes.append("Could not wire overlay_video into the display node.")
        self._connect(engine, connection_cls, created["oneclick"], "pose_path", created["view3d"], "pose_path", notes)
        self._connect(engine, connection_cls, created["oneclick"], "glb_path", created["view3d"], "glb_path", notes)
        self._connect(engine, connection_cls, created["oneclick"], "pose_path", created["viewface"], "pose_path", notes)

        created_text = ", ".join(f"{key}={name}" for key, name in created.items() if not key.startswith("_"))
        status = f"Dropped compact SAM 3D Body graph ({len([k for k in created if not k.startswith('_')])} nodes)."
        self.parameter_output_values["status"] = status
        self.parameter_output_values["created_nodes"] = created_text
        self.log_params.append_to_logs(status + "\n")
        self.log_params.append_to_logs(created_text + "\n")
        for note in notes:
            self.log_params.append_to_logs(note + "\n")

        # Remove this drop node so the graph can't be accidentally spawned twice.
        try:
            engine.handle_request(node_events.DeleteNodeRequest(node_name=self.name))
        except Exception as exc:
            self.log_params.append_to_logs(f"Could not remove drop node: {exc}\n")

    def _group_frame_metadata(self, x1: float, y1: float, x2: float, y2: float) -> dict[str, Any]:
        """Metadata for the standard black semi-transparent Group frame covering the given bounds."""
        size = {"width": int(x2 - x1) + 2 * GROUP_PAD, "height": int(y2 - y1) + 2 * GROUP_PAD + GROUP_HEADER}
        return {
            "library": "Griptape Nodes Library",
            "node_type": "Group",
            "is_node_group": True,
            "executable": False,
            "hideaddparameter": True,
            "showConnectionsCollapsed": True,
            "group_settings_params": ["description"],
            "color": "#000000",
            "opacity": 0.4,
            "position": {"x": int(x1) - GROUP_PAD, "y": int(y1) - GROUP_PAD - GROUP_HEADER},
            "size": size,
            "expanded_dimensions": dict(size),
        }

    def _create_group_frame(
        self,
        engine: Any,
        node_events: Any,
        title: str,
        planned_keys: list[str],
        base_x: int,
        base_y: int,
        flow_name: str | None,
        notes: list[str],
    ) -> tuple[str, tuple[int, int] | None]:
        """Create the (empty) Group frame first, sized from the planned layout,
        so nodes can be created directly inside it via parent_group_name.

        Returns the group name and the frame's canvas origin (top-left); child
        positions must be expressed relative to that origin."""
        x1 = y1 = x2 = y2 = None
        for key in planned_keys:
            dx, dy, width, height = COMPACT_LAYOUT.get(key, (0, 0, None, None))
            width = width or 460
            height = height or 620
            nx, ny = base_x + dx, base_y + dy
            x1 = nx if x1 is None else min(x1, nx)
            y1 = ny if y1 is None else min(y1, ny)
            x2 = nx + width if x2 is None else max(x2, nx + width)
            y2 = ny + height if y2 is None else max(y2, ny + height)
        if x1 is None:
            return "", None
        origin = (int(x1) - GROUP_PAD, int(y1) - GROUP_PAD - GROUP_HEADER)
        kwargs: dict[str, Any] = {
            "node_type": "Group",
            "specific_library_name": "Griptape Nodes Library",
            "node_name": title,
            "metadata": self._group_frame_metadata(x1, y1, x2, y2),
        }
        if flow_name:
            kwargs["override_parent_flow_name"] = flow_name
        try:
            result = engine.handle_request(node_events.CreateNodeRequest(**kwargs))
            if hasattr(result, "failed") and result.failed():
                raise ValueError(getattr(result, "result_details", "") or "group creation failed")
            return str(getattr(result, "node_name", None) or title), origin
        except Exception as exc:
            notes.append(f"Could not create group frame: {exc}")
            return "", None

    def _add_nodes_to_group(
        self,
        engine: Any,
        node_events: Any,
        group_name: str,
        member_names: list[str],
        flow_name: str | None,
        notes: list[str],
    ) -> bool:
        """Explicit group membership request (the mechanism the editor uses)."""
        try:
            add_cls = getattr(node_events, "AddNodesToNodeGroupRequest", None)
            if add_cls is None:
                notes.append("Engine has no AddNodesToNodeGroupRequest; group is visual only.")
                return False
            add_kwargs: dict[str, Any] = {"node_names": list(member_names), "node_group_name": group_name}
            if flow_name:
                add_kwargs["flow_name"] = flow_name
            result = engine.handle_request(add_cls(**add_kwargs))
            if hasattr(result, "failed") and result.failed():
                notes.append(f"Group membership failed: {getattr(result, 'result_details', '')}")
                return False
            return True
        except Exception as exc:
            notes.append(f"Group membership failed: {exc}")
            return False

    def _group_members(self, engine: Any, group_name: str) -> set[str]:
        """The group's own membership list (metadata node_names_in_group)."""
        try:
            from griptape_nodes.retained_mode.events.node_events import GetNodeMetadataRequest

            result = engine.handle_request(GetNodeMetadataRequest(node_name=group_name))
            meta = getattr(result, "metadata", None) or {}
            return set(meta.get("node_names_in_group") or [])
        except Exception:
            return set()

    def _set_node_position(self, engine: Any, node_events: Any, node_name: str, position: dict[str, Any], notes: list[str]) -> None:
        """Rewrite one node's canvas position, preserving the rest of its metadata."""
        get_cls = getattr(node_events, "GetNodeMetadataRequest", None)
        set_cls = getattr(node_events, "SetNodeMetadataRequest", None)
        if get_cls is None or set_cls is None:
            return
        try:
            current = dict(getattr(engine.handle_request(get_cls(node_name=node_name)), "metadata", None) or {})
            current["position"] = dict(position)
            engine.handle_request(set_cls(node_name=node_name, metadata=current))
        except Exception as exc:
            notes.append(f"Could not reposition {node_name}: {exc}")

    def _shift_members_relative(self, engine: Any, node_events: Any, member_names: list[str], origin: tuple[int, int], notes: list[str]) -> None:
        """Rewrite members' positions from canvas coordinates to group-relative
        (the editor renders group members relative to the frame's origin)."""
        get_cls = getattr(node_events, "GetNodeMetadataRequest", None)
        set_cls = getattr(node_events, "SetNodeMetadataRequest", None)
        if get_cls is None or set_cls is None:
            notes.append("Engine lacks node metadata requests; grouped nodes may render offset.")
            return
        for name in member_names:
            try:
                meta = dict(getattr(engine.handle_request(get_cls(node_name=name)), "metadata", None) or {})
                position = dict(meta.get("position") or {})
                position["x"] = float(position.get("x", 0)) - origin[0]
                position["y"] = float(position.get("y", 0)) - origin[1]
                meta["position"] = position
                engine.handle_request(set_cls(node_name=name, metadata=meta))
            except Exception as exc:
                notes.append(f"Could not reposition {name} inside group: {exc}")

    def _nodes_bounds(self, engine: Any, member_names: list[str]) -> tuple[int, int, int, int] | None:
        """Bounding box (x1, y1, x2, y2) of the members on the canvas."""
        try:
            from griptape_nodes.retained_mode.events.node_events import GetNodeMetadataRequest
        except Exception:
            return None
        x1 = y1 = None
        x2 = y2 = None
        for name in member_names:
            try:
                result = engine.handle_request(GetNodeMetadataRequest(node_name=name))
                meta = getattr(result, "metadata", None) or {}
                position = meta.get("position") or {}
                size = meta.get("size") or {}
                x = float(position.get("x", 0))
                y = float(position.get("y", 0))
                w = float(size.get("width") or 460)
                h = float(size.get("height") or 620)
            except Exception:
                continue
            x1 = x if x1 is None else min(x1, x)
            y1 = y if y1 is None else min(y1, y)
            x2 = x + w if x2 is None else max(x2, x + w)
            y2 = y + h if y2 is None else max(y2, y + h)
        if x1 is None:
            return None
        return (int(x1), int(y1), int(x2), int(y2))

    def _group_nodes(
        self,
        engine: Any,
        node_events: Any,
        group_name: str,
        member_names: list[str],
        flow_name: str | None,
        notes: list[str],
    ) -> str:
        """Wrap the spawned nodes in the standard organizational Group frame
        (the black semi-transparent one), sized to fit around them. Purely
        cosmetic, so any failure is logged and ignored."""
        if not member_names:
            return ""
        bounds = self._nodes_bounds(engine, member_names)
        if bounds is None:
            bounds = (0, 0, 460, 620)
        metadata = self._group_frame_metadata(*bounds)
        kwargs: dict[str, Any] = {
            "node_type": "Group",
            "specific_library_name": "Griptape Nodes Library",
            "node_name": group_name,
            "metadata": metadata,
        }
        if flow_name:
            kwargs["override_parent_flow_name"] = flow_name
        try:
            result = engine.handle_request(node_events.CreateNodeRequest(**kwargs))
            if hasattr(result, "failed") and result.failed():
                raise ValueError(getattr(result, "result_details", "") or "group creation failed")
            actual_name = str(getattr(result, "node_name", None) or group_name)
        except Exception as exc:
            notes.append(f"Could not group nodes: {exc}")
            return ""
        # Real membership (not just a frame drawn behind the nodes). Members'
        # positions then have to become group-relative or the editor shifts
        # them by the frame's canvas position.
        if self._add_nodes_to_group(engine, node_events, actual_name, member_names, flow_name, notes):
            origin = (bounds[0] - GROUP_PAD, bounds[1] - GROUP_PAD - GROUP_HEADER)
            self._shift_members_relative(engine, node_events, member_names, origin, notes)
        return actual_name

    @staticmethod
    def _create_with_optional_group(
        engine: Any,
        node_events: Any,
        kwargs: dict[str, Any],
        parent_group: str | None,
        ungrouped: list[str] | None,
        node_name: str,
    ) -> Any:
        """CreateNodeRequest with parent_group_name, retrying without it on old
        engines. Nodes the engine did not parent get recorded in `ungrouped`."""
        if parent_group:
            try:
                result = engine.handle_request(node_events.CreateNodeRequest(**kwargs, parent_group_name=parent_group))
                grouped = str(getattr(result, "parent_group_name", "") or "") == parent_group
                if not grouped and ungrouped is not None and not (hasattr(result, "failed") and result.failed()):
                    ungrouped.append(str(getattr(result, "node_name", None) or node_name))
                return result
            except TypeError:
                pass  # Engine predates parent_group_name; create plainly, group later.
        result = engine.handle_request(node_events.CreateNodeRequest(**kwargs))
        if parent_group and ungrouped is not None and not (hasattr(result, "failed") and result.failed()):
            ungrouped.append(str(getattr(result, "node_name", None) or node_name))
        return result

    def _create_node(
        self,
        engine: Any,
        node_events: Any,
        *,
        node_type: str,
        node_name: str,
        flow_name: str | None,
        metadata: dict[str, Any] | None = None,
        parent_group: str | None = None,
        ungrouped: list[str] | None = None,
    ) -> str:
        kwargs: dict[str, Any] = {
            "node_type": node_type,
            "specific_library_name": LIBRARY_NAME,
            "node_name": node_name,
        }
        if flow_name:
            kwargs["override_parent_flow_name"] = flow_name
        if metadata:
            kwargs["metadata"] = metadata
        result = self._create_with_optional_group(engine, node_events, kwargs, parent_group, ungrouped, node_name)
        if hasattr(result, "failed") and result.failed():
            details = getattr(result, "result_details", "") or result
            raise ValueError(f"Could not create {node_type}: {details}")
        return str(getattr(result, "node_name", None) or node_name)

    def _create_candidate_node(
        self,
        engine: Any,
        node_events: Any,
        candidates: tuple[tuple[str, str], ...],
        node_name: str,
        flow_name: str | None,
        metadata: dict[str, Any] | None = None,
        parent_group: str | None = None,
        ungrouped: list[str] | None = None,
    ) -> str:
        for library, node_type in candidates:
            kwargs: dict[str, Any] = {
                "node_type": node_type,
                "specific_library_name": library,
                "node_name": node_name,
            }
            if flow_name:
                kwargs["override_parent_flow_name"] = flow_name
            if metadata:
                kwargs["metadata"] = metadata
            try:
                result = self._create_with_optional_group(engine, node_events, kwargs, parent_group, ungrouped, node_name)
            except Exception:
                continue
            if hasattr(result, "failed") and result.failed():
                continue
            return str(getattr(result, "node_name", None) or node_name)
        return ""

    def _connect(
        self,
        engine: Any,
        connection_cls: Any,
        source_node: str,
        source_param: str,
        target_node: str,
        target_param: str,
        notes: list[str],
        *,
        optional: bool = False,
    ) -> bool:
        try:
            result = engine.handle_request(
                connection_cls(
                    source_node_name=source_node,
                    source_parameter_name=source_param,
                    target_node_name=target_node,
                    target_parameter_name=target_param,
                )
            )
        except Exception as exc:
            if not optional:
                notes.append(f"Could not wire {source_node}.{source_param} → {target_node}.{target_param}: {exc}")
            return False
        if hasattr(result, "failed") and result.failed():
            if not optional:
                details = getattr(result, "result_details", "") or result
                notes.append(f"Could not wire {source_node}.{source_param} → {target_node}.{target_param}: {details}")
            return False
        return True

    def _connect_first(
        self,
        engine: Any,
        connection_cls: Any,
        source_node: str,
        source_params: tuple[str, ...],
        target_node: str,
        target_param: str,
        notes: list[str],
    ) -> None:
        for source_param in source_params:
            if self._connect(engine, connection_cls, source_node, source_param, target_node, target_param, notes, optional=True):
                return
        notes.append(f"Could not wire {source_node} → {target_node}.{target_param} (tried {', '.join(source_params)}).")

    def _copy_runtime_from_load(self, engine: Any, parameter_events: Any, load_name: str, target_name: str, notes: list[str]) -> None:
        for source_param, target_param in (
            ("python_executable", "python_executable"),
            ("repo_dir", "repo_dir"),
        ):
            try:
                get_result = engine.handle_request(
                    parameter_events.GetParameterValueRequest(node_name=load_name, parameter_name=source_param)
                )
                value = getattr(get_result, "value", None)
                if value in (None, ""):
                    continue
                engine.handle_request(
                    parameter_events.SetParameterValueRequest(node_name=target_name, parameter_name=target_param, value=value)
                )
            except Exception as exc:
                notes.append(f"Could not copy {source_param} onto {target_name}: {exc}")

    def _self_position(self, engine: Any) -> tuple[int, int]:
        """Canvas position of this drop node; spawned nodes are placed relative to it."""
        try:
            from griptape_nodes.retained_mode.events.node_events import GetNodeMetadataRequest

            result = engine.handle_request(GetNodeMetadataRequest(node_name=self.name))
            meta = getattr(result, "metadata", None)
            if isinstance(meta, dict):
                position = meta.get("position") or {}
                return (int(position.get("x", 0)), int(position.get("y", 0)))
        except Exception:
            pass
        return (0, 0)

    def _find_existing_setup(self, engine: Any) -> str:
        try:
            from griptape_nodes.exe_types.node_types import BaseNode
        except Exception:
            BaseNode = object  # type: ignore[misc, assignment]
        try:
            nodes = engine.ObjectManager().get_filtered_subset(type=BaseNode)
        except Exception:
            return ""
        items = nodes.items() if isinstance(nodes, dict) else [(getattr(n, "name", ""), n) for n in (nodes or [])]
        for name, node in items:
            if type(node).__name__ == "SAM3DBodySetupNode" and str(name) != str(self.name):
                return str(name)
        return ""

    def _current_flow_name(self, engine: Any, flow_events: Any) -> str | None:
        for attr in ("flow_name", "parent_flow_name", "_parent_flow_name"):
            value = getattr(self, attr, None)
            if value:
                return str(value)
        try:
            from griptape_nodes.retained_mode.events.node_events import GetNodeMetadataRequest

            result = engine.handle_request(GetNodeMetadataRequest(node_name=self.name))
            meta = getattr(result, "metadata", None)
            if isinstance(meta, dict):
                for key in ("flow_name", "parent_flow_name", "flow"):
                    if meta.get(key):
                        return str(meta[key])
        except Exception:
            pass
        try:
            result = engine.handle_request(flow_events.ListFlowsInFlowRequest(parent_flow_name=None))
            names = list(getattr(result, "flow_names", []) or [])
            if names:
                return str(names[0])
        except Exception:
            pass
        return None

    @staticmethod
    def _engine() -> Any:
        try:
            from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

            return GriptapeNodes
        except Exception:
            return None

    @staticmethod
    def _event_types() -> tuple[Any, Any, Any, Any, Any]:
        import griptape_nodes.retained_mode.events.flow_events as flow_events
        import griptape_nodes.retained_mode.events.node_events as node_events
        import griptape_nodes.retained_mode.events.parameter_events as parameter_events

        connection_cls = None
        try:
            from griptape_nodes.retained_mode.events.connection_events import CreateConnectionRequest

            connection_cls = CreateConnectionRequest
        except Exception:
            connection_cls = getattr(node_events, "CreateConnectionRequest", None)
        if connection_cls is None:
            raise ValueError("CreateConnectionRequest is not available in this engine version.")
        autolayout_cls = getattr(flow_events, "AutoLayoutFlowRequest", None)
        return node_events, flow_events, parameter_events, connection_cls, autolayout_cls
