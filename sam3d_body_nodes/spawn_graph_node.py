from __future__ import annotations

import uuid
from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, ControlNode
from griptape_nodes.exe_types.param_components.log_parameter import LogParameter
from griptape_nodes.traits.options import Options

try:
    from sam3d_body_nodes._common import add_logs_group
except ImportError:
    from _common import add_logs_group  # type: ignore

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
# Offsets and sizes lifted 1:1 from the user's hand-arranged workflow
# (sam_bodyflow.py), anchored on the Load Video node. Two ComfyUI-style rows:
# Render Body Video -> Display Video on top, Create 3D Animation -> 3D Body
# Viewer below, with the settings nodes in the middle column.
COMPACT_LAYOUT = {
    "setup": (0, -1150, None, None),
    "video": (0, 0, 1097, 900),
    "oneclick": (1178, -71, 600, 1043),
    "posevideo": (1924, -305, 606, 434),
    "display": (2658, -613, 1209, 811),
    "create3d": (1924, 405, None, None),
    "view3d": (2658, 339, 1227, 840),
}
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
        add_logs_group(self)

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

        def meta(key: str) -> dict[str, Any]:
            dx, dy, width, height = COMPACT_LAYOUT.get(key, (0, 0, None, None))
            data: dict[str, Any] = {"position": {"x": base_x + dx, "y": base_y + dy}}
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
            )
            created[key] = name
            return name

        if include_setup:
            spawn("setup", "SAM3DBodySetupNode")
        # One-Click runs the whole pipeline; previews are separate connected nodes.
        spawn("oneclick", "SAM3DBodyOneClickNode")
        spawn("create3d", "CreateSAM3DAnimationNode")
        spawn("view3d", "ViewSAM3DBodyNode")
        spawn("posevideo", "RenderSAM3DPoseVideoNode")

        if bool(self.get_parameter_value("include_load_video")):
            video_name = self._create_candidate_node(
                engine, node_events, VIDEO_NODE_CANDIDATES, f"SAM3D_video_{tag}", flow_name, metadata=meta("video")
            )
            if video_name:
                created["video"] = video_name
            else:
                notes.append("No Load Video node type found; set media_input on the One-Click node yourself.")

        display_name = self._create_candidate_node(
            engine, node_events, DISPLAY_VIDEO_CANDIDATES, f"SAM3D_video_out_{tag}", flow_name, metadata=meta("display")
        )
        if display_name:
            created["display"] = display_name
        else:
            notes.append("No Display Video node type found; connect the render node's video output to your own display node.")

        existing_setup = created.get("setup")
        if not existing_setup and bool(self.get_parameter_value("reuse_existing_setup")):
            existing_setup = self._find_existing_setup(engine)
        if existing_setup:
            self._connect(engine, connection_cls, existing_setup, "repo_dir_out", created["oneclick"], "repo_dir", notes)
            self._connect(engine, connection_cls, existing_setup, "python_executable_out", created["oneclick"], "python_executable", notes)

        # Data connections only; upstream nodes resolve automatically, no exec chain needed.
        if "video" in created:
            self._connect_first(engine, connection_cls, created["video"], VIDEO_SOURCE_PARAMS, created["oneclick"], "media_input", notes)
        # The render node takes the source video FROM One-Click's media_input
        # pass-through port, so the wire runs left -> right through the node
        # instead of arcing over it from Load Video.
        self._connect(engine, connection_cls, created["oneclick"], "media_input", created["posevideo"], "media_input", notes)
        # ComfyUI-style rows: small settings node -> its own output node.
        self._connect(engine, connection_cls, created["oneclick"], "pose_path", created["create3d"], "pose_path", notes)
        self._connect(engine, connection_cls, created["create3d"], "viewer", created["view3d"], "viewer", notes)
        self._connect(engine, connection_cls, created["oneclick"], "pose_path", created["posevideo"], "pose_path", notes)
        # Pass the already-rendered overlay through so mesh style is instant.
        self._connect(engine, connection_cls, created["oneclick"], "overlay_path", created["posevideo"], "overlay_path", notes)
        if "display" in created:
            for target_param in DISPLAY_VIDEO_INPUT_PARAMS:
                if self._connect(engine, connection_cls, created["posevideo"], "video", created["display"], target_param, notes, optional=True):
                    break
            else:
                notes.append("Could not wire the render node's video output into the display node.")

        # LAST step before handing the canvas back: re-assert every node's
        # planned position. Wiring connections (value pushes into display
        # nodes) has been observed to reset a node's position, so this must
        # run after all connects.
        for key, name in created.items():
            if key.startswith("_") or key not in COMPACT_LAYOUT:
                continue
            self._set_node_position(engine, node_events, name, meta(key)["position"], notes)

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

    def _create_node(
        self,
        engine: Any,
        node_events: Any,
        *,
        node_type: str,
        node_name: str,
        flow_name: str | None,
        metadata: dict[str, Any] | None = None,
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
        result = engine.handle_request(node_events.CreateNodeRequest(**kwargs))
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
                result = engine.handle_request(node_events.CreateNodeRequest(**kwargs))
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
