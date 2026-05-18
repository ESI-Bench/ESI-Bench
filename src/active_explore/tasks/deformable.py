from __future__ import annotations

import math
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

os.environ.setdefault("OG_DISABLE_EMITTER_APIS", "1")

import omnigibson as og
import torch as th
import yaml
from omnigibson.macros import gm
from omnigibson.objects.dataset_object import DatasetObject
from omnigibson.utils.constants import PrimType

from utils import normalize_prediction_to_option, normalize_text, option_lines, resolve_path


gm.USE_GPU_DYNAMICS = True
gm.ENABLE_FLATCACHE = False
gm.ENABLE_OBJECT_STATES = True
gm.ENABLE_TRANSITION_RULES = False

TASK_NAME = "deformable"
DEFAULT_MODEL = "gemini-2.5-flash"
DISABLE_VIEWER_CAMERA_MODALITIES = True
SKIP_INITIAL_SETTLE = True

RENDER_OBJECT_PREFIX = "cover_small_item_render_"
DEFAULT_TASK_TYPE = "cover_small_item_cloth"
DEFAULT_ITEM_DROP_HEIGHT_M = 0.15
DEFAULT_CLOTH_CLEARANCE_ABOVE_ITEM_M = 0.10
DEFAULT_CLOTH_MASS_KG = 1.0
DEFAULT_CLOTH_DOWNWARD_SPEED_MPS = 1.75
DEFAULT_CAMERA_FOV_DEG = 70.0
DEFAULT_CAPTURE_WIDTH = 1280
DEFAULT_CAPTURE_HEIGHT = 720
VIEWER_FRAME_RENDER_STEPS = 12
VIEWER_FRAME_MAX_RETRIES = 3
VIEWER_FRAME_RETRY_SLEEP_SEC = 0.15

VALID_ACTIONS = {
    "move_forward",
    "move_backward",
    "move_left",
    "move_right",
    "move_up",
    "move_down",
    "turn_left",
    "turn_right",
    "turn_up",
    "turn_down",
    "stop",
}

ACTION_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string"},
        "reasoning": {"type": "string"},
        "answer": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["action", "reasoning", "answer", "confidence"],
}

FINAL_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "confidence": {"type": "number"},
        "reasoning": {"type": "string"},
    },
    "required": ["answer", "confidence", "reasoning"],
}


@dataclass
class RuntimeObjectRecord:
    name: str
    category: str
    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]
    in_rooms: tuple[str, ...]
    obj: object = field(repr=False)

    @property
    def center(self) -> tuple[float, float, float]:
        return tuple((lo + hi) * 0.5 for lo, hi in zip(self.bbox_min, self.bbox_max))

    @property
    def extents(self) -> tuple[float, float, float]:
        return tuple(hi - lo for lo, hi in zip(self.bbox_min, self.bbox_max))


def _to_float_list(value) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [float(v) for v in value]


def _tensor_to_tuple3(value) -> tuple[float, float, float]:
    vals = _to_float_list(value)
    return float(vals[0]), float(vals[1]), float(vals[2])


def _read_current_aabb(obj) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    bbox_min, bbox_max = [_tensor_to_tuple3(x) for x in obj.aabb]
    return bbox_min, bbox_max


def _step_sim(steps: int) -> None:
    for _ in range(max(int(steps), 0)):
        og.sim.step()


def _render_only(frames: int = 4) -> None:
    for _ in range(max(int(frames), 0)):
        try:
            og.sim.render()
        except Exception:
            break


def _warmup_render_pipeline(steps: int = 2, renders: int = 4) -> None:
    for _ in range(max(int(steps), 0)):
        try:
            og.sim.step()
        except Exception:
            break
    _render_only(renders)


def _configure_sim_for_cloth_drop() -> None:
    try:
        og.sim.stop()
    except Exception:
        pass
    try:
        og.sim.set_simulation_dt(physics_dt=1.0 / 240.0, rendering_dt=1.0 / 60.0, sim_step_dt=1.0 / 60.0)
    except Exception:
        pass
    try:
        og.sim.play()
    except Exception:
        pass


def _set_viewer_camera_fov(fov_deg: float) -> None:
    cam = getattr(og.sim, "viewer_camera", None) or getattr(og.sim, "_viewer_camera", None)
    if cam is None:
        return
    try:
        aperture_mm = float(cam.horizontal_aperture)
        cam.focal_length = aperture_mm / (2.0 * math.tan(math.radians(float(fov_deg)) * 0.5))
    except Exception:
        pass


def _set_velocity_zero(obj) -> None:
    try:
        obj.root_link.set_linear_velocity(th.tensor([0.0, 0.0, 0.0], dtype=th.float32))
    except Exception:
        pass
    try:
        obj.root_link.set_angular_velocity(th.tensor([0.0, 0.0, 0.0], dtype=th.float32))
    except Exception:
        pass


def _reset_cloth_to_best_configuration(cloth_obj) -> str | None:
    try:
        available = list(cloth_obj.root_link.get_available_configurations())
    except Exception:
        available = []
    preferred = "settled" if "settled" in available else ("default" if "default" in available else None)
    if preferred is None:
        return None
    try:
        cloth_obj.root_link.reset_points_to_configuration(preferred)
        return preferred
    except Exception:
        return None


def _get_scene_objects(scene) -> list[object]:
    raw_objects = getattr(scene, "objects", [])
    if isinstance(raw_objects, dict):
        return list(raw_objects.values())
    return list(raw_objects)


def build_env_config(
    scene_name: str,
    room_name: str | None,
    robot: str,
    objects: list[dict[str, Any]] | None = None,
    full_scene: bool = False,
) -> dict[str, Any]:
    cfg_file = Path(og.example_config_path) / f"{robot.lower()}_primitives.yaml"
    with cfg_file.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config.setdefault("scene", {})
    config["scene"]["scene_model"] = scene_name
    if room_name and not full_scene:
        config["scene"]["load_room_instances"] = [str(room_name)]
    else:
        config["scene"].pop("load_room_instances", None)
        config["scene"].pop("load_room_types", None)
    config["robots"] = []
    config["objects"] = []
    env_cfg = config.setdefault("env", {})
    env_cfg["physics_frequency"] = 240.0
    env_cfg["rendering_frequency"] = 60.0
    env_cfg["action_frequency"] = 60.0
    return config


def _should_keep_room_object(category: str, in_rooms: tuple[str, ...], room_name: str | None) -> bool:
    if room_name is None:
        return True
    if room_name in in_rooms:
        return True
    if category == "floors":
        return True
    if not in_rooms and category in {"walls", "ceilings", "door", "sliding_door"}:
        return True
    return False


def _collect_room_objects(scene, room_name: str | None) -> list[RuntimeObjectRecord]:
    robot_names = {robot.name for robot in getattr(scene, "robots", [])}
    output = []
    for obj in _get_scene_objects(scene):
        if getattr(obj, "name", None) in robot_names:
            continue
        category = str(getattr(obj, "category", "object"))
        in_rooms = tuple(str(room) for room in (getattr(obj, "in_rooms", None) or []))
        if not _should_keep_room_object(category, in_rooms, room_name):
            continue
        try:
            bbox_min, bbox_max = _read_current_aabb(obj)
        except Exception:
            continue
        output.append(
            RuntimeObjectRecord(
                name=str(getattr(obj, "name", "")),
                category=category,
                bbox_min=bbox_min,
                bbox_max=bbox_max,
                in_rooms=in_rooms,
                obj=obj,
            )
        )
    output.sort(key=lambda record: (record.category, record.name))
    return output


def _distance_xy(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return math.sqrt((float(left[0]) - float(right[0])) ** 2 + (float(left[1]) - float(right[1])) ** 2)


def _floor_selection_sort_key(floor: RuntimeObjectRecord, agent_pos, room_name: str | None):
    room_match = 0 if room_name is not None and room_name in floor.in_rooms else 1
    xy_contains = 0 if (
        floor.bbox_min[0] <= agent_pos[0] <= floor.bbox_max[0]
        and floor.bbox_min[1] <= agent_pos[1] <= floor.bbox_max[1]
    ) else 1
    z_gap = abs(float(floor.bbox_max[2]) - float(agent_pos[2]))
    area = float(floor.extents[0]) * float(floor.extents[1])
    xy_gap = _distance_xy(floor.center, agent_pos)
    return room_match, xy_contains, z_gap, xy_gap, -area, floor.name


def _select_floor(room_objects: list[RuntimeObjectRecord], floor_name: str | None, room_name: str | None = None):
    floors = [obj for obj in room_objects if obj.category == "floors"]
    if floor_name:
        for floor in floors:
            if floor.name == floor_name:
                return floor
        raise ValueError(f"Floor '{floor_name}' not found among loaded room objects.")
    if not floors:
        raise ValueError("No floor object found in loaded room.")
    floors.sort(key=lambda floor: _floor_selection_sort_key(floor, (0.0, 0.0, 0.0), room_name))
    return floors[0]


def _runtime_steps(payload: dict[str, Any]) -> dict[str, int]:
    setup = payload.get("camera_setup")
    if not isinstance(setup, dict):
        setup = payload.get("camera_poses")
    raw_steps = (setup or {}).get("runtime_steps") if isinstance(setup, dict) else {}
    raw_steps = raw_steps or {}
    defaults = {
        "scene_warmup_steps": 20,
        "item_add_steps": 12,
        "item_settle_steps": 30,
        "post_item_freeze_steps": 4,
        "cloth_add_steps": 16,
        "cloth_settle_steps": 40,
    }
    return {key: int(raw_steps.get(key, value) or value) for key, value in defaults.items()}


def _camera_setup(payload: dict[str, Any]) -> dict[str, Any]:
    setup = payload.get("camera_setup")
    if isinstance(setup, dict):
        return setup
    setup = payload.get("camera_poses")
    return setup if isinstance(setup, dict) else {}


def _snapshot_object_state(obj) -> dict[str, Any]:
    payload = {}
    try:
        pos, quat = obj.get_position_orientation()
        payload["pose"] = {"position": _to_float_list(pos), "quaternion_xyzw": _to_float_list(quat)}
    except Exception:
        pass
    try:
        bbox_min, bbox_max = _read_current_aabb(obj)
        payload["bbox"] = {
            "min": [float(v) for v in bbox_min],
            "max": [float(v) for v in bbox_max],
            "center": [float((lo + hi) * 0.5) for lo, hi in zip(bbox_min, bbox_max)],
        }
    except Exception as exc:
        payload["bbox_error"] = f"{exc.__class__.__name__}: {exc}"
    return payload


def scene_room(payload: dict[str, Any]) -> tuple[str, str]:
    return normalize_text(payload.get("scene")), normalize_text(payload.get("room")) or "scene_wide"


def question_id(payload: dict[str, Any], source_path: Path) -> str:
    raw = normalize_text(payload.get("question_id")) or source_path.stem
    return raw.replace("\\", "/").split("/")[-1]


def task_type(payload: dict[str, Any]) -> str:
    return normalize_text(payload.get("task_type")) or DEFAULT_TASK_TYPE


def question_text(payload: dict[str, Any]) -> str:
    return normalize_text((payload.get("qa") or {}).get("question"))


def answer_option_id(payload: dict[str, Any]) -> str:
    return normalize_text((payload.get("qa") or {}).get("answer_option_id"))


def answer_text(payload: dict[str, Any]) -> str:
    return normalize_text((payload.get("qa") or {}).get("answer_text"))


def preprocess(payload: dict[str, Any], source_json: Path, config=None) -> dict[str, Any]:
    main_view = (payload.get("render") or {}).get("main_view") or {}
    data_root = getattr(config, "json_root", None)
    reference_image = resolve_path(main_view.get("image_path"), source_json, data_root=data_root)
    return {
        "source_json": str(source_json),
        "reference_image_path": str(reference_image) if reference_image is not None else None,
        "dynamic_object_names": [],
    }


def reference_image_path(payload: dict[str, Any], task_state: dict[str, Any] | None = None) -> Path | None:
    path = (task_state or {}).get("reference_image_path")
    return Path(path) if path else None


def _viewer_camera():
    cam = getattr(og.sim, "_viewer_camera", None) or getattr(og.sim, "viewer_camera", None)
    if cam is None:
        raise RuntimeError("Viewer camera is unavailable; cannot capture RGB frames.")
    return cam


def _prepare_capture_camera(task_state: dict[str, Any]) -> object:
    cam = task_state.get("capture_camera")
    if cam is None:
        cam = _viewer_camera()
        task_state["capture_camera"] = cam
    try:
        if not cam.initialized:
            cam.initialize()
    except Exception:
        pass
    try:
        cam.add_modality("rgb")
    except Exception:
        pass
    try:
        cam.image_height = int(DEFAULT_CAPTURE_HEIGHT)
    except Exception:
        pass
    try:
        cam.image_width = int(DEFAULT_CAPTURE_WIDTH)
    except Exception:
        pass
    try:
        cam.initialize_sensors(names=["rgb"])
    except Exception:
        pass
    try:
        cam.clipping_range = th.tensor([0.001, 1000.0], dtype=th.float32)
    except Exception:
        pass
    _warmup_render_pipeline(steps=2, renders=4)
    return cam


def _set_capture_camera_pose(cam, pos: np.ndarray, quat: np.ndarray) -> None:
    if getattr(cam, "_prim", None) is None:
        raise RuntimeError("Capture camera prim was not loaded into the stage before configuration.")
    cam.set_position_orientation(
        position=th.tensor([float(v) for v in pos], dtype=th.float32),
        orientation=th.tensor([float(v) for v in quat], dtype=th.float32),
    )
    try:
        cam.clipping_range = th.tensor([0.01, 100.0], dtype=th.float32)
    except Exception:
        pass
    _warmup_render_pipeline(steps=2, renders=4)


def _camera_rgb_obs(cam) -> np.ndarray:
    result = cam.get_obs()
    obs = result[0] if isinstance(result, tuple) else result
    if not isinstance(obs, dict):
        raise RuntimeError(f"get_obs() returned non-dict obs: {type(obs)}")
    if "rgb" not in obs:
        raise RuntimeError(f"RGB not found in obs. keys={list(obs.keys())}")
    frame = obs["rgb"]
    if hasattr(frame, "detach"):
        frame = frame.detach()
    if hasattr(frame, "cpu"):
        frame = frame.cpu()
    image = np.array(frame)
    if not (image.ndim == 3 and image.shape[2] in (3, 4) and image.shape[0] > 0 and image.shape[1] > 0):
        raise ValueError(f"Capture camera returned invalid rgb shape {getattr(image, 'shape', None)}")
    image = image[..., :3]
    if np.issubdtype(image.dtype, np.floating):
        if image.max() <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)
    return image


def _write_rgb(path: Path, image: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return path


def _write_placeholder_observation(path: Path, label: str = "deformable observation unavailable") -> Path:
    image = np.full((DEFAULT_CAPTURE_HEIGHT, DEFAULT_CAPTURE_WIDTH, 3), 242, dtype=np.uint8)
    cv2.putText(
        image,
        label,
        (48, DEFAULT_CAPTURE_HEIGHT // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (32, 32, 32),
        2,
        cv2.LINE_AA,
    )
    return _write_rgb(path, image)


def capture_image(
    env,
    payload: dict[str, Any],
    camera_info: dict[str, Any],
    pos: np.ndarray,
    quat: np.ndarray,
    image_path: Path,
    task_state: dict[str, Any] | None = None,
) -> Path:
    state = task_state if task_state is not None else {}
    cam = _prepare_capture_camera(state)
    _set_capture_camera_pose(cam, pos, quat)
    last_exc = None
    for attempt in range(1, VIEWER_FRAME_MAX_RETRIES + 1):
        try:
            _warmup_render_pipeline(steps=2, renders=VIEWER_FRAME_RENDER_STEPS)
            return _write_rgb(Path(image_path), _camera_rgb_obs(cam))
        except Exception as exc:
            last_exc = exc
            if attempt < VIEWER_FRAME_MAX_RETRIES:
                time.sleep(VIEWER_FRAME_RETRY_SLEEP_SEC)
    fallback = state.get("reference_image_path")
    if fallback:
        fallback_path = Path(fallback)
        if fallback_path.exists():
            output_path = Path(image_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(fallback_path, output_path)
            return output_path
    return _write_placeholder_observation(Path(image_path))


def initial_camera(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    pose = ((payload.get("render") or {}).get("main_view") or {}).get("camera_pose") or {}
    pos = np.array(pose.get("position", [0.0, 0.0, 1.0]), dtype=float)
    quat = np.array(pose.get("quaternion_xyzw", [0.0, 0.0, 0.0, 1.0]), dtype=float)
    return pos, quat, {"camera_pose": pose}


def build_env_objects(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return []


def postprocess_env(env, payload: dict[str, Any], camera_info: dict[str, Any] | None = None, task_state: dict[str, Any] | None = None) -> dict[str, Any]:
    task_state = task_state if task_state is not None else {}
    scene = env.scene
    room_name = normalize_text(payload.get("room")) or None
    floor_name = normalize_text(payload.get("floor_name")) or None
    small_item = payload.get("small_item") or {}
    cloth = payload.get("cloth") or {}
    steps = _runtime_steps(payload)
    seed = int(payload.get("seed", 0) or 0)

    if not small_item.get("category") or not small_item.get("model"):
        raise ValueError("Missing deformable small_item category/model")
    if not cloth.get("category") or not cloth.get("model"):
        raise ValueError("Missing deformable cloth category/model")

    _configure_sim_for_cloth_drop()
    _set_viewer_camera_fov(float(_camera_setup(payload).get("fov_deg", DEFAULT_CAMERA_FOV_DEG) or DEFAULT_CAMERA_FOV_DEG))
    _step_sim(steps["scene_warmup_steps"])

    room_objects = _collect_room_objects(scene, room_name)
    floor_record = _select_floor(room_objects, floor_name, room_name)

    item_name = f"{RENDER_OBJECT_PREFIX}item_{seed:010d}"
    cloth_name = f"{RENDER_OBJECT_PREFIX}cloth_{seed:010d}"
    placement_xy = small_item.get("placement_xy") or [0.0, 0.0]

    item_obj = DatasetObject(name=item_name, category=small_item["category"], model=small_item["model"])
    scene.add_object(item_obj)
    _step_sim(steps["item_add_steps"])
    item_obj.set_position_orientation(
        position=th.tensor(
            [
                float(placement_xy[0]),
                float(placement_xy[1]),
                float(floor_record.bbox_max[2]) + DEFAULT_ITEM_DROP_HEIGHT_M,
            ],
            dtype=th.float32,
        ),
        orientation=th.tensor([0.0, 0.0, 0.0, 1.0], dtype=th.float32),
    )
    _step_sim(steps["item_settle_steps"])
    _set_velocity_zero(item_obj)
    _step_sim(steps["post_item_freeze_steps"])

    item_bbox_min, item_bbox_max = _read_current_aabb(item_obj)
    item_center = [
        float((item_bbox_min[0] + item_bbox_max[0]) * 0.5),
        float((item_bbox_min[1] + item_bbox_max[1]) * 0.5),
        float((item_bbox_min[2] + item_bbox_max[2]) * 0.5),
    ]
    item_top_z = float(item_bbox_max[2])

    cloth_obj = DatasetObject(
        name=cloth_name,
        category=cloth["category"],
        model=cloth["model"],
        prim_type=PrimType.CLOTH,
        abilities={"cloth": {}},
        load_config={"default_configuration": "settled"},
    )
    scene.add_object(cloth_obj)
    _step_sim(steps["cloth_add_steps"])
    cloth_configuration_used = _reset_cloth_to_best_configuration(cloth_obj)
    try:
        cloth_obj.root_link.mass = float(cloth.get("mass_kg", DEFAULT_CLOTH_MASS_KG) or DEFAULT_CLOTH_MASS_KG)
    except Exception:
        pass

    cloth_drop_pos = [
        float(item_center[0]),
        float(item_center[1]),
        float(item_top_z) + float(cloth.get("drop_clearance_above_item_m", DEFAULT_CLOTH_CLEARANCE_ABOVE_ITEM_M)),
    ]
    cloth_obj.set_position_orientation(
        position=th.tensor(cloth_drop_pos, dtype=th.float32),
        orientation=th.tensor([0.0, 0.0, 0.0, 1.0], dtype=th.float32),
    )
    try:
        cloth_obj.root_link.set_linear_velocity(
            th.tensor(
                [0.0, 0.0, -float(cloth.get("initial_downward_speed_mps", DEFAULT_CLOTH_DOWNWARD_SPEED_MPS))],
                dtype=th.float32,
            )
        )
    except Exception:
        _set_velocity_zero(cloth_obj)
    _step_sim(steps["cloth_settle_steps"])
    _render_only(4)
    try:
        capture_cam = _prepare_capture_camera(task_state)
        main_pos, main_quat, _ = initial_camera(payload)
        _set_capture_camera_pose(capture_cam, main_pos, main_quat)
    except Exception:
        pass
    try:
        og.sim.stop()
    except Exception:
        pass

    dynamic_names = [item_name, cloth_name]
    task_state["dynamic_object_names"] = dynamic_names
    return {
        "dynamic_object_names": dynamic_names,
        "floor_name": floor_record.name,
        "runtime_steps": steps,
        "item_after_settle": _snapshot_object_state(item_obj),
        "cloth_after_settle": _snapshot_object_state(cloth_obj),
        "cloth_configuration_used": cloth_configuration_used,
    }


def cleanup_runtime(env, payload: dict[str, Any] | None = None, task_state: dict[str, Any] | None = None) -> bool:
    state = task_state or {}
    try:
        if og.sim is not None:
            og.sim.stop()
    except Exception:
        pass

    if env is not None:
        scene = getattr(env, "scene", None)
        if scene is not None:
            for name in list(state.get("dynamic_object_names") or []):
                try:
                    obj = scene.object_registry("name", name)
                except Exception:
                    obj = None
                if obj is None:
                    continue
                try:
                    scene.remove_object(obj)
                except Exception:
                    pass
        try:
            if og.sim is not None:
                og.sim.stop()
        except Exception:
            pass
        try:
            env.close()
        except Exception:
            pass

    try:
        if og.sim is not None:
            og.sim.stop()
    except Exception:
        pass
    try:
        og.clear()
    except Exception:
        pass
    return False


def _build_options_block(payload: dict[str, Any]) -> str:
    return "\n".join(option_lines((payload.get("qa") or {}).get("options")))


def build_system_prompt(
    payload: dict[str, Any],
    threshold: float,
    min_steps: int,
    camera_info: dict[str, Any] | None = None,
    task_state: dict[str, Any] | None = None,
) -> str:
    return (
        "You are an embodied visual reasoning agent exploring a 3D indoor scene.\n"
        f"Task type: {task_type(payload)}\n"
        f"Question: {question_text(payload)}\n"
        f"Options:\n{_build_options_block(payload)}\n"
        "You will receive the original question reference image, recent views, and then the CURRENT view.\n"
        "Output EXACTLY one JSON object and nothing else:\n"
        "{\n"
        '  "action": "<move_forward|move_backward|move_left|move_right|move_up|move_down|turn_left|turn_right|turn_up|turn_down|stop>",\n'
        '  "reasoning": "<brief explanation>",\n'
        '  "answer": "<exactly one option id from the list, such as A or B or C or D>",\n'
        '  "confidence": <float 0.0-1.0>\n'
        "}\n"
        "Rules:\n"
        "  - Use the cloth shape, room context, and object priors from multiple views.\n"
        "  - If current evidence is insufficient, keep exploring instead of stopping.\n"
        f"  - Before step {min_steps}, confidence should usually stay low unless the answer is very obvious.\n"
        f"  - Do not stop early unless confidence is at least {threshold:.2f} or there is no useful exploration left.\n"
        "  - The answer field must always be exactly one option id from the list, even when uncertain.\n"
        "  - Never answer with option text, 'not sure', 'unknown', or 'cannot answer'. Give your best guess as a single option id.\n"
    )


def build_force_choice_prompt(
    payload: dict[str, Any],
    camera_info: dict[str, Any] | None = None,
    task_state: dict[str, Any] | None = None,
) -> str:
    return (
        "Exploration budget is exhausted.\n"
        f"Question: {question_text(payload)}\n"
        f"Options:\n{_build_options_block(payload)}\n"
        "You must choose exactly one option.\n"
        "Output EXACTLY one JSON object and nothing else:\n"
        '{"answer": "<exactly one option id>", "confidence": <float 0.0-1.0>, "reasoning": "<brief explanation>"}'
    )


def parse_model_output(parsed: dict[str, Any]) -> dict[str, Any]:
    action = normalize_text(parsed.get("action")).lower() or "move_forward"
    if action not in VALID_ACTIONS:
        action = "move_forward"
    try:
        confidence = float(parsed.get("confidence", 0.0))
    except Exception:
        confidence = 0.0
    answer = normalize_text(parsed.get("answer"))
    if answer.lower() in {"", "not sure", "unsure", "unknown"}:
        options = (parsed.get("options") or [])
        options = options if isinstance(options, list) else []
    if answer.lower() in {"", "not sure", "unsure", "unknown"}:
        qa_options = []
        # parse_model_output is intentionally payload-free, so this final fallback
        # is limited to preserving a valid option-id shape for forced answers.
        answer = "A"
    return {
        **parsed,
        "action": action,
        "answer": answer,
        "confidence": max(0.0, min(1.0, confidence)),
        "reasoning": normalize_text(parsed.get("reasoning")) or "no reasoning provided",
    }


def should_stop(parsed: dict[str, Any], history: list[dict[str, Any]], step: int, max_steps: int, min_steps: int, threshold: float) -> tuple[bool, str]:
    if float(parsed.get("confidence", 0.0)) >= threshold and step >= min_steps:
        return True, "confidence_threshold"
    if parsed.get("action") == "stop":
        return True, "model_stop"
    if step == max_steps:
        return True, "max_steps"
    return False, ""


def resolve_final_answer(history: list[dict[str, Any]]) -> tuple[str, int]:
    if not history:
        return "not sure", -1
    latest = history[-1]
    return normalize_text(latest.get("answer")) or "not sure", int(latest["step"])


def needs_force_final_choice(answer: str, stop_reason: str) -> bool:
    normalized = normalize_text(answer).lower()
    return stop_reason == "max_steps" or normalized in {"", "not sure", "unsure", "unknown"}


def score(
    payload: dict[str, Any],
    final_answer: dict[str, Any],
    camera_info: dict[str, Any] | None = None,
    task_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_prediction = normalize_prediction_to_option(payload, {**(final_answer or {}), "action": "stop"}.get("answer"))
    gt_option_id = answer_option_id(payload)
    return {
        "task_type": task_type(payload),
        "question": question_text(payload),
        "options": (payload.get("qa") or {}).get("options", []),
        "ground_truth_answer_option_id": gt_option_id,
        "ground_truth_answer_text": answer_text(payload),
        "normalized_prediction": {
            "raw_answer": (final_answer or {}).get("answer"),
            "normalized_answer_option_id": normalized_prediction.get("answer_option_id"),
            "normalized_answer_text": normalized_prediction.get("answer_text"),
            "matched_option": normalized_prediction.get("matched", False),
            "confidence": (final_answer or {}).get("confidence"),
            "reasoning": (final_answer or {}).get("reasoning"),
            "action": "stop",
        },
        "correct": normalized_prediction.get("answer_option_id") == gt_option_id,
        "reference_image_path": (task_state or {}).get("reference_image_path"),
    }
