from __future__ import annotations

import argparse
import inspect
import importlib
import json
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("OG_DISABLE_EMITTER_APIS", "1")

import cv2
import numpy as np
import omnigibson as og
import torch as th
import yaml
from omnigibson.macros import gm
from omnigibson.utils.object_state_utils import sample_kinematics
from scipy.spatial.transform import Rotation

from models.gemini import GeminiModel
from models.gpt import GPTModel


gm.ENABLE_FLATCACHE = False
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_OBJECT_STATES = True
gm.ENABLE_TRANSITION_RULES = False

MOVE_STEP = 0.25
TURN_STEP = 15.0
SQUARE_ORI = [0.0, 0.0, 0.0, 1.0]
SETTLE_STEPS = 60
PLACEMENT_RETRIES = 5
CAMERA_ACTIONS = {
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
}


@dataclass
class ActiveExploreConfig:
    task: str
    metadata: Path
    question_index: int
    json_root: Path | None
    results_root: Path
    step_image_root: Path
    provider: str
    model: str | None
    api_key: str | None
    max_steps: int
    min_steps: int
    threshold: float
    max_new_tokens: int
    temperature: float
    top_p: float
    robot: str
    overwrite: bool


@dataclass
class SelectedQuestion:
    payload: dict[str, Any]
    source_path: Path
    source_label: str


def normalize_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def parse_args(argv: list[str] | None = None) -> ActiveExploreConfig:
    parser = argparse.ArgumentParser(description="Unified BEHAVIOR-NEW active exploration pipeline.")
    parser.add_argument("--task", required=True, help="Task module name under active_explore/tasks, e.g. counting or action.")
    parser.add_argument("--metadata", type=Path, required=True, help="Metadata JSON containing json_paths, or a single question JSON.")
    parser.add_argument("--question-index", type=int, default=0, help="Question index for metadata json_paths.")
    parser.add_argument("--json-root", type=Path, default=None, help="Optional root used to resolve relative json_paths.")
    parser.add_argument("--results-root", type=Path, default=Path("BEHAVIOR-NEW/active_results"))
    parser.add_argument("--step-image-root", type=Path, default=Path("BEHAVIOR-NEW/active_steps"))
    parser.add_argument("--provider", choices=["gemini", "gpt"], default="gemini")
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--min-steps", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--robot", default="R1")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    return ActiveExploreConfig(
        task=args.task,
        metadata=args.metadata.expanduser().resolve(),
        question_index=args.question_index,
        json_root=args.json_root.expanduser().resolve() if args.json_root else None,
        results_root=args.results_root.expanduser().resolve(),
        step_image_root=args.step_image_root.expanduser().resolve(),
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        max_steps=args.max_steps,
        min_steps=args.min_steps,
        threshold=args.threshold,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        robot=args.robot,
        overwrite=args.overwrite,
    )


def load_task_module(task_name: str):
    return importlib.import_module(f"tasks.{task_name}")


def build_model_client(provider: str, api_key: str | None, model: str):
    if provider == "gemini":
        return GeminiModel(api_key=api_key, model=model)
    if provider == "gpt":
        return GPTModel(api_key=api_key, model=model)
    raise ValueError(f"Unsupported provider: {provider}")


def resolve_json_path(raw_path: str, metadata_path: Path, json_root: Path | None) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute() and path.exists():
        return path.resolve()
    roots = [metadata_path.parent]
    if json_root is not None:
        roots.append(json_root)
    roots.append(Path.cwd())
    for root in roots:
        candidate = (root / path).resolve()
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve question JSON path: {raw_path}")


def decode_json_field(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def is_hf_question_row(value: Any) -> bool:
    return isinstance(value, dict) and "metadata_json" in value and "big_task" in value and "small_task" in value


def payload_from_hf_row(row: dict[str, Any]) -> dict[str, Any]:
    metadata = decode_json_field(row.get("metadata_json"))
    payload = dict(metadata) if isinstance(metadata, dict) else {}
    row_answer = decode_json_field(row.get("answer"))
    row_options = decode_json_field(row.get("options_json"))
    row_image_paths = decode_json_field(row.get("image_paths_json"))

    payload["scene"] = row.get("scene") or payload.get("scene") or "unknown_scene"
    payload["room"] = row.get("room") or payload.get("room") or "unknown_room"
    if row.get("question") and not payload.get("_question"):
        payload["_question"] = row["question"]
    if row.get("answer") not in (None, "") and "_ground_truth" not in payload:
        payload["_ground_truth"] = row_answer
    if row.get("answer") not in (None, "") and "answer" not in payload:
        payload["answer"] = row_answer

    payload["_hf_id"] = row.get("id")
    payload["_hf_big_task"] = row.get("big_task")
    payload["_hf_small_task"] = row.get("small_task")
    payload["_hf_runner_task"] = row.get("runner_task")
    payload["_hf_options"] = row_options if row_options is not None else []
    payload["_hf_image_paths"] = row_image_paths if row_image_paths is not None else []
    return payload


def load_question_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if is_hf_question_row(payload):
        return payload_from_hf_row(payload)
    return payload


def select_question(metadata_path: Path, question_index: int, json_root: Path | None) -> SelectedQuestion:
    if metadata_path.suffix.lower() == ".jsonl":
        raise ValueError("questions.jsonl is an HF export artifact, not a runner input. Use dataset/json_clean/*.json or dataset/json_clean/<Big Task>.json.")

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    if is_hf_question_row(metadata):
        return SelectedQuestion(payload=payload_from_hf_row(metadata), source_path=metadata_path, source_label=str(metadata_path))
    if isinstance(metadata, dict) and isinstance(metadata.get("json_paths"), list):
        json_paths = metadata["json_paths"]
    elif isinstance(metadata, list):
        json_paths = metadata
    else:
        return SelectedQuestion(payload=metadata, source_path=metadata_path, source_label=str(metadata_path))
    if not json_paths:
        raise ValueError(f"No json_paths found in metadata: {metadata_path}")
    if question_index < 0 or question_index >= len(json_paths):
        raise IndexError(f"question_index={question_index} out of range for {len(json_paths)} paths")
    selected = json_paths[question_index]
    question_json = resolve_json_path(str(selected), metadata_path, json_root)
    return SelectedQuestion(payload=load_question_json(question_json), source_path=question_json, source_label=str(question_json))


def rotate_vec(vec: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    return Rotation.from_quat(quat_xyzw).apply(vec)


def apply_camera_action(pos: np.ndarray, quat: np.ndarray, action: str) -> tuple[np.ndarray, np.ndarray]:
    pos = pos.copy()
    quat = quat.copy()
    right = rotate_vec(np.array([1.0, 0.0, 0.0]), quat)
    up = np.array([0.0, 0.0, 1.0])
    forward = np.cross(right, up)

    def flat(vec):
        out = vec.copy()
        out[2] = 0.0
        norm = np.linalg.norm(out)
        return out / norm if norm > 1e-9 else out

    if action == "move_forward":
        pos += flat(forward) * MOVE_STEP
    elif action == "move_backward":
        pos -= flat(forward) * MOVE_STEP
    elif action == "move_right":
        pos += flat(right) * MOVE_STEP
    elif action == "move_left":
        pos -= flat(right) * MOVE_STEP
    elif action == "move_up":
        pos[2] += MOVE_STEP
    elif action == "move_down":
        pos[2] = max(0.01, pos[2] - MOVE_STEP)
    elif action == "turn_left":
        quat = (Rotation.from_rotvec([0, 0, np.radians(TURN_STEP)]) * Rotation.from_quat(quat)).as_quat()
    elif action == "turn_right":
        quat = (Rotation.from_rotvec([0, 0, -np.radians(TURN_STEP)]) * Rotation.from_quat(quat)).as_quat()
    elif action == "turn_up":
        axis = rotate_vec(np.array([1.0, 0.0, 0.0]), quat)
        quat = (Rotation.from_rotvec(axis * np.radians(TURN_STEP)) * Rotation.from_quat(quat)).as_quat()
    elif action == "turn_down":
        axis = rotate_vec(np.array([1.0, 0.0, 0.0]), quat)
        quat = (Rotation.from_rotvec(axis * -np.radians(TURN_STEP)) * Rotation.from_quat(quat)).as_quat()
    return pos, quat


def set_camera(pos: np.ndarray, quat: np.ndarray) -> None:
    og.sim._viewer_camera.set_position_orientation(position=pos, orientation=quat)
    for _ in range(10):
        og.sim.render()


def capture_image(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(10):
        og.sim.render()
    rgb = og.sim._viewer_camera.get_obs()[0]["rgb"].cpu().numpy()[:, :, :3].astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return path


def build_env_config(
    scene_name: str,
    room_name: str | None,
    robot: str,
    objects: list[dict[str, Any]],
    full_scene: bool = False,
) -> dict[str, Any]:
    cfg_file = Path(og.example_config_path) / f"{robot.lower()}_primitives.yaml"
    if cfg_file.exists():
        with cfg_file.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    else:
        config = {"scene": {"type": "InteractiveTraversableScene"}, "robots": [], "objects": []}
    config.setdefault("scene", {})
    config["scene"]["scene_model"] = scene_name
    config["scene"]["not_load_object_categories"] = ["ceilings", "carpet", "walls"]
    if room_name and not full_scene:
        config["scene"]["load_room_instances"] = [room_name]
    config["objects"] = objects
    return config


def task_build_env_config(
    task_module,
    scene_name: str,
    room_name: str | None,
    robot: str,
    objects: list[dict[str, Any]],
    full_scene: bool = False,
) -> dict[str, Any]:
    hook = getattr(task_module, "build_env_config", None)
    if hook is not None:
        return call_with_supported_args(
            hook,
            scene_name,
            room_name,
            robot,
            objects,
            full_scene=full_scene,
        )
    return build_env_config(scene_name, room_name, robot, objects, full_scene=full_scene)


def task_initial_settle_steps(task_module) -> int:
    if getattr(task_module, "SKIP_INITIAL_SETTLE", False):
        return 0
    return int(getattr(task_module, "INITIAL_SETTLE_STEPS", 30))


def collect_contents(
    image_path: Path,
    history: list[dict[str, Any]],
    prompt: str,
    reference_image_paths: list[Path] | None = None,
    reference_image_path: Path | None = None,
) -> list[Any]:
    contents: list[Any] = []
    if history:
        summary = "\n".join(
            f"Step {item['step']}: action={item['action']} answer={item['answer']} "
            f"conf={item['confidence']:.2f} reasoning={item['reasoning']}"
            for item in history
        )
        contents.append("Action history so far:\n" + summary)
    references = list(reference_image_paths or [])
    if reference_image_path is not None:
        references.append(reference_image_path)
    for index, reference in enumerate(references, start=1):
        if reference.exists():
            label = "[QUESTION REFERENCE IMAGE - dataset render]" if len(references) == 1 else f"[QUESTION REFERENCE IMAGE {index}]"
            contents.append(label)
            contents.append(reference)
    recent = history[-5:]
    for item in recent:
        past = image_path.parent / item["image"]
        if past.exists():
            contents.append(f"[Past view from step {item['step']}]")
            contents.append(past)
        for extra_index, extra_path in enumerate(item.get("extra_image_paths") or [], start=1):
            extra = Path(extra_path)
            if extra.exists():
                contents.append(f"[Past extra view from step {item['step']} #{extra_index}]")
                contents.append(extra)
    contents.append(f"[CURRENT VIEW - step {len(history) + 1}]")
    contents.append(image_path)
    contents.append(prompt)
    return contents


def resolve_final_answer(history: list[dict[str, Any]]) -> tuple[str, int]:
    for item in reversed(history):
        answer = normalize_text(item.get("answer"))
        if answer and answer.lower() not in {"not sure", "unsure", "unknown"}:
            return answer, int(item["step"])
    if history:
        return normalize_text(history[-1].get("answer")) or "not sure", int(history[-1]["step"])
    return "not sure", -1


def output_path_for(task_module, payload: dict[str, Any], source_json: Path, results_root: Path) -> Path:
    scene, room = task_module.scene_room(payload)
    qid = task_module.question_id(payload, source_json)
    return results_root / task_module.TASK_NAME / scene / room / f"{qid}_answer.json"


def step_image_dir_for(task_module, payload: dict[str, Any], source_json: Path, step_image_root: Path) -> Path:
    scene, room = task_module.scene_room(payload)
    qid = task_module.question_id(payload, source_json)
    return step_image_root / task_module.TASK_NAME / scene / room / qid


def call_with_supported_args(func, *args, **kwargs):
    params = inspect.signature(func).parameters
    filtered_kwargs = {key: value for key, value in kwargs.items() if key in params}
    return func(*args, **filtered_kwargs)


def task_preprocess(task_module, payload: dict[str, Any], question_json: Path, config: ActiveExploreConfig) -> dict[str, Any]:
    hook = getattr(task_module, "preprocess", None)
    if hook is None:
        return {}
    state = call_with_supported_args(hook, payload, source_json=question_json, config=config)
    return state if isinstance(state, dict) else {}


def task_postprocess_env(task_module, env, payload: dict[str, Any], camera_info: dict[str, Any], task_state: dict[str, Any]) -> dict[str, Any]:
    hook = getattr(task_module, "postprocess_env", None)
    if hook is None:
        return {}
    result = call_with_supported_args(hook, env, payload, camera_info, task_state=task_state)
    return result if isinstance(result, dict) else {}


def task_reference_image_paths(task_module, payload: dict[str, Any], task_state: dict[str, Any]) -> list[Path]:
    hook = getattr(task_module, "reference_image_paths", None)
    if hook is not None:
        paths = call_with_supported_args(hook, payload, task_state=task_state)
        if paths is None:
            return []
        return [Path(path) for path in paths if path]
    hook = getattr(task_module, "reference_image_path", None)
    if hook is None:
        return []
    path = call_with_supported_args(hook, payload, task_state=task_state)
    return [Path(path)] if path else []


def task_build_system_prompt(task_module, payload: dict[str, Any], threshold: float, min_steps: int, camera_info: dict[str, Any], task_state: dict[str, Any]) -> str:
    return call_with_supported_args(
        task_module.build_system_prompt,
        payload,
        threshold,
        min_steps,
        camera_info,
        task_state=task_state,
    )


def task_build_force_choice_prompt(task_module, payload: dict[str, Any], camera_info: dict[str, Any], task_state: dict[str, Any]) -> str:
    return call_with_supported_args(task_module.build_force_choice_prompt, payload, camera_info, task_state=task_state)


def task_parse_model_output(task_module, parsed: dict[str, Any], payload: dict[str, Any], task_state: dict[str, Any]) -> dict[str, Any]:
    return call_with_supported_args(task_module.parse_model_output, parsed, payload=payload, task_state=task_state)


def task_score(task_module, payload: dict[str, Any], final_answer: dict[str, Any], camera_info: dict[str, Any], task_state: dict[str, Any]) -> dict[str, Any]:
    return call_with_supported_args(task_module.score, payload, final_answer, camera_info=camera_info, task_state=task_state)


def force_final_choice(
    task_module,
    model_client,
    payload: dict[str, Any],
    camera_info: dict[str, Any],
    image_path: Path,
    history: list[dict[str, Any]],
    config: ActiveExploreConfig,
    task_state: dict[str, Any],
    reference_image_paths: list[Path] | None = None,
) -> dict[str, Any]:
    prompt = task_build_force_choice_prompt(task_module, payload, camera_info, task_state)
    parsed, raw_text, _ = model_client.generate_json(
        contents=collect_contents(image_path, history, prompt, reference_image_paths=reference_image_paths),
        system_instruction="Return exactly one valid JSON object and nothing else.",
        response_schema=getattr(task_module, "FINAL_RESPONSE_SCHEMA", None),
        max_output_tokens=max(96, min(config.max_new_tokens, 192)),
        temperature=0.0,
        top_p=1.0,
        fallback={"answer": "not sure", "confidence": 0.0, "reasoning": "forced choice parse fallback"},
    )
    parsed = task_parse_model_output(task_module, parsed, payload, task_state)
    return {
        "answer": parsed.get("answer", "not sure"),
        "confidence": float(parsed.get("confidence", 0.0)),
        "reasoning": normalize_text(parsed.get("reasoning")) or "forced choice fallback",
        "raw_output": raw_text,
    }


def task_post_action_query(
    task_module,
    model_client,
    payload: dict[str, Any],
    camera_info: dict[str, Any],
    image_path: Path,
    history: list[dict[str, Any]],
    action_result: dict[str, Any],
    config: ActiveExploreConfig,
    task_state: dict[str, Any],
    reference_image_paths: list[Path] | None = None,
) -> dict[str, Any]:
    hook = getattr(task_module, "post_action_query", None)
    if hook is None:
        return {}
    result = call_with_supported_args(
        hook,
        model_client,
        payload,
        camera_info,
        image_path,
        history,
        action_result,
        config=config,
        task_state=task_state,
        reference_image_paths=reference_image_paths,
    )
    return result if isinstance(result, dict) else {}


def task_should_stop(task_module, parsed: dict[str, Any], history: list[dict[str, Any]], step: int, config: ActiveExploreConfig) -> tuple[bool, str]:
    hook = getattr(task_module, "should_stop", None)
    if hook is not None:
        return hook(parsed=parsed, history=history, step=step, max_steps=config.max_steps, min_steps=config.min_steps, threshold=config.threshold)

    confidence = float(parsed.get("confidence", 0.0))
    action = normalize_text(parsed.get("action"))
    if confidence >= config.threshold and step >= config.min_steps:
        return True, "confidence_threshold"
    if action == "stop":
        return True, "model_stop"
    if step == config.max_steps:
        return True, "max_steps"
    return False, ""


def task_resolve_final_answer(task_module, history: list[dict[str, Any]]) -> tuple[str, int]:
    hook = getattr(task_module, "resolve_final_answer", None)
    if hook is not None:
        return hook(history)
    return resolve_final_answer(history)


def task_needs_force_final_choice(task_module, answer: str, stop_reason: str) -> bool:
    hook = getattr(task_module, "needs_force_final_choice", None)
    if hook is not None:
        return bool(hook(answer=answer, stop_reason=stop_reason))
    return normalize_text(answer).lower() in {"not sure", "unsure", "unknown", ""}


def step_env(env, n: int = 10) -> None:
    idle = th.zeros(env.robots[0].action_dim, dtype=th.float32) if env.robots else []
    for _ in range(int(n)):
        env.step(idle)


def place_on_top(obj, target, env) -> bool:
    square_ori = th.tensor(SQUARE_ORI, dtype=th.float32)
    for _attempt in range(PLACEMENT_RETRIES):
        try:
            ok = sample_kinematics("onTop", obj, target, use_last_ditch_effort=True, use_trav_map=False)
            if ok:
                pos, _ = obj.get_position_orientation()
                obj.set_position_orientation(position=pos, orientation=square_ori)
                obj.keep_still()
                step_env(env, SETTLE_STEPS)
                return True
        except Exception:
            pass
    return False


def reset_object_to_pose(obj, pose: dict[str, Any], env) -> bool:
    if not pose or "position" not in pose:
        return False
    obj.set_position_orientation(
        position=th.tensor(pose["position"], dtype=th.float32),
        orientation=th.tensor(SQUARE_ORI, dtype=th.float32),
    )
    obj.keep_still()
    step_env(env, 20)
    return True


def object_display_names(task_module, payload: dict[str, Any], camera_info: dict[str, Any]) -> dict[str, str]:
    provider = getattr(task_module, "object_display_names", None)
    if provider is not None:
        return dict(provider(payload, camera_info))
    output = {}
    for spec in task_module.build_env_objects(payload):
        name = normalize_text(spec.get("name"))
        category = normalize_text(spec.get("category")).replace("_", " ")
        if name:
            output[name] = category or name
    return output


def resolve_scene_object_name(task_module, text: str, payload: dict[str, Any], camera_info: dict[str, Any]) -> str | None:
    text = normalize_text(text).lower()
    display = object_display_names(task_module, payload, camera_info)
    for object_name, display_name in sorted(display.items(), key=lambda item: -len(item[1])):
        lowered_display = display_name.lower()
        if lowered_display in text or text in lowered_display or object_name.lower() in text:
            return object_name
    for object_name, display_name in display.items():
        if any(word in text for word in display_name.lower().split() if len(word) > 3):
            return object_name
    return None


def execute_physical_action(task_module, env, payload: dict[str, Any], camera_info: dict[str, Any], action: str) -> dict[str, Any]:
    scene = env.scene
    action_lower = action.lower().strip()
    initial_poses = payload.get("initial_poses") or {}

    if action_lower.startswith("pick up "):
        obj_name = resolve_scene_object_name(task_module, action_lower[len("pick up "):], payload, camera_info)
        return {"handled": True, "operation": "pick_up", "object": obj_name}

    if "place" in action_lower and "on top of" in action_lower:
        left, right = action_lower.split("on top of", 1)
        obj_name = resolve_scene_object_name(task_module, left.replace("place", "").strip(), payload, camera_info)
        target_name = resolve_scene_object_name(task_module, right.strip(), payload, camera_info)
        obj = scene.object_registry("name", obj_name) if obj_name else None
        target = scene.object_registry("name", target_name) if target_name else None
        success = bool(obj is not None and target is not None and place_on_top(obj, target, env))
        return {
            "handled": True,
            "operation": "place_on_top",
            "object": obj_name,
            "target": target_name,
            "success": success,
        }

    if action_lower.startswith("put back "):
        obj_name = resolve_scene_object_name(task_module, action_lower[len("put back "):], payload, camera_info)
        obj = scene.object_registry("name", obj_name) if obj_name else None
        success = bool(obj is not None and reset_object_to_pose(obj, initial_poses.get(obj_name), env))
        return {"handled": True, "operation": "put_back", "object": obj_name, "success": success}

    return {"handled": False}


def execute_action(
    task_module,
    env,
    payload: dict[str, Any],
    camera_info: dict[str, Any],
    pos: np.ndarray,
    quat: np.ndarray,
    action: str,
    task_state: dict[str, Any] | None = None,
    step: int | None = None,
    step_image_dir: Path | None = None,
):
    task_action = getattr(task_module, "execute_task_action", None)
    if task_action is not None:
        result = call_with_supported_args(
            task_action,
            env,
            payload,
            camera_info,
            action,
            pos=pos,
            quat=quat,
            task_state=task_state or {},
            step=step,
            step_image_dir=step_image_dir,
        )
        if isinstance(result, dict) and result.get("handled"):
            next_pos = np.array(result.get("position", pos), dtype=float)
            next_quat = np.array(result.get("quaternion_xyzw", quat), dtype=float)
            return next_pos, next_quat, result
    if action in CAMERA_ACTIONS:
        next_pos, next_quat = apply_camera_action(pos, quat, action)
        return next_pos, next_quat, {"handled": True, "operation": "camera", "action": action}
    info = execute_physical_action(task_module, env, payload, camera_info, action)
    if info.get("handled"):
        return pos, quat, info
    return pos, quat, {"handled": False, "operation": "noop", "action": action}


def task_capture_image(
    task_module,
    env,
    payload: dict[str, Any],
    camera_info: dict[str, Any],
    pos: np.ndarray,
    quat: np.ndarray,
    image_path: Path,
    task_state: dict[str, Any],
) -> Path:
    hook = getattr(task_module, "capture_image", None)
    if hook is not None:
        result = call_with_supported_args(
            hook,
            env,
            payload,
            camera_info,
            pos,
            quat,
            image_path,
            task_state=task_state,
        )
        return Path(result) if result is not None else image_path
    set_camera(pos, quat)
    return capture_image(image_path)


def task_cleanup_runtime(task_module, env, payload: dict[str, Any], task_state: dict[str, Any]) -> bool:
    hook = getattr(task_module, "cleanup_runtime", None)
    if hook is None:
        return False
    result = call_with_supported_args(hook, env, payload, task_state=task_state)
    return bool(result)


def run_one(config: ActiveExploreConfig) -> dict[str, Any]:
    task_module = load_task_module(config.task)
    selected_question = select_question(config.metadata, config.question_index, config.json_root)
    question_json = selected_question.source_path
    source_label = selected_question.source_label
    payload = selected_question.payload

    output_path = output_path_for(task_module, payload, question_json, config.results_root)
    if output_path.exists() and not config.overwrite:
        with output_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    scene_name, room_name = task_module.scene_room(payload)
    task_state = task_preprocess(task_module, payload, question_json, config)
    if task_state.get("skip_reason"):
        return {
            "source_json": source_label,
            "source_metadata": str(config.metadata),
            "question_index": config.question_index,
            "task": getattr(task_module, "TASK_NAME", config.task),
            "scene": payload.get("scene"),
            "room": payload.get("room"),
            "question_id": task_module.question_id(payload, question_json),
            "skipped": True,
            "skip_reason": task_state["skip_reason"],
            "correct": None,
        }
    pos, quat, camera_info = task_module.initial_camera(payload)
    objects = task_module.build_env_objects(payload)
    env = None
    history: list[dict[str, Any]] = []
    final_answer: dict[str, Any] | None = None
    env_postprocess: dict[str, Any] = {}
    step_image_dir = step_image_dir_for(task_module, payload, question_json, config.step_image_root)
    default_model = "gpt-5" if config.provider == "gpt" else getattr(task_module, "DEFAULT_MODEL", "gemini-2.5-flash")
    model_name = config.model or default_model
    model_client = build_model_client(config.provider, config.api_key, model_name)

    try:
        full_scene = bool(getattr(task_module, "FULL_SCENE", False))
        env = og.Environment(configs=task_build_env_config(task_module, scene_name, room_name, config.robot, objects, full_scene=full_scene))
        if not getattr(task_module, "DISABLE_VIEWER_CAMERA_MODALITIES", False):
            for modality in ["seg_semantic", "seg_instance", "seg_instance_id"]:
                try:
                    og.sim._viewer_camera.add_modality(modality)
                except Exception:
                    pass
        for _ in range(task_initial_settle_steps(task_module)):
            og.sim.step()
        env_postprocess = task_postprocess_env(task_module, env, payload, camera_info, task_state)
        camera_override = env_postprocess.get("camera_override") if isinstance(env_postprocess, dict) else None
        if isinstance(camera_override, dict) and camera_override.get("position") and camera_override.get("quaternion_xyzw"):
            pos = np.array(camera_override["position"], dtype=float)
            quat = np.array(camera_override["quaternion_xyzw"], dtype=float)
            camera_info = {**camera_info, "postprocess_camera_override": camera_override}
        reference_image_paths = task_reference_image_paths(task_module, payload, task_state)

        for step in range(1, config.max_steps + 1):
            image_path = task_capture_image(
                task_module,
                env,
                payload,
                camera_info,
                pos,
                quat,
                step_image_dir / f"step_{step:03d}.png",
                task_state,
            )
            prompt = task_build_system_prompt(task_module, payload, config.threshold, config.min_steps, camera_info, task_state)
            parsed, raw_text, finish_reason = model_client.generate_json(
                contents=collect_contents(image_path, history, prompt, reference_image_paths=reference_image_paths),
                system_instruction="Return exactly one valid JSON object and nothing else.",
                response_schema=getattr(task_module, "ACTION_RESPONSE_SCHEMA", None),
                max_output_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                fallback={"action": "move_forward", "answer": "not sure", "confidence": 0.0, "reasoning": "parse fallback"},
            )
            parsed = task_parse_model_output(task_module, parsed, payload, task_state)
            action = parsed["action"]
            answer = parsed["answer"]
            confidence = float(parsed["confidence"])
            reasoning = parsed["reasoning"]
            pos_before = pos.copy()
            quat_before = quat.copy()

            should_stop, stop_reason = task_should_stop(task_module, parsed, history, step, config)

            action_result = {"handled": False}
            if not should_stop:
                pos, quat, action_result = execute_action(
                    task_module,
                    env,
                    payload,
                    camera_info,
                    pos,
                    quat,
                    action,
                    task_state=task_state,
                    step=step,
                    step_image_dir=step_image_dir,
                )

            history.append({
                "step": step,
                "action": action,
                "answer": answer,
                "confidence": confidence,
                "reasoning": reasoning,
                "image": image_path.name,
                "camera": {"position": pos_before.tolist(), "quaternion_xyzw": quat_before.tolist()},
                "action_result": action_result,
                "extra_image_paths": action_result.get("extra_image_paths") or [],
                "raw_output": raw_text,
                "finish_reason": finish_reason,
            })

            post_action = {}
            if not should_stop:
                post_action = task_post_action_query(
                    task_module,
                    model_client,
                    payload,
                    camera_info,
                    image_path,
                    history,
                    action_result,
                    config,
                    task_state,
                    reference_image_paths=reference_image_paths,
                )
                if post_action.get("history_update"):
                    history[-1].update(post_action["history_update"])
                    answer = history[-1].get("answer", answer)
                    confidence = float(history[-1].get("confidence", confidence))
                    reasoning = history[-1].get("reasoning", reasoning)
                if post_action.get("final_answer"):
                    final_answer = post_action["final_answer"]
                    break
                if post_action.get("should_stop"):
                    should_stop = True
                    stop_reason = normalize_text(post_action.get("stop_reason")) or "post_action_query"

            if should_stop:
                resolved_answer, answer_step = task_resolve_final_answer(task_module, history)
                if task_needs_force_final_choice(task_module, resolved_answer, stop_reason):
                    forced = force_final_choice(
                        task_module,
                        model_client,
                        payload,
                        camera_info,
                        image_path,
                        history,
                        config,
                        task_state,
                        reference_image_paths=reference_image_paths,
                    )
                    resolved_answer = forced["answer"]
                    answer_step = step
                    confidence = forced["confidence"]
                    reasoning = forced["reasoning"]
                    history.append({
                        "step": step,
                        "action": "force_final_choice",
                        "answer": resolved_answer,
                        "confidence": confidence,
                        "reasoning": reasoning,
                        "image": image_path.name,
                        "camera": {"position": pos.tolist(), "quaternion_xyzw": quat.tolist()},
                        "raw_output": forced["raw_output"],
                    })
                    stop_reason = f"{stop_reason}_forced_choice"
                final_answer = {
                    "answer": resolved_answer,
                    "answer_step": answer_step,
                    "confidence": confidence,
                    "reasoning": reasoning,
                    "steps": step,
                    "stopped_by": stop_reason,
                }
                break
            time.sleep(0.1)

        final_answer = final_answer or {"answer": "not sure", "answer_step": -1, "confidence": 0.0, "reasoning": "no answer", "steps": len(history), "stopped_by": "empty"}
        score = task_score(task_module, payload, final_answer, camera_info, task_state)
        result = {
            "source_json": source_label,
            "source_metadata": str(config.metadata),
            "question_index": config.question_index,
            "task": task_module.TASK_NAME,
            "scene": scene_name,
            "room": room_name,
            "question_id": task_module.question_id(payload, question_json),
            "provider": config.provider,
            "model": model_name,
            "final_answer": final_answer,
            "history": history,
            "step_image_dir": str(step_image_dir),
            "camera_info": camera_info,
            "env_postprocess": env_postprocess,
            **score,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        return result
    finally:
        active_exception = sys.exc_info()[0] is not None
        handled_cleanup = False
        try:
            handled_cleanup = task_cleanup_runtime(task_module, env, payload if "payload" in locals() else {}, task_state if "task_state" in locals() else {})
        except Exception:
            handled_cleanup = False
        try:
            if not active_exception and not handled_cleanup and getattr(og, "app", None) is not None:
                og.shutdown()
        except BaseException:
            pass


def main(argv: list[str] | None = None) -> int:
    config = parse_args(argv)
    try:
        result = run_one(config)
    except Exception as exc:
        print(f"[fatal] {type(exc).__name__}: {exc}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        return 2
    correct = result.get("correct")
    print(json.dumps({
        "source_json": result.get("source_json"),
        "skipped": result.get("skipped", False),
        "skip_reason": result.get("skip_reason"),
        "final_answer": result.get("final_answer"),
        "correct": correct,
        "step_image_dir": result.get("step_image_dir"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
