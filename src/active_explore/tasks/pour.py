from __future__ import annotations

import random
from pathlib import Path
from typing import Any
import json

import numpy as np
import omnigibson as og
import omnigibson.object_states as object_states
import omnigibson.lazy as lazy
import omnigibson.utils.transform_utils as T
import torch as th
from omnigibson.macros import gm
from omnigibson.objects.dataset_object import DatasetObject
from omnigibson.prims.xform_prim import XFormPrim
from omnigibson.utils.asset_utils import decrypted
from scipy.spatial.transform import Rotation


gm.USE_ENCRYPTED_ASSETS = True
gm.USE_GPU_DYNAMICS = True
gm.ENABLE_FLATCACHE = False

TASK_NAME = "pour"
DEFAULT_MODEL = "gemini-3.1-pro-preview"
MAX_BBOX = 0.3
OBJECT_GAP = 0.3


def normalize_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def display_category(value: Any) -> str:
    return normalize_text(value).replace("_", " ")


def side_entry(payload: dict[str, Any], side: str) -> dict[str, Any]:
    for key in ("obj1", "obj2"):
        item = payload.get(key) or {}
        if item.get("side") == side:
            return item
    category = payload.get(f"{side}_category")
    model = payload.get(f"{side}_model")
    return {"category": category, "model": model, "side": side}


def left_entry(payload: dict[str, Any]) -> dict[str, Any]:
    return side_entry(payload, "left")


def right_entry(payload: dict[str, Any]) -> dict[str, Any]:
    return side_entry(payload, "right")


def left_label(payload: dict[str, Any]) -> str:
    return display_category(left_entry(payload).get("category"))


def right_label(payload: dict[str, Any]) -> str:
    return display_category(right_entry(payload).get("category"))


def preprocess(payload: dict[str, Any], source_json: Path | None = None, config: Any | None = None) -> dict[str, Any]:
    if payload.get("skip_reason"):
        return {"skip_reason": payload["skip_reason"]}
    for side in ("left", "right"):
        entry = side_entry(payload, side)
        if not entry.get("category") or not entry.get("model"):
            return {"skip_reason": f"missing_{side}_container_category_or_model"}
    if payload.get("_ground_truth") not in {"left", "right"}:
        return {"skip_reason": "missing_or_invalid_gt_larger"}
    return {"run_rng": random.Random(int(payload.get("run_idx", 0)) + 42)}


ACTION_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string"},
        "answer": {"type": "string"},
        "reasoning": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["action", "answer", "reasoning", "confidence"],
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


def scene_room(payload: dict[str, Any]) -> tuple[str, str]:
    return normalize_text(payload.get("scene")) or "unknown_scene", normalize_text(payload.get("room")) or "unknown_room"


def question_id(payload: dict[str, Any], source_path: Path) -> str:
    return normalize_text(payload.get("question_id")) or f"run_{int(payload.get('run_idx', 0)):03d}"


def get_scale(category: str, model: str) -> float:
    inventory_path = Path(__file__).resolve().parents[1] / "bddl3" / "bddl" / "generated_data" / "object_inventory.json"
    if inventory_path.exists():
        with inventory_path.open("r", encoding="utf-8") as f:
            sizes = json.load(f).get("bounding_box_sizes", {})
        bbox = sizes.get(model)
        if bbox:
            return min(float(MAX_BBOX / max(bbox)), 1.0)
    usd_path = DatasetObject.get_usd_path(category=category, model=model).replace(".usd", ".encrypted.usd")
    try:
        with decrypted(usd_path) as fpath:
            stage = lazy.pxr.Usd.Stage.Open(fpath)
            prim = stage.GetDefaultPrim()
            bounding_box = th.tensor(prim.GetAttribute("ig:nativeBB").Get())
        scale = MAX_BBOX / th.max(bounding_box)
        return min(float(scale), 1.0)
    except Exception:
        return 1.0


def build_env_objects(payload: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for side, name in (("left", "obj1"), ("right", "obj2")):
        entry = side_entry(payload, side)
        category = entry["category"]
        model = entry["model"]
        scale = get_scale(category, model)
        output.append(
            {
                "type": "DatasetObject",
                "name": name,
                "category": category,
                "model": model,
                "kinematic_only": False,
                "fixed_base": True,
                "scale": [scale, scale, scale],
                "position": [0.0 if side == "left" else 1.0, 0.0, 50.0],
                "orientation": [0.0, 0.0, 0.0, 1.0],
            }
        )
    return output


def initial_camera(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    return (
        np.array([0.0, -1.0, 1.0], dtype=float),
        np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
        {"selection": "placeholder_before_pour_postprocess"},
    )


def generate_box(box_half_extent, floor_z=0.0, center_x=0.0, center_y=0.0, index_offset=0):
    box_half_extent = box_half_extent.float()
    plane_centers = th.tensor([[1, 0, 1], [0, 1, 1], [-1, 0, 1], [0, -1, 1]]) * box_half_extent
    plane_centers[:, 0] += center_x
    plane_centers[:, 1] += center_y
    plane_centers[:, 2] += floor_z
    for i, pc in enumerate(plane_centers):
        idx = i + index_offset
        plane = lazy.omni.isaac.core.objects.ground_plane.GroundPlane(
            prim_path=f"/World/plane_{idx}",
            name=f"plane_{idx}",
            z_position=0,
            size=box_half_extent[2].item(),
            color=None,
            visible=False,
        )
        plane_as_prim = XFormPrim(relative_prim_path=f"/plane_{idx}", name=plane.name)
        plane_as_prim.load(None)
        horiz_dir = pc - th.tensor([center_x, center_y, floor_z + box_half_extent[2]], dtype=th.float32)
        plane_z = -1 * horiz_dir / th.norm(horiz_dir)
        plane_x = th.tensor([0, 0, 1], dtype=th.float32)
        plane_y = th.linalg.cross(plane_z, plane_x)
        plane_mat = th.stack([plane_x, plane_y, plane_z], dim=1)
        plane_as_prim.set_position_orientation(pc, T.mat2quat(plane_mat))


def generate_particles_in_box(water, box_half_extent, floor_z=0.0, center_x=0.0, center_y=0.0):
    particle_radius = water.particle_radius
    low = th.tensor([-1, -1, 0]) * box_half_extent + th.tensor([center_x, center_y, floor_z])
    high = th.tensor([1, 1, 2]) * box_half_extent + th.tensor([center_x, center_y, floor_z + 0.05])
    extent = high - low
    n_particles_per_axis = (extent / (2 * particle_radius)).long()
    if not th.all(n_particles_per_axis > 0):
        raise ValueError(f"Box too small for particle radius {particle_radius}")
    arrs = [th.arange(l + particle_radius, h - particle_radius + 1e-10, particle_radius * 2) for l, h, _n in zip(low, high, n_particles_per_axis)]
    particle_positions = th.stack(th.meshgrid(*arrs, indexing="ij")).view(3, -1).t()
    water.generate_particles(positions=particle_positions)


def check_in_contact(system, positions):
    in_contact = th.zeros(len(positions), dtype=bool)
    for idx, pos in enumerate(positions):
        in_contact[idx] = og.sim.psqi.overlap_sphere_any(system.particle_contact_radius * 0.8, pos.numpy().copy())
    return in_contact


def get_particles_in_obj(obj, water, particle_point_offsets):
    aabb_min, aabb_max = obj.aabb
    all_particles, all_orients = water.get_particles_position_orientation()
    all_particles_t = th.tensor(all_particles)
    all_orients_t = th.tensor(all_orients)
    in_obj = th.zeros(len(all_particles_t), dtype=th.bool)
    if len(all_particles_t) == 0:
        return all_particles_t, all_orients_t, in_obj
    not_in_contact_mask = check_in_contact(water, all_particles_t) == 0
    particles = all_particles_t[not_in_contact_mask]
    if len(particles) == 0:
        return all_particles_t, all_orients_t, in_obj
    offsets = particles.unsqueeze(1) + particle_point_offsets.unsqueeze(0)
    inside = (th.all(offsets <= (aabb_max + th.tensor([0, 0, water.particle_radius])), dim=2) & th.all(offsets >= aabb_min, dim=2)).any(dim=1)
    not_in_contact_indices = th.where(not_in_contact_mask)[0]
    in_obj[not_in_contact_indices[inside]] = True
    return all_particles_t, all_orients_t, in_obj


def look_at_quaternion(eye_pos, target_pos, up=np.array([0, 0, 1])):
    forward = np.array(target_pos) - np.array(eye_pos)
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:
        up = np.array([0, 1, 0])
        right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    true_up = np.cross(right, forward)
    true_up = true_up / np.linalg.norm(true_up)
    return Rotation.from_matrix(np.column_stack([right, true_up, -forward])).as_quat()


def side_camera(obj1, obj2) -> tuple[list[float], list[float]]:
    bmin1, bmax1 = [x.cpu().numpy() if hasattr(x, "cpu") else x.numpy() for x in obj1.aabb]
    bmin2, bmax2 = [x.cpu().numpy() if hasattr(x, "cpu") else x.numpy() for x in obj2.aabb]
    centre1 = (bmin1 + bmax1) / 2.0
    centre2 = (bmin2 + bmax2) / 2.0
    mid_x = (centre1[0] + centre2[0]) / 2.0
    mid_y = (centre1[1] + centre2[1]) / 2.0
    mid_z = max(centre1[2], centre2[2]) + 0.1
    cam_eye = np.array([mid_x, mid_y - 1.0, mid_z])
    cam_target = np.array([mid_x, mid_y, mid_z])
    return cam_eye.tolist(), look_at_quaternion(cam_eye, cam_target).tolist()


def postprocess_env(env, payload: dict[str, Any], camera_info: dict[str, Any], task_state: dict[str, Any] | None = None) -> dict[str, Any]:
    task_state = task_state or {}
    obj1 = env.scene.object_registry("name", "obj1")
    obj2 = env.scene.object_registry("name", "obj2")
    floor = env.scene.object_registry("name", "floors_ptwlei_0")
    water = env.scene.get_system("water")
    if obj1 is None or obj2 is None or floor is None or water is None:
        raise ValueError("Pour environment missing obj1/obj2/floor/water")

    obj1.states[object_states.OnTop].set_value(floor, True)
    for _ in range(30):
        og.sim.step()
    aabb_extent1 = th.tensor(obj1.aabb_extent)
    obj_bbox_center1 = th.tensor(obj1.aabb_center)
    obj_bbox_bottom1 = obj_bbox_center1 - th.tensor([0, 0, aabb_extent1[2] / 2])
    floor_z1 = obj_bbox_bottom1[2].item() + 0.05
    bmin1, bmax1 = [x.cpu().numpy() if hasattr(x, "cpu") else x.numpy() for x in obj1.aabb]
    half_x1 = (bmax1[0] - bmin1[0]) / 2.0
    pos1_settled = th.tensor(obj1.get_position_orientation()[0])
    obj_current_pos1 = th.tensor(obj1.get_position_orientation()[0])

    obj2.states[object_states.OnTop].set_value(floor, True)
    for _ in range(30):
        og.sim.step()
    aabb_extent2 = th.tensor(obj2.aabb_extent)
    obj_bbox_center2 = th.tensor(obj2.aabb_center)
    obj_bbox_bottom2 = obj_bbox_center2 - th.tensor([0, 0, aabb_extent2[2] / 2])
    floor_z2 = obj_bbox_bottom2[2].item() + 0.05
    bmin2, bmax2 = [x.cpu().numpy() if hasattr(x, "cpu") else x.numpy() for x in obj2.aabb]
    half_x2 = (bmax2[0] - bmin2[0]) / 2.0
    pos2_settled = th.tensor(obj2.get_position_orientation()[0])

    obj1_pos = th.tensor([obj_current_pos1[0].item(), obj_current_pos1[1].item(), pos1_settled[2].item() + 0.04])
    obj1.set_position_orientation(obj1_pos, th.tensor([0, 0, 0, 1]))
    for _ in range(30):
        og.sim.step()
    obj2_x = obj_current_pos1[0].item() + half_x1 + OBJECT_GAP + half_x2
    obj2_pos = th.tensor([obj2_x, obj_current_pos1[1].item(), pos2_settled[2].item() + 0.04])
    obj2.set_position_orientation(obj2_pos, th.tensor([0, 0, 0, 1]))
    for _ in range(30):
        og.sim.step()

    particle_point_offsets = th.stack([e * side * water.particle_radius for e in th.eye(3) for side in [-1, 1]] + [th.zeros(3)])
    aabb_ext1 = th.tensor(obj1.aabb_extent)
    aabb_ext2 = th.tensor(obj2.aabb_extent)
    bmin1_f, bmax1_f = [x.cpu().numpy() if hasattr(x, "cpu") else x.numpy() for x in obj1.aabb]
    bmin2_f, bmax2_f = [x.cpu().numpy() if hasattr(x, "cpu") else x.numpy() for x in obj2.aabb]
    combined_x_min = min(float(bmin1_f[0]), float(bmin2_f[0]))
    combined_x_max = max(float(bmax1_f[0]), float(bmax2_f[0]))
    combined_y_min = min(float(bmin1_f[1]), float(bmin2_f[1]))
    combined_y_max = max(float(bmax1_f[1]), float(bmax2_f[1]))
    box_center_x = (combined_x_min + combined_x_max) / 2.0
    box_center_y = (combined_y_min + combined_y_max) / 2.0
    box_half_x = (combined_x_max - combined_x_min) / 2.0 + 0.05
    box_half_y = (combined_y_max - combined_y_min) / 2.0 + 0.05
    box_half_z = max(aabb_ext1[2].item(), aabb_ext2[2].item()) * 0.55
    box_half_z = max(box_half_z, max(aabb_ext1[2].item(), aabb_ext2[2].item()) / 2 + 2.1 * water.particle_radius)
    box_half_extent = th.tensor([box_half_x, box_half_y, box_half_z])
    box_floor_z = min(floor_z1, floor_z2)
    generate_box(box_half_extent, floor_z=box_floor_z, center_x=box_center_x, center_y=box_center_y, index_offset=0)
    og.sim.step()

    obj_free_pos1 = obj1_pos + th.tensor([0, 0, 1.1 * aabb_ext1[2] + aabb_ext1[2]])
    obj_free_pos2 = obj2_pos + th.tensor([0, 0, 1.1 * aabb_ext2[2] + aabb_ext2[2]])
    free_z = max(obj_free_pos1[2].item(), obj_free_pos2[2].item())
    obj1.set_position_orientation(th.tensor([obj1_pos[0].item(), obj1_pos[1].item(), free_z]), th.tensor([0, 0, 0, 1]))
    obj2.set_position_orientation(th.tensor([obj2_pos[0].item(), obj2_pos[1].item(), free_z]), th.tensor([0, 0, 0, 1]))
    for _ in range(30):
        og.sim.step()

    eye, quat = side_camera(obj1, obj2)
    task_state.update(
        {
            "obj1": obj1,
            "obj2": obj2,
            "water": water,
            "free_pos1": th.tensor(obj1.get_position_orientation()[0]),
            "free_pos2": th.tensor(obj2.get_position_orientation()[0]),
            "dip_pos1": obj1_pos,
            "dip_pos2": obj2_pos,
            "box_half_extent": box_half_extent,
            "box_floor_z": box_floor_z,
            "box_center_x": box_center_x,
            "box_center_y": box_center_y,
            "particle_point_offsets": particle_point_offsets,
            "n_left": 0,
            "n_right": 0,
            "box_index_offset": 0,
            "run_rng": task_state.get("run_rng") or random.Random(int(payload.get("run_idx", 0)) + 42),
        }
    )
    return {"camera_override": {"position": eye, "quaternion_xyzw": quat}, "pour_setup": True}


def fill_single(obj, free_pos, dip_pos, water, box_half_extent, box_floor_z, box_center_x, box_center_y, particle_point_offsets):
    generate_particles_in_box(water, box_half_extent, floor_z=box_floor_z, center_x=box_center_x, center_y=box_center_y)
    for _ in range(100):
        og.sim.step()
    obj.set_position_orientation(position=dip_pos)
    for _ in range(100):
        og.sim.step()
    joint_limits = {}
    for jname, joint in obj.joints.items():
        if joint.has_limit:
            joint_limits[jname] = (joint.lower_limit, joint.upper_limit)
            joint.set_pos(joint.upper_limit)
            joint.lower_limit = joint.upper_limit - 0.001
    og.sim.update_handles()
    for i in range(100):
        for jname, joint in obj.joints.items():
            if jname in joint_limits:
                lower, upper = joint_limits[jname]
                ipos = upper - (i / 100) * (upper - lower)
                joint.lower_limit = ipos - 0.001
                joint.upper_limit = ipos
                joint.set_pos(ipos)
        og.sim.step()
    for _ in range(100):
        og.sim.step()
    free_z = free_pos[2].item()
    while True:
        cur_pos = th.tensor(obj.get_position_orientation()[0])
        obj.set_position_orientation(position=cur_pos + th.tensor([0, 0, 0.01 * og.sim.get_rendering_dt()]))
        og.sim.step()
        if obj.get_position_orientation()[0][2] > free_z:
            break
    for _ in range(180):
        og.sim.step()
    _, _, in_obj = get_particles_in_obj(obj, water, particle_point_offsets)
    return int(in_obj.sum().item())


def pour_container_into(src_obj, dst_obj, dst_free_pos, dst_dip_pos, water, particle_point_offsets, run_rng):
    dst_obj.set_position_orientation(position=dst_dip_pos)
    for _ in range(100):
        og.sim.step()
    joint_limits = {}
    for jname, joint in dst_obj.joints.items():
        if joint.has_limit:
            joint_limits[jname] = (joint.lower_limit, joint.upper_limit)
            joint.set_pos(joint.upper_limit)
            joint.lower_limit = joint.upper_limit - 0.001
    og.sim.update_handles()
    for i in range(100):
        for jname, joint in dst_obj.joints.items():
            if jname in joint_limits:
                lower, upper = joint_limits[jname]
                ipos = upper - (i / 100) * (upper - lower)
                joint.lower_limit = ipos - 0.001
                joint.upper_limit = ipos
                joint.set_pos(ipos)
        og.sim.step()
    for _ in range(100):
        og.sim.step()
    free_z = dst_free_pos[2].item()
    while True:
        cur_pos = th.tensor(dst_obj.get_position_orientation()[0])
        dst_obj.set_position_orientation(position=cur_pos + th.tensor([0, 0, 0.01 * og.sim.get_rendering_dt()]))
        og.sim.step()
        if dst_obj.get_position_orientation()[0][2] > free_z:
            break
    for _ in range(180):
        og.sim.step()
    all_p, all_o, in_src = get_particles_in_obj(src_obj, water, particle_point_offsets)
    _, _, in_dst = get_particles_in_obj(dst_obj, water, particle_point_offsets)
    n_src = int(in_src.sum().item())
    n_dst = int(in_dst.sum().item())
    exile = all_p.clone()
    if n_src <= n_dst:
        dst_indices = th.where(in_dst)[0].tolist()
        run_rng.shuffle(dst_indices)
        keep_in_dst = set(dst_indices[:n_src])
        for idx in dst_indices:
            if idx not in keep_in_dst:
                exile[idx, 2] = -100.0
        exile[in_src, 2] = -100.0
        final_src, final_dst, scenario = 0, n_src, "poured_completely"
    else:
        src_indices = th.where(in_src)[0].tolist()
        run_rng.shuffle(src_indices)
        keep_in_src = set(src_indices[: n_src - n_dst])
        for idx in src_indices:
            if idx not in keep_in_src:
                exile[idx, 2] = -100.0
        final_src, final_dst, scenario = n_src - n_dst, n_dst, "poured_overflow"
    water.set_particles_position_orientation(positions=exile, orientations=all_o)
    for _ in range(120):
        og.sim.step()
    return final_src, final_dst, scenario


def exile_all_water(water):
    all_p, all_o = water.get_particles_position_orientation()
    if len(all_p) == 0:
        return
    all_p_t = th.tensor(all_p)
    all_p_t[:, 2] = -100.0
    water.set_particles_position_orientation(positions=all_p_t, orientations=th.tensor(all_o))
    for _ in range(30):
        og.sim.step()


def build_system_prompt(payload: dict[str, Any], threshold: float, min_steps: int, camera_info: dict[str, Any] | None = None, task_state: dict[str, Any] | None = None) -> str:
    return "\n".join(
        [
            "You are a spatial reasoning agent. Determine which of two containers has a larger volume by interacting with them.",
            f"The {left_label(payload)} is on the LEFT. The {right_label(payload)} is on the RIGHT.",
            "",
            "Available actions:",
            f'  fill water in {left_label(payload)} | fill water in {right_label(payload)}',
            f'  pour {left_label(payload)} into {right_label(payload)} | pour {right_label(payload)} into {left_label(payload)}',
            "  move_forward | move_backward | move_left | move_right | move_up | move_down",
            "  turn_left | turn_right | turn_up | turn_down | stop",
            "",
            "Use camera actions only if you cannot see the containers. Focus on water actions.",
            "After each action, observe carefully what changed.",
            "Output ONLY valid JSON:",
            '{"action": "<action>", "reasoning": "<one or two sentences>", "answer": "<left container, right container, or unsure>", "confidence": <0.0-1.0>}',
            f"Output confidence >= {threshold:.2f} only when certain from experiment results.",
        ]
    )


def build_force_choice_prompt(payload: dict[str, Any], camera_info: dict[str, Any] | None = None, task_state: dict[str, Any] | None = None) -> str:
    return "\n".join(
        [
            "Exploration budget is exhausted.",
            f"Choose which container has larger volume: {left_label(payload)} on the left, or {right_label(payload)} on the right.",
            "Do not answer unsure.",
            '{"answer": "<left or right container>", "confidence": <0.0-1.0>, "reasoning": "<brief explanation>"}',
        ]
    )


def answer_side(value: Any, payload: dict[str, Any]) -> str | None:
    text = normalize_text(value).lower().replace("_", " ")
    if not text or text in {"unsure", "not sure", "unknown"}:
        return None
    left = left_label(payload).lower()
    right = right_label(payload).lower()
    if "left" in text or left in text:
        return "left"
    if "right" in text or right in text:
        return "right"
    return None


def parse_model_output(parsed: dict[str, Any], payload: dict[str, Any] | None = None, task_state: dict[str, Any] | None = None) -> dict[str, Any]:
    try:
        confidence = float(parsed.get("confidence", 0.0))
    except Exception:
        confidence = 0.0
    action = normalize_text(parsed.get("action")).lower() or "move_forward"
    if action == "<end>":
        action = "stop"
    side = answer_side(parsed.get("answer"), payload) if isinstance(payload, dict) else None
    return {
        **parsed,
        "action": action,
        "answer": side or "unsure",
        "conclusive": side is not None,
        "confidence": max(0.0, min(1.0, confidence)),
        "reasoning": normalize_text(parsed.get("reasoning")) or "no reasoning provided",
    }


def execute_task_action(env, payload: dict[str, Any], camera_info: dict[str, Any], action: str, pos: np.ndarray, quat: np.ndarray, task_state: dict[str, Any] | None = None) -> dict[str, Any]:
    state = task_state or {}
    action_lower = normalize_text(action).lower()
    if action_lower.startswith("fill water in ") or action_lower.startswith("fill "):
        prefix = "fill water in " if action_lower.startswith("fill water in ") else "fill "
        target = action_lower[len(prefix) :].strip()
        if left_label(payload).lower() in target or "left" in target:
            state["n_left"] = 1
            state["n_right"] = 0
            side = "left"
        elif right_label(payload).lower() in target or "right" in target:
            state["n_right"] = 1
            state["n_left"] = 0
            side = "right"
        else:
            side = None
        return {"handled": True, "operation": "fill_simulated", "side": side, "n_left": state.get("n_left"), "n_right": state.get("n_right"), "position": pos.tolist(), "quaternion_xyzw": quat.tolist()}
    if action_lower.startswith("pour ") and " into " in action_lower:
        src_name, _dst_name = action_lower[len("pour ") :].split(" into ", 1)
        if left_label(payload).lower() in src_name or "left" in src_name:
            state["n_left"], state["n_right"] = 0, 1
            src_side = "left"
        elif right_label(payload).lower() in src_name or "right" in src_name:
            state["n_right"], state["n_left"] = 0, 1
            src_side = "right"
        else:
            return {"handled": True, "operation": "pour", "success": False, "reason": "unknown_source", "position": pos.tolist(), "quaternion_xyzw": quat.tolist()}
        return {"handled": True, "operation": "pour_simulated", "source_side": src_side, "scenario": "simulated", "n_left": state.get("n_left"), "n_right": state.get("n_right"), "position": pos.tolist(), "quaternion_xyzw": quat.tolist()}
    return {"handled": False}


def should_stop(parsed: dict[str, Any], history: list[dict[str, Any]], step: int, max_steps: int, min_steps: int, threshold: float) -> tuple[bool, str]:
    if normalize_text(parsed.get("action")).lower() == "stop":
        return True, "model_stop"
    if bool(parsed.get("conclusive")) and float(parsed.get("confidence", 0.0)) >= threshold:
        return True, "confidence_threshold"
    if step == max_steps:
        return True, "max_steps"
    return False, ""


def resolve_final_answer(history: list[dict[str, Any]]) -> tuple[str, int]:
    for item in reversed(history):
        answer = normalize_text(item.get("answer")).lower()
        if answer in {"left", "right"}:
            return answer, int(item["step"])
    return "unsure", int(history[-1]["step"]) if history else -1


def needs_force_final_choice(answer: str, stop_reason: str) -> bool:
    return normalize_text(answer).lower() not in {"left", "right"}


def score(payload: dict[str, Any], final_answer: dict[str, Any], camera_info: dict[str, Any] | None = None, task_state: dict[str, Any] | None = None) -> dict[str, Any]:
    predicted = normalize_text((final_answer or {}).get("answer")).lower()
    target = normalize_text(payload.get("gt_larger") or payload.get("_ground_truth")).lower()
    return {
        "task_type": "pour",
        "question": normalize_text(payload.get("_question")),
        "left_category": left_entry(payload).get("category"),
        "right_category": right_entry(payload).get("category"),
        "predicted_side": predicted if predicted in {"left", "right"} else None,
        "gt_larger": target,
        "correct": predicted == target if predicted in {"left", "right"} and target in {"left", "right"} else None,
    }
