"""
batch_pour_compare.py
"""

import argparse
import json
import os
import random
import sys
import traceback

import cv2
import torch as th
import numpy as np
from scipy.spatial.transform import Rotation

import omnigibson as og
import omnigibson.object_states as object_states
from omnigibson.macros import gm
from omnigibson.objects.dataset_object import DatasetObject
from omnigibson.prims.xform_prim import XFormPrim
from omnigibson.utils.asset_utils import decrypted
import omnigibson.utils.transform_utils as T
import omnigibson.lazy as lazy

gm.HEADLESS             = False
gm.USE_ENCRYPTED_ASSETS = True
gm.USE_GPU_DYNAMICS     = True
gm.ENABLE_FLATCACHE     = False

MAX_BBOX        = 0.3
OBJECT_GAP      = 0.3
TARGET_CAP_DIFF = 200


def generate_box(box_half_extent, floor_z=0.0, center_x=0.0, center_y=0.0, index_offset=0):
    plane_centers = (
        th.tensor([[1, 0, 1], [0, 1, 1], [-1, 0, 1], [0, -1, 1]])
        * box_half_extent
    )
    plane_centers[:, 0] += center_x
    plane_centers[:, 1] += center_y
    plane_centers[:, 2] += floor_z
    for i, pc in enumerate(plane_centers):
        idx = i + index_offset
        plane = lazy.omni.isaac.core.objects.ground_plane.GroundPlane(
            prim_path=f"/World/plane_{idx}", name=f"plane_{idx}",
            z_position=0, size=box_half_extent[2].item(), color=None, visible=False,
        )
        plane_as_prim = XFormPrim(relative_prim_path=f"/plane_{idx}", name=plane.name)
        plane_as_prim.load(None)
        horiz_dir  = pc - th.tensor([center_x, center_y, floor_z + box_half_extent[2]])
        plane_z    = -1 * horiz_dir / th.norm(horiz_dir)
        plane_x    = th.tensor([0, 0, 1], dtype=th.float32)
        plane_y    = th.cross(plane_z, plane_x)
        plane_mat  = th.stack([plane_x, plane_y, plane_z], dim=1)
        plane_quat = T.mat2quat(plane_mat)
        plane_as_prim.set_position_orientation(pc, plane_quat)


def generate_particles_in_box(water, box_half_extent, floor_z=0.0, center_x=0.0, center_y=0.0):
    particle_radius = water.particle_radius
    low  = th.tensor([-1, -1, 0]) * box_half_extent + th.tensor([center_x, center_y, floor_z])
    high = th.tensor([1, 1, 2])   * box_half_extent + th.tensor([center_x, center_y, floor_z + 0.05])
    extent = high - low
    n_particles_per_axis = (extent / (2 * particle_radius)).long()
    assert th.all(n_particles_per_axis > 0), \
        f"Box too small for particle radius {particle_radius}."
    arrs = [
        th.arange(l + particle_radius, h - particle_radius + 1e-10, particle_radius * 2)
        for l, h, n in zip(low, high, n_particles_per_axis)
    ]
    particle_positions = th.stack(th.meshgrid(*arrs, indexing="ij")).view(3, -1).t()
    water.generate_particles(positions=particle_positions)


def check_in_contact(system, positions):
    in_contact = th.zeros(len(positions), dtype=bool)
    for idx, pos in enumerate(positions):
        in_contact[idx] = og.sim.psqi.overlap_sphere_any(
            system.particle_contact_radius * 0.8, pos.numpy().copy()
        )
    return in_contact


def look_at_quaternion(eye_pos, target_pos, up=np.array([0, 0, 1])):
    forward = np.array(target_pos) - np.array(eye_pos)
    forward = forward / np.linalg.norm(forward)
    right   = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:
        up    = np.array([0, 1, 0])
        right = np.cross(forward, up)
    right   = right / np.linalg.norm(right)
    true_up = np.cross(right, forward)
    true_up = true_up / np.linalg.norm(true_up)
    rot_matrix = np.column_stack([right, true_up, -forward])
    return Rotation.from_matrix(rot_matrix).as_quat()


def get_scale(cat, mdl):
    usd_path = DatasetObject.get_usd_path(category=cat, model=mdl)
    usd_path = usd_path.replace(".usd", ".encrypted.usd")
    with decrypted(usd_path) as fpath:
        stage = lazy.pxr.Usd.Stage.Open(fpath)
        prim  = stage.GetDefaultPrim()
        bounding_box = th.tensor(prim.GetAttribute("ig:nativeBB").Get())
    scale = MAX_BBOX / th.max(bounding_box)
    return min(float(scale), 1.0)


def capture_image(out_path: str):
    for _ in range(10):
        og.sim.render()
    image = og.sim._viewer_camera.get_obs()[0]["rgb"].cpu().numpy()[:, :, :3].astype(np.uint8)
    cv2.imwrite(out_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    print(f"[render] Saved -> {out_path}")


def set_side_camera(obj1, obj2):
    bmin1, bmax1 = [x.numpy() for x in obj1.aabb]
    bmin2, bmax2 = [x.numpy() for x in obj2.aabb]
    centre1 = (bmin1 + bmax1) / 2.0
    centre2 = (bmin2 + bmax2) / 2.0
    mid_x = (centre1[0] + centre2[0]) / 2.0
    mid_y = (centre1[1] + centre2[1]) / 2.0
    mid_z = max(centre1[2], centre2[2]) + 0.1
    cam_eye    = np.array([mid_x, mid_y - 1.0, mid_z])
    cam_target = np.array([mid_x, mid_y,        mid_z])
    cam_quat   = look_at_quaternion(cam_eye, cam_target)
    og.sim._viewer_camera.set_position_orientation(cam_eye, cam_quat)


def get_particles_in_obj(obj, water, particle_point_offsets):
    aabb_min, aabb_max = obj.aabb
    all_particles, all_orients = water.get_particles_position_orientation()
    all_particles_t = th.tensor(all_particles)
    all_orients_t   = th.tensor(all_orients)
    in_obj = th.zeros(len(all_particles_t), dtype=th.bool)
    if len(all_particles_t) == 0:
        return all_particles_t, all_orients_t, in_obj
    not_in_contact_mask = check_in_contact(water, all_particles_t) == 0
    particles = all_particles_t[not_in_contact_mask]
    if len(particles) == 0:
        return all_particles_t, all_orients_t, in_obj
    offsets = particles.unsqueeze(1) + particle_point_offsets.unsqueeze(0)
    inside  = (
        th.all(offsets <= (aabb_max + th.tensor([0, 0, water.particle_radius])), dim=2) &
        th.all(offsets >= aabb_min, dim=2)
    ).any(dim=1)
    not_in_contact_indices = th.where(not_in_contact_mask)[0]
    in_obj[not_in_contact_indices[inside]] = True
    return all_particles_t, all_orients_t, in_obj


def fill_both(obj1, obj2, water, floor_z1, floor_z2, particle_point_offsets):
    """
    One water system — generate one big box covering both objects,
    dip both simultaneously, lift both simultaneously.
    Returns (n1, n2).
    """
    aabb_extent1    = th.tensor(obj1.aabb_extent)
    aabb_extent2    = th.tensor(obj2.aabb_extent)
    obj_current_pos1 = th.tensor(obj1.get_position_orientation()[0])
    obj_current_pos2 = th.tensor(obj2.get_position_orientation()[0])

    obj_dipped_pos1 = obj_current_pos1
    obj_dipped_pos2 = obj_current_pos2
    obj_free_pos1   = obj_current_pos1 + th.tensor([0, 0, 1.1 * aabb_extent1[2] + aabb_extent1[2]])
    obj_free_pos2   = obj_current_pos2 + th.tensor([0, 0, 1.1 * aabb_extent2[2] + aabb_extent2[2]])
    # Both lift until the higher of the two free positions
    free_z = max(obj_free_pos1[2].item(), obj_free_pos2[2].item())

    # ── One big box covering both objects in XY ───────────────────────────────
    # X: from left of obj1 to right of obj2
    # Y: max half-extent of either object
    # Z: use the lower of the two floor_z so both are submerged
    bmin1, bmax1 = [x.numpy() for x in obj1.aabb]
    bmin2, bmax2 = [x.numpy() for x in obj2.aabb]
    combined_x_min = float(bmin1[0])
    combined_x_max = float(bmax2[0])
    combined_y_min = min(float(bmin1[1]), float(bmin2[1]))
    combined_y_max = max(float(bmax1[1]), float(bmax2[1]))
    center_x = (combined_x_min + combined_x_max) / 2.0
    center_y = (combined_y_min + combined_y_max) / 2.0
    half_x   = (combined_x_max - combined_x_min) / 2.0 + 0.05
    half_y   = (combined_y_max - combined_y_min) / 2.0 + 0.05
    half_z   = max(aabb_extent1[2].item(), aabb_extent2[2].item()) * 0.55
    half_z   = max(half_z, max(aabb_extent1[2].item(), aabb_extent2[2].item()) / 2 + 2.1 * water.particle_radius)
    box_half_extent = th.tensor([half_x, half_y, half_z])
    floor_z = min(floor_z1, floor_z2)

    generate_box(box_half_extent, floor_z=floor_z,
                 center_x=center_x, center_y=center_y,
                 index_offset=0)
    og.sim.step()
    generate_particles_in_box(water, box_half_extent, floor_z=floor_z,
                               center_x=center_x, center_y=center_y)
    for _ in range(100):
        og.sim.step()

    # ── Dip both simultaneously ───────────────────────────────────────────────
    obj1.set_position_orientation(position=obj_dipped_pos1)
    obj2.set_position_orientation(position=obj_dipped_pos2)
    for _ in range(100):
        og.sim.step()

    # Open joints for both
    joint_limits1 = {}
    for jname, joint in obj1.joints.items():
        if joint.has_limit:
            joint_limits1[jname] = (joint.lower_limit, joint.upper_limit)
            joint.set_pos(joint.upper_limit)
            joint.lower_limit = joint.upper_limit - 0.001
    joint_limits2 = {}
    for jname, joint in obj2.joints.items():
        if joint.has_limit:
            joint_limits2[jname] = (joint.lower_limit, joint.upper_limit)
            joint.set_pos(joint.upper_limit)
            joint.lower_limit = joint.upper_limit - 0.001
    og.sim.update_handles()

    n_steps_for_close = 100
    for i in range(n_steps_for_close):
        for jname, joint in obj1.joints.items():
            if jname in joint_limits1:
                lower, upper = joint_limits1[jname]
                ratio = i / n_steps_for_close
                ipos  = upper - ratio * (upper - lower)
                joint.lower_limit = ipos - 0.001
                joint.upper_limit = ipos
                joint.set_pos(ipos)
        for jname, joint in obj2.joints.items():
            if jname in joint_limits2:
                lower, upper = joint_limits2[jname]
                ratio = i / n_steps_for_close
                ipos  = upper - ratio * (upper - lower)
                joint.lower_limit = ipos - 0.001
                joint.upper_limit = ipos
                joint.set_pos(ipos)
        og.sim.step()

    for _ in range(100):
        og.sim.step()

    # ── Lift both simultaneously until higher free_z ──────────────────────────
    lin_vel = 0.01
    while True:
        delta_z  = lin_vel * og.sim.get_rendering_dt()
        cur_pos1 = th.tensor(obj1.get_position_orientation()[0])
        cur_pos2 = th.tensor(obj2.get_position_orientation()[0])
        obj1.set_position_orientation(position=cur_pos1 + th.tensor([0, 0, delta_z]))
        obj2.set_position_orientation(position=cur_pos2 + th.tensor([0, 0, delta_z]))
        og.sim.step()
        if (obj1.get_position_orientation()[0][2] > free_z and
            obj2.get_position_orientation()[0][2] > free_z):
            break

    for _ in range(180):
        og.sim.step()

    all_p, all_o, in_obj1 = get_particles_in_obj(obj1, water, particle_point_offsets)
    _,     _,     in_obj2 = get_particles_in_obj(obj2, water, particle_point_offsets)
    n1 = int(in_obj1.sum().item())
    n2 = int(in_obj2.sum().item())
    print(f"  [fill] obj1: {n1}  obj2: {n2}")
    return n1, n2


def pick_pair(instances, run_idx):
    rng = random.Random(run_idx)
    obj1_entry = rng.choice(instances)

    best      = None
    best_dist = float("inf")
    for candidate in instances:
        if candidate["category"] == obj1_entry["category"]:
            continue
        dist = abs(abs(candidate["particles_in_container"] - obj1_entry["particles_in_container"]) - TARGET_CAP_DIFF)
        if dist < best_dist:
            best_dist = dist
            best      = candidate
    obj2_entry = best

    if rng.random() < 0.5:
        left, right = obj1_entry, obj2_entry
    else:
        left, right = obj2_entry, obj1_entry

    print(f"[pair] left ={left['category']}/{left['model']}  cap={left['particles_in_container']}")
    print(f"[pair] right={right['category']}/{right['model']}  cap={right['particles_in_container']}")
    print(f"[pair] diff (left-right) = {left['particles_in_container'] - right['particles_in_container']}")
    return left, right


def process(obj1_entry, obj2_entry, out_dir, run_idx):
    os.makedirs(out_dir, exist_ok=True)

    cat1, mdl1 = obj1_entry["category"], obj1_entry["model"]
    cat2, mdl2 = obj2_entry["category"], obj2_entry["model"]
    cap1_expected = obj1_entry["particles_in_container"]
    cap2_expected = obj2_entry["particles_in_container"]

    scale1 = get_scale(cat1, mdl1)
    scale2 = get_scale(cat2, mdl2)

    cfg = {
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": "Rs_int",
            "not_load_object_categories": [
                "pot_plant", "coffee_table", "sofa", "breakfast_table",
                "straight_chair", "floor_lamp", "table_lamp", "swivel_chair",
                "bookcase", "standing_tv", "laptop", "picture", "countertop",
                "ottoman", "bottom_cabinet",
            ],
            "load_room_instances": ["living_room_0"],
        },
        "objects": [
            {
                "type": "DatasetObject", "name": "obj1",
                "category": cat1, "model": mdl1,
                "kinematic_only": False, "fixed_base": True,
                "scale": [scale1, scale1, scale1],
            },
            {
                "type": "DatasetObject", "name": "obj2",
                "category": cat2, "model": mdl2,
                "kinematic_only": False, "fixed_base": True,
                "scale": [scale2, scale2, scale2],
            },
        ],
    }

    env   = og.Environment(configs=cfg)
    og.sim.step()

    obj1  = env.scene.object_registry("name", "obj1")
    obj2  = env.scene.object_registry("name", "obj2")
    floor = env.scene.object_registry("name", "floors_ptwlei_0")
    water = env.scene.get_system("water")

    # ── Place obj1 on floor — same as batch_water_fill.py ─────────────────────
    obj1.states[object_states.OnTop].set_value(floor, True)
    for _ in range(30):
        og.sim.step()

    obj_current_pos1 = th.tensor(obj1.get_position_orientation()[0])
    aabb_extent1     = th.tensor(obj1.aabb_extent)
    obj_bbox_center1 = th.tensor(obj1.aabb_center)
    obj_bbox_bottom1 = obj_bbox_center1 - th.tensor([0, 0, aabb_extent1[2] / 2])
    floor_z1         = obj_bbox_bottom1[2].item() + 0.05
    bmin1, bmax1     = [x.numpy() for x in obj1.aabb]
    half_x1          = (bmax1[0] - bmin1[0]) / 2.0
    pos1_settled     = th.tensor(obj1.get_position_orientation()[0])

    # ── Place obj2 on floor — same as batch_water_fill.py ─────────────────────
    obj2.states[object_states.OnTop].set_value(floor, True)
    for _ in range(30):
        og.sim.step()

    aabb_extent2     = th.tensor(obj2.aabb_extent)
    obj_bbox_center2 = th.tensor(obj2.aabb_center)
    obj_bbox_bottom2 = obj_bbox_center2 - th.tensor([0, 0, aabb_extent2[2] / 2])
    floor_z2         = obj_bbox_bottom2[2].item() + 0.05
    bmin2, bmax2     = [x.numpy() for x in obj2.aabb]
    half_x2          = (bmax2[0] - bmin2[0]) / 2.0
    pos2_settled     = th.tensor(obj2.get_position_orientation()[0])

    # ── Move obj2 to the right of obj1 in X, keep its own Z ──────────────────
    obj1_x   = obj_current_pos1[0].item()
    obj1_pos = th.tensor([obj1_x, obj_current_pos1[1].item(), pos1_settled[2].item() + 0.04])
    obj1.set_position_orientation(obj1_pos, th.tensor([0, 0, 0, 1]))
    for _ in range(30):
        og.sim.step()

    # ── Move obj2 to the right of obj1 in X, keep its own Z ──────────────────
    obj2_x   = obj_current_pos1[0].item() + half_x1 + OBJECT_GAP + half_x2
    obj2_pos = th.tensor([obj2_x, obj_current_pos1[1].item(), pos2_settled[2].item() + 0.04])
    obj2.set_position_orientation(obj2_pos, th.tensor([0, 0, 0, 1]))
    for _ in range(30):
        og.sim.step()

    particle_point_offsets = th.stack(
        [e * side * water.particle_radius for e in th.eye(3) for side in [-1, 1]]
        + [th.zeros(3)]
    )

    og.sim._viewer_camera.add_modality("rgb")

    # ── Photo 0: both empty ───────────────────────────────────────────────────
    set_side_camera(obj1, obj2)
    capture_image(os.path.join(out_dir, "00_before_fill_left.png"))

    # ── Fill both simultaneously with one water box ───────────────────────────
    print("[step] Filling both objects simultaneously...")
    n1, n2 = fill_both(obj1, obj2, water, floor_z1, floor_z2, particle_point_offsets)
    pos1_after_fill = th.tensor(obj1.get_position_orientation()[0])

    # ── Photo 1: both filled (no separate left-only photo since simultaneous) ─
    set_side_camera(obj1, obj2)
    capture_image(os.path.join(out_dir, "01_after_fill_left.png"))

    # ── Photo 2: same state, both filled ─────────────────────────────────────
    capture_image(os.path.join(out_dir, "02_after_fill_right.png"))

    print(f"[counts] n1(left)={n1}  n2(right)={n2}")

    # ── Simulate pour ─────────────────────────────────────────────────────────
    all_p, all_o, in_obj1 = get_particles_in_obj(obj1, water, particle_point_offsets)
    _,     _,     in_obj2 = get_particles_in_obj(obj2, water, particle_point_offsets)

    exile = all_p.clone()
    rng   = random.Random(run_idx + 42)

    if n1 <= n2:
        obj2_indices = th.where(in_obj2)[0].tolist()
        rng.shuffle(obj2_indices)
        keep_in_obj2 = set(obj2_indices[:n1])
        for idx in obj2_indices:
            if idx not in keep_in_obj2:
                exile[idx, 2] = -100.0
        exile[in_obj1, 2] = -100.0
        final_n1 = 0
        final_n2 = n1
        scenario = "left_poured_into_right_completely"
        print(f"  [pour] n1<=n2: left emptied, right trimmed to {n1}")
    else:
        obj1_indices = th.where(in_obj1)[0].tolist()
        rng.shuffle(obj1_indices)
        keep_n       = n1 - n2
        keep_in_obj1 = set(obj1_indices[:keep_n])
        for idx in obj1_indices:
            if idx not in keep_in_obj1:
                exile[idx, 2] = -100.0
        final_n1 = n1 - n2
        final_n2 = n2
        scenario = "left_poured_into_right_overflow"
        print(f"  [pour] n1>n2: left has {n1-n2} leftover, right stays at {n2}")

    water.set_particles_position_orientation(positions=exile, orientations=all_o)
    for _ in range(120):
        og.sim.step()

    # ── Snap both to obj1's post-fill Z for final photo ───────────────────────
    ref_z = pos1_after_fill[2].item()
    cur1  = th.tensor(obj1.get_position_orientation()[0])
    cur2  = th.tensor(obj2.get_position_orientation()[0])
    obj1.set_position_orientation(th.tensor([cur1[0].item(), cur1[1].item(), ref_z]), th.tensor([0, 0, 0, 1]))
    obj2.set_position_orientation(th.tensor([cur2[0].item(), cur2[1].item(), ref_z]), th.tensor([0, 0, 0, 1]))
    for _ in range(30):
        og.sim.step()

    # ── Photo 3: final state ──────────────────────────────────────────────────
    set_side_camera(obj1, obj2)
    capture_image(os.path.join(out_dir, "03_final.png"))

    result = {
        "run_idx":  run_idx,
        "scenario": scenario,
        "obj1": {
            "category":           cat1,
            "model":              mdl1,
            "side":               "left",
            "capacity_from_json": cap1_expected,
            "capacity_measured":  n1,
        },
        "obj2": {
            "category":           cat2,
            "model":              mdl2,
            "side":               "right",
            "capacity_from_json": cap2_expected,
            "capacity_measured":  n2,
        },
        "capacity_diff_json":     cap1_expected - cap2_expected,
        "capacity_diff_measured": n1 - n2,
        "final_particles_obj1":   final_n1,
        "final_particles_obj2":   final_n2,
        "answer": (
            "obj1 poured completely into obj2"
            if scenario == "left_poured_into_right_completely"
            else "obj1 poured into obj2 but obj1 still has leftover water"
        ),
    }

    result_path = os.path.join(out_dir, "result.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[result] Saved -> {result_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capacity_json", type=str, default="fillable_capacity.json")
    parser.add_argument("--out_dir",       type=str, required=True)
    parser.add_argument("--run_idx",       type=int, default=0)
    args = parser.parse_args()

    with open(args.capacity_json) as f:
        data = json.load(f)
    instances = data["instances"]

    obj1_entry, obj2_entry = pick_pair(instances, args.run_idx)

    if og.sim:
        og.clear()
    else:
        og.launch()
    if og.sim.is_playing():
        og.sim.stop()

    try:
        process(obj1_entry, obj2_entry, args.out_dir, args.run_idx)
        sys.exit(0)
    except Exception as e:
        print(f"[ERROR] {e}")
        traceback.print_exc()
        os.makedirs(args.out_dir, exist_ok=True)
        with open(os.path.join(args.out_dir, "result.json"), "w") as f:
            json.dump({"success": False, "error": str(e)}, f, indent=2)
        sys.exit(2)


if __name__ == "__main__":
    main()