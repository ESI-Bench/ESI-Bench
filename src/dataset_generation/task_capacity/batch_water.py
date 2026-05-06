"""
batch_water_fill.py

Processes a single fillable object (category + model) through the water-dip
pipeline and saves:
  <output_root>/<category>/<model>/
      side.png          – side view camera (same as original script)
      top.png           – top-down camera looking straight down at object centre
      result.json       – category, model, particle counts, AABB, success flag

Usage:
  python batch_water_fill.py --category bowl --model abcdef [--output_root water_fill_results]

Exit codes:
  0 — success (particles found in container)
  1 — no particles captured (object may not be fillable)
  2 — fatal error
"""

import argparse
import json
import os
import sys
import traceback

import torch as th
import numpy as np
from scipy.spatial import KDTree
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from scipy.spatial.transform import Rotation

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.objects.dataset_object import DatasetObject
from omnigibson.prims.xform_prim import XFormPrim
from omnigibson.utils.asset_utils import decrypted
import omnigibson.utils.transform_utils as T
import omnigibson.lazy as lazy

gm.HEADLESS      = False          # headless for batch — flip to False to debug visually
gm.USE_ENCRYPTED_ASSETS = True
gm.USE_GPU_DYNAMICS     = True
gm.ENABLE_FLATCACHE     = False

MAX_BBOX = 0.3


# ── Helpers (identical to your original script) ───────────────────────────────

def find_largest_connected_component(points, d):
    points_np = points.numpy().copy()
    tree  = KDTree(points_np)
    pairs = tree.query_pairs(r=d, output_type="ndarray")
    n_points = points.shape[0]
    adjacency_matrix = csr_matrix(
        (np.ones(pairs.shape[0]), (pairs[:, 0], pairs[:, 1])),
        shape=(n_points, n_points),
    )
    adjacency_matrix = adjacency_matrix + adjacency_matrix.T
    _, labels = connected_components(
        csgraph=adjacency_matrix, directed=False, return_labels=True
    )
    largest_component_label   = th.argmax(th.bincount(th.tensor(labels)))
    largest_component_indices = th.where(th.tensor(labels) == largest_component_label)[0]
    return points[largest_component_indices]


def generate_box(box_half_extent, floor_z=0.0, center_x=0.0, center_y=0.0):
    plane_centers = (
        th.tensor([[1, 0, 1], [0, 1, 1], [-1, 0, 1], [0, -1, 1]])
        * box_half_extent
    )
    plane_centers[:, 0] += center_x
    plane_centers[:, 1] += center_y
    plane_centers[:, 2] += floor_z
    for i, pc in enumerate(plane_centers):
        plane = lazy.omni.isaac.core.objects.ground_plane.GroundPlane(
            prim_path=f"/World/plane_{i}", name=f"plane_{i}",
            z_position=0, size=box_half_extent[2].item(), color=None, visible=False,
        )
        plane_as_prim = XFormPrim(relative_prim_path=f"/plane_{i}", name=plane.name)
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
    """Render a frame and save to out_path as PNG."""
    import cv2
    for _ in range(10):
        og.sim.render()
    image = og.sim._viewer_camera.get_obs()[0]["rgb"].cpu().numpy()[:, :, :3].astype(np.uint8)
    cv2.imwrite(out_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    print(f"[render] Saved -> {out_path}")


# ── Core pipeline ─────────────────────────────────────────────────────────────

def process_object(cat, mdl, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    if og.sim:
        og.clear()
    else:
        og.launch()

    if og.sim.is_playing():
        og.sim.stop()

    scale = get_scale(cat, mdl)

    cfg = {
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": "Rs_int",
            "not_load_object_categories": [
                "pot_plant", "coffee_table", "sofa", "breakfast_table",
                "straight_chair", "floor_lamp", "table_lamp", "swivel_chair",
                "bookcase", "standing_tv", "laptop", "picture", "countertop", "ottoman", "bottom_cabinet"
            ],
            "load_room_instances": ["living_room_0"],
        },
        "objects": [
            {
                "type":         "DatasetObject",
                "name":         "fillable",
                "category":     cat,
                "model":        mdl,
                "kinematic_only": False,
                "fixed_base":   True,
                "scale":        [scale, scale, scale],
            },
        ],
    }

    import omnigibson.object_states as object_states

    env      = og.Environment(configs=cfg)
    og.sim.step()

    fillable = env.scene.object_registry("name", "fillable")
    floor    = env.scene.object_registry("name", "floors_ptwlei_0")

    # ── Place on floor ─────────────────────────────────────────────────────────
    fillable.states[object_states.OnTop].set_value(floor, True)
    for _ in range(30):
        og.sim.step()

    # Open all joints fully (measuring-cup-style lids)
    joint_limits = {}
    for joint_name, joint in fillable.joints.items():
        if joint.has_limit:
            joint_limits[joint_name] = (joint.lower_limit, joint.upper_limit)
            joint.set_pos(joint.upper_limit)
            joint.lower_limit = joint.upper_limit - 0.001
    og.sim.update_handles()

    # ── Geometry ───────────────────────────────────────────────────────────────
    aabb_extent    = th.tensor(fillable.aabb_extent)
    obj_bbox_center = th.tensor(fillable.aabb_center)
    obj_bbox_bottom = obj_bbox_center - th.tensor([0, 0, aabb_extent[2] / 2])
    obj_current_pos = th.tensor(fillable.get_position_orientation()[0])

    obj_dipped_pos = obj_current_pos 
    obj_free_pos   = obj_current_pos + th.tensor([0, 0, 1.1 * aabb_extent[2] + aabb_extent[2]])
    floor_z        = obj_bbox_bottom[2]

    # ── Generate water box ────────────────────────────────────────────────────
    water = env.scene.get_system("water")
    box_half_extent = th.maximum(
        aabb_extent * 0.55,
        aabb_extent / 2 + 2.1 * water.particle_radius,
    )
    center_x = obj_dipped_pos[0].item()
    center_y = obj_dipped_pos[1].item()

    generate_box(box_half_extent, floor_z=floor_z, center_x=center_x, center_y=center_y)
    og.sim.step()
    generate_particles_in_box(water, box_half_extent, floor_z=floor_z,
                               center_x=center_x, center_y=center_y)
    for _ in range(100):
        og.sim.step()

    # Dip
    fillable.set_position_orientation(position=obj_dipped_pos)
    for _ in range(100):
        og.sim.step()

    # Slowly close joints (lids)
    n_steps_for_close = 100
    for i in range(n_steps_for_close):
        for joint_name, joint in fillable.joints.items():
            if joint_name in joint_limits:
                lower, upper = joint_limits[joint_name]
                openness_ratio    = i / n_steps_for_close
                interpolated_pos  = upper - openness_ratio * (upper - lower)
                joint.lower_limit = interpolated_pos - 0.001
                joint.upper_limit = interpolated_pos
                joint.set_pos(interpolated_pos)
        og.sim.step()

    for _ in range(100):
        og.sim.step()

    # Lift out
    lin_vel = 0.01
    while True:
        delta_z  = lin_vel * og.sim.get_rendering_dt()
        cur_pos  = th.tensor(fillable.get_position_orientation()[0])
        new_pos  = cur_pos + th.tensor([0, 0, delta_z])
        fillable.set_position_orientation(position=new_pos)
        og.sim.step()
        if fillable.get_position_orientation()[0][2] > obj_free_pos[2]:
            break

    for _ in range(180):
        og.sim.step()

    # ── Identify in-container particles ───────────────────────────────────────
    aabb_min, aabb_max = fillable.aabb
    all_particles, all_orients = water.get_particles_position_orientation()
    all_particles_t = th.tensor(all_particles)
    all_orients_t   = th.tensor(all_orients)

    not_in_contact_mask = check_in_contact(water, all_particles_t) == 0
    particles = all_particles_t[not_in_contact_mask]

    particle_point_offsets = th.stack(
        [e * side * water.particle_radius for e in th.eye(3) for side in [-1, 1]]
        + [th.zeros(3)]
    )

    particles_in_container = 0
    in_glass_global = th.zeros(len(all_particles_t), dtype=th.bool)

    if len(particles) > 0:
        offsets = particles.unsqueeze(1) + particle_point_offsets.unsqueeze(0)
        inside  = (
            th.all(offsets <= (aabb_max + th.tensor([0, 0, water.particle_radius])), dim=2) &
            th.all(offsets >= aabb_min, dim=2)
        ).any(dim=1)

        not_in_contact_indices = th.where(not_in_contact_mask)[0]
        in_glass_global[not_in_contact_indices[inside]] = True
        particles_in_container = int(in_glass_global.sum().item())

    # # Banish stray particles
    # exile_positions = all_particles_t.clone()
    # exile_positions[~in_glass_global, 2] = -100.0
    # water.set_particles_position_orientation(
    #     positions=exile_positions, orientations=all_orients_t
    # )
    # print(f"Particles in container: {particles_in_container}  |  stray banished: {(~in_glass_global).sum().item()}")
    # og.sim.step()

    # ── Lower back to floor ────────────────────────────────────────────────────
    # while True:
    #     delta_z  = lin_vel * og.sim.get_rendering_dt()
    #     cur_pos  = th.tensor(fillable.get_position_orientation()[0])
    #     new_pos  = cur_pos - th.tensor([0, 0, delta_z])
    #     fillable.set_position_orientation(position=new_pos)
    #     og.sim.step()
    #     if fillable.get_position_orientation()[0][2] <= obj_dipped_pos[2]:
    #         break

    # fillable.set_position_orientation(obj_dipped_pos, th.tensor([0, 0, 0, 1]))
    # for _ in range(180):
    #     og.sim.step()

    # ── Add render modalities ──────────────────────────────────────────────────
    og.sim._viewer_camera.add_modality("rgb")

    # ── Side camera ───────────────────────────────────────────────────────────
    aabb_min_np, aabb_max_np = [x.numpy() for x in fillable.aabb]
    obj_centre = (aabb_min_np + aabb_max_np) / 2.0

    cam_eye    = np.array([obj_centre[0], obj_centre[1] - 0.5, obj_centre[2]])
    cam_target = np.array([obj_centre[0], obj_centre[1],       obj_centre[2]])
    cam_quat   = look_at_quaternion(cam_eye, cam_target)
    og.sim._viewer_camera.set_position_orientation(cam_eye, cam_quat)
    capture_image(os.path.join(out_dir, "side.png"))

    # ── Top-down camera ────────────────────────────────────────────────────────
    # Straight above, looking down at the object centre
    top_height = aabb_max_np[2] + 0.6          # 0.6 m above the rim
    cam_eye_top    = np.array([obj_centre[0], obj_centre[1], top_height])
    cam_target_top = np.array([obj_centre[0], obj_centre[1], obj_centre[2]])
    # Use Y-forward as "up" hint so the image isn't spinning randomly
    cam_quat_top   = look_at_quaternion(cam_eye_top, cam_target_top,
                                         up=np.array([0, 1, 0]))
    og.sim._viewer_camera.set_position_orientation(cam_eye_top, cam_quat_top)
    capture_image(os.path.join(out_dir, "top.png"))

    # ── Write result JSON ──────────────────────────────────────────────────────
    pos_final, quat_final = fillable.get_position_orientation()
    result = {
        "category":            cat,
        "model":               mdl,
        "scale":               scale,
        "particles_in_container": particles_in_container,
        "total_particles_generated": int(len(all_particles_t)),
        "success":             particles_in_container > 0,
        "aabb_min":            aabb_min_np.tolist(),
        "aabb_max":            aabb_max_np.tolist(),
        "aabb_extent":         aabb_extent.tolist(),
        "position":            pos_final.tolist(),
        "quaternion_xyzw":     quat_final.tolist(),
        "particle_radius":     float(water.particle_radius),
        "output_dir":          out_dir,
    }

    result_path = os.path.join(out_dir, "result.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[result] Saved -> {result_path}")

    return particles_in_container > 0


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Water-fill a single fillable object.")
    parser.add_argument("--category", type=str, required=True)
    parser.add_argument("--model",    type=str, required=True)
    parser.add_argument("--out_dir",  type=str, required=True)
    args = parser.parse_args()

    cat     = args.category.lower()
    mdl     = args.model
    out_dir = args.out_dir

    try:
        success = process_object(cat, mdl, out_dir)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"[ERROR] {cat}/{mdl}: {e}")
        traceback.print_exc()
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "result.json"), "w") as f:
            json.dump({
                "category": cat, "model": mdl,
                "success": False, "error": str(e),
            }, f, indent=2)
        sys.exit(2)


if __name__ == "__main__":
    main()