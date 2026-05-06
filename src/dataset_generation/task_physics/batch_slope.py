"""
batch_slope.py - single run: scene/room, tilted slope, random object, 30 steps, 4 views/step.
Called by run_slope.sh.
"""

import os, json, math, random, argparse
import numpy as np
import torch as th
import cv2

import omnigibson as og
from omnigibson.macros import gm
import omnigibson.utils.transform_utils as T
from scipy.spatial.transform import Rotation

gm.ENABLE_FLATCACHE        = False
gm.USE_GPU_DYNAMICS        = False
gm.ENABLE_OBJECT_STATES    = True
gm.ENABLE_TRANSITION_RULES = False

SLOPE_BASE_HALF_X = 0.20
SLOPE_HALF_Y      = 0.15
SLOPE_HALF_Z      = 0.01
SLOPE_ANGLE_DEG   = 20.0
NUM_VIEWS         = 4
CAMERA_RADIUS     = 0.60
CAMERA_HEIGHT     = 0.40
LOOK_HEIGHT       = 0.10
NUM_STEPS         = 30
SLIDE_THRESH      = 0.03
FALL_THRESH       = 0.05   # object fallen if z drops more than this below slope bottom
WALL_MARGIN       = 0.30
OBJ_MARGIN        = 0.10
SKIP_CATS         = {"floors", "ceilings", "walls"}
ROOM_OBJECTS_JSON = "bddl3/bddl/generated_data/combined_room_object_list_future.json"
SCENES_DIR        = "scenes5"


def look_at_quaternion(eye_pos, target_pos, up=np.array([0, 0, 1])):
    forward = np.array(target_pos) - np.array(eye_pos)
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:
        up = np.array([0, 1, 0]);  right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    true_up = np.cross(right, forward)
    true_up = true_up / np.linalg.norm(true_up)
    rot_matrix = np.column_stack([right, true_up, -forward])
    return Rotation.from_matrix(rot_matrix).as_quat()


def apply_friction(obj, sf, df):
    import omnigibson.lazy as lazy
    mat = lazy.isaacsim.core.api.materials.PhysicsMaterial(
        prim_path=f"{obj.prim_path}/Looks/{obj.name}_friction_mat",
        name=f"{obj.name}_friction_mat",
        static_friction=sf, dynamic_friction=df, restitution=0.1,
    )
    for link in obj.links.values():
        for msh in link.collision_meshes.values():
            msh.apply_physics_material(mat)
    print(f"[friction] {obj.name}: static={sf:.3f}  dynamic={df:.3f}")


def get_model_for_category(category, rng):
    for path in [
        "bddl3/bddl/generated_data/object_inventory.json",
        os.path.join(os.path.dirname(__file__), "object_inventory.json"),
    ]:
        if not os.path.exists(path): continue
        with open(path) as f: inv = json.load(f)
        providers = inv.get("providers", inv)
        matches = [k for k in providers if k.startswith(f"{category}-")]
        if matches:
            model_id = rng.choice(matches).split("-", 1)[1]
            print(f"[model] {category} -> {model_id}")
            return model_id
        return None
    return None


def get_scene_bboxes(scene_name, room_name):
    path = os.path.join(SCENES_DIR, f"{scene_name}_scene_dict.json")
    with open(path) as f:
        sd = json.load(f)
    bboxes = []
    for cat, entries in sd.get(room_name, {}).items():
        if cat in SKIP_CATS: continue
        for bmin, bmax in entries:
            bboxes.append((np.array(bmin), np.array(bmax)))
    return bboxes


def find_clear_xy(half_x, half_y, fx_min, fx_max, fy_min, fy_max, scene_bboxes, rng, max_tries=200):
    for _ in range(max_tries):
        cx = rng.uniform(fx_min + WALL_MARGIN + half_x, fx_max - WALL_MARGIN - half_x)
        cy = rng.uniform(fy_min + WALL_MARGIN + half_y, fy_max - WALL_MARGIN - half_y)
        ok = True
        for bmin, bmax in scene_bboxes:
            if (cx - half_x - OBJ_MARGIN < bmax[0] and
                cx + half_x + OBJ_MARGIN > bmin[0] and
                cy - half_y - OBJ_MARGIN < bmax[1] and
                cy + half_y + OBJ_MARGIN > bmin[1]):
                ok = False; break
        if ok:
            return cx, cy
    raise RuntimeError("Could not find clear XY for slope")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene",        required=True)
    parser.add_argument("--room",         required=True)
    parser.add_argument("--floor",        required=True)
    parser.add_argument("--run_idx",      type=int, default=0)
    parser.add_argument("--output_root",  default="renders_slope")
    parser.add_argument("--objects_json", default="slope_objects.json")
    args = parser.parse_args()

    seed = hash((args.scene, args.room, args.run_idx)) & 0xFFFFFFFF
    rng  = random.Random(seed)

    run_dir = os.path.join(args.output_root, args.scene, args.room, f"run_{args.run_idx:03d}")
    os.makedirs(run_dir, exist_ok=True)

    # ── Random params ─────────────────────────────────────────────────────────
    slope_half_x     = rng.uniform(SLOPE_BASE_HALF_X, SLOPE_BASE_HALF_X * 1.5)
    slope_angle_deg  = rng.uniform(10.0, 45.0)
    static_friction  = rng.uniform(0.3, 3.0)
    dynamic_friction = rng.uniform(0.3, static_friction)

    with open(args.objects_json) as f:
        obj_categories = json.load(f)
    rng.shuffle(obj_categories)
    obj_category = obj_model = None
    for cat in obj_categories:
        model = get_model_for_category(cat, rng)
        if model:
            obj_category = cat;  obj_model = model;  break
    assert obj_category, "No valid category found"

    print(f"[params] slope_half_x={slope_half_x:.3f}  sf={static_friction:.3f}  df={dynamic_friction:.3f}")
    print(f"[object] {obj_category} / {obj_model}")

    angle_rad  = math.radians(slope_angle_deg)
    slope_quat = T.euler2quat(th.tensor([0., angle_rad, 0.])).tolist()
    room_type  = "_".join(args.room.split("_")[:-1]) if args.room[-1].isdigit() else args.room

    cfg = {
        "env":    {"action_timestep": 1/60., "physics_timestep": 1/240.},
        "render": {"viewer_width": 1280, "viewer_height": 720},
        "scene":  {"type": "InteractiveTraversableScene",
                   "scene_model": args.scene, "load_room_types": [room_type]},
        "robots": [],
        "objects": [
            {"type": "PrimitiveObject", "name": "slope", "primitive_type": "Cube",
             "fixed_base": True,
             "scale": [slope_half_x*2, SLOPE_HALF_Y*2, SLOPE_HALF_Z*2],
             "position": [0., 0., 50.], "orientation": slope_quat,
             "visual_only": False, "rgba": [0.6, 0.55, 0.45, 1.0]},
            {"type": "DatasetObject", "name": "task_obj",
             "category": obj_category, "model": obj_model,
             "position": [0., 0., 50.], "orientation": [0., 0., 0., 1.]},
        ],
    }

    env      = og.Environment(configs=cfg)
    scene    = env.scene
    slope    = scene.object_registry("name", "slope")
    task_obj = scene.object_registry("name", "task_obj")

    floor_obj = scene.object_registry("name", args.floor)
    assert floor_obj is not None, f"Floor '{args.floor}' not found!"
    for _ in range(10): og.sim.step()

    floor_bmin, floor_bmax = [x.cpu().numpy() for x in floor_obj.aabb]
    fx_min, fx_max = floor_bmin[0], floor_bmax[0]
    fy_min, fy_max = floor_bmin[1], floor_bmax[1]
    floor_z        = float(floor_bmax[2])
    print(f"[floor] top_z={floor_z:.4f}  x=[{fx_min:.2f},{fx_max:.2f}]  y=[{fy_min:.2f},{fy_max:.2f}]")

    scene_bboxes       = get_scene_bboxes(args.scene, args.room)
    slope_cx, slope_cy = find_clear_xy(slope_half_x, SLOPE_HALF_Y, fx_min, fx_max, fy_min, fy_max, scene_bboxes, rng)
    slope_z            = floor_z + slope_half_x * math.sin(angle_rad) + SLOPE_HALF_Z * math.cos(angle_rad)

    slope.set_position_orientation(
        position=th.tensor([slope_cx, slope_cy, slope_z], dtype=th.float32),
        orientation=th.tensor(slope_quat, dtype=th.float32),
    )
    slope.keep_still()
    for _ in range(10): og.sim.step()

    slope_bmin, slope_bmax = [x.cpu().numpy() for x in slope.aabb]
    slope_centre = ((slope_bmin + slope_bmax) / 2.).tolist()
    print(f"[slope] xy=({slope_cx:.3f},{slope_cy:.3f})  aabb_min={slope_bmin.round(3).tolist()}  aabb_max={slope_bmax.round(3).tolist()}")

    apply_friction(slope,    static_friction, dynamic_friction)
    apply_friction(task_obj, static_friction, dynamic_friction)

    slope_mid_x = float((slope_bmin[0] + slope_bmax[0]) / 2.)
    slope_mid_y = float((slope_bmin[1] + slope_bmax[1]) / 2.)
    slope_mid_z = float((slope_bmin[2] + slope_bmax[2]) / 2.)
    # ── Camera setup (shared for both phases) ────────────────────────────────
    cx          = slope_mid_x
    cy          = slope_mid_y
    look_target = np.array([cx, cy, floor_z + LOOK_HEIGHT])

    cam_eyes = []
    cam_quats = []
    for vi in range(NUM_VIEWS):
        az   = math.radians(vi * 90)
        eye  = np.array([cx + CAMERA_RADIUS * math.cos(az),
                         cy + CAMERA_RADIUS * math.sin(az),
                         floor_z + CAMERA_HEIGHT])
        quat = look_at_quaternion(eye, look_target)
        cam_eyes.append(eye)
        cam_quats.append(quat)

    def set_view(vi):
        og.sim._viewer_camera.set_position_orientation(cam_eyes[vi], cam_quats[vi])
        og.sim.render()

    def capture(step_idx, vi, prefix="step"):
        rgb = og.sim._viewer_camera.get_obs()[0].get("rgb")
        if rgb is not None:
            img = cv2.cvtColor(np.array(rgb)[:, :, :3], cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(run_dir, f"{prefix}_{step_idx:04d}_view_{vi}.png"), img)

    # ── Phase 1: place object on floor 0.02m from front face of slope (Y axis) ──
    # Slope tilts along X, front face visible along Y = slope_bmax[1]
    obj_floor_x = slope_mid_x
    obj_floor_y = float(slope_bmax[1]) + 0.02
    task_obj.set_position_orientation(
        position=th.tensor([obj_floor_x, obj_floor_y, floor_z + 0.3], dtype=th.float32),
        orientation=th.tensor([0., 0., 0., 1.], dtype=th.float32),
    )
    task_obj.keep_still()
    for _ in range(20): og.sim.step()   # let it fall and settle on floor

    pos_floor = task_obj.get_position_orientation()[0].cpu().numpy().copy()
    obj_bmin_floor, obj_bmax_floor = [x.cpu().numpy() for x in task_obj.aabb]
    print(f"[phase1] obj on floor at {pos_floor.round(3)}")

    # Camera looks at midpoint between object and slope so both are visible
    floor_look = np.array([
        slope_mid_x,
        (float(slope_bmax[1]) + pos_floor[1]) / 2.,
        floor_z + LOOK_HEIGHT,
    ])
    floor_cam_eyes  = []
    floor_cam_quats = []
    for vi in range(NUM_VIEWS):
        az   = math.radians(vi * 90)
        eye  = np.array([floor_look[0] + CAMERA_RADIUS * math.cos(az),
                         floor_look[1] + CAMERA_RADIUS * math.sin(az),
                         floor_z + CAMERA_HEIGHT])
        quat = look_at_quaternion(eye, floor_look)
        floor_cam_eyes.append(eye)
        floor_cam_quats.append(quat)
        og.sim._viewer_camera.set_position_orientation(eye, quat)
        og.sim.render()
        capture(0, vi, prefix="floor")

    # ── Phase 2: place object on slope centre, run 30 steps ──────────────────
    task_obj.set_position_orientation(
        position=th.tensor([slope_mid_x, slope_mid_y, slope_mid_z + 0.05], dtype=th.float32),
        orientation=th.tensor([0., 0., 0., 1.], dtype=th.float32),
    )
    task_obj.keep_still()

    # Settle a few steps so pos_init reflects where object actually rests on slope
    for _ in range(5): og.sim.step()
    pos_init = task_obj.get_position_orientation()[0].cpu().numpy().copy()
    obj_bmin_init, obj_bmax_init = [x.cpu().numpy() for x in task_obj.aabb]
    print(f"[phase2] obj settled on slope at {pos_init.round(3)}")

    exist_flags  = {}
    camera_poses = []

    # Record slope-phase camera poses (fixed across all steps)
    for vi in range(NUM_VIEWS):
        camera_poses.append({
            "view_idx":        vi,
            "azimuth_deg":     vi * 90,
            "position":        [round(v, 4) for v in cam_eyes[vi].tolist()],
            "quaternion_xyzw": [round(v, 4) for v in cam_quats[vi].tolist()],
            "target":          [round(v, 4) for v in look_target.tolist()],
        })

    for step in range(1, NUM_STEPS + 1):
        og.sim.step()
        for vi in range(NUM_VIEWS):
            set_view(vi); capture(step, vi, prefix="step")
            exist_flags[f"exist_step{step:04d}_view{vi}"] = True  # simplified

    pos_final = task_obj.get_position_orientation()[0].cpu().numpy()
    obj_bmin_final, obj_bmax_final = [x.cpu().numpy() for x in task_obj.aabb]

    slide_dist = float(np.linalg.norm(pos_final[:2] - pos_init[:2]))
    slid       = slide_dist > SLIDE_THRESH
    # Fallen = object Z dropped below slope bottom by more than threshold
    fallen = bool(obj_bmin_final[2] < floor_z + FALL_THRESH)

    print(f"[result] slide_dist={slide_dist:.4f}m  slid={slid}  fallen={fallen}")

    # ── Camera poses (one entry per view, fixed across all steps) ─────────────
    for vi in range(NUM_VIEWS):
        camera_poses.append({
            "view_idx":    vi,
            "azimuth_deg": vi * 90,
            "position":    [round(v, 4) for v in cam_eyes[vi].tolist()],
            "quaternion_xyzw": [round(v, 4) for v in cam_quats[vi].tolist()],
            "target":      [round(v, 4) for v in look_target.tolist()],
        })

    # ── Metadata ──────────────────────────────────────────────────────────────
    metadata = {
        "scene":             args.scene,
        "room":              args.room,
        "run_idx":           args.run_idx,
        "seed":              seed,
        "floor_name":        args.floor,
        "slope_angle_deg":   round(slope_angle_deg, 2),
        "slope_half_x":      round(slope_half_x, 4),
        "slope_half_y":      SLOPE_HALF_Y,
        "slope_half_z":      SLOPE_HALF_Z,
        "static_friction":   round(static_friction, 4),
        "dynamic_friction":  round(dynamic_friction, 4),
        "obj_category":      obj_category,
        "obj_model":         obj_model,
        "num_steps":         NUM_STEPS,
        "slide_dist_m":      round(slide_dist, 5),
        "ground_truth_slid": slid,
        "ground_truth_fallen": fallen,
        "slope": {
            "position":  [round(v, 4) for v in [slope_cx, slope_cy, slope_z]],
            "aabb_min":  slope_bmin.round(4).tolist(),
            "aabb_max":  slope_bmax.round(4).tolist(),
            "centre":    [round(v, 4) for v in slope_centre],
        },
        "objects": {
            "task_obj": {
                "category":    obj_category,
                "model":       obj_model,
                "pos_on_floor": pos_floor.round(4).tolist(),
                "pos_init":    pos_init.round(4).tolist(),
                "pos_final":   pos_final.round(4).tolist(),
                "aabb_min_init":  obj_bmin_init.round(4).tolist(),
                "aabb_max_init":  obj_bmax_init.round(4).tolist(),
                "aabb_min_final": obj_bmin_final.round(4).tolist(),
                "aabb_max_final": obj_bmax_final.round(4).tolist(),
            }
        },
        "camera_poses_floor": [
            {"view_idx":        vi,
             "azimuth_deg":     vi * 90,
             "position":        [round(v,4) for v in floor_cam_eyes[vi].tolist()],
             "quaternion_xyzw": [round(v,4) for v in floor_cam_quats[vi].tolist()],
             "target":          [round(v,4) for v in floor_look.tolist()],
             "filename":        f"floor_0000_view_{vi}.png"}
            for vi in range(NUM_VIEWS)
        ],
        "slope_scale": [round(slope_half_x*2, 4), round(SLOPE_HALF_Y*2, 4), round(SLOPE_HALF_Z*2, 4)],
        "slope_quaternion": [round(v, 4) for v in slope_quat],
        "camera_poses":  camera_poses,
        **exist_flags,
    }

    with open(os.path.join(run_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[done] -> {run_dir}/")
    raise SystemExit(0)


if __name__ == "__main__":
    main()