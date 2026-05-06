"""
Generate all mirror question families from one runtime OmniGibson scene load.

This script keeps the current standalone task logic for:
- mirror_object_reality
- mirror_distance
- mirror_correspondence

but runs all three inside a single loaded room so the simulator does not need
to restart between task families.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import traceback

import batch_mirror_correspondence as correspondence_mod
import batch_mirror_distance as distance_mod
import batch_mirror_object_reality as object_reality_mod


TASK_SPECS = (
    ("mirror_object_reality", object_reality_mod),
    ("mirror_distance", distance_mod),
    ("mirror_correspondence", correspondence_mod),
)
MIN_ROOM_AREA_M2 = 6.0
SKIPPED_ROOM_MARKER = "mirror_room_skipped.json"
ATTEMPTED_ROOM_MARKER = "mirror_room_attempted.json"
RENDER_OBJECT_NAME_PREFIX = "render_item_"


def _write_json(path: str, payload: dict) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _log_exception(context: str, exc: Exception) -> None:
    print(f"[batch_mirror_merge] {context}: {exc.__class__.__name__}: {exc}")
    traceback.print_exc()


def _write_room_attempted_marker(room_run_root: str, scene_name: str, payload: dict) -> None:
    scene_root = os.path.join(room_run_root, scene_name)
    os.makedirs(scene_root, exist_ok=True)
    _write_json(os.path.join(scene_root, ATTEMPTED_ROOM_MARKER), payload)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate all mirror task families from one loaded OmniGibson room.")
    parser.add_argument("--scene", default="Rs_int", help="Scene model name")
    parser.add_argument("--room", action="append", default=None, help="Optional room instance name. Repeat to process multiple rooms in one scene load.")
    parser.add_argument("--floor", action="append", default=None, help="Optional floor object name paired with --room. Repeat to process multiple rooms in one scene load.")
    parser.add_argument("--agent_position", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    parser.add_argument("--keys_json", type=str, default=distance_mod.DEFAULT_KEYS_JSON)
    parser.add_argument("--robot", type=str, default="R1")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run_idx", type=int, default=0, help="Run index used to preserve per-room output layout.")
    parser.add_argument("--max_objects", type=int, default=4)
    parser.add_argument("--max_questions_per_type", type=int, default=8)
    parser.add_argument("--output_root", type=str, default="renders_mirror")
    parser.add_argument("--skip_render", action="store_true")
    parser.add_argument(
        "--disable_trav_map_occlusion_check",
        action="store_true",
        help="Disable the secondary 2D occlusion check based on scene traversability pixels.",
    )
    parser.add_argument(
        "--exit_on_finish",
        action="store_true",
        help="Exit immediately after generation. By default, keep simulator alive.",
    )
    return parser


def _normalize_room_floor_args(args) -> list[tuple[str | None, str | None]]:
    rooms = [] if args.room is None else list(args.room)
    floors = [] if args.floor is None else list(args.floor)

    if not rooms:
        rooms = [None]
    if not floors:
        floors = [None] * len(rooms)
    if len(floors) == 1 and len(rooms) > 1:
        floors = floors * len(rooms)
    if len(rooms) != len(floors):
        raise ValueError(f"--room count ({len(rooms)}) must match --floor count ({len(floors)}).")

    return list(zip(rooms, floors))


def _room_output_paths(output_root: str, scene_name: str, room_name: str | None, run_idx: int) -> tuple[str, str, str]:
    room_key = "scene_wide" if room_name is None else str(room_name)
    room_run_root = os.path.join(output_root, scene_name, room_key, f"run_{int(run_idx):04d}")
    room_scene_root = os.path.join(room_run_root, scene_name)
    question_json_root = os.path.join(room_scene_root, "mirror_question_jsons")
    render_root = os.path.join(room_scene_root, "mirror_renders")
    return room_run_root, question_json_root, render_root


def _build_task_metadata(task_name: str, module, args, resolved_seed: int, floor_record, preferred_camera_xy, trav_map, trav_map_img, resolved_room_instance):
    camera_setup = {
        "mode": "per_question_reset",
        "preferred_xy": None if preferred_camera_xy is None else [float(v) for v in preferred_camera_xy],
        "height_m": module.CAMERA_HEIGHT,
        "fov_deg": module.CAMERA_FOV_DEG,
        "look_target": "mirror_center",
        "trav_map_pixel_occlusion_check": bool(trav_map is not None and trav_map_img is not None),
    }

    if task_name == "mirror_distance":
        camera_setup["render_view_groups"] = {
            "single_view": "observer_view",
            "multi_view": {
                "type": "arc_around_mirror_center",
                "angle_range_deg": [-module.MULTI_VIEW_ARC_MAX_DEG, module.MULTI_VIEW_ARC_MAX_DEG],
                "step_deg": module.MULTI_VIEW_STEP_DEG,
            },
            "gt_view": {
                "type": "observer_perpendicular_side_views",
                "side_offset_m": module.GT_SIDE_OFFSET_M,
                "look_target": "observer_position",
            },
        }
    elif task_name == "mirror_correspondence":
        camera_setup["context_render_enabled"] = False
        camera_setup["clear_radius_m"] = module.CONTEXT_CLEAR_RADIUS_M

    return {
        "scene": args.scene,
        "room": args.room if task_name == "mirror_object_reality" else (resolved_room_instance or args.room),
        "floor_name": floor_record.name,
        "seed": resolved_seed,
        "camera_setup": camera_setup,
        "mirror_setup": {
            "name": "render_mirror_main",
            "category": "mirror",
            "model": "tytkbq",
            "mode": "per_question_reset",
            "camera_forward_distance_m": module.MIRROR_AHEAD_DISTANCE,
        },
    }


def _ensure_mirror(scene):
    mirror_obj = scene.object_registry("name", "render_mirror_main")
    if mirror_obj is None:
        mirror_obj = distance_mod.DatasetObject(
            name="render_mirror_main",
            category="mirror",
            model="tytkbq",
            visual_only=True,
        )
        scene.add_object(mirror_obj)
    try:
        mirror_obj.visual_only = True
    except Exception:
        pass
    distance_mod._step_sim(20)
    distance_mod._park_object(mirror_obj, 999)
    distance_mod._step_sim(10)
    return mirror_obj


def _cleanup_render_objects(scene) -> int:
    removed = 0
    for obj in list(scene.objects):
        name = str(getattr(obj, "name", ""))
        if not name.startswith(RENDER_OBJECT_NAME_PREFIX):
            continue
        try:
            scene.remove_object(obj)
            removed += 1
        except Exception:
            pass
    if removed > 0:
        distance_mod._step_sim(5)
    return removed


def _process_room(args, env, scene, room_name, floor_name, resolved_seed: int) -> dict:
    room_run_root, question_json_root, render_root = _room_output_paths(
        output_root=args.output_root,
        scene_name=args.scene,
        room_name=room_name,
        run_idx=args.run_idx,
    )
    os.makedirs(room_run_root, exist_ok=True)
    os.makedirs(os.path.join(room_run_root, args.scene), exist_ok=True)

    room_label = room_name if room_name is not None else "__scene__"
    room_args_value = args.room
    floor_args_value = args.floor
    args.room = room_name
    args.floor = floor_name

    try:
        agent_pos = distance_mod._resolve_agent_position(env, args.agent_position)
        room_objects = distance_mod._collect_room_objects(scene, room_name)
        floor_record = distance_mod._select_floor(room_objects, floor_name, agent_pos, room_name=room_name)
        room_area_m2 = float(floor_record.extents[0]) * float(floor_record.extents[1])
        if room_area_m2 < MIN_ROOM_AREA_M2:
            skip_payload = {
                "scene": args.scene,
                "room": room_name,
                "seed": resolved_seed,
                "floor_name": floor_record.name,
                "room_area_m2": room_area_m2,
                "skipped": True,
                "skip_reason": f"room area below {MIN_ROOM_AREA_M2:.1f} m^2",
                "task_types": [task_name for task_name, _ in TASK_SPECS],
            }
            _write_json(os.path.join(os.path.join(room_run_root, args.scene), SKIPPED_ROOM_MARKER), skip_payload)
            print(json.dumps(skip_payload, indent=2, ensure_ascii=False))
            return {
                "room": room_name,
                "floor": floor_name,
                "status": "skipped",
                "question_summary": {task_name: 0 for task_name, _ in TASK_SPECS},
                "task_errors": None,
            }

        if args.agent_position is None and not getattr(env, "robots", []):
            agent_pos = (
                float(floor_record.center[0]),
                float(floor_record.center[1]),
                float(floor_record.bbox_max[2]) + distance_mod.CAMERA_HEIGHT,
            )

        room_bbox_world_xy, resolved_room_instance = distance_mod._resolve_room_bbox_world_xy(scene, room_name, floor_record)
        mirror_obj = _ensure_mirror(scene)

        summary = {}
        task_errors = {}
        for task_name, module in TASK_SPECS:
            _cleanup_render_objects(scene)
            module._set_viewer_camera_fov()
            task_rng = random.Random(resolved_seed)
            blockers, wall_blockers, trav_map, trav_map_img, free_positions, placeable_pool, preferred_camera_xy = _task_setup(
                module=module,
                args=args,
                env=env,
                scene=scene,
                room_objects=room_objects,
                floor_record=floor_record,
                agent_pos=agent_pos,
            )
            question_scene_metadata = _build_task_metadata(
                task_name=task_name,
                module=module,
                args=args,
                resolved_seed=resolved_seed,
                floor_record=floor_record,
                preferred_camera_xy=preferred_camera_xy,
                trav_map=trav_map,
                trav_map_img=trav_map_img,
                resolved_room_instance=resolved_room_instance,
            )

            generate_kwargs = {
                "scene": scene,
                "rng": task_rng,
                "mirror_obj": mirror_obj,
                "floor_record": floor_record,
                "free_positions": free_positions,
                "blockers": blockers,
                "wall_blockers": wall_blockers,
                "placeable_pool": placeable_pool,
                "max_q": max(1, args.max_questions_per_type),
                "render_root": render_root,
                "enable_render": not args.skip_render,
                "trav_map": trav_map,
                "trav_map_img": trav_map_img,
                "preferred_camera_xy": preferred_camera_xy,
                "task_types": {task_name},
                "question_json_root": question_json_root,
                "question_scene_metadata": question_scene_metadata,
            }
            if task_name != "mirror_object_reality":
                generate_kwargs["room_bbox_world_xy"] = room_bbox_world_xy
                generate_kwargs["room_instance_name"] = resolved_room_instance

            try:
                qa = module._generate_questions_with_per_question_placement(**generate_kwargs)
                summary[task_name] = len(qa.get(task_name, []))
            except Exception as exc:
                summary[task_name] = 0
                task_errors[task_name] = f"{exc.__class__.__name__}: {exc}"
                print(f"[batch_mirror_merge] room='{room_label}' task='{task_name}' failed: {exc.__class__.__name__}: {exc}", flush=True)
            finally:
                _cleanup_render_objects(scene)

        room_result = {
            "scene": args.scene,
            "room": room_name,
            "floor": floor_name,
            "seed": resolved_seed,
            "question_summary": summary,
            "task_errors": task_errors or None,
            "question_json_root": question_json_root,
            "render_root": None if args.skip_render else render_root,
        }
        _write_room_attempted_marker(
            room_run_root,
            args.scene,
            {
                **room_result,
                "status": "attempted",
                "attempted": True,
            },
        )
        print(json.dumps(room_result, indent=2, ensure_ascii=False))
        return {
            "room": room_name,
            "floor": floor_name,
            "status": "ok",
            "question_summary": summary,
            "task_errors": task_errors or None,
        }
    finally:
        _cleanup_render_objects(scene)
        args.room = room_args_value
        args.floor = floor_args_value


def _task_setup(module, args, env, scene, room_objects, floor_record, agent_pos):
    blockers = [obj for obj in room_objects if module._is_floor_blocker(obj, floor_record.bbox_max[2])]
    wall_blockers = [obj for obj in room_objects if module._is_wall_occluder(obj)]
    trav_map = None
    trav_map_img = None
    if not args.disable_trav_map_occlusion_check:
        trav_map, trav_map_img = module._trav_map_floor_image(scene, floor_idx=0, scene_name=args.scene)
    free_positions = module._generate_free_positions(
        floor_record=floor_record,
        blockers=blockers,
        agent_pos=agent_pos,
        count=220,
        target_radius=module.DEFAULT_TARGET_RADIUS,
    )
    if not free_positions:
        raise RuntimeError(f"No free floor positions found for task '{module.__name__}'.")
    placeable_pool = module._build_placeable_category_pool(args.keys_json)
    if not placeable_pool:
        raise RuntimeError(f"No placeable categories found for task '{module.__name__}'.")
    preferred_camera_xy = None
    if args.agent_position is not None:
        preferred_camera_xy = [float(args.agent_position[0]), float(args.agent_position[1])]
    return blockers, wall_blockers, trav_map, trav_map_img, free_positions, placeable_pool, preferred_camera_xy


def main(argv=None):
    args = _build_parser().parse_args(argv)
    room_specs = _normalize_room_floor_args(args)

    resolved_seed = args.seed if args.seed is not None else random.SystemRandom().randrange(0, 2**32)

    config = distance_mod._build_config(args)
    try:
        env = distance_mod.og.Environment(configs=config)
        distance_mod._set_viewer_camera_fov()
        scene = env.scene
        room_results = []
        for room_name, floor_name in room_specs:
            print(
                f"[batch_mirror_merge] scene={args.scene} room={room_name} floor={floor_name} run_idx={args.run_idx}",
                flush=True,
            )
            try:
                room_results.append(_process_room(args, env, scene, room_name, floor_name, resolved_seed))
            except Exception as exc:
                _log_exception(f"room={room_name} floor={floor_name}", exc)
                room_run_root, question_json_root, render_root = _room_output_paths(
                    output_root=args.output_root,
                    scene_name=args.scene,
                    room_name=room_name,
                    run_idx=args.run_idx,
                )
                _write_room_attempted_marker(
                    room_run_root,
                    args.scene,
                    {
                        "scene": args.scene,
                        "room": room_name,
                        "floor": floor_name,
                        "seed": resolved_seed,
                        "status": "error",
                        "attempted": True,
                        "question_summary": {task_name: 0 for task_name, _ in TASK_SPECS},
                        "task_errors": {"__room__": f"{exc.__class__.__name__}: {exc}"},
                        "question_json_root": question_json_root,
                        "render_root": None if args.skip_render else render_root,
                    },
                )
                room_results.append(
                    {
                        "room": room_name,
                        "floor": floor_name,
                        "status": "error",
                        "question_summary": {task_name: 0 for task_name, _ in TASK_SPECS},
                        "task_errors": {"__room__": f"{exc.__class__.__name__}: {exc}"},
                    }
                )

        print(json.dumps({"scene": args.scene, "seed": resolved_seed, "run_idx": args.run_idx, "rooms": room_results}, indent=2, ensure_ascii=False))

        if not args.exit_on_finish:
            print("[batch_mirror_merge] Generation finished. Simulator is kept alive. Press Ctrl+C to exit.")
            try:
                while True:
                    distance_mod.og.sim.render()
                    distance_mod._step_sim(1)
            except KeyboardInterrupt:
                print("[batch_mirror_merge] Exit requested by user (Ctrl+C).")
    except Exception as exc:
        _log_exception("main", exc)
        raise


if __name__ == "__main__":
    main()
