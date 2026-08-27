from .common import *  # noqa: F401,F403
from . import common

TASK_NAME = "category_ambiguity"
FULL_SCENE = True
TASK_LABEL = "Category Ambiguity"
TASK_CASE = "semantic_fault"
TASK_FOCUS = "Count only the requested semantic target while rejecting visually similar confuser categories."
TASK_RULES = [
    "This case intentionally includes confuser categories that look semantically similar.",
    "Use the question's target category, not visually similar distractors, when counting.",
]


def build_env_config(
    scene_name,
    room_name,
    robot,
    objects=None,
    full_scene=False,
    payload=None,
):
    """Restore room context for the only Enumerative case without anchors."""
    config = common.build_env_config(
        scene_name,
        room_name,
        robot,
        objects,
        full_scene=full_scene,
        payload=payload,
    )
    scene = config.setdefault("scene", {})
    scene.pop("load_object_categories", None)
    # Keep walls while avoiding the usual ceiling / carpet rendering problems.
    scene["not_load_object_categories"] = ["ceilings", "carpet"]
    return config


def _look_at_quaternion(eye, target):
    forward = common.np.asarray(target, dtype=float) - common.np.asarray(eye, dtype=float)
    forward /= max(float(common.np.linalg.norm(forward)), 1e-8)
    right = common.np.cross(forward, common.np.asarray([0.0, 0.0, 1.0], dtype=float))
    if float(common.np.linalg.norm(right)) <= 1e-8:
        right = common.np.cross(forward, common.np.asarray([0.0, 1.0, 0.0], dtype=float))
    right /= max(float(common.np.linalg.norm(right)), 1e-8)
    true_up = common.np.cross(right, forward)
    matrix = common.np.asarray(
        [
            [right[0], true_up[0], -forward[0]],
            [right[1], true_up[1], -forward[1]],
            [right[2], true_up[2], -forward[2]],
        ],
        dtype=float,
    )
    trace = float(common.np.trace(matrix))
    if trace > 0.0:
        scale = (trace + 1.0) ** 0.5 * 2.0
        quat = [
            (matrix[2, 1] - matrix[1, 2]) / scale,
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[1, 0] - matrix[0, 1]) / scale,
            0.25 * scale,
        ]
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = (1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) ** 0.5 * 2.0
        quat = [
            0.25 * scale,
            (matrix[0, 1] + matrix[1, 0]) / scale,
            (matrix[0, 2] + matrix[2, 0]) / scale,
            (matrix[2, 1] - matrix[1, 2]) / scale,
        ]
    elif matrix[1, 1] > matrix[2, 2]:
        scale = (1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) ** 0.5 * 2.0
        quat = [
            (matrix[0, 1] + matrix[1, 0]) / scale,
            0.25 * scale,
            (matrix[1, 2] + matrix[2, 1]) / scale,
            (matrix[0, 2] - matrix[2, 0]) / scale,
        ]
    else:
        scale = (1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) ** 0.5 * 2.0
        quat = [
            (matrix[0, 2] + matrix[2, 0]) / scale,
            (matrix[1, 2] + matrix[2, 1]) / scale,
            0.25 * scale,
            (matrix[1, 0] - matrix[0, 1]) / scale,
        ]
    quat = common.np.asarray(quat, dtype=float)
    return quat / max(float(common.np.linalg.norm(quat)), 1e-8)


def _fallback_camera(payload, camera_info):
    pose = (camera_info or {}).get("camera_pose") or common.extract_primary_camera_pose(payload)
    eye = common.np.asarray(pose.get("position"), dtype=float)
    target = common.np.asarray(pose.get("look_target"), dtype=float)
    if eye.shape != (3,) or target.shape != (3,):
        return None
    horizontal_distance = float(common.np.linalg.norm(eye[:2] - target[:2]))
    pitch_deg = float(
        pose.get("pitch_deg")
        or common.np.degrees(
            common.np.arctan2(max(0.0, eye[2] - target[2]), max(horizontal_distance, 1e-6))
        )
    )
    if horizontal_distance >= 0.75 and pitch_deg <= 45.0:
        # Keep valid saved camera positions, but tilt Category Ambiguity views
        # slightly downward because the generated targets sit on the floor and
        # otherwise tend to be clipped by the bottom edge of the image.
        lowered_target = target.copy()
        lowered_target[2] = max(0.25, float(target[2]) - 0.4)
        if lowered_target[2] >= float(target[2]) - 1e-6:
            return None
        return {
            "position": eye.tolist(),
            "quaternion_xyzw": _look_at_quaternion(eye, lowered_target).tolist(),
            "look_target": lowered_target.tolist(),
            "source": "category_ambiguity_downward_tilt",
            "replaced_pitch_deg": pitch_deg,
            "look_target_z_offset_m": float(lowered_target[2] - target[2]),
        }

    render = ((payload.get("question_data") or {}).get("render") or {})
    resolved = render.get("resolved_objects") or {}
    generated_positions = []
    for group_name in ("targets", "confusers"):
        for item in resolved.get(group_name) or []:
            position = item.get("position") or item.get("requested_position")
            if isinstance(position, (list, tuple)) and len(position) == 3:
                generated_positions.append([float(value) for value in position])
    if generated_positions:
        generated = common.np.asarray(generated_positions, dtype=float)
        target = common.np.asarray(
            [
                float(common.np.mean(generated[:, 0])),
                float(common.np.mean(generated[:, 1])),
                max(0.18, float(common.np.percentile(generated[:, 2], 75)) + 0.12),
            ],
            dtype=float,
        )

    room_bbox = render.get("room_bbox") or {}
    bounds = room_bbox.get("expanded_bbox_world_xy") or room_bbox.get("bbox_world_xy")
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 4:
        return None
    xmin, ymin, xmax, ymax = [float(value) for value in bounds]
    inset = min(0.45, max(0.2, 0.12 * min(xmax - xmin, ymax - ymin)))
    x0, x1 = xmin + inset, xmax - inset
    y0, y1 = ymin + inset, ymax - inset
    if x1 <= x0 or y1 <= y0:
        return None
    candidates = [
        (x0, y0),
        (x0, y1),
        (x1, y0),
        (x1, y1),
        (x0, 0.5 * (y0 + y1)),
        (x1, 0.5 * (y0 + y1)),
        (0.5 * (x0 + x1), y0),
        (0.5 * (x0 + x1), y1),
    ]
    valid = [
        point
        for point in candidates
        if float(common.np.linalg.norm(common.np.asarray(point) - target[:2])) >= 1.25
    ]
    if not valid:
        return None
    chosen = max(
        valid,
        key=lambda point: (
            float(common.np.linalg.norm(common.np.asarray(point) - target[:2])),
            float(common.np.linalg.norm(common.np.asarray(point) - eye[:2])),
        ),
    )
    fallback_eye = common.np.asarray([chosen[0], chosen[1], max(float(eye[2]), 1.4)], dtype=float)
    return {
        "position": fallback_eye.tolist(),
        "quaternion_xyzw": _look_at_quaternion(fallback_eye, target).tolist(),
        "look_target": target.tolist(),
        "source": "category_ambiguity_room_edge_fallback",
        "replaced_pitch_deg": pitch_deg,
        "replaced_horizontal_distance_m": horizontal_distance,
    }


def postprocess_env(env, payload, camera_info=None):
    result = common.postprocess_env(env, payload, camera_info)
    category_counts = {}
    for obj in env.scene.objects:
        category = str(getattr(obj, "category", "") or "")
        category_counts[category] = category_counts.get(category, 0) + 1
    result.update(
        {
            "loaded_wall_count": category_counts.get("walls", 0),
            "loaded_scene_object_count": len(env.scene.objects),
            "loaded_scene_category_count": len(category_counts),
        }
    )
    camera_override = _fallback_camera(payload, camera_info)
    if camera_override is not None:
        result["camera_override"] = camera_override
    return result


def build_system_prompt(payload, threshold, min_steps, camera_info=None):
    return common.build_system_prompt_for_task(
        payload, threshold, min_steps, camera_info, TASK_LABEL, TASK_FOCUS, TASK_CASE, TASK_RULES
    )


def build_force_choice_prompt(payload, camera_info=None):
    return common.build_force_choice_prompt_for_task(payload, camera_info, TASK_LABEL, TASK_FOCUS, TASK_CASE, TASK_RULES)


def score(payload, final_answer, camera_info=None):
    return common.score_for_task(payload, final_answer, camera_info, TASK_LABEL, TASK_FOCUS, TASK_CASE, TASK_RULES)
